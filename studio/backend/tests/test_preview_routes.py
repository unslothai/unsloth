# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Security smoke for the public /p preview routes.

Exercises the route layer with a real ``preview_router`` while stubbing the
expensive model calls (``load_model`` / ``openai_chat_completions``). Covers the
public-surface guarantees: HMAC capability gating (a valid ``?k=`` token or
Bearer credential is required; missing/invalid/wrong-ref tokens 404 before any
model load), path-traversal rejection, request sanitization (tools / provider
routing / use_adapter / generation clamp), asset-path containment, the page CSP
+ no-referrer headers and HTML escaping, and that the preview lock is held until
a streaming response is fully drained.
"""

import asyncio
import json
from pathlib import Path
import sys
import types as _types

import pytest


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Mirror test_preview.py: the real `loggers` package pulls in heavy handlers.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

import routes.preview as preview
import utils.preview_token as preview_token
from models.inference import ChatCompletionRequest


# A fixed secret keeps signing deterministic and avoids touching auth.db.
_TEST_SECRET = b"unit-test-preview-secret-0123456789"


def _use_test_secret(monkeypatch) -> None:
    monkeypatch.setattr(
        preview_token, "get_or_create_preview_link_secret", lambda: _TEST_SECRET
    )


def _sig(ref: str) -> str:
    """Valid capability token for ``ref`` under the patched test secret."""
    return preview_token.sign_preview_ref(ref)


def _make_run(outputs: Path, name: str = "demorun") -> Path:
    run = outputs / name
    run.mkdir(parents = True)
    (run / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "HuggingFaceTB/SmolLM-135M"})
    )
    ckpt = run / "checkpoint-1"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")
    return run


@pytest.fixture
def captured():
    return {}


@pytest.fixture
def client(tmp_path, monkeypatch, captured):
    outputs = tmp_path / "outputs"
    _make_run(outputs)

    _use_test_secret(monkeypatch)

    # Public sharing on by default; reset the per-IP rate buckets each test.
    monkeypatch.setattr(preview, "get_preview_sharing_enabled", lambda: True)
    import utils.preview_rate_limit as _rl

    _rl.reset()

    # resolve_preview_checkpoint -> resolve_output_dir -> outputs_root().
    from utils.paths import storage_roots as _sr

    monkeypatch.setattr(_sr, "outputs_root", lambda: outputs)

    async def _fake_load_model(load_req, request, subject):
        captured["load_path"] = load_req.model_path
        return None

    async def _fake_chat(payload, request, subject):
        captured["payload"] = payload
        return {"ok": True}

    monkeypatch.setattr(preview, "load_model", _fake_load_model)
    monkeypatch.setattr(preview, "openai_chat_completions", _fake_chat)

    app = FastAPI()
    app.include_router(preview.router, prefix = "/p")
    app.dependency_overrides[preview.get_current_subject] = lambda: "admin"
    # raise_server_exceptions=False so a 5xx surfaces as a response, not a throw.
    return TestClient(app, raise_server_exceptions = False)


# ── Page rendering ────────────────────────────────────────────────────────


def test_page_renders_with_csp(client):
    r = client.get(f"/p/demorun?k={_sig('demorun')}")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    csp = r.headers.get("content-security-policy", "")
    assert "default-src 'self'" in csp
    assert "base-uri 'none'" in csp
    # Token rides in the query string; keep it out of the Referer header.
    assert r.headers.get("referrer-policy") == "no-referrer"


def test_page_escapes_title(tmp_path, monkeypatch, captured):
    outputs = tmp_path / "outputs"
    # Run dir name carries an HTML-special char; the page must escape it.
    _make_run(outputs, name = "a<b")
    _use_test_secret(monkeypatch)
    from utils.paths import storage_roots as _sr

    monkeypatch.setattr(_sr, "outputs_root", lambda: outputs)

    app = FastAPI()
    app.include_router(preview.router, prefix = "/p")
    c = TestClient(app, raise_server_exceptions = False)

    # Sign the decoded canonical ref ("a<b"), not the %-encoded path segment.
    r = c.get(f"/p/a%3Cb?k={_sig('a<b')}")
    assert r.status_code == 200
    assert "a<b" not in r.text
    assert "a&lt;b" in r.text


def test_models_endpoint_shape(client):
    r = client.get(f"/p/demorun/v1/models?k={_sig('demorun')}")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "demorun"
    assert body["data"][0]["owned_by"] == "unsloth-studio"


def test_list_previews_builds_urls(client, monkeypatch):
    monkeypatch.setattr(
        preview,
        "list_preview_targets",
        lambda: [{"ref": "demorun", "is_latest": True}],
    )
    r = client.get("/p")
    assert r.status_code == 200
    data = r.json()["data"]
    assert data[0]["url"].endswith("/p/demorun/v1")
    # The listing hands the authenticated owner a usable capability.
    assert data[0]["key"] == _sig("demorun")
    assert data[0]["share_url"].endswith(f"/p/demorun?k={_sig('demorun')}")


def test_list_previews_omits_capability_when_sharing_disabled(client, monkeypatch):
    monkeypatch.setattr(
        preview,
        "list_preview_targets",
        lambda: [{"ref": "demorun", "is_latest": True}],
    )
    monkeypatch.setattr(preview, "get_preview_sharing_enabled", lambda: False)
    r = client.get("/p")
    assert r.status_code == 200
    body = r.json()
    # Don't hand out credentials that 404; signal the disabled state instead.
    assert body["sharing_enabled"] is False
    assert body["data"][0]["key"] is None
    assert body["data"][0]["share_url"] is None


# ── Path traversal / containment ────────────────────────────────────────────


@pytest.mark.parametrize(
    "path",
    [
        "/p/..",  # parent segment as run
        "/p/%2e%2e/etc",  # encoded traversal
        "/p/..%2f..%2fetc/v1/models",  # encoded slash traversal
        "/p/does-not-exist",  # unknown run
    ],
)
def test_traversal_and_missing_rejected(client, path):
    r = client.get(path)
    assert r.status_code in (400, 404), (path, r.status_code)


def test_chat_traversal_rejected(client):
    r = client.post(
        "/p/..%2f..%2fetc/v1/chat/completions",
        json = {"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code in (400, 404)


# ── Asset containment ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "asset",
    [
        "../../../../etc/passwd",  # escapes dist
        "secrets.txt",  # non-allowlisted suffix
        "nope.png",  # allowlisted suffix but missing
    ],
)
def test_asset_path_contained(client, asset):
    r = client.get(f"/p/_assets/{asset}")
    assert r.status_code == 404


# ── Request sanitization ─────────────────────────────────────────────────────


def test_chat_payload_sanitized(client, captured):
    r = client.post(
        f"/p/demorun/v1/chat/completions?k={_sig('demorun')}",
        json = {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {"type": "function", "function": {"name": "rm", "parameters": {}}}
            ],
            "enable_tools": True,
            "enabled_tools": ["python"],
            "mcp_enabled": True,
            "bypass_permissions": True,
            "provider_id": "p1",
            "provider_type": "custom",
            "provider_base_url": "http://evil.example/v1",
            "external_model": "gpt-4o",
            "use_adapter": False,
            "confirm_tool_calls": True,
            "session_id": "abc",
            "rag_scope": {"project_id": "x"},
        },
    )
    assert r.status_code == 200
    p = captured["payload"]
    assert isinstance(p, ChatCompletionRequest)
    # Tools / code-exec off.
    assert p.tools is None
    assert p.enable_tools is False
    assert p.enabled_tools is None
    assert p.mcp_enabled is False
    assert p.bypass_permissions is False
    # Tool-loop levers neutralized regardless of the tool gate.
    assert p.confirm_tool_calls is False
    assert p.session_id is None
    assert p.rag_scope is None
    # Provider routing stripped so /p can't proxy an arbitrary endpoint.
    assert p.provider_id is None
    assert p.provider_type is None
    assert p.provider_base_url is None
    assert p.external_model is None
    # Adapter pinned on for LoRA: a caller can't flip the shared backend to base.
    assert p.use_adapter is True
    # Generation cost capped on this public surface (no override sent -> ceiling).
    assert p.max_tokens == preview._PREVIEW_MAX_OUTPUT_TOKENS
    assert p.max_completion_tokens == preview._PREVIEW_MAX_OUTPUT_TOKENS
    assert p.n == 1
    # Loads the resolved checkpoint dir, not an attacker-supplied path.
    assert captured["load_path"].endswith("demorun")


def test_merged_checkpoint_strips_use_adapter(tmp_path, monkeypatch, captured):
    # Merged (non-LoRA) checkpoint: no adapter to toggle, so use_adapter -> None.
    outputs = tmp_path / "outputs"
    merged = outputs / "mergedrun"
    merged.mkdir(parents = True)
    (merged / "config.json").write_text(json.dumps({"_name_or_path": "some/base"}))

    _use_test_secret(monkeypatch)
    from utils.paths import storage_roots as _sr

    monkeypatch.setattr(_sr, "outputs_root", lambda: outputs)

    async def _fake_load(load_req, request, subject):
        return None

    async def _fake_chat(payload, request, subject):
        captured["payload"] = payload
        return {"ok": True}

    monkeypatch.setattr(preview, "load_model", _fake_load)
    monkeypatch.setattr(preview, "openai_chat_completions", _fake_chat)

    app = FastAPI()
    app.include_router(preview.router, prefix = "/p")
    c = TestClient(app, raise_server_exceptions = False)
    r = c.post(
        f"/p/mergedrun/v1/chat/completions?k={_sig('mergedrun')}",
        json = {"messages": [{"role": "user", "content": "hi"}], "use_adapter": False},
    )
    assert r.status_code == 200
    assert captured["payload"].use_adapter is None


# ── Streaming lock lifetime ──────────────────────────────────────────────────


def test_streaming_holds_lock_until_drained(tmp_path, monkeypatch, captured):
    outputs = tmp_path / "outputs"
    _make_run(outputs)
    from utils.paths import storage_roots as _sr

    monkeypatch.setattr(_sr, "outputs_root", lambda: outputs)

    async def _fake_load_model(load_req, request, subject):
        return None

    async def _gen():
        yield b"data: {}\n\n"
        yield b"data: [DONE]\n\n"

    async def _fake_chat(payload, request, subject):
        return StreamingResponse(_gen())

    monkeypatch.setattr(preview, "load_model", _fake_load_model)
    monkeypatch.setattr(preview, "openai_chat_completions", _fake_chat)

    async def _run():
        assert not preview._preview_lock.locked()
        payload = ChatCompletionRequest(messages = [{"role": "user", "content": "hi"}])
        resp = await preview._serve_chat("demorun", None, payload, request = None)
        # Lock must still be held: a second checkpoint must not swap the backend
        # mid-stream.
        assert preview._preview_lock.locked()
        chunks = [c async for c in resp.body_iterator]
        # Released only after the stream fully drains.
        assert not preview._preview_lock.locked()
        return chunks

    chunks = asyncio.run(_run())
    assert any(b"[DONE]" in c for c in chunks)
    assert not preview._preview_lock.locked()


# ── Capability gating ────────────────────────────────────────────────────────


def test_chat_without_token_404_and_no_load(client, captured):
    r = client.post(
        "/p/demorun/v1/chat/completions",
        json = {"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 404
    # Verified before any model work: nothing loaded, nothing generated.
    assert "load_path" not in captured
    assert "payload" not in captured


def test_chat_with_invalid_token_404(client, captured):
    r = client.post(
        "/p/demorun/v1/chat/completions?k=not-a-valid-token",
        json = {"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 404
    assert "load_path" not in captured


def test_token_for_other_ref_rejected(client, captured):
    # A capability minted for a different ref must not unlock demorun.
    r = client.post(
        f"/p/demorun/v1/chat/completions?k={_sig('otherrun')}",
        json = {"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 404
    assert "load_path" not in captured


def test_models_without_token_404(client):
    assert client.get("/p/demorun/v1/models").status_code == 404


def test_page_without_token_404(client):
    assert client.get("/p/demorun").status_code == 404


def test_checkpoint_route_with_valid_sig(client, captured):
    # Nested ref: the signed/verified/resolved canonical ref is "run/checkpoint".
    sig = _sig("demorun/checkpoint-1")
    r = client.post(
        f"/p/demorun/checkpoint-1/v1/chat/completions?k={sig}",
        json = {"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert captured["load_path"].endswith("checkpoint-1")


def test_checkpoint_token_does_not_unlock_bare_run(client, captured):
    # A token minted for the nested checkpoint must not unlock the run ref.
    r = client.post(
        f"/p/demorun/v1/chat/completions?k={_sig('demorun/checkpoint-1')}",
        json = {"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 404
    assert "load_path" not in captured


def test_bearer_token_accepted(client, captured):
    # OpenAI-compatible clients pass the capability as the api_key (Bearer header).
    r = client.post(
        "/p/demorun/v1/chat/completions",
        headers = {"Authorization": f"Bearer {_sig('demorun')}"},
        json = {"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert captured["load_path"].endswith("demorun")


def test_generation_clamp_caps_overrides(client, captured):
    r = client.post(
        f"/p/demorun/v1/chat/completions?k={_sig('demorun')}",
        json = {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 999999,
            "max_completion_tokens": 888888,
            "n": 64,
        },
    )
    assert r.status_code == 200
    p = captured["payload"]
    assert p.max_tokens == preview._PREVIEW_MAX_OUTPUT_TOKENS
    assert p.max_completion_tokens == preview._PREVIEW_MAX_OUTPUT_TOKENS
    assert p.n == 1


def test_generation_clamp_honors_lower_legacy_max_tokens(client, captured):
    # A caller asking for fewer tokens via the legacy field must not be bumped up
    # to the ceiling: _effective_max_tokens prefers max_completion_tokens, so both
    # fields have to carry the lower value.
    r = client.post(
        f"/p/demorun/v1/chat/completions?k={_sig('demorun')}",
        json = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16},
    )
    assert r.status_code == 200
    p = captured["payload"]
    assert p.max_tokens == 16
    assert p.max_completion_tokens == 16


def test_generation_clamp_honors_lower_completion_tokens(client, captured):
    r = client.post(
        f"/p/demorun/v1/chat/completions?k={_sig('demorun')}",
        json = {
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 32,
        },
    )
    assert r.status_code == 200
    p = captured["payload"]
    assert p.max_tokens == 32
    assert p.max_completion_tokens == 32


# ── Public-sharing kill switch ───────────────────────────────────────────────


def test_chat_blocked_when_sharing_disabled(client, monkeypatch, captured):
    # Admin turned public sharing off: even a valid token 404s, with no model load.
    monkeypatch.setattr(preview, "get_preview_sharing_enabled", lambda: False)
    r = client.post(
        f"/p/demorun/v1/chat/completions?k={_sig('demorun')}",
        json = {"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 404
    assert "load_path" not in captured


def test_page_blocked_when_sharing_disabled(client, monkeypatch):
    monkeypatch.setattr(preview, "get_preview_sharing_enabled", lambda: False)
    assert client.get(f"/p/demorun?k={_sig('demorun')}").status_code == 404


# ── Rate limiting ────────────────────────────────────────────────────────────


def test_chat_rate_limited_returns_429(client, monkeypatch):
    import utils.preview_rate_limit as rl

    monkeypatch.setattr(rl, "_MAX_REQUESTS", 2)
    rl.reset()
    url = f"/p/demorun/v1/chat/completions?k={_sig('demorun')}"
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert client.post(url, json = body).status_code == 200
    assert client.post(url, json = body).status_code == 200
    r = client.post(url, json = body)
    assert r.status_code == 429
    assert r.headers.get("retry-after")
