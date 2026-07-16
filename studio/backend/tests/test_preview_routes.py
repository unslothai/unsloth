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
    monkeypatch.setattr(preview_token, "get_or_create_preview_link_secret", lambda: _TEST_SECRET)


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

    monkeypatch.setattr(preview, "load_model_for_preview", _fake_load_model)
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


def test_page_renders_friendly_busy_message(client):
    r = client.get(f"/p/demorun?k={_sig('demorun')}")
    assert "Studio is currently using another model" in r.text


def test_page_escapes_title(tmp_path, monkeypatch, captured):
    outputs = tmp_path / "outputs"
    # Run dir name carries an HTML-special char; the page must escape it.
    _make_run(outputs, name = "a<b")
    _use_test_secret(monkeypatch)
    monkeypatch.setattr(preview, "get_preview_sharing_enabled", lambda: True)
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
            "tools": [{"type": "function", "function": {"name": "rm", "parameters": {}}}],
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
            "enable_thinking": True,
            "reasoning_effort": "high",
            "preserve_thinking": True,
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
    assert p.enable_thinking is False
    assert p.reasoning_effort == "none"
    assert p.preserve_thinking is False
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
    monkeypatch.setattr(preview, "get_preview_sharing_enabled", lambda: True)
    from utils.paths import storage_roots as _sr

    monkeypatch.setattr(_sr, "outputs_root", lambda: outputs)

    async def _fake_load(load_req, request, subject):
        return None

    async def _fake_chat(payload, request, subject):
        captured["payload"] = payload
        return {"ok": True}

    monkeypatch.setattr(preview, "load_model_for_preview", _fake_load)
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

    monkeypatch.setattr(preview, "load_model_for_preview", _fake_load_model)
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
        json = {"messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 32},
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


# ── Model-slot guard ─────────────────────────────────────────────────────────

from types import SimpleNamespace

from fastapi import HTTPException

import routes.inference as inference
from core.inference import llama_keepwarm
from models.inference import LoadRequest


@pytest.fixture
def slot_state():
    def _reset():
        with inference._preview_slot_lock:
            inference._preview_resident_ident = None

    _reset()
    yield
    _reset()


@pytest.fixture
def fake_slot(slot_state, monkeypatch):
    state = {"ident": None, "loads": []}

    async def _fake_impl(load_req, fastapi_request, subject):
        state["loads"].append(load_req.model_path)
        if state.get("fail_load"):
            # Mirror a load that tears the old model down and then fails.
            state["ident"] = None
            raise HTTPException(status_code = 500, detail = "load failed")
        state["ident"] = load_req.model_path

    monkeypatch.setattr(inference, "_load_model_impl", _fake_impl)
    monkeypatch.setattr(inference, "_loaded_slot_ident", lambda: state["ident"])
    monkeypatch.setattr(
        llama_keepwarm, "other_inference_request_count", lambda **kw: state.get("busy", 0)
    )
    monkeypatch.setattr(
        llama_keepwarm,
        "other_preview_inflight_count",
        lambda **kw: state.get("other_previews", 0),
    )
    return state


def test_preview_load_refused_when_studio_model_is_loaded(fake_slot):
    # An owner model stays protected even while idle: a preview must not swap it
    # out merely because the user is between Studio messages.
    fake_slot["ident"] = "owner-model"

    async def _run():
        with pytest.raises(HTTPException) as exc:
            await inference.load_model_for_preview(
                LoadRequest(model_path = "/outputs/run/ckpt"),
                SimpleNamespace(app = None),
                "admin",
            )
        return exc.value

    exc = asyncio.run(_run())
    assert exc.status_code == 503
    assert exc.headers.get("Retry-After")
    assert fake_slot["loads"] == []
    assert fake_slot["ident"] == "owner-model"


def test_preview_load_allowed_when_checkpoint_already_loaded_and_idle(fake_slot):
    # A resident model that already serves this exact checkpoint is borrowed as-is:
    # no _load_model_impl call (which could reload/reconfigure a Studio-owned GGUF
    # whose live settings differ from the bare preview request, #5401), and a
    # Studio-owned model is not silently claimed for preview.
    fake_slot["ident"] = "/outputs/run/ckpt"

    asyncio.run(
        inference.load_model_for_preview(
            LoadRequest(model_path = "/outputs/run/ckpt"),
            SimpleNamespace(app = None),
            "admin",
        )
    )
    assert fake_slot["loads"] == []  # borrowed, not reloaded
    assert not inference._is_preview_resident("/outputs/run/ckpt")  # stays Studio-owned


def test_preview_load_refused_on_same_checkpoint_when_studio_busy(fake_slot):
    # same_target doesn't guarantee a no-op reload (GGUF settings can still force
    # a restart), so busy Studio traffic blocks even a same-checkpoint request.
    fake_slot["ident"] = "/outputs/run/ckpt"
    fake_slot["busy"] = 1

    async def _run():
        with pytest.raises(HTTPException) as exc:
            await inference.load_model_for_preview(
                LoadRequest(model_path = "/outputs/run/ckpt"),
                SimpleNamespace(app = None),
                "admin",
            )
        return exc.value

    exc = asyncio.run(_run())
    assert exc.status_code == 503
    assert fake_slot["loads"] == []


def test_preview_waiters_do_not_trip_busy_guard(fake_slot):
    # Busy traffic fully accounted for by other previews isn't foreign traffic, so
    # the request proceeds; the same-checkpoint model is then borrowed (no reload).
    fake_slot["ident"] = "/outputs/run/ckpt"
    fake_slot["busy"] = 1
    fake_slot["other_previews"] = 1
    asyncio.run(
        inference.load_model_for_preview(
            LoadRequest(model_path = "/outputs/run/ckpt"),
            SimpleNamespace(app = None),
            "admin",
        )
    )
    assert fake_slot["loads"] == []  # borrowed same-checkpoint, not reloaded


def test_preview_can_swap_out_prior_preview_model(fake_slot):
    for path in ("/outputs/run/ckpt-a", "/outputs/run/ckpt-b"):
        asyncio.run(
            inference.load_model_for_preview(
                LoadRequest(model_path = path), SimpleNamespace(app = None), "admin"
            )
        )
    assert fake_slot["loads"] == ["/outputs/run/ckpt-a", "/outputs/run/ckpt-b"]
    assert fake_slot["ident"] == "/outputs/run/ckpt-b"


def test_preview_swap_refused_while_studio_traffic_in_flight(fake_slot):
    asyncio.run(
        inference.load_model_for_preview(
            LoadRequest(model_path = "/outputs/run/ckpt-a"), SimpleNamespace(app = None), "admin"
        )
    )
    fake_slot["busy"] = 1

    async def _run():
        with pytest.raises(HTTPException) as exc:
            await inference.load_model_for_preview(
                LoadRequest(model_path = "/outputs/run/ckpt-b"),
                SimpleNamespace(app = None),
                "admin",
            )
        return exc.value

    exc = asyncio.run(_run())
    assert exc.status_code == 503
    assert fake_slot["ident"] == "/outputs/run/ckpt-a"


def test_borrowed_studio_model_stays_studio_owned(fake_slot):
    fake_slot["ident"] = "/outputs/run/ckpt-a"
    asyncio.run(
        inference.load_model_for_preview(
            LoadRequest(model_path = "/outputs/run/ckpt-a"), SimpleNamespace(app = None), "admin"
        )
    )

    async def _run():
        with pytest.raises(HTTPException) as exc:
            await inference.load_model_for_preview(
                LoadRequest(model_path = "/outputs/run/ckpt-b"),
                SimpleNamespace(app = None),
                "admin",
            )
        return exc.value

    exc = asyncio.run(_run())
    assert exc.status_code == 503
    assert fake_slot["ident"] == "/outputs/run/ckpt-a"


def test_studio_noop_load_reclaims_preview_marker(slot_state, monkeypatch):
    # A Studio /load that returns already_loaded must still claim a preview-owned
    # checkpoint, or a later preview could swap out the model Studio just loaded.
    inference._set_preview_resident("mymodel")
    backend = SimpleNamespace(active_model_name = "mymodel", models = {})
    monkeypatch.setattr(inference, "get_inference_backend", lambda: backend)
    monkeypatch.setattr(
        inference, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False)
    )
    monkeypatch.setattr(
        inference,
        "_resolve_model_identifier_for_request",
        lambda req, operation: ("mymodel", "mymodel", False),
    )
    monkeypatch.setattr(inference, "resolve_effective_chat_template_override", lambda **kw: None)
    monkeypatch.setattr(inference, "load_inference_config", lambda name: {})
    monkeypatch.setattr(
        inference,
        "_detect_safetensors_features",
        lambda backend, tpl: {
            "supports_reasoning": False,
            "reasoning_style": "enable_thinking",
            "reasoning_always_on": False,
            "supports_preserve_thinking": False,
            "supports_tools": False,
        },
    )
    monkeypatch.setattr(inference, "_resolve_loaded_trust_remote_code", lambda *a: False)

    resp = asyncio.run(
        inference._load_model_impl(
            LoadRequest(model_path = "mymodel"), SimpleNamespace(app = None), "admin"
        )
    )
    assert resp.status == "already_loaded"
    assert not inference._is_preview_resident("mymodel")


def test_failed_load_preserves_preview_marker(slot_state):
    # A real load that fails validation before touching the backend must not
    # clear the resident marker: the old preview-owned checkpoint is still
    # loaded, unchanged.
    inference._set_preview_resident("/outputs/run/ckpt-a")

    async def _run():
        with pytest.raises(HTTPException) as exc:
            await inference._load_model_impl(
                LoadRequest(
                    model_path = "/outputs/run/ckpt-a", llama_extra_args = ["--host", "0.0.0.0"]
                ),
                SimpleNamespace(app = None),
                "admin",
            )
        return exc.value

    exc = asyncio.run(_run())
    assert exc.status_code == 400
    assert inference._is_preview_resident("/outputs/run/ckpt-a")


def test_chat_returns_503_when_model_busy(tmp_path, monkeypatch, slot_state):
    # Route-level: the guard's 503 (with Retry-After) reaches the public caller
    # through the real _serve_chat, which must release the preview lock.
    outputs = tmp_path / "outputs"
    _make_run(outputs)
    _use_test_secret(monkeypatch)
    monkeypatch.setattr(preview, "get_preview_sharing_enabled", lambda: True)
    from utils.paths import storage_roots as _sr

    monkeypatch.setattr(_sr, "outputs_root", lambda: outputs)

    async def _fake_impl(load_req, fastapi_request, subject):
        raise AssertionError("must not load while busy")

    monkeypatch.setattr(inference, "_load_model_impl", _fake_impl)
    monkeypatch.setattr(inference, "_loaded_slot_ident", lambda: "owner-model")
    monkeypatch.setattr(llama_keepwarm, "other_inference_request_count", lambda **kw: 1)

    app = FastAPI()
    app.include_router(preview.router, prefix = "/p")
    c = TestClient(app, raise_server_exceptions = False)
    r = c.post(
        f"/p/demorun/v1/chat/completions?k={_sig('demorun')}",
        json = {"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 503
    assert r.headers.get("retry-after")
    assert not preview._preview_lock.locked()


def test_preview_load_refused_during_sidecar_swap(fake_slot, monkeypatch):
    # Preview loads bypass load_model(), so they must re-apply its sidecar-install
    # guard: a public preview must not complete a load mid transformers install.
    import utils.transformers_version as tv

    monkeypatch.setattr(tv, "sidecar_swap_in_progress", lambda: True)

    async def _run():
        with pytest.raises(HTTPException) as exc:
            await inference.load_model_for_preview(
                LoadRequest(model_path = "/outputs/run/ckpt"),
                SimpleNamespace(app = None, scope = {"path": "/p/run/v1/chat/completions"}),
                "admin",
            )
        return exc.value

    exc = asyncio.run(_run())
    assert exc.status_code == 409
    assert fake_slot["loads"] == []  # never touched the backend


def test_preview_refusal_untracks_and_balances_counters(fake_slot):
    # A refused preview never touches the model, so it must untrack itself (no
    # keep-warm activity stamp) and balance BOTH in-flight counters -- otherwise
    # public /p spam could pin an idle Studio model and skew the busy guard.
    fake_slot["ident"] = "owner-model"  # a different model is resident -> refuse
    llama_keepwarm._inflight = 1
    llama_keepwarm._preview_inflight = 1
    scope = {"type": "http", "path": "/p/run/v1/chat/completions"}

    async def _run():
        with pytest.raises(HTTPException) as exc:
            await inference.load_model_for_preview(
                LoadRequest(model_path = "/outputs/run/ckpt"),
                SimpleNamespace(app = None, scope = scope),
                "admin",
            )
        return exc.value

    exc = asyncio.run(_run())
    assert exc.status_code == 503
    assert scope.get(llama_keepwarm._UNTRACKED_SCOPE_KEY) is True
    assert llama_keepwarm._inflight == 0
    assert llama_keepwarm._preview_inflight == 0
    assert fake_slot["loads"] == []
    llama_keepwarm._inflight = 0
    llama_keepwarm._preview_inflight = 0


def test_preview_reuses_own_checkpoint_without_reload(fake_slot):
    # A preview re-requesting the checkpoint it already loaded borrows it (no second
    # _load_model_impl call, which could reconfigure it) and keeps preview ownership.
    asyncio.run(
        inference.load_model_for_preview(
            LoadRequest(model_path = "/outputs/run/ckpt-a"), SimpleNamespace(app = None), "admin"
        )
    )
    assert fake_slot["loads"] == ["/outputs/run/ckpt-a"]
    assert inference._is_preview_resident("/outputs/run/ckpt-a")
    asyncio.run(
        inference.load_model_for_preview(
            LoadRequest(model_path = "/outputs/run/ckpt-a"), SimpleNamespace(app = None), "admin"
        )
    )
    assert fake_slot["loads"] == ["/outputs/run/ckpt-a"]  # borrowed, no second load
    assert inference._is_preview_resident("/outputs/run/ckpt-a")


def test_generation_paths_rely_on_middleware_claim():
    # Native generation entries no longer claim the resident model directly: claiming
    # before generation could strand a preview-owned checkpoint as Studio-owned if the
    # generation (or an upstream llama-server call) then fails. The keep-warm
    # middleware claims the slot on a successful 2xx response instead. generate_stream
    # additionally enforces the preview-swap reject, since it does not run through
    # _maybe_auto_switch_model where that guard otherwise fires.
    import inspect

    stream_src = inspect.getsource(inference.generate_stream)
    audio_src = inspect.getsource(inference.generate_audio)
    assert "_claim_slot_for_non_preview(" not in stream_src
    assert "_claim_slot_for_non_preview(" not in audio_src
    assert "_PREVIEW_SWAP_REJECT_SCOPE_KEY" in stream_src


def test_preview_swap_proceeds_when_studio_request_only_queued(fake_slot):
    # Codex P2 (round 19): a Studio request merely QUEUED on the lifecycle gate (in
    # _pending, not yet generating) must NOT block a preview swap. The keep-warm
    # middleware bumps _pending before FastAPI auth, so counting it would let an
    # unauthenticated probe starve public previews. The queued request is instead
    # protected by the swap-reject: when it wakes after the load it sees the swap
    # generation advance and gets a retryable 503 (in _maybe_auto_switch_model /
    # generate_stream) rather than running against the swapped-in checkpoint. (In-flight
    # Studio traffic still blocks -- see test_preview_swap_refused_while_studio_traffic_in_flight.)
    from core.inference import llama_keepwarm as kw

    asyncio.run(
        inference.load_model_for_preview(
            LoadRequest(model_path = "/outputs/run/ckpt-a"),
            SimpleNamespace(app = None, scope = {"path": "/p/a/v1/chat/completions"}),
            "admin",
        )
    )
    assert inference._is_preview_resident("/outputs/run/ckpt-a")
    kw._pending = 1  # a queued Studio (non-preview) request, not yet in flight
    kw._preview_pending = 0
    try:
        asyncio.run(
            inference.load_model_for_preview(
                LoadRequest(model_path = "/outputs/run/ckpt-b"),
                SimpleNamespace(app = None, scope = {"path": "/p/b/v1/chat/completions"}),
                "admin",
            )
        )
        # The swap proceeds: B loads and becomes the preview-owned resident.
        assert fake_slot["loads"] == ["/outputs/run/ckpt-a", "/outputs/run/ckpt-b"]
        assert inference._is_preview_resident("/outputs/run/ckpt-b")
        assert not inference._is_preview_resident("/outputs/run/ckpt-a")
    finally:
        kw._pending = 0
        kw._preview_pending = 0
        kw._inflight = 0
        kw._preview_inflight = 0


def test_claim_slot_for_non_preview_gates_on_preview_path(slot_state):
    # A non-preview local request claims the resident model (clears the marker); a
    # /p preview (including an audio preview that reaches generate_audio via
    # openai_chat_completions) keeps its ownership.
    inference._set_preview_resident("/outputs/run/ckpt")
    inference._claim_slot_for_non_preview(
        SimpleNamespace(scope = {"path": "/api/inference/generate/stream"})
    )
    assert not inference._is_preview_resident("/outputs/run/ckpt")

    inference._set_preview_resident("/outputs/run/ckpt")
    inference._claim_slot_for_non_preview(
        SimpleNamespace(scope = {"path": "/p/run/v1/chat/completions"})
    )
    assert inference._is_preview_resident("/outputs/run/ckpt")
    inference._set_preview_resident(None)


def test_preview_same_target_compares_case_sensitively(fake_slot):
    # Two checkpoints differing only by case are different files on a case-sensitive
    # filesystem, so a preview for /outputs/run must not borrow a resident /outputs/Run
    # (serving the wrong checkpoint); it must load the requested path.
    asyncio.run(
        inference.load_model_for_preview(
            LoadRequest(model_path = "/outputs/Run"),
            SimpleNamespace(app = None, scope = {"path": "/p/a/v1/chat/completions"}),
            "admin",
        )
    )
    assert fake_slot["loads"] == ["/outputs/Run"]
    assert inference._is_preview_resident("/outputs/Run")
    asyncio.run(
        inference.load_model_for_preview(
            LoadRequest(model_path = "/outputs/run"),
            SimpleNamespace(app = None, scope = {"path": "/p/b/v1/chat/completions"}),
            "admin",
        )
    )
    # Not borrowed: the differently-cased path was actually loaded, not served stale.
    assert fake_slot["loads"] == ["/outputs/Run", "/outputs/run"]
    assert inference._is_preview_resident("/outputs/run")


def test_preview_reload_failure_restores_prior_ownership(slot_state, monkeypatch):
    # _load_model_impl clears the preview marker mid-load (reclaiming the slot for
    # Studio). If it then fails while the prior preview model is still resident, the
    # marker must be restored to that model, not left cleared -- otherwise the next
    # preview for a different checkpoint sees a still-preview model as Studio-owned and
    # 503s.
    resident = {"ident": "/outputs/run/ckpt-A"}
    monkeypatch.setattr(inference, "_loaded_slot_ident", lambda: resident["ident"])
    monkeypatch.setattr(llama_keepwarm, "other_inference_request_count", lambda **kw: 0)
    monkeypatch.setattr(llama_keepwarm, "other_preview_inflight_count", lambda **kw: 0)
    llama_keepwarm._pending = 0
    llama_keepwarm._preview_pending = 0
    inference._set_preview_resident("/outputs/run/ckpt-A")  # A is preview-owned

    async def _clear_then_fail(load_req, fastapi_request, subject):
        inference._set_preview_resident(None)  # mirror _load_model_impl reclaiming slot
        raise HTTPException(status_code = 500, detail = "spawn failed")  # A still resident

    monkeypatch.setattr(inference, "_load_model_impl", _clear_then_fail)

    async def _run():
        with pytest.raises(HTTPException) as exc:
            await inference.load_model_for_preview(
                LoadRequest(model_path = "/outputs/run/ckpt-B"),
                SimpleNamespace(app = None, scope = {"path": "/p/b/v1/chat/completions"}),
                "admin",
            )
        return exc.value

    exc = asyncio.run(_run())
    assert exc.status_code == 500
    # A is still resident and its preview ownership is restored (not Studio-owned).
    assert inference._is_preview_resident("/outputs/run/ckpt-A")


def test_generate_stream_image_on_text_model_preserves_preview_marker(slot_state, monkeypatch):
    # An image request against a text-only resident model is rejected (400) before
    # generation. The ownership claim must run only after that modality check, so a
    # preview-owned checkpoint is not stranded as Studio-owned by a request that
    # never generated.
    from models.inference import GenerateRequest

    backend = SimpleNamespace(
        active_model_name = "/outputs/run/ckpt-a",
        models = {"/outputs/run/ckpt-a": {"is_vision": False}},
    )
    monkeypatch.setattr(inference, "get_inference_backend", lambda: backend)
    inference._set_preview_resident("/outputs/run/ckpt-a")
    req = GenerateRequest(messages = [{"role": "user", "content": "hi"}], image_base64 = "aGVsbG8=")
    fake_req = SimpleNamespace(scope = {"path": "/api/inference/generate/stream"})
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference.generate_stream(req, fake_req, "tester"))
    assert exc.value.status_code == 400
    # No direct claim runs, so ownership is intact after the failed modality check.
    assert inference._is_preview_resident("/outputs/run/ckpt-a")


def test_generate_stream_rejects_after_preview_swap(slot_state, monkeypatch):
    # Codex P1: generate_stream does not run through _maybe_auto_switch_model, so it
    # must enforce the preview-swap reject itself. If a public preview loaded a
    # different checkpoint (GGUF or Unsloth/LoRA) while this native request waited on
    # the keep-warm gate (the middleware flagged the scope), generate_stream must 503
    # instead of streaming from the preview's model.
    from models.inference import GenerateRequest
    from core.inference import llama_keepwarm as kw

    backend = SimpleNamespace(
        active_model_name = "/outputs/run/ckpt-a",
        models = {"/outputs/run/ckpt-a": {"is_vision": False}},
    )
    monkeypatch.setattr(inference, "get_inference_backend", lambda: backend)
    req = GenerateRequest(messages = [{"role": "user", "content": "hi"}])
    fake_req = SimpleNamespace(
        scope = {
            "path": "/api/inference/generate/stream",
            kw._PREVIEW_SWAP_REJECT_SCOPE_KEY: True,
        }
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference.generate_stream(req, fake_req, "tester"))
    assert exc.value.status_code == 503


def test_generate_stream_swap_reject_precedes_no_model_check(slot_state, monkeypatch):
    # Codex P2: the preview-swap reject must run before the loaded-model / capability
    # checks. If a preview loaded a GGUF (no active_model_name) while this request was
    # queued, the request must get the retryable 503, not a hard 400 "No model loaded"
    # from a backend-state check against the swapped-in model.
    from models.inference import GenerateRequest
    from core.inference import llama_keepwarm as kw

    backend = SimpleNamespace(active_model_name = None, models = {})
    monkeypatch.setattr(inference, "get_inference_backend", lambda: backend)
    req = GenerateRequest(messages = [{"role": "user", "content": "hi"}])
    fake_req = SimpleNamespace(
        scope = {
            "path": "/api/inference/generate/stream",
            kw._PREVIEW_SWAP_REJECT_SCOPE_KEY: True,
        }
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference.generate_stream(req, fake_req, "tester"))
    assert exc.value.status_code == 503  # retryable swap reject, not the 400 no-model


def test_openai_stream_error_sse_flags_current_response_failed(slot_state):
    # Codex P2: every OpenAI-family streaming error (local chat/completions/responses,
    # admission failures, passthrough relays) is emitted through _openai_stream_error_sse
    # after the 200 headers, so it must flag the current response failed via the
    # contextvar the middleware sets -- otherwise the middleware claims a preview-owned
    # model for Studio on a stream that errored.
    from core.inference import llama_keepwarm as kw

    scope = {"type": "http", "path": "/v1/chat/completions"}
    kw.set_current_response_scope(scope)
    try:
        out = inference._openai_stream_error_sse({"error": {"message": "boom"}})
        assert "boom" in out  # still returns the error SSE
        assert scope.get(kw._RESPONSE_FAILED_SCOPE_KEY) is True
    finally:
        kw.set_current_response_scope(None)


def test_anthropic_stream_error_event_flags_current_response_failed(slot_state):
    # Codex P2: the local /v1/messages and passthrough Anthropic stream errors emit via
    # _anthropic_stream_error_event after the 200 headers, so it must flag the current
    # response failed (via the contextvar the middleware sets), or the middleware would
    # claim a preview-owned model for Studio on a failed stream.
    from core.inference import llama_keepwarm as kw

    scope = {"type": "http", "path": "/v1/messages"}
    kw.set_current_response_scope(scope)
    try:
        ev = inference._anthropic_stream_error_event(RuntimeError("boom"), force = True)
        assert ev is not None
        assert scope.get(kw._RESPONSE_FAILED_SCOPE_KEY) is True
    finally:
        kw.set_current_response_scope(None)


def test_preview_load_assembly_failure_marks_new_checkpoint(fake_slot, monkeypatch):
    # Codex P2: if _load_model_impl makes checkpoint B resident and then raises while
    # assembling the load response (loaded_ok stays false), the slot holds B, so it
    # must be marked preview-owned -- it was loaded only by this preview -- not
    # restored to the prior checkpoint A (which would leave B looking Studio-owned).
    inference._set_preview_resident("/outputs/run/ckpt-a")  # prior preview A

    async def _load_then_assembly_fail(load_req, fastapi_request, subject):
        fake_slot["loads"].append(load_req.model_path)
        fake_slot["ident"] = load_req.model_path  # B is now the resident slot
        raise HTTPException(status_code = 500, detail = "assembly failed after load")

    monkeypatch.setattr(inference, "_load_model_impl", _load_then_assembly_fail)

    with pytest.raises(HTTPException):
        asyncio.run(
            inference.load_model_for_preview(
                LoadRequest(model_path = "/outputs/run/ckpt-b"),
                SimpleNamespace(app = None, scope = {"path": "/p/b/v1/chat/completions"}),
                "admin",
            )
        )
    assert inference._is_preview_resident("/outputs/run/ckpt-b")  # B is preview-owned
    assert not inference._is_preview_resident("/outputs/run/ckpt-a")


def test_same_loaded_identifier_is_filesystem_aware():
    # The already-loaded dedup must be filesystem-aware: on a case-sensitive filesystem
    # two checkpoint paths differing only by case are different models and must reload.
    import os.path

    assert inference._same_loaded_identifier("/outputs/Run", "/outputs/Run")
    assert not inference._same_loaded_identifier(None, "/outputs/Run")
    assert not inference._same_loaded_identifier("", "/outputs/Run")
    # Case-distinct local paths dedup only where the filesystem is case-insensitive.
    expected = os.path.normcase("/outputs/Run") == os.path.normcase("/outputs/run")
    assert inference._same_loaded_identifier("/outputs/Run", "/outputs/run") is expected
    # Hugging Face repo IDs are resolved case-insensitively, so they still dedup
    # regardless of case (no unnecessary reload of the same repo).
    assert inference._same_loaded_identifier("Unsloth/Foo", "unsloth/foo")
    assert inference._same_loaded_identifier("unsloth/foo", "unsloth/foo")
    assert not inference._same_loaded_identifier("unsloth/foo", "unsloth/bar")


def test_same_loaded_identifier_treats_existing_relative_paths_as_filesystem(tmp_path, monkeypatch):
    # Codex P2: an existing RELATIVE checkpoint path is a local filesystem path (as
    # ModelConfig.from_identifier resolves it), not a repo id, so it must compare
    # filesystem-aware. Two relative paths differing only by case must not dedup to the
    # already-loaded fast path on a case-sensitive filesystem.
    import os

    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "Run").mkdir(parents = True)
    # Only "outputs/Run" exists, so this is classified local and compared normcase.
    expected = os.path.normcase("outputs/Run") == os.path.normcase("outputs/run")
    assert inference._same_loaded_identifier("outputs/Run", "outputs/run") is expected
    # An identical existing relative path still dedups.
    assert inference._same_loaded_identifier("outputs/Run", "outputs/Run")


def test_completions_embeddings_defer_claim_to_middleware():
    # Codex P2: /v1/completions and /v1/embeddings must not claim the resident model
    # before the llama-server call. llama-server can still reject a valid body (e.g. a
    # no-pooling error for embeddings against a non-embedding GGUF), which would strand
    # a preview-owned checkpoint as Studio-owned; the keep-warm middleware claims on a
    # successful 2xx instead.
    import inspect
    for fn in (inference.openai_completions, inference.openai_embeddings):
        assert "_claim_slot_for_non_preview(" not in inspect.getsource(fn), fn.__name__


def test_load_restores_preview_marker_on_late_companion_reject():
    # Codex P2: _load_model_impl clears the preview marker before the native vision
    # companion validation, which can still reject after the clear. On that late reject
    # the resident preview model was never torn down, so preview ownership is restored
    # rather than left marked Studio-owned.
    import inspect

    src = inspect.getsource(inference._load_model_impl)
    assert "_prior_preview_marker = _get_preview_resident()" in src
    assert "_set_preview_resident(_prior_preview_marker)" in src


def _run_middleware(app, path):
    mw = llama_keepwarm.LlamaKeepWarmMiddleware(app)
    scope = {"type": "http", "method": "POST", "path": path}
    sent = []

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def _send(msg):
        sent.append(msg)

    asyncio.run(mw(scope, _receive, _send))
    return sent


def _reset_keepwarm_counters():
    llama_keepwarm._pending = 0
    llama_keepwarm._preview_pending = 0
    llama_keepwarm._inflight = 0
    llama_keepwarm._preview_inflight = 0


def test_studio_request_arriving_during_preview_swap_flags_scope(slot_state, monkeypatch):
    # A non-preview request blocked on the lifecycle gate while a preview swap loaded a
    # new checkpoint (swap-generation advanced between entry and gate) must not run
    # against the swapped-in preview model. The middleware flags the scope; the local
    # inference path (_maybe_auto_switch_model) does the reject. Deferring it to the
    # route -- not a 503 in the middleware -- lets an external-provider request untrack
    # and return before the check, so it is never rejected for a swap it never touches.
    _reset_keepwarm_counters()
    gens = iter([5, 6])  # capture=5 before the gate, check=6 after (a swap completed)
    monkeypatch.setattr(llama_keepwarm, "_preview_swap_gen", lambda: next(gens, 6))

    seen = {}

    async def _app(scope, receive, send):
        seen["flagged"] = bool(scope.get(llama_keepwarm._PREVIEW_SWAP_REJECT_SCOPE_KEY))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    _run_middleware(_app, "/v1/chat/completions")
    # The middleware ran the app (no early 503) but flagged the scope so the route
    # rejects only once external-provider requests have had a chance to untrack.
    assert seen["flagged"] is True
    _reset_keepwarm_counters()


def test_maybe_auto_switch_rejects_when_preview_swap_flagged():
    # The deferred half of the swap-race guard: _maybe_auto_switch_model raises 503 when
    # the middleware flagged the scope (a preview swapped the model out from under this
    # request while it waited on the gate).
    from fastapi import HTTPException

    scope = {
        "type": "http",
        "path": "/v1/chat/completions",
        llama_keepwarm._PREVIEW_SWAP_REJECT_SCOPE_KEY: True,
    }
    req = _types.SimpleNamespace(scope = scope)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference._maybe_auto_switch_model("some-model", req, "tester"))
    assert exc.value.status_code == 503


def test_maybe_auto_switch_no_reject_without_swap_flag(slot_state):
    # Without the flag the swap-race guard is inert: an omitted-model request returns
    # via the normal no-op resident-use path (no 503), so external-provider traffic
    # during a preview swap is unaffected by a swap it never touches.
    scope = {"type": "http", "path": "/v1/chat/completions"}
    req = _types.SimpleNamespace(scope = scope)
    # Omitted model -> the no-op resident-use path returns without raising.
    asyncio.run(inference._maybe_auto_switch_model(None, req, "tester"))


def test_studio_request_during_in_progress_swap_flags_scope(slot_state, monkeypatch):
    # Codex P1: a non-preview request that arrives while a preview swap is in progress
    # must be rejected via the in-progress flag, not only via a swap-counter change --
    # the counter may already have bumped but the gate not yet released, so a request
    # capturing the advanced counter would otherwise see no change and run against the
    # just-loaded preview checkpoint.
    _reset_keepwarm_counters()
    # The swap counter does NOT change across this request (captured == checked).
    monkeypatch.setattr(llama_keepwarm, "_preview_swap_gen", lambda: 7)
    llama_keepwarm.note_preview_swap_begin()  # a swap is in progress
    assert llama_keepwarm._preview_swap_active() is True
    try:
        seen = {}

        async def _app(scope, receive, send):
            seen["flagged"] = bool(scope.get(llama_keepwarm._PREVIEW_SWAP_REJECT_SCOPE_KEY))
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"{}", "more_body": False})

        _run_middleware(_app, "/v1/chat/completions")
        assert seen["flagged"] is True  # rejected on the in-progress flag alone
    finally:
        llama_keepwarm.note_preview_swap_end()
    assert llama_keepwarm._preview_swap_active() is False
    _reset_keepwarm_counters()


def test_failed_stream_response_does_not_claim_slot(slot_state):
    # Codex P2: a streaming response that returns 200 headers then encodes a failure
    # (marks the scope failed) must NOT be treated as a successful completion, so the
    # preview marker stays and a later preview for another checkpoint is not 503'd.
    _reset_keepwarm_counters()
    inference._set_preview_resident("/outputs/run/ckpt")

    async def _app(scope, receive, send):
        llama_keepwarm.mark_response_failed(scope)  # mid-stream failure after 200
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    _run_middleware(_app, "/v1/chat/completions")
    assert inference._is_preview_resident("/outputs/run/ckpt")  # not claimed on failure
    _reset_keepwarm_counters()


def test_non_gguf_load_failure_restores_preview_marker():
    # Codex P2: a failed non-GGUF (Unsloth/LoRA) load unloads only the new entry, so
    # the prior preview checkpoint can still be resident even though the marker was
    # cleared; _load_model_impl restores its ownership on both a raised and a
    # falsy-return load failure.
    import inspect

    src = inspect.getsource(inference._load_model_impl)
    assert "_restore_marker_if_prior_preview_still_resident" in src
    # Restored on both failure paths: the except (raise) and the falsy-success branch.
    assert src.count("_restore_marker_if_prior_preview_still_resident()") >= 2


def test_gguf_load_failure_restores_preview_marker():
    # Codex P2 (round 17): a GGUF load can raise before it tears down the old
    # llama-server (e.g. an update-in-progress guard fires before _kill_process),
    # leaving the prior preview-owned GGUF resident while the marker was already
    # cleared -- later previews for another checkpoint would then 503 against it.
    # _load_model_impl restores preview ownership on both the raised and the
    # falsy-return GGUF load failures, mirroring the non-GGUF path via a shared helper.
    import inspect

    src = inspect.getsource(inference._load_model_impl)
    # The restore helper is defined once and shared by the GGUF + non-GGUF paths.
    assert src.count("def _restore_marker_if_prior_preview_still_resident") == 1
    # Non-GGUF (2) + GGUF (2) failure sites all restore via the shared helper.
    assert src.count("_restore_marker_if_prior_preview_still_resident()") >= 5  # incl def
    # The GGUF load_with_tensor_fallback is wrapped so a raise AND a falsy return both
    # restore the marker before propagating the failure.
    gguf_fail_region = src[
        src.index("load_with_tensor_fallback(") : src.index("Failed to load GGUF model")
    ]
    assert "except Exception:" in gguf_fail_region
    assert "_restore_marker_if_prior_preview_still_resident()" in gguf_fail_region


def test_monitor_openai_sse_event_reports_error_status():
    # Codex P2 (round 17): the legacy /v1/completions relay forwards an upstream
    # HTTP-200 SSE `data: {"error": ...}` event. _monitor_openai_sse_event must report
    # "error" so the relay can mark the response failed and stop the keep-warm
    # middleware claiming a preview-owned slot on a failed completion stream.
    err = inference._monitor_openai_sse_event(
        "mon-round17", b'data: {"error": {"message": "boom"}}', None
    )
    assert err == "error"
    done = inference._monitor_openai_sse_event("mon-round17", b"data: [DONE]", None)
    assert done == "done"
    normal = inference._monitor_openai_sse_event(
        "mon-round17", b'data: {"choices": [{"delta": {"content": "hi"}}]}', None
    )
    assert normal is None


def test_completions_relay_marks_scope_failed_on_sse_error():
    # The legacy /v1/completions relay must mark the response failed when a relayed
    # event is an upstream error, so a preview-owned model is not claimed for Studio
    # after a failed completion stream (the middleware skips its claim on a failed scope).
    import inspect

    src = inspect.getsource(inference.openai_completions)
    # Format-independent: the relay checks the event status for an error and marks the
    # scope failed (assertions must survive an autoformatter reflow of the condition).
    assert "_monitor_openai_sse_event(" in src
    assert '== "error"' in src
    assert 'mark_response_failed(getattr(request, "scope", None))' in src


def test_anthropic_stream_error_event_marks_failed_even_when_unclassified(slot_state):
    # Codex P2 (round 18): an unclassified local /v1/messages generator error returns
    # None (no in-band error event emitted) and the caller swallows it, then finishes
    # the stream normally. The response must still be flagged failed so the middleware
    # does not claim a preview-owned model for Studio on a stream that errored.
    from core.inference import llama_keepwarm as kw

    scope = {"type": "http", "path": "/v1/messages"}
    kw.set_current_response_scope(scope)
    try:
        ev = inference._anthropic_stream_error_event(RuntimeError("boom"))  # unclassified
        assert ev is None  # no in-band error event for an unclassified error
        assert scope.get(kw._RESPONSE_FAILED_SCOPE_KEY) is True
    finally:
        kw.set_current_response_scope(None)


def test_openai_passthrough_transport_error_marks_failed():
    # Codex P2 (round 18): the passthrough httpx transport except (RemoteProtocolError/
    # ReadError/CloseError) re-raises after the 200 headers without emitting an error
    # SSE, so it must flag the response failed first or the middleware claims a
    # preview-owned slot on an interrupted stream.
    import inspect

    src = inspect.getsource(inference._openai_passthrough_stream_admitted)
    idx = src.index("httpx.RemoteProtocolError, httpx.ReadError, httpx.CloseError")
    # Bound the slice to the transport-error branch (up to the next except clause).
    branch = src[idx : src.index("except HTTPException", idx)]
    assert 'mark_response_failed(getattr(request, "scope", None))' in branch
    assert "raise" in branch


def test_anthropic_passthrough_non_200_marks_failed():
    # Codex P2 (round 18): the passthrough delivers an upstream 4xx/5xx as an in-band
    # Anthropic error event under the outer 200 stream headers, so it must flag the
    # response failed before yielding it, or the middleware claims a preview-owned slot
    # on a passthrough whose upstream errored before any SSE.
    import inspect

    src = inspect.getsource(inference._anthropic_passthrough_stream)
    idx = src.index("anthropic passthrough upstream error")
    # Bound the slice to the non-200 branch (up to the in-band error event it yields).
    branch = src[idx : src.index("yield build_anthropic_sse_event", idx)]
    assert 'mark_response_failed(getattr(request, "scope", None))' in branch


def test_admission_cancel_marks_response_failed():
    # Codex P2 (round 19): a streaming request cancelled before it leases an upstream
    # returns 200 with no body via the caller's `except LlamaAdmissionCancelled` path,
    # which never sets _RESPONSE_FAILED_SCOPE_KEY. _raise_if_openai_admission_cancelled
    # is the single choke point for that cancellation, so it must flag the response
    # failed -- otherwise the middleware claims a preview-owned slot for a stream that
    # never ran. (Harmless for non-streaming callers, which surface a non-2xx.)
    import inspect
    src = inspect.getsource(inference._raise_if_openai_admission_cancelled)
    # Flagged on both cancellation branches (already-cancelled and pre-header cancel).
    assert src.count("mark_current_response_failed()") >= 2


def test_preview_not_blocked_by_pending_non_preview_waiter(fake_slot):
    # Codex P2 (round 19): a queued (pending) non-preview request -- possibly an
    # unauthenticated probe that will 401 and never touch the model, since the
    # middleware bumps _pending before auth -- must not block a preview. Genuine queued
    # Studio requests are covered by the swap-reject when they wake, not by the busy
    # guard counting all pending requests blindly.
    from core.inference import llama_keepwarm as kw
    kw._note_pending(is_preview = False)  # queued non-preview waiter
    try:
        asyncio.run(
            inference.load_model_for_preview(
                LoadRequest(model_path = "/outputs/run/ckpt"),
                SimpleNamespace(app = None, scope = {"path": "/p/run/v1/chat/completions"}),
                "admin",
            )
        )
        assert fake_slot["loads"] == ["/outputs/run/ckpt"]  # not blocked by the waiter
    finally:
        kw._note_unpending(is_preview = False)


def test_responses_stream_failure_paths_mark_response_failed():
    # Codex P2 (round 20): _responses_stream emits its own response.failed SSE events
    # (admission timeout, upstream unreachable, non-200, transport error, generic
    # exception) after the 200 headers. With claim_resident=False the middleware would
    # otherwise treat that 2xx as a successful Studio generation and clear preview
    # ownership, so every failed-response builder / inline emitter flags the response
    # failed before yielding.
    import inspect

    src = inspect.getsource(inference._responses_stream)
    # 2 failure-only builders (admission-failed + failed_response_payload, the latter
    # covering both the transport-error and generic-exception yields) plus 2 inline
    # emitters (upstream-unreachable, non-200) all mark the response failed.
    assert src.count('mark_response_failed(getattr(request, "scope", None))') >= 4


def test_generate_stream_cancel_marks_response_failed():
    # Codex P2 (round 20): generate_stream's `if cancel_event.is_set(): ... break` ends
    # a 200 stream with no completion (client disconnect, possibly before the first
    # token) without going through the except path, so it must flag the response failed
    # or the middleware claims a preview-owned model for a stream that never completed.
    import inspect

    src = inspect.getsource(inference.generate_stream)
    # Marked in the cancel-break branch in addition to the existing except handler.
    assert src.count("mark_response_failed(_gs_scope)") >= 2
    cancel_idx = src.index("if cancel_event.is_set():")
    branch = src[cancel_idx : src.index("chunk = await asyncio.to_thread", cancel_idx)]
    assert "mark_response_failed(_gs_scope)" in branch


def test_middleware_claims_slot_on_successful_non_preview_response(slot_state):
    # A non-preview inference that returns 2xx adopts the resident model for Studio,
    # so the preview-ownership marker is cleared on completion.
    _reset_keepwarm_counters()
    inference._set_preview_resident("/outputs/run/ckpt-a")

    async def _app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    _run_middleware(_app, "/v1/chat/completions")
    assert not inference._is_preview_resident("/outputs/run/ckpt-a")  # claimed for Studio
    _reset_keepwarm_counters()


def test_middleware_preview_response_keeps_ownership(slot_state):
    # A /p preview response (is_preview) must not claim the slot: the preview keeps its
    # own ownership so a later preview can still swap it.
    _reset_keepwarm_counters()
    inference._set_preview_resident("/outputs/run/ckpt")

    async def _app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    _run_middleware(_app, "/p/demo/v1/chat/completions")
    assert inference._is_preview_resident("/outputs/run/ckpt")  # ownership preserved
    _reset_keepwarm_counters()


def test_middleware_rejected_response_keeps_preview_ownership(slot_state):
    # A non-preview request rejected by a per-route capability check (non-2xx) never
    # ran against the model, so it must not claim -- the preview keeps ownership.
    _reset_keepwarm_counters()
    inference._set_preview_resident("/outputs/run/ckpt")

    async def _app(scope, receive, send):
        await send({"type": "http.response.start", "status": 400, "headers": []})
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    _run_middleware(_app, "/v1/chat/completions")
    assert inference._is_preview_resident("/outputs/run/ckpt")  # not claimed on 4xx
    _reset_keepwarm_counters()


def test_slot_claim_happens_before_inflight_decrement(slot_state, monkeypatch):
    # Codex P2: the middleware must clear preview ownership BEFORE decrementing the
    # in-flight count. Doing it after opens a window where a preview for another
    # checkpoint sees no non-preview traffic and a still-preview-owned slot, swaps its
    # model in, and then the delayed claim clears ownership on the wrong model.
    _reset_keepwarm_counters()
    inference._set_preview_resident("/outputs/run/ckpt-a")
    observed = {}
    real_claim = llama_keepwarm._claim_non_preview_slot

    def _spy():
        observed["inflight_at_claim"] = llama_keepwarm._inflight
        real_claim()

    monkeypatch.setattr(llama_keepwarm, "_claim_non_preview_slot", _spy)

    async def _app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    _run_middleware(_app, "/v1/chat/completions")
    # The request is still counted in-flight when the claim runs, so the preview busy
    # guard would refuse a concurrent swap; only then is the count decremented.
    assert observed["inflight_at_claim"] == 1
    assert llama_keepwarm._inflight == 0  # decremented afterwards
    _reset_keepwarm_counters()


def test_rejected_preview_request_does_not_refresh_idle_timer(slot_state):
    # Codex P2: a public preview POST rejected before it loads the model (rate-limit
    # 429, bad capability token 404, body-validation 4xx) never served tokens, so it
    # must not refresh the idle timer -- otherwise repeated rejected preview POSTs pin
    # an otherwise idle model in VRAM and idle-unload can never free it.
    _reset_keepwarm_counters()
    llama_keepwarm._last_active = 0.0

    async def _rejected(scope, receive, send):
        await send({"type": "http.response.start", "status": 429, "headers": []})
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    _run_middleware(_rejected, "/p/demo/v1/chat/completions")
    assert llama_keepwarm._last_active == 0.0  # untracked end -> no activity stamp
    assert llama_keepwarm._inflight == 0  # still balanced
    _reset_keepwarm_counters()


def test_successful_preview_request_refreshes_idle_timer(slot_state):
    # Contrast: a preview that actually served (2xx) DOES stamp activity, so a live
    # preview stream keeps the model warm for its duration.
    _reset_keepwarm_counters()
    llama_keepwarm._last_active = 0.0

    async def _ok(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    _run_middleware(_ok, "/p/demo/v1/chat/completions")
    assert llama_keepwarm._last_active > 0.0  # _note_end stamped activity
    _reset_keepwarm_counters()
