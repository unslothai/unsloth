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


def test_generation_paths_claim_via_gated_helper():
    # Every local generation entry (native stream/audio + OpenAI chat) must adopt
    # the resident model for Studio through the gated helper, so a preview cannot
    # swap it out between turns -- while an audio preview routed through
    # openai_chat_completions -> generate_audio still keeps its ownership.
    import inspect
    for fn in (
        inference.generate_stream,
        inference.generate_audio,
        inference.openai_chat_completions,
    ):
        assert "_claim_slot_for_non_preview(" in inspect.getsource(fn)


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
