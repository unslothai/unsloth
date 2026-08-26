# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Security smoke for the public /p preview routes.

Exercises the route layer with a real ``preview_router`` while stubbing the
expensive model calls (``load_model_for_preview`` / ``openai_chat_completions``). Covers the
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
    app.dependency_overrides[preview.authenticated_without_credential] = lambda: False
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
    assert "messages: msgs,\n              stream: true," in r.text


def test_page_renders_friendly_busy_message(client):
    response = client.get(f"/p/demorun?k={_sig('demorun')}")
    assert "Unsloth is currently using another model" in response.text


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


def test_list_previews_omits_capability_for_keyless_caller(client, monkeypatch):
    monkeypatch.setattr(
        preview,
        "list_preview_targets",
        lambda: [{"ref": "demorun", "is_latest": True}],
    )
    client.app.dependency_overrides[preview.authenticated_without_credential] = lambda: True
    body = client.get("/p").json()
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


# Model-slot ownership regressions.
import threading
from types import SimpleNamespace

from fastapi import HTTPException

import routes.inference as inference
from core.inference import llama_keepwarm
from models.inference import LoadRequest


@pytest.fixture(autouse = True)
def reset_admitted_inference():
    with llama_keepwarm._lock:
        llama_keepwarm._admitted_inference = 0
    yield
    with llama_keepwarm._lock:
        llama_keepwarm._admitted_inference = 0


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
    state = {"ident": None, "loads": [], "load_kwargs": []}

    async def _fake_impl(load_req, fastapi_request, subject, **kwargs):
        state["loads"].append(load_req.model_path)
        state["load_kwargs"].append(kwargs)
        if state.get("fail_load"):
            state["ident"] = None
            raise HTTPException(status_code = 500, detail = "load failed")
        state["ident"] = load_req.model_path

    monkeypatch.setattr(inference, "_load_model_impl", _fake_impl)
    monkeypatch.setattr(inference, "_loaded_slot_ident", lambda: state["ident"])
    monkeypatch.setattr(
        llama_keepwarm, "other_admitted_inference_count", lambda: state.get("busy", 0)
    )
    return state


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


def test_preview_load_refused_when_studio_model_is_loaded(fake_slot):
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


def test_preview_does_not_borrow_studio_owned_lora(fake_slot, tmp_path):
    checkpoint = tmp_path / "lora-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding = "utf-8")
    fake_slot["ident"] = str(checkpoint)

    async def _run():
        with pytest.raises(HTTPException) as exc:
            await inference.load_model_for_preview(
                LoadRequest(model_path = str(checkpoint)),
                SimpleNamespace(app = None),
                "admin",
            )
        return exc.value

    exc = asyncio.run(_run())
    assert exc.status_code == 503
    assert fake_slot["loads"] == []
    assert not inference._is_preview_resident(str(checkpoint))


def test_preview_can_swap_out_prior_preview_model(fake_slot):
    for path in ("/outputs/run/ckpt-a", "/outputs/run/ckpt-b"):
        asyncio.run(
            inference.load_model_for_preview(
                LoadRequest(model_path = path), SimpleNamespace(app = None), "admin"
            )
        )
    assert fake_slot["loads"] == ["/outputs/run/ckpt-a", "/outputs/run/ckpt-b"]
    assert fake_slot["ident"] == "/outputs/run/ckpt-b"


@pytest.mark.parametrize("owner", ["diffusion", "video"])
def test_preview_load_refused_while_image_or_video_owns_gpu(fake_slot, monkeypatch, owner):
    from core.inference import gpu_arbiter

    monkeypatch.setattr(gpu_arbiter, "current_owner", lambda: owner)

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
    assert fake_slot["loads"] == []  # never reached the load, so nothing was evicted


def test_preview_maps_atomic_gpu_refusal_to_503(fake_slot, monkeypatch):
    from core.inference import gpu_arbiter

    monkeypatch.setattr(gpu_arbiter, "current_owner", lambda: None)
    swap_notes = []
    monkeypatch.setattr(llama_keepwarm, "note_preview_swap", lambda: swap_notes.append(True))

    async def _lose_gpu_ownership(*args, **kwargs):
        assert kwargs["allow_gpu_owner_eviction"] is False
        raise gpu_arbiter.GpuOwnerBusyError(gpu_arbiter.DIFFUSION)

    monkeypatch.setattr(inference, "_load_model_impl", _lose_gpu_ownership)

    async def _run():
        with pytest.raises(HTTPException) as excinfo:
            await inference.load_model_for_preview(
                LoadRequest(model_path = "/outputs/run/ckpt-a"),
                SimpleNamespace(scope = {"path": "/p/a/v1/chat/completions"}),
                "admin",
            )
        return excinfo.value

    exc = asyncio.run(_run())
    assert exc.status_code == 503
    assert "image or video" in exc.detail
    assert inference._get_preview_resident() is None
    assert swap_notes == []


def test_preview_reload_failure_restores_prior_ownership(slot_state, monkeypatch):
    resident = {"ident": "/outputs/run/ckpt-A"}
    monkeypatch.setattr(inference, "_loaded_slot_ident", lambda: resident["ident"])
    monkeypatch.setattr(llama_keepwarm, "other_admitted_inference_count", lambda: 0)
    llama_keepwarm._pending = 0
    llama_keepwarm._preview_pending = 0
    inference._set_preview_resident("/outputs/run/ckpt-A")  # A is preview-owned

    async def _clear_then_fail(load_req, fastapi_request, subject, **kwargs):
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
    assert inference._is_preview_resident("/outputs/run/ckpt-A")


def test_cancelled_json_response_does_not_claim_slot(slot_state):
    import inspect
    import threading

    src = inspect.getsource(inference.openai_chat_completions)
    assert src.count("_mark_cancelled_json_response_failed(request, cancel_event)") == 3

    _reset_keepwarm_counters()
    inference._set_preview_resident("/outputs/run/ckpt")

    async def _app(scope, receive, send):
        cancelled = threading.Event()
        cancelled.set()
        inference._mark_cancelled_json_response_failed(
            _types.SimpleNamespace(scope = scope), cancelled
        )
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    _run_middleware(_app, "/v1/chat/completions")
    assert inference._is_preview_resident("/outputs/run/ckpt")
    _reset_keepwarm_counters()


def test_queued_preview_does_not_deadlock_studio_switch():
    from core.inference import llama_keepwarm as kw
    async def _run():
        _reset_keepwarm_counters()
        kw._admitted_inference = 0
        preview._preview_lock = asyncio.Lock()
        assert not inference._auto_switch_process_lock.locked()

        studio_holds_gate = asyncio.Event()
        queued_has_serializer = asyncio.Event()
        await preview._preview_lock.acquire()
        kw._note_start(is_preview = True)

        async def _receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def _send(_message):
            return None

        async def queued_preview_app(scope, receive, send):
            serializer_waiting = kw.begin_preview_serializer_wait(scope)
            locked = False
            try:
                await preview._preview_lock.acquire()
                locked = True
                queued_has_serializer.set()
                await kw.resume_preview_after_serializer(scope)
                serializer_waiting = False
                await inference._acquire_swap_gate()
                try:
                    await send({"type": "http.response.start", "status": 200})
                    await send({"type": "http.response.body", "body": b""})
                finally:
                    inference._auto_switch_process_lock.release()
            finally:
                if serializer_waiting:
                    kw.cancel_preview_serializer_wait(scope)
                if locked:
                    preview._preview_lock.release()

        async def studio_switch_app(scope, receive, send):
            kw.note_admitted_inference(scope)
            await inference._acquire_swap_gate()
            try:
                async with kw.inference_lifecycle_gate():
                    studio_holds_gate.set()
                    await inference._wait_for_model_switch_idle(current_request_counted = True)
            finally:
                inference._auto_switch_process_lock.release()
            await send({"type": "http.response.start", "status": 200})
            await send({"type": "http.response.body", "body": b""})

        preview_scope = {
            "type": "http",
            "method": "POST",
            "path": "/p/run/ckpt/v1/chat/completions",
            "headers": [],
        }
        studio_scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"authorization", b"Bearer valid")],
        }
        queued_preview = asyncio.create_task(
            kw.LlamaKeepWarmMiddleware(queued_preview_app)(preview_scope, _receive, _send)
        )
        while kw._preview_pending != 1:
            await asyncio.sleep(0)
        assert kw._preview_inflight == 1  # only active preview A, not queued B

        studio_switch = asyncio.create_task(
            kw.LlamaKeepWarmMiddleware(studio_switch_app)(studio_scope, _receive, _send)
        )
        await asyncio.wait_for(studio_holds_gate.wait(), 1)

        preview._preview_lock.release()
        await asyncio.wait_for(queued_has_serializer.wait(), 1)
        kw._note_end(is_preview = True)

        await asyncio.wait_for(asyncio.gather(queued_preview, studio_switch), 2)
        assert kw._inflight == 0
        assert kw._pending == 0
        assert kw._preview_inflight == 0
        assert kw._preview_pending == 0
        assert kw._admitted_inference == 0

    asyncio.run(_run())


def test_admitted_inference_counter_excludes_previews():
    from core.inference import llama_keepwarm as kw

    kw._admitted_inference = 0
    kw._inflight += 1  # non-preview request tracked pre-auth (never reached the hook)
    try:
        assert kw.other_admitted_inference_count() == 0  # unadmitted in-flight not counted
        scope = {"path": "/v1/chat/completions"}
        kw.note_admitted_inference(scope)  # passed auth, reached the inference hook
        assert kw.other_admitted_inference_count() == 1
        kw.note_admitted_inference(scope)  # idempotent per scope
        assert kw.other_admitted_inference_count() == 1
        kw.note_admitted_inference({"path": "/p/run/v1/chat/completions"})
        assert kw.other_admitted_inference_count() == 1
        kw._note_admitted_end()  # middleware _finish balances the admit
        assert kw.other_admitted_inference_count() == 0
    finally:
        kw._inflight = 0
        kw._admitted_inference = 0


@pytest.mark.parametrize("status, claimed", [(200, True), (400, False)])
def test_middleware_claims_slot_only_on_success(slot_state, status, claimed):
    _reset_keepwarm_counters()
    checkpoint = "/outputs/run/ckpt-a"
    inference._set_preview_resident(checkpoint)

    async def _app(scope, receive, send):
        await send({"type": "http.response.start", "status": status, "headers": []})
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    _run_middleware(_app, "/v1/chat/completions")
    assert inference._is_preview_resident(checkpoint) is (not claimed)
    _reset_keepwarm_counters()


def test_slot_claim_happens_before_admitted_decrement(slot_state, monkeypatch):
    _reset_keepwarm_counters()
    llama_keepwarm._admitted_inference = 0
    inference._set_preview_resident("/outputs/run/ckpt-a")
    observed = {}
    real_claim = llama_keepwarm._claim_non_preview_slot

    def _spy():
        observed["admitted_at_claim"] = llama_keepwarm._admitted_inference
        real_claim()

    monkeypatch.setattr(llama_keepwarm, "_claim_non_preview_slot", _spy)

    async def _app(scope, receive, send):
        llama_keepwarm.note_admitted_inference(scope)  # passed auth, reached the inference hook
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    _run_middleware(_app, "/v1/chat/completions")
    assert observed["admitted_at_claim"] == 1
    assert llama_keepwarm._admitted_inference == 0  # decremented afterwards
    assert not inference._is_preview_resident("/outputs/run/ckpt-a")  # claimed for Unsloth
    _reset_keepwarm_counters()
    llama_keepwarm._admitted_inference = 0


def test_preview_rechecks_ownership_after_admitted_count(fake_slot, monkeypatch):
    fake_slot["ident"] = "/outputs/run/ckpt-a"
    inference._set_preview_resident("/outputs/run/ckpt-a")

    def _finish_studio_request() -> int:
        thread = threading.Thread(target = inference._set_preview_resident, args = (None,))
        thread.start()
        thread.join()
        return 0

    monkeypatch.setattr(llama_keepwarm, "other_admitted_inference_count", _finish_studio_request)

    async def _run():
        with pytest.raises(HTTPException) as excinfo:
            await inference.load_model_for_preview(
                LoadRequest(model_path = "/outputs/run/ckpt-b"),
                SimpleNamespace(scope = {"path": "/p/b/v1/chat/completions"}),
                "admin",
            )
        return excinfo.value

    exc = asyncio.run(_run())
    assert exc.status_code == 503
    assert fake_slot["loads"] == []


def test_preview_swap_marker_skipped_for_same_target_borrow():
    import inspect

    src = inspect.getsource(inference.load_model_for_preview)
    guard = src.index("if not same_target:")
    begin = src.index("note_preview_swap_begin()", guard)
    check = src.index("other_admitted_inference_count()")
    assert guard < begin < check
    assert "note_preview_swap_begin()" in src[guard : guard + 200]
