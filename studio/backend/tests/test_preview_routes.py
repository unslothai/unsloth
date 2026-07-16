# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Security smoke for the public /p preview routes.

Real ``preview_router`` with stubbed model calls. Covers HMAC capability gating
(missing/invalid/wrong-ref tokens 404 before any load), path-traversal rejection,
request sanitization (tools / provider routing / use_adapter / generation clamp),
asset-path containment, the page CSP + no-referrer headers and HTML escaping, and
that the preview lock is held until a streaming response fully drains.
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


# Fixed secret: deterministic signing, no auth.db.
_TEST_SECRET = b"unit-test-preview-secret-0123456789"


def _use_test_secret(monkeypatch) -> None:
    monkeypatch.setattr(preview_token, "get_or_create_preview_link_secret", lambda: _TEST_SECRET)


def _sig(ref: str) -> str:
    """Valid capability token for ``ref`` under the test secret."""
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

    # Sharing on by default; reset the per-IP rate buckets each test.
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
    # Run dir name has an HTML-special char; the page must escape it.
    _make_run(outputs, name = "a<b")
    _use_test_secret(monkeypatch)
    monkeypatch.setattr(preview, "get_preview_sharing_enabled", lambda: True)
    from utils.paths import storage_roots as _sr

    monkeypatch.setattr(_sr, "outputs_root", lambda: outputs)

    app = FastAPI()
    app.include_router(preview.router, prefix = "/p")
    c = TestClient(app, raise_server_exceptions = False)

    # Sign the decoded canonical ref ("a<b"), not the %-encoded segment.
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
    # The listing hands the owner a usable capability.
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
    # Don't hand out credentials that 404; signal the disabled state.
    assert body["sharing_enabled"] is False
    assert body["data"][0]["key"] is None
    assert body["data"][0]["share_url"] is None


# ── Path traversal / containment ────────────────────────────────────────────


@pytest.mark.parametrize(
    "path",
    [
        "/p/..",  # parent segment as run
        "/p/%2e%2e/etc",  # encoded traversal
        "/p/..%2f..%2fetc/v1/models",  # encoded-slash traversal
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
        "nope.png",  # allowlisted suffix, missing file
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
    # Tool-loop levers neutralized regardless of the gate.
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
    # Adapter pinned on for LoRA: a caller can't flip the backend to base.
    assert p.use_adapter is True
    # Generation capped on this public surface (no override -> ceiling).
    assert p.max_tokens == preview._PREVIEW_MAX_OUTPUT_TOKENS
    assert p.max_completion_tokens == preview._PREVIEW_MAX_OUTPUT_TOKENS
    assert p.n == 1
    # Loads the resolved checkpoint dir, not an attacker path.
    assert captured["load_path"].endswith("demorun")


def test_merged_checkpoint_strips_use_adapter(tmp_path, monkeypatch, captured):
    # Merged (non-LoRA) checkpoint: no adapter, so use_adapter -> None.
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
        # Lock still held: a second checkpoint must not swap the backend mid-stream.
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
    # Rejected before any model work: nothing loaded, nothing generated.
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
    # A capability for a different ref must not unlock demorun.
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
    # Nested ref: the canonical ref is "run/checkpoint".
    sig = _sig("demorun/checkpoint-1")
    r = client.post(
        f"/p/demorun/checkpoint-1/v1/chat/completions?k={sig}",
        json = {"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert captured["load_path"].endswith("checkpoint-1")


def test_checkpoint_token_does_not_unlock_bare_run(client, captured):
    # A token for the nested checkpoint must not unlock the run ref.
    r = client.post(
        f"/p/demorun/v1/chat/completions?k={_sig('demorun/checkpoint-1')}",
        json = {"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 404
    assert "load_path" not in captured


def test_bearer_token_accepted(client, captured):
    # OpenAI clients pass the capability as the api_key (Bearer header).
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
    # A caller asking for fewer tokens via the legacy field must not be bumped to the
    # ceiling: _effective_max_tokens prefers max_completion_tokens, so both fields carry
    # the lower value.
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
    # Sharing off: even a valid token 404s, with no model load.
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
            # Mirror a load that tears the old model down, then fails.
            state["ident"] = None
            raise HTTPException(status_code = 500, detail = "load failed")
        state["ident"] = load_req.model_path

    monkeypatch.setattr(inference, "_load_model_impl", _fake_impl)
    monkeypatch.setattr(inference, "_loaded_slot_ident", lambda: state["ident"])
    # The busy guard counts admitted (post-auth) non-preview inference; "busy" drives it.
    monkeypatch.setattr(
        llama_keepwarm, "other_admitted_inference_count", lambda: state.get("busy", 0)
    )
    return state


def test_preview_load_refused_when_studio_model_is_loaded(fake_slot):
    # An owner model stays protected even while idle: a preview must not swap it out
    # just because the user is between Studio messages.
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
    # A resident model already serving this exact checkpoint is borrowed as-is: no
    # _load_model_impl call (which could reload/reconfigure a Studio-owned GGUF whose
    # live settings differ from the bare preview request, #5401), and a Studio-owned
    # model isn't silently claimed for preview.
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
    # same_target doesn't guarantee a no-op reload (GGUF settings can force a restart),
    # so busy Studio traffic blocks even a same-checkpoint request.
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
    # Preview (/p/) in-flight traffic is never admitted local inference, so it doesn't
    # count toward the busy guard: with no admitted non-preview inference the request
    # borrows the same checkpoint (no reload). See test_admitted_inference_counter_excludes_previews.
    fake_slot["ident"] = "/outputs/run/ckpt"
    fake_slot["busy"] = 0  # no admitted non-preview inference (e.g. only previews)
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
    # A Studio /load returning already_loaded must still claim a preview-owned checkpoint,
    # or a later preview could swap out the model Studio just loaded.
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
    # A load that fails validation before touching the backend must not clear the
    # resident marker: the old preview-owned checkpoint is still loaded.
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
    # Route-level: the guard's 503 (with Retry-After) reaches the caller through the
    # real _serve_chat, which must release the preview lock.
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
    monkeypatch.setattr(llama_keepwarm, "other_admitted_inference_count", lambda: 1)

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
    # Preview loads bypass load_model(), so they re-apply its sidecar-install guard: a
    # public preview must not complete a load mid transformers install.
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
    # A refused preview never touches the model, so it untracks itself (no activity stamp)
    # and balances BOTH in-flight counters, else public /p spam could pin an idle Studio
    # model and skew the busy guard.
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
    # A preview re-requesting its own loaded checkpoint borrows it (no second
    # _load_model_impl call) and keeps preview ownership.
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
    # before generation could strand a preview-owned checkpoint as Studio-owned if
    # generation then fails. The middleware claims on a 2xx instead. generate_stream also
    # enforces the preview-swap reject, since it doesn't run through _maybe_auto_switch_model.
    import inspect

    stream_src = inspect.getsource(inference.generate_stream)
    audio_src = inspect.getsource(inference.generate_audio)
    assert "_claim_slot_for_non_preview(" not in stream_src
    assert "_claim_slot_for_non_preview(" not in audio_src
    assert "preview_swapped_since_entry(" in stream_src


def test_preview_swap_proceeds_when_studio_request_only_queued(fake_slot):
    # Codex P2 (round 19): a Studio request merely QUEUED on the lifecycle gate (in
    # _pending, not yet generating) must NOT block a preview swap: the middleware bumps
    # _pending before auth, so counting it would let an unauthenticated probe starve
    # previews. The queued request is protected by the swap-reject instead: on waking it
    # sees the generation advance and gets a retryable 503. (In-flight Studio traffic still
    # blocks -- see test_preview_swap_refused_while_studio_traffic_in_flight.)
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
    # A non-preview local request claims the resident model (clears the marker); a /p
    # preview (including an audio preview via openai_chat_completions) keeps ownership.
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
    # Two checkpoints differing only by case are different files on a case-sensitive FS,
    # so a preview for /outputs/run must not borrow a resident /outputs/Run; it must load
    # the requested path.
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
    # Not borrowed: the differently-cased path was actually loaded.
    assert fake_slot["loads"] == ["/outputs/Run", "/outputs/run"]
    assert inference._is_preview_resident("/outputs/run")


def test_preview_reload_failure_restores_prior_ownership(slot_state, monkeypatch):
    # _load_model_impl clears the preview marker mid-load (reclaiming the slot). If it
    # then fails while the prior preview model is still resident, the marker must be
    # restored to it, else the next preview for a different checkpoint sees a still-
    # preview model as Studio-owned and 503s.
    resident = {"ident": "/outputs/run/ckpt-A"}
    monkeypatch.setattr(inference, "_loaded_slot_ident", lambda: resident["ident"])
    monkeypatch.setattr(llama_keepwarm, "other_admitted_inference_count", lambda: 0)
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
    # A is still resident and its preview ownership is restored.
    assert inference._is_preview_resident("/outputs/run/ckpt-A")


def test_generate_stream_image_on_text_model_preserves_preview_marker(slot_state, monkeypatch):
    # An image request against a text-only resident model 400s before generation. The
    # ownership claim runs only after that modality check, so a preview-owned checkpoint
    # isn't stranded as Studio-owned by a request that never generated.
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
    # No direct claim runs, so ownership is intact after the failed check.
    assert inference._is_preview_resident("/outputs/run/ckpt-a")


def test_generate_stream_rejects_after_preview_swap(slot_state, monkeypatch):
    # Codex P1: generate_stream doesn't run through _maybe_auto_switch_model, so it
    # enforces the preview-swap reject itself. If a preview loaded a different checkpoint
    # while this native request waited on the gate (the middleware flagged the scope),
    # generate_stream must 503 instead of streaming from the preview's model.
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
    # Codex P2: the preview-swap reject runs before the loaded-model / capability checks.
    # If a preview loaded a GGUF (no active_model_name) while this request was queued, it
    # must get the retryable 503, not a hard 400 "No model loaded" from a backend-state check.
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
    # Codex P2: every OpenAI-family streaming error is emitted through _openai_stream_error_sse
    # after the 200 headers, so it must flag the current response failed via the middleware's
    # contextvar, else the middleware claims a preview-owned model on a stream that errored.
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
    # Codex P2: Anthropic stream errors emit via _anthropic_stream_error_event after the
    # 200 headers, so it must flag the current response failed (via the middleware's
    # contextvar), else the middleware claims a preview-owned model on a failed stream.
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
    # Codex P2: if _load_model_impl makes B resident then raises while assembling the load
    # response (loaded_ok false), the slot holds B, so B must be marked preview-owned (it
    # was loaded only by this preview), not restored to A (which would leave B Studio-owned).
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
    # The already-loaded dedup is filesystem-aware: on a case-sensitive FS two paths
    # differing only by case are different models and must reload.
    import os.path

    assert inference._same_loaded_identifier("/outputs/Run", "/outputs/Run")
    assert not inference._same_loaded_identifier(None, "/outputs/Run")
    assert not inference._same_loaded_identifier("", "/outputs/Run")
    # Case-distinct local paths dedup only where the filesystem is case-insensitive.
    expected = os.path.normcase("/outputs/Run") == os.path.normcase("/outputs/run")
    assert inference._same_loaded_identifier("/outputs/Run", "/outputs/run") is expected
    # HF repo IDs resolve case-insensitively, so they dedup regardless of case.
    assert inference._same_loaded_identifier("Unsloth/Foo", "unsloth/foo")
    assert inference._same_loaded_identifier("unsloth/foo", "unsloth/foo")
    assert not inference._same_loaded_identifier("unsloth/foo", "unsloth/bar")


def test_same_loaded_identifier_treats_existing_relative_paths_as_filesystem(tmp_path, monkeypatch):
    # Codex P2: an existing RELATIVE checkpoint path is a local filesystem path (as
    # ModelConfig.from_identifier resolves it), not a repo id, so it compares filesystem-
    # aware: two case-differing relative paths must not dedup on a case-sensitive FS.
    import os

    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs" / "Run").mkdir(parents = True)
    # Only "outputs/Run" exists, so it is classified local and compared normcase.
    expected = os.path.normcase("outputs/Run") == os.path.normcase("outputs/run")
    assert inference._same_loaded_identifier("outputs/Run", "outputs/run") is expected
    # An identical existing relative path still dedups.
    assert inference._same_loaded_identifier("outputs/Run", "outputs/Run")


def test_completions_embeddings_defer_claim_to_middleware():
    # Codex P2: /v1/completions and /v1/embeddings must not claim the resident model before
    # the llama-server call, which can still reject a valid body (e.g. a no-pooling error
    # for embeddings against a non-embedding GGUF) and strand a preview-owned checkpoint.
    # The middleware claims on a 2xx instead.
    import inspect
    for fn in (inference.openai_completions, inference.openai_embeddings):
        assert "_claim_slot_for_non_preview(" not in inspect.getsource(fn), fn.__name__


def test_load_restores_preview_marker_on_late_companion_reject():
    # Codex P2: _load_model_impl clears the preview marker before the native vision
    # companion validation, which can reject after the clear. On that late reject the
    # resident preview model was never torn down, so preview ownership is restored.
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
    # A non-preview request blocked on the gate while a preview swap loaded a new checkpoint
    # (generation advanced between entry and gate) must not run against the swapped-in model.
    # The middleware flags the scope; the route (_maybe_auto_switch_model) does the reject.
    # Deferring to the route -- not a middleware 503 -- lets an external-provider request
    # untrack and return first, so it's never rejected for a swap it never touches.
    _reset_keepwarm_counters()
    gens = iter([5, 6])  # capture=5 before the gate, check=6 after (swap completed)
    monkeypatch.setattr(llama_keepwarm, "_preview_swap_gen", lambda: next(gens, 6))

    seen = {}

    async def _app(scope, receive, send):
        seen["flagged"] = bool(scope.get(llama_keepwarm._PREVIEW_SWAP_REJECT_SCOPE_KEY))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    _run_middleware(_app, "/v1/chat/completions")
    # The middleware ran the app (no early 503) but flagged the scope so the route rejects
    # only after external-provider requests have had a chance to untrack.
    assert seen["flagged"] is True
    _reset_keepwarm_counters()


def test_maybe_auto_switch_rejects_when_preview_swap_flagged():
    # The deferred half of the swap-race guard: _maybe_auto_switch_model raises 503 when the
    # middleware flagged the scope (a preview swapped the model out while this request waited).
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
    # Without the flag the guard is inert: an omitted-model request returns via the normal
    # no-op resident-use path (no 503), so external traffic during a swap is unaffected.
    scope = {"type": "http", "path": "/v1/chat/completions"}
    req = _types.SimpleNamespace(scope = scope)
    # Omitted model -> no-op resident-use path returns without raising.
    asyncio.run(inference._maybe_auto_switch_model(None, req, "tester"))


def test_studio_request_during_in_progress_swap_flags_scope(slot_state, monkeypatch):
    # Codex P1: a non-preview request arriving while a swap is in progress is rejected via
    # the in-progress flag, not only a counter change -- the counter may have bumped but the
    # gate not yet released, so a request capturing the advanced counter would see no change
    # and run against the just-loaded checkpoint.
    _reset_keepwarm_counters()
    # The swap counter does NOT change here (captured == checked).
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
    # Codex P2: a stream that returns 200 headers then marks the scope failed must NOT be
    # treated as a successful completion, so the preview marker stays and a later preview
    # isn't 503'd.
    _reset_keepwarm_counters()
    inference._set_preview_resident("/outputs/run/ckpt")

    async def _app(scope, receive, send):
        llama_keepwarm.mark_response_failed(scope)  # mid-stream failure after 200
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    _run_middleware(_app, "/v1/chat/completions")
    assert inference._is_preview_resident("/outputs/run/ckpt")  # not claimed on failure
    _reset_keepwarm_counters()


def test_disconnect_before_terminal_frame_does_not_claim_slot(slot_state):
    # Codex P2 (round 26): a client disconnect after the 200 headers raises before the terminal
    # body frame (an OSError _SameTaskStreamingResponse converts to a CancelledError, whose
    # handler finishes the monitor and re-raises without flagging the scope). The middleware
    # only claims on a clean completion, so this cancelled 2xx preserves preview ownership.
    _reset_keepwarm_counters()
    inference._set_preview_resident("/outputs/run/ckpt")

    async def _app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        # A content chunk streams, then the body is cut off mid-stream: no terminal
        # more_body False frame is ever sent (the client-disconnect-after-headers path).
        await send({"type": "http.response.body", "body": b"data: hi\n\n", "more_body": True})
        raise RuntimeError("client disconnected mid-stream")

    with pytest.raises(RuntimeError):
        _run_middleware(_app, "/v1/chat/completions")
    assert inference._is_preview_resident("/outputs/run/ckpt")  # cancelled stream did not claim
    _reset_keepwarm_counters()


def test_clean_stream_completion_claims_slot(slot_state):
    # Positive control for the completion gate: a streaming 200 that sends its terminal frame
    # completed cleanly, so it adopts the resident model for Studio (clears the preview marker).
    _reset_keepwarm_counters()
    inference._set_preview_resident("/outputs/run/ckpt")

    async def _app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"data: hi\n\n", "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    _run_middleware(_app, "/v1/chat/completions")
    assert not inference._is_preview_resident("/outputs/run/ckpt")  # clean completion claimed
    _reset_keepwarm_counters()


def test_disconnect_on_terminal_frame_write_does_not_claim_slot(slot_state):
    # Codex P2 (round 27): the claim must run only after the terminal body frame is actually
    # delivered. If the client disconnects on that final write, send() raises, so the stream
    # did not complete cleanly and preview ownership is preserved.
    _reset_keepwarm_counters()
    inference._set_preview_resident("/outputs/run/ckpt")

    async def _app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"data: hi\n\n", "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    mw = llama_keepwarm.LlamaKeepWarmMiddleware(_app)
    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions"}

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def _send(msg):
        # Emulate a client disconnect on the terminal frame's write.
        if msg.get("type") == "http.response.body" and not msg.get("more_body", False):
            raise OSError("client disconnected on final write")

    with pytest.raises(OSError):
        asyncio.run(mw(scope, _receive, _send))
    assert inference._is_preview_resident("/outputs/run/ckpt")  # terminal write failed -> no claim
    _reset_keepwarm_counters()


def test_passthrough_cancel_marks_response_failed():
    # Codex P2 (round 27): an explicit /inference/cancel ends a passthrough stream with a clean
    # terminal frame, which the completion gate treats as success. Both the OpenAI and Anthropic
    # passthrough streams flag the response failed in their finally when cancel_event is set.
    import inspect
    import re
    for fn in (
        inference._openai_passthrough_stream_admitted,
        inference._anthropic_passthrough_stream,
    ):
        src = inspect.getsource(fn)
        assert re.search(
            r"if cancel_event\.is_set\(\):\n(?:\s*#.*\n|\s*from core[^\n]*\n|\s*\n)*"
            r'\s*mark_response_failed\(getattr\(request, "scope", None\)\)',
            src,
        ), fn.__name__


def test_non_gguf_load_failure_restores_preview_marker():
    # Codex P2: a failed non-GGUF (Unsloth/LoRA) load unloads only the new entry, so the
    # prior preview checkpoint can still be resident though the marker was cleared;
    # _load_model_impl restores its ownership on both a raised and a falsy-return failure.
    import inspect

    src = inspect.getsource(inference._load_model_impl)
    assert "_restore_marker_if_prior_preview_still_resident" in src
    # Restored on both failure paths: the except (raise) and the falsy-success branch.
    assert src.count("_restore_marker_if_prior_preview_still_resident()") >= 2


def test_gguf_load_failure_restores_preview_marker():
    # Codex P2 (round 17): a GGUF load can raise before it tears down the old llama-server
    # (e.g. an update-in-progress guard fires before _kill_process), leaving the prior
    # preview-owned GGUF resident while the marker was cleared -- later previews would 503
    # against it. _load_model_impl restores ownership on both raised and falsy-return GGUF
    # failures, mirroring the non-GGUF path via a shared helper.
    import inspect

    src = inspect.getsource(inference._load_model_impl)
    # The restore helper is defined once and shared by the GGUF + non-GGUF paths.
    assert src.count("def _restore_marker_if_prior_preview_still_resident") == 1
    # Non-GGUF (2) + GGUF (2) failure sites all restore via the shared helper.
    assert src.count("_restore_marker_if_prior_preview_still_resident()") >= 5  # incl def
    # load_with_tensor_fallback is wrapped so both a raise and a falsy return restore the
    # marker before propagating the failure.
    gguf_fail_region = src[
        src.index("load_with_tensor_fallback(") : src.index("Failed to load GGUF model")
    ]
    assert "except Exception:" in gguf_fail_region
    assert "_restore_marker_if_prior_preview_still_resident()" in gguf_fail_region


def test_preload_unload_teardown_restores_preview_marker():
    # Codex P2 (round 29): before loading the new model each path tears down the other backend's
    # active model, and that teardown runs after the marker was cleared. If the unload raises
    # with the prior preview-owned checkpoint still resident, the marker must be restored or a
    # later preview for another checkpoint is 503'd against a model Studio never adopted.
    import inspect

    src = inspect.getsource(inference._load_model_impl)
    # Non-GGUF path: the pre-load llama_backend.unload_model() teardown is wrapped so a raise
    # restores the marker before propagating.
    non_gguf = src[src.index("Unloading GGUF model before loading Unsloth model") :]
    non_gguf = non_gguf[: non_gguf.index("Shut down any export subprocess")]
    assert "llama_backend.unload_model()" in non_gguf
    assert "_restore_marker_if_prior_preview_still_resident()" in non_gguf
    assert "raise" in non_gguf
    # GGUF path: the pre-load unsloth_backend.unload_model teardown is wrapped the same way.
    gguf = src[src.index("before loading GGUF") :]
    gguf = gguf[: gguf.index("Inherit llama_extra_args")]
    assert "_restore_marker_if_prior_preview_still_resident()" in gguf
    assert "raise" in gguf


def test_monitor_openai_sse_event_reports_error_status():
    # Codex P2 (round 17): the legacy /v1/completions relay forwards an upstream HTTP-200
    # SSE `data: {"error": ...}` event. _monitor_openai_sse_event must report "error" so the
    # relay marks the response failed and the middleware doesn't claim a slot on a failed stream.
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
    # The relay must mark the response failed when a relayed event is an upstream error, so
    # a preview-owned model isn't claimed after a failed completion stream.
    import inspect

    src = inspect.getsource(inference.openai_completions)
    # Format-independent: the relay checks the event status for an error and marks the scope
    # failed (assertions survive an autoformatter reflow of the condition).
    assert "_monitor_openai_sse_event(" in src
    assert '== "error"' in src
    assert 'mark_response_failed(getattr(request, "scope", None))' in src


def test_anthropic_stream_error_event_marks_failed_even_when_unclassified(slot_state):
    # Codex P2 (round 18): an unclassified /v1/messages generator error returns None (no
    # in-band error event) and the caller swallows it, finishing the stream normally. The
    # response must still be flagged failed so the middleware doesn't claim a slot.
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
    # ReadError/CloseError) re-raises after the 200 headers without an error SSE, so it must
    # flag the response failed first or the middleware claims a slot on an interrupted stream.
    import inspect

    src = inspect.getsource(inference._openai_passthrough_stream_admitted)
    idx = src.index("httpx.RemoteProtocolError, httpx.ReadError, httpx.CloseError")
    # Bound the slice to the transport-error branch (up to the next except).
    branch = src[idx : src.index("except HTTPException", idx)]
    assert 'mark_response_failed(getattr(request, "scope", None))' in branch
    assert "raise" in branch


def test_anthropic_passthrough_non_200_marks_failed():
    # Codex P2 (round 18): the passthrough delivers an upstream 4xx/5xx as an in-band
    # Anthropic error event under the outer 200 headers, so it must flag the response failed
    # before yielding it, or the middleware claims a slot on a passthrough that errored.
    import inspect

    src = inspect.getsource(inference._anthropic_passthrough_stream)
    idx = src.index("anthropic passthrough upstream error")
    # Bound the slice to the non-200 branch (up to the in-band error event).
    branch = src[idx : src.index("yield build_anthropic_sse_event", idx)]
    assert 'mark_response_failed(getattr(request, "scope", None))' in branch


def test_admission_cancel_marks_response_failed():
    # Codex P2 (round 19): a stream cancelled before leasing an upstream returns 200 with no
    # body via the caller's `except LlamaAdmissionCancelled` path, which never sets
    # _RESPONSE_FAILED_SCOPE_KEY. _raise_if_openai_admission_cancelled is the single choke
    # point for that cancellation, so it must flag the response failed, else the middleware
    # claims a slot for a stream that never ran. (Harmless for non-streaming callers.)
    import inspect
    src = inspect.getsource(inference._raise_if_openai_admission_cancelled)
    # Flagged on both branches (already-cancelled and pre-header cancel).
    assert src.count("mark_current_response_failed()") >= 2


def test_preview_not_blocked_by_pending_non_preview_waiter(fake_slot):
    # Codex P2 (round 19): a queued (pending) non-preview request -- possibly an
    # unauthenticated probe that 401s and never touches the model, since the middleware
    # bumps _pending before auth -- must not block a preview. Genuine queued Studio requests
    # are covered by the swap-reject on waking, not by the busy guard counting all pending.
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


def test_admitted_inference_counter_excludes_previews():
    # Codex P2 (round 21): the middleware tracks a POST in _inflight before auth, so the busy
    # guard counts only ADMITTED (post-auth) non-preview inference, not raw _inflight. An
    # in-flight-but-unadmitted request isn't counted; note_admitted_inference counts a non-
    # preview scope and skips a preview one; the finish decrement balances it.
    from core.inference import llama_keepwarm as kw

    # Reset residue: a direct _maybe_auto_switch_model unit call increments the counter
    # without the middleware _finish that decrements it in production.
    kw._admitted_inference = 0
    kw._inflight += 1  # non-preview request tracked pre-auth (never reached the hook)
    try:
        assert kw.other_admitted_inference_count() == 0  # unadmitted in-flight not counted
        scope = {"path": "/v1/chat/completions"}
        kw.note_admitted_inference(scope)  # passed auth, reached the inference hook
        assert kw.other_admitted_inference_count() == 1
        kw.note_admitted_inference(scope)  # idempotent per scope
        assert kw.other_admitted_inference_count() == 1
        # A preview scope is never admitted (it carries its own ownership).
        kw.note_admitted_inference({"path": "/p/run/v1/chat/completions"})
        assert kw.other_admitted_inference_count() == 1
        kw._note_admitted_end()  # middleware _finish balances the admit
        assert kw.other_admitted_inference_count() == 0
    finally:
        kw._inflight = 0
        kw._admitted_inference = 0


def test_preview_not_blocked_by_unadmitted_inflight_request(fake_slot, monkeypatch):
    # Codex P2 (round 21): a pre-auth non-preview request sits in _inflight (tracked before
    # auth) but never calls note_admitted_inference, so the busy guard (admitted-only) lets a
    # preview proceed. Use the real counter here, not the fake_slot stub.
    from core.inference import llama_keepwarm as kw

    monkeypatch.setattr(
        llama_keepwarm, "other_admitted_inference_count", kw.other_admitted_inference_count
    )
    kw._admitted_inference = 0  # reset residue from prior direct _maybe_auto_switch calls
    kw._inflight += 1  # unauthenticated non-preview request, tracked but never admitted
    try:
        asyncio.run(
            inference.load_model_for_preview(
                LoadRequest(model_path = "/outputs/run/ckpt"),
                SimpleNamespace(app = None, scope = {"path": "/p/run/v1/chat/completions"}),
                "admin",
            )
        )
        assert fake_slot["loads"] == ["/outputs/run/ckpt"]  # not blocked
    finally:
        kw._inflight = 0
        kw._admitted_inference = 0


def test_preview_swapped_since_entry_catches_pre_admission_swap():
    # Codex P2 (round 22): a non-preview request that passed the gate BEFORE a preview swap
    # never gets _PREVIEW_SWAP_REJECT_SCOPE_KEY (it didn't wait on the gate during the swap).
    # preview_swapped_since_entry compares the generation captured at entry against the
    # current one, so admission still rejects it after the swap; an in-progress swap and the
    # gate-wait flag also reject.
    from core.inference import llama_keepwarm as kw

    kw._preview_swap_generation = 5
    kw._preview_swap_inflight = 0
    try:
        # Entered at generation 5, no swap since: not rejected.
        scope = {"path": "/v1/chat/completions", kw._SWAP_GEN_AT_ENTRY_KEY: 5}
        assert kw.preview_swapped_since_entry(scope) is False
        # A swap completes (generation advances) while it is pre-admission.
        kw._preview_swap_generation = 6
        assert kw.preview_swapped_since_entry(scope) is True
        # An in-progress swap rejects even before the generation advances.
        scope2 = {"path": "/v1/chat/completions", kw._SWAP_GEN_AT_ENTRY_KEY: 6}
        assert kw.preview_swapped_since_entry(scope2) is False
        kw._preview_swap_inflight = 1
        assert kw.preview_swapped_since_entry(scope2) is True
        # The explicit gate-wait flag still rejects (checked first).
        kw._preview_swap_inflight = 0
        assert kw.preview_swapped_since_entry({kw._PREVIEW_SWAP_REJECT_SCOPE_KEY: True}) is True
        # A non-dict scope is never rejected.
        assert kw.preview_swapped_since_entry(None) is False
        # Codex P1 (round 23): a preview scope is NEVER rejected -- load_model_for_preview
        # bumps the generation before serving the preview's own chat, so the preview must not
        # reject itself after loading.
        kw._preview_swap_inflight = 1
        kw._preview_swap_generation = 42
        prev = {"path": "/p/run/v1/chat/completions", kw._SWAP_GEN_AT_ENTRY_KEY: 5}
        assert kw.preview_swapped_since_entry(prev) is False
    finally:
        kw._preview_swap_generation = 0
        kw._preview_swap_inflight = 0


def test_anthropic_stream_cancel_marks_response_failed():
    # Codex P2 (round 23): _anthropic_tool_stream / _anthropic_plain_stream return on
    # cancel_event / disconnect with a 200 and no completion, so they must flag the response
    # failed or the middleware claims a slot for a cancelled stream.
    import inspect
    for fn in (inference._anthropic_tool_stream, inference._anthropic_plain_stream):
        src = inspect.getsource(fn)
        idx = src.index("cancel_event.is_set() or await request.is_disconnected()")
        branch = src[idx : src.index("_sentinel)", idx)]
        assert 'mark_response_failed(getattr(request, "scope", None))' in branch, fn.__name__


def test_audio_input_stream_cancel_marks_response_failed():
    # Codex P2 (round 23): the audio-input streaming path's cancel_event break and disconnect
    # return end a 200 with no completion, so both must flag the response failed (alongside the
    # exception path) or a cancelled audio stream claims a slot.
    import inspect

    src = inspect.getsource(inference.openai_chat_completions)
    assert "async def audio_input_stream()" in src
    a_src = src[src.index("async def audio_input_stream()") :]
    # cancel break + disconnect return + except path all mark failed.
    assert a_src.count("mark_current_response_failed()") >= 3


def test_responses_stream_failure_paths_mark_response_failed():
    # Codex P2 (round 20): _responses_stream emits its own response.failed SSE events
    # (admission timeout, upstream unreachable, non-200, transport error, generic exception)
    # after the 200 headers. With claim_resident=False the middleware would otherwise treat
    # that 2xx as a successful generation and clear preview ownership, so every failed-response
    # builder / inline emitter flags the response failed before yielding.
    import inspect
    src = inspect.getsource(inference._responses_stream)
    # 2 failure-only builders (admission-failed + failed_response_payload, covering the
    # transport-error and generic-exception yields) plus 2 inline emitters (upstream-
    # unreachable, non-200) all mark the response failed.
    assert src.count('mark_response_failed(getattr(request, "scope", None))') >= 4


def test_responses_stream_disconnect_and_error_chunk_mark_failed():
    # Codex P2 (round 25): _responses_stream's post-loop and except-branch disconnect returns
    # ended a 200 with no completion, and an upstream HTTP-200 data:{"error"} SSE chunk (no
    # choices) was treated as a usage-only frame -- both let the middleware claim a preview-owned
    # slot on a stream that never completed. Mark the disconnect returns failed, and convert the
    # error chunk to response.failed before any response.completed can fire.
    import inspect

    src = inspect.getsource(inference._responses_stream)
    # The two round-20 marks (4) plus the two round-25 disconnect-return marks -> >= 6.
    assert src.count('mark_response_failed(getattr(request, "scope", None))') >= 6
    # The chunk loop detects an upstream error payload and converts it to response.failed.
    assert "_monitor_openai_error_message(chunk_data)" in src
    err_idx = src.index("_monitor_openai_error_message(chunk_data)")
    err_branch = src[err_idx : src.index('_apply_usage(chunk_data.get("usage"))', err_idx)]
    assert "_failed_response_payload(RuntimeError(error_message)" in err_branch
    assert "return" in err_branch


def test_anthropic_passthrough_error_chunk_marks_failed():
    # Codex P2 (round 28): the Anthropic passthrough ignored HTTP-200 data:{"error"} chunks
    # (emitter.feed_chunk drops chunks without choices), so the stream finished cleanly and the
    # middleware claimed a preview-owned model. The loop now detects the error before feed_chunk,
    # flags the response failed, and surfaces an Anthropic error event before returning.
    import inspect

    src = inspect.getsource(inference._anthropic_passthrough_stream)
    assert "_monitor_openai_error_message(chunk)" in src
    err_idx = src.index("_monitor_openai_error_message(chunk)")
    err_branch = src[err_idx : src.index("emitter.feed_chunk(chunk)", err_idx)]
    assert 'mark_response_failed(getattr(request, "scope", None))' in err_branch
    assert "_anthropic_stream_error_event(" in err_branch
    assert "return" in err_branch


def test_generate_stream_cancel_marks_response_failed():
    # Codex P2 (round 20): generate_stream's `if cancel_event.is_set(): ... break` ends a 200
    # stream with no completion (client disconnect, possibly before the first token) without
    # the except path, so it must flag the response failed or the middleware claims a slot.
    import inspect

    src = inspect.getsource(inference.generate_stream)
    # Marked in the cancel-break branch as well as the except handler.
    assert src.count("mark_response_failed(_gs_scope)") >= 2
    cancel_idx = src.index("if cancel_event.is_set():")
    branch = src[cancel_idx : src.index("chunk = await asyncio.to_thread", cancel_idx)]
    assert "mark_response_failed(_gs_scope)" in branch


def test_preheader_cancel_marks_response_failed():
    # Codex P2 (round 24): _send_stream_with_preheader_cancel returns None when the client
    # cancels/disconnects before the upstream response starts. Every caller then ends its 200
    # stream without serving tokens, so the helper flags the scope failed once (covering all
    # callers -- legacy completions, responses, Anthropic/OpenAI passthrough) or the middleware
    # claims a preview-owned slot for a request that never used the model.
    import inspect

    src = inspect.getsource(inference._send_stream_with_preheader_cancel)
    assert 'mark_response_failed(getattr(request, "scope", None))' in src
    # Both return-None paths flag the scope: the pre-send check and the post-send cancel race.
    assert src.count("_mark_preheader_cancel()") >= 2


def test_native_chat_stream_cancels_mark_response_failed():
    # Codex P2 (round 24): the GGUF/safetensors chat streaming loops end a 200 with no
    # completion on cancel_event (client disconnect via the watcher, checked first) or on
    # is_disconnected, so each cancel-break and disconnect-return must flag the response failed
    # or the middleware claims a preview-owned slot for a cancelled stream. Round 29 adds the
    # post-loop finalize mark: a cancel observed inside the threaded next() call returns the
    # sentinel and breaks without hitting the pre-next check, so the finalize marks it too.
    import inspect
    src = inspect.getsource(inference.openai_chat_completions)
    for gen in ("gguf_tool_stream", "gguf_stream_chunks", "sf_tool_stream", "stream_chunks"):
        start = src.index(f"async def {gen}()")
        try:
            nxt = src.index("async def ", start + 1)
        except ValueError:
            nxt = len(src)
        block = src[start:nxt]
        # cancel-break + disconnect-return + post-loop sentinel-break all mark failed.
        assert block.count("mark_current_response_failed()") >= 3, gen
        # The post-loop mark sits just before the "completed" finalize, guarding the
        # sentinel-break exit that the pre-next cancel check cannot catch.
        finalize = block.index('"cancelled" if cancel_event.is_set() else "completed"')
        guard = block.rindex("if cancel_event.is_set():", 0, finalize)
        assert "mark_current_response_failed()" in block[guard:finalize], gen


def test_non_streaming_tool_cancel_marks_response_failed():
    # Codex P2 (round 30): the GGUF and safetensors tool drains break mid-generation on an
    # explicit /inference/cancel but still return a JSON 200, so each must flag the response
    # failed before returning or the middleware claims a preview-owned slot for a cancelled
    # completion (round 29 covered their streaming siblings; these are the non-streaming ones).
    import inspect

    src = inspect.getsource(inference.openai_chat_completions)
    sentinel = '"cancelled" if cancel_event.is_set() else "completed"'
    non_streaming = 0
    pos = 0
    while (j := src.find(sentinel, pos)) != -1:
        pos = j + len(sentinel)
        # The non-streaming finalizes are the ones whose next statement returns a JSON body.
        if "return _model_json_response(response)" in src[j : j + 700]:
            assert "mark_current_response_failed()" in src[max(0, j - 400) : j]
            non_streaming += 1
    assert non_streaming >= 2  # GGUF tool drain + safetensors tool drain


def test_deferred_claim_load_leaves_new_model_preview_owned():
    # Codex P2 (round 24): a claim_resident=False auto-switch load clears the preview marker
    # mid-load, but the route still runs post-switch checks that can 4xx. On a rejection the
    # middleware skips its 2xx claim, so the freshly loaded model must be marked preview-owned
    # (evictable by the next preview) rather than stranded as Studio-owned.
    import inspect

    src = inspect.getsource(inference._maybe_auto_switch_model)
    after_load = src[src.index("await _load_model_impl(") :]
    assert "if not claim_resident:" in after_load
    assert "_set_preview_resident(_loaded_slot_ident())" in after_load
    # Round 27: the load is wrapped in try/finally so a post-load raise (the new model is
    # resident, then _load_model_impl raises while building the response) still marks the
    # resident slot preview-owned, or restores the prior owner if the target never loaded.
    assert "finally:" in after_load
    assert "_set_preview_resident(_switch_prior_marker)" in after_load


def test_middleware_claims_slot_on_successful_non_preview_response(slot_state):
    # A non-preview inference returning 2xx adopts the resident model, so the preview
    # marker is cleared on completion.
    _reset_keepwarm_counters()
    inference._set_preview_resident("/outputs/run/ckpt-a")

    async def _app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    _run_middleware(_app, "/v1/chat/completions")
    assert not inference._is_preview_resident("/outputs/run/ckpt-a")  # claimed for Studio
    _reset_keepwarm_counters()


def test_middleware_preview_response_keeps_ownership(slot_state):
    # A /p preview response must not claim the slot: it keeps its own ownership so a later
    # preview can still swap it.
    _reset_keepwarm_counters()
    inference._set_preview_resident("/outputs/run/ckpt")

    async def _app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    _run_middleware(_app, "/p/demo/v1/chat/completions")
    assert inference._is_preview_resident("/outputs/run/ckpt")  # ownership preserved
    _reset_keepwarm_counters()


def test_middleware_rejected_response_keeps_preview_ownership(slot_state):
    # A non-preview request rejected by a capability check (non-2xx) never ran, so it must
    # not claim -- the preview keeps ownership.
    _reset_keepwarm_counters()
    inference._set_preview_resident("/outputs/run/ckpt")

    async def _app(scope, receive, send):
        await send({"type": "http.response.start", "status": 400, "headers": []})
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    _run_middleware(_app, "/v1/chat/completions")
    assert inference._is_preview_resident("/outputs/run/ckpt")  # not claimed on 4xx
    _reset_keepwarm_counters()


def test_slot_claim_happens_before_inflight_decrement(slot_state, monkeypatch):
    # Codex P2: the middleware clears preview ownership BEFORE decrementing in-flight. Doing
    # it after opens a window where a preview for another checkpoint sees no non-preview
    # traffic and a still-preview-owned slot, swaps in, and the delayed claim clears the wrong model.
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
    # Still counted in-flight when the claim runs, so the busy guard refuses a concurrent
    # swap; only then is the count decremented.
    assert observed["inflight_at_claim"] == 1
    assert llama_keepwarm._inflight == 0  # decremented afterwards
    _reset_keepwarm_counters()


def test_slot_claim_happens_before_admitted_decrement(slot_state, monkeypatch):
    # Codex P2 (round 29): the middleware clears preview ownership BEFORE dropping the admitted
    # count. load_model_for_preview's busy guard keys on other_admitted_inference_count(), so
    # decrementing first opens a window where a preview sees no admitted Studio traffic and a
    # still-preview-owned slot, swaps in, and the delayed claim then clears the wrong checkpoint.
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
    # Still counted admitted when the claim runs, so a concurrent preview's busy guard refuses
    # the swap; only then is the admit balanced.
    assert observed["admitted_at_claim"] == 1
    assert llama_keepwarm._admitted_inference == 0  # decremented afterwards
    assert not inference._is_preview_resident("/outputs/run/ckpt-a")  # claimed for Studio
    _reset_keepwarm_counters()
    llama_keepwarm._admitted_inference = 0


def test_preview_swap_marked_before_admitted_check():
    # Codex P2 (round 30): load_model_for_preview must set the swap-in-progress marker BEFORE
    # reading other_admitted_inference_count(), so the check and marker are atomic. A Studio
    # request admitted before the marker is caught by the busy check; one admitted after is
    # rejected by preview_swapped_since_entry (which keys on the live _preview_swap_inflight).
    # Setting the marker only after the check left a gap where a Studio request admitted
    # between the two ran against the slot this preview was about to replace.
    import inspect

    src = inspect.getsource(inference.load_model_for_preview)
    begin = src.index("note_preview_swap_begin()")
    check = src.index("other_admitted_inference_count()")
    assert begin < check, "swap marker must be set before the admitted-count busy check"


def test_preview_swap_marker_skipped_for_same_target_borrow():
    # Codex P2 (round 31): a same-target borrow changes nothing (no swap counter bump), so it
    # must not set the swap-in-progress marker -- doing so 503s concurrent Studio requests via
    # preview_swapped_since_entry for no reason. The marker is gated on not same_target and
    # still precedes the admitted-count check for the real-load path.
    import inspect

    src = inspect.getsource(inference.load_model_for_preview)
    guard = src.index("if not same_target:")
    begin = src.index("note_preview_swap_begin()", guard)
    check = src.index("other_admitted_inference_count()")
    # The marker is the first statement under the not-same_target guard, still before the check.
    assert guard < begin < check
    assert "note_preview_swap_begin()" in src[guard : guard + 200]


def test_preview_same_checkpoint_matches_equivalent_spellings(tmp_path):
    # Codex P2 (round 31): Studio may load a checkpoint via a spelling (a relative outputs/run)
    # different from the absolute path the preview resolver produces. _preview_same_checkpoint
    # borrows the slot for an equivalent filesystem path while never matching a distinct
    # checkpoint or a non-path identifier (an HF repo id resolves under the cwd).
    ckpt = tmp_path / "outputs" / "run"
    ckpt.mkdir(parents = True)
    abs_path = str(ckpt)
    # Exact string match (fast path, no resolve).
    assert inference._preview_same_checkpoint(abs_path, abs_path)
    # Equivalent spelling with a redundant "." segment resolves to the same path.
    dotted = str(tmp_path / "outputs" / "." / "run")
    assert inference._preview_same_checkpoint(dotted, abs_path)
    # A different checkpoint under the same root must not match.
    assert not inference._preview_same_checkpoint(str(tmp_path / "outputs" / "other"), abs_path)
    # A non-path identifier (HF repo id) never matches a filesystem checkpoint.
    assert not inference._preview_same_checkpoint("org/model", abs_path)


def test_rejected_preview_request_does_not_refresh_idle_timer(slot_state):
    # Codex P2: a preview POST rejected before loading (429, bad-token 404, body 4xx) never
    # served tokens, so it must not refresh the idle timer -- else repeated rejected POSTs
    # pin an idle model in VRAM and idle-unload can never free it.
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
    # Contrast: a preview that served (2xx) DOES stamp activity, keeping the model warm
    # for its duration.
    _reset_keepwarm_counters()
    llama_keepwarm._last_active = 0.0

    async def _ok(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    _run_middleware(_ok, "/p/demo/v1/chat/completions")
    assert llama_keepwarm._last_active > 0.0  # _note_end stamped activity
    _reset_keepwarm_counters()
