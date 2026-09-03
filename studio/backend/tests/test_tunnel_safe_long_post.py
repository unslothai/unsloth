# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A load slower than a proxy's idle timer still reaches the client.

Cloudflare quick tunnels (`--secure`) drop a request whose origin has sent no body
bytes for ~100s, and a 600 GB GGUF load runs 100-330s, so the browser reported
"Request failed" on loads the server completed with a 200. Measured on a real
quick tunnel:

  * no body for 150s                -> 524
  * headers at t=0, no body         -> 524 (headers are NOT enough)
  * one byte at t=90s, then silence -> killed ~125s later, client sees 200 with an
                                       EMPTY body
  * one space every 20s             -> survives, body intact

So the padding must be continuous, and a failure found after the status commits
can only travel in the body.

Two consequences below: the padded reply is a StreamingResponse, so an in-process
caller that drains no body awaits ``load_model_gated`` instead; and a client that
treats any 200 as success has to learn the in-band failure key.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

_backend_root = Path(__file__).resolve().parent.parent
_repo_root = _backend_root.parents[1]
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))


@pytest.fixture
def route(monkeypatch):
    from routes import inference as route_mod

    # Keep the test fast; the real defaults are guarded by the last test below.
    monkeypatch.setattr(route_mod, "_TUNNEL_KEEPALIVE_AFTER_S", 0.05)
    monkeypatch.setattr(route_mod, "_TUNNEL_KEEPALIVE_EVERY_S", 0.02)
    return route_mod


async def _collect(response):
    """Chunks a client would receive, in order."""
    return [chunk async for chunk in response.body_iterator]


async def _collect_into(response, sink):
    """As _collect, but visible to another thread while the stream is still open."""
    async for chunk in response.body_iterator:
        sink.append(chunk)
    return sink


def test_a_fast_call_keeps_the_plain_response(route):
    async def quick():
        return {"status": "loaded"}

    result = asyncio.run(route._tunnel_safe_json(quick(), label = "t"))
    # Not a Response: FastAPI serialises it through response_model as before.
    assert result == {"status": "loaded"}


def test_a_fast_failure_keeps_its_status_code(route):
    async def boom():
        raise HTTPException(status_code = 409, detail = "busy")

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(route._tunnel_safe_json(boom(), label = "t"))
    assert excinfo.value.status_code == 409
    assert excinfo.value.detail == "busy"


def test_a_slow_call_pads_then_sends_valid_json(route):
    async def slow():
        await asyncio.sleep(0.2)
        return {"status": "loaded", "model": "unsloth/Kimi-K3-GGUF"}

    async def run():
        response = await route._tunnel_safe_json(slow(), label = "t")
        return response, await _collect(response)

    response, chunks = asyncio.run(run())
    assert response.media_type == "application/json"
    assert response.headers["x-accel-buffering"] == "no"
    assert chunks[0] == b" "
    assert len(chunks) > 2, "a call this slow must be padded more than once"
    assert all(c == b" " for c in chunks[:-1])
    body = b"".join(chunks)
    # Leading whitespace is legal JSON, which is why no client had to change.
    assert json.loads(body) == {"status": "loaded", "model": "unsloth/Kimi-K3-GGUF"}


def test_a_slow_failure_travels_in_the_body(route):
    async def slow_boom():
        await asyncio.sleep(0.2)
        raise HTTPException(status_code = 500, detail = "llama-server died")

    async def run():
        return await _collect(await route._tunnel_safe_json(slow_boom(), label = "t"))

    body = json.loads(b"".join(asyncio.run(run())))
    assert body["_deferred_error"] == {"status_code": 500, "detail": "llama-server died"}


def test_an_unexpected_slow_failure_becomes_a_500(route):
    async def slow_boom():
        await asyncio.sleep(0.2)
        raise RuntimeError("out of VRAM")

    async def run():
        return await _collect(await route._tunnel_safe_json(slow_boom(), label = "t"))

    body = json.loads(b"".join(asyncio.run(run())))
    assert body["_deferred_error"]["status_code"] == 500
    assert "out of VRAM" in body["_deferred_error"]["detail"]


def test_the_work_survives_a_client_disconnect(route):
    """Abandoning the stream must not cancel the load: the model still lands."""
    finished = []

    async def slow():
        await asyncio.sleep(0.2)
        finished.append(True)
        return {"status": "loaded"}

    async def run():
        response = await route._tunnel_safe_json(slow(), label = "t")
        it = response.body_iterator
        assert await it.__anext__() == b" "  # one pad, then the client vanishes
        await it.aclose()
        await asyncio.sleep(0.4)

    asyncio.run(run())
    assert finished == [True]


# ── Direct (in-process) callers must not get the padded response ──────────────


@pytest.fixture
def slow_load(route, monkeypatch):
    """/load whose work outruns the keepalive timer; records the finished model paths."""
    finished = []

    async def _slow_impl(request, fastapi_request, current_subject, **kwargs):
        await asyncio.sleep(0.2)  # >> the fixture's 0.05s keepalive threshold
        finished.append(request.model_path)
        return {"status": "loaded", "model": request.model_path}

    monkeypatch.setattr(route, "_load_model_impl", _slow_impl)
    # The sidecar-swap guard reads real install state; it has its own tests.
    monkeypatch.setattr(route, "_raise_if_sidecar_swap_in_progress", lambda: None)
    return finished


def _load_request(model_path = "unsloth/Kimi-K3-GGUF"):
    from models.inference import LoadRequest
    return LoadRequest(model_path = model_path)


def test_an_in_process_caller_gets_the_real_result(route, slow_load):
    """preview awaits load_model_gated, so a slow load blocks until the model is
    resident; awaiting the route would hand back an undrained StreamingResponse."""
    result = asyncio.run(route.load_model_gated(_load_request(), None, "admin"))

    assert not isinstance(result, StreamingResponse)
    assert result == {"status": "loaded", "model": "unsloth/Kimi-K3-GGUF"}
    # Returned only after the load finished, not at the 15s mark.
    assert slow_load == ["unsloth/Kimi-K3-GGUF"]


def test_an_in_process_caller_sees_a_late_failure(route, monkeypatch):
    """A late failure raises for a direct caller, not a 200 carrying `_deferred_error`."""

    async def _slow_boom(request, fastapi_request, current_subject, **kwargs):
        await asyncio.sleep(0.2)
        raise HTTPException(status_code = 507, detail = "CUDA out of memory")

    monkeypatch.setattr(route, "_load_model_impl", _slow_boom)
    monkeypatch.setattr(route, "_raise_if_sidecar_swap_in_progress", lambda: None)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(route.load_model_gated(_load_request(), None, "admin"))
    assert excinfo.value.status_code == 507
    assert excinfo.value.detail == "CUDA out of memory"


def test_the_route_still_pads_the_same_slow_load(route, slow_load):
    """The other half of the split: real HTTP clients keep the padding."""

    async def run():
        response = await route.load_model(_load_request(), None, "admin")
        return response, b"".join(await _collect(response))

    response, body = asyncio.run(run())
    assert isinstance(response, StreamingResponse)
    assert body.startswith(b" ")
    assert json.loads(body) == {"status": "loaded", "model": "unsloth/Kimi-K3-GGUF"}


def test_only_the_route_pads(route):
    """Padding belongs to the route: a gated-coroutine caller must not inherit a stream."""
    import inspect

    assert "_tunnel_safe_json" in inspect.getsource(route.load_model)
    assert "_tunnel_safe_json" not in inspect.getsource(route.load_model_gated)
    # /unload has no in-process caller today; _unload_model_impl is its equivalent.
    assert "_tunnel_safe_json" in inspect.getsource(route.unload_model)
    assert "_tunnel_safe_json" not in inspect.getsource(route._unload_model_impl)


def test_preview_awaits_the_gated_load(route):
    """preview.py is the one in-process caller; it must bypass the padding."""
    src = (_backend_root / "routes" / "preview.py").read_text(encoding = "utf-8")
    assert "load_model_gated(" in src
    assert re.search(r"\bawait load_model\(", src) is None, (
        "preview must not await the padded /load route: its StreamingResponse "
        "returns before the checkpoint is loaded"
    )


def test_preview_chat_waits_for_a_slow_checkpoint_load(route, slow_load, monkeypatch, tmp_path):
    """End to end: /p must not start generating while the checkpoint is still loading."""
    import routes.preview as preview
    from models.inference import ChatCompletionRequest

    checkpoint = tmp_path / "run" / "checkpoint-1"
    checkpoint.mkdir(parents = True)
    monkeypatch.setattr(preview, "_resolve_or_4xx", lambda run, cp: checkpoint)

    loaded_when_chat_started = []

    async def _fake_chat(payload, request, subject):
        loaded_when_chat_started.append(list(slow_load))
        return {"ok": True}

    monkeypatch.setattr(preview, "openai_chat_completions", _fake_chat)

    async def run():
        return await preview._serve_chat(
            "run",
            "checkpoint-1",
            ChatCompletionRequest(messages = [{"role": "user", "content": "hi"}]),
            request = None,
        )

    assert asyncio.run(run()) == {"ok": True}
    # Pre-fix this was empty: the padded StreamingResponse returned at the threshold.
    assert loaded_when_chat_started == [[str(checkpoint)]]
    assert not preview._preview_lock.locked()


# ── The slow teardown must not sit on the event loop ──────────────────────────
#
# ``LlamaCppBackend.unload_model`` is a plain ``def`` and a 600 GB teardown measures
# ~160s. Called bare it blocks _tunnel_safe_json's own timer and pad generator, so
# zero bytes leave and the proxy 524s anyway: padding dead on the two slowest paths.


class _SlowSyncTeardown:
    """A synchronous GGUF teardown that returns only once the padding has flowed.

    On the event loop it never sees a pad byte and falls out on ``cap_s`` with none.
    """

    def __init__(
        self,
        chunks,
        *,
        want = 3,
        cap_s = 2.0,
    ):
        self._chunks = chunks
        self._want = want
        self._cap_s = cap_s
        self.thread = None
        self.pads_while_running = None

    def __call__(self, *args, **kwargs):
        self.thread = threading.current_thread()
        deadline = time.monotonic() + self._cap_s
        while len(self._chunks) < self._want and time.monotonic() < deadline:
            time.sleep(0.005)
        self.pads_while_running = len(self._chunks)
        return True

    def assert_padded_off_the_loop(self, label):
        assert self.thread is not None, f"{label} never ran"
        assert self.thread is not threading.main_thread(), f"{label} ran on the event loop thread"
        assert self.pads_while_running is not None and self.pads_while_running >= self._want, (
            f"{label} blocked the padding: {self.pads_while_running} pad bytes went out "
            "while it ran, so a proxy would have timed the response out"
        )


def _no_active_generation_checks(route, monkeypatch):
    """Neutralise the chat-cancellation gate; test_active_generations owns it."""
    monkeypatch.setattr(route, "_raise_or_cancel_active_generations", lambda **kwargs: 0)

    async def _drain(**kwargs):
        return None

    monkeypatch.setattr(route, "_drain_and_recancel_before_teardown", _drain)


def _stub_unsloth_load_over_a_resident_gguf(route, monkeypatch, *, teardown):
    """POST /load over a resident GGUF: the branch that tears llama-server down first."""
    import core.export

    from core.inference import llama_keepwarm as kw

    _no_active_generation_checks(route, monkeypatch)
    monkeypatch.setattr(route, "_raise_if_sidecar_swap_in_progress", lambda: None)
    monkeypatch.setattr(route, "validate_extra_args", lambda args: [])
    monkeypatch.setattr(
        route,
        "_resolve_model_identifier_for_request",
        lambda request, operation: (request.model_path, request.model_path, False),
    )
    monkeypatch.setattr(
        route,
        "resolve_effective_chat_template_override",
        lambda model_identifier = None, user_override = None: None,
    )
    # Both guards: the load path resolves its config under the per-model one.
    monkeypatch.setattr(route, "_hf_offline_if_unreachable", contextlib.nullcontext)
    monkeypatch.setattr(route, "_hf_offline_if_unreachable_for", contextlib.nullcontext)
    monkeypatch.setattr(route, "_mlx_distributed_launch_detected", lambda: False)
    monkeypatch.setattr(
        route.ModelConfig,
        "from_identifier",
        staticmethod(
            lambda **kwargs: SimpleNamespace(
                is_gguf = False,
                identifier = "org/A",
                display_name = "A",
                is_vision = False,
                is_lora = False,
                is_audio = False,
                audio_type = None,
                has_audio_input = False,
                is_local = False,
                gguf_hf_repo = None,
                gguf_variant = None,
            )
        ),
    )
    monkeypatch.setattr(route, "_effective_load_in_4bit", lambda config, requested: False)
    monkeypatch.setattr(route, "_resolve_inherited_extra_args", lambda *a, **k: None)
    monkeypatch.setattr(route, "_guard_chat_load_against_training", lambda *a, **k: None)
    monkeypatch.setattr(route, "load_inference_config", lambda name: {})
    monkeypatch.setattr(
        route,
        "_detect_safetensors_features",
        lambda backend, template, tools = None: {
            "supports_reasoning": False,
            "reasoning_style": "enable_thinking",
            "reasoning_effort_levels": [],
            "reasoning_always_on": False,
            "supports_preserve_thinking": False,
            "supports_tools": False,
        },
    )
    monkeypatch.setattr(route, "_resolve_loaded_trust_remote_code", lambda *a, **k: False)
    monkeypatch.setattr(
        core.export, "get_export_backend", lambda: SimpleNamespace(current_checkpoint = None)
    )
    monkeypatch.setattr(kw, "note_model_loaded", lambda *a, **k: None)
    monkeypatch.setattr(
        route,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(
            is_loaded = True,  # a GGUF is resident and must come down first
            is_active = True,
            hf_variant = None,
            model_identifier = "org/OLD-GGUF",
            layer_preserves_tensor_intent = False,
            unload_model = teardown,
        ),
    )
    monkeypatch.setattr(
        route,
        "get_inference_backend",
        lambda: SimpleNamespace(
            active_model_name = "org/OTHER",
            models = {},
            load_model = lambda **kwargs: True,
        ),
    )


def test_a_slow_gguf_teardown_before_an_unsloth_load_still_pads(route, monkeypatch):
    """POST /load replacing a resident GGUF with a non-GGUF model."""
    from models.inference import LoadRequest

    chunks: list[bytes] = []
    teardown = _SlowSyncTeardown(chunks)
    _stub_unsloth_load_over_a_resident_gguf(route, monkeypatch, teardown = teardown)

    async def run():
        response = await route.load_model(LoadRequest(model_path = "org/A"), None, "tester")
        if not isinstance(response, StreamingResponse):
            return response
        await _collect_into(response, chunks)
        return response

    response = asyncio.run(run())
    # Pre-fix the teardown ran in the task's first step: a plain LoadResponse, no pads.
    assert isinstance(
        response, StreamingResponse
    ), "a load whose teardown blocks the loop can never be padded"
    teardown.assert_padded_off_the_loop("the pre-load GGUF teardown")
    assert json.loads(b"".join(chunks))["status"] == "loaded"


def _stub_gguf_unload(route, monkeypatch, *, teardown):
    """POST /unload for the already-loaded GGUF: the manual teardown branch."""
    from core.inference import llama_keepwarm as kw

    _no_active_generation_checks(route, monkeypatch)
    monkeypatch.setattr(route, "is_registered_native_path_label", lambda *a: False)
    monkeypatch.setattr(kw, "note_model_unloaded", lambda: None)
    monkeypatch.setattr(
        route,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(
            is_active = True,
            is_loaded = True,
            model_identifier = "org/A-GGUF",
            hf_variant = "UD-Q4_K_XL",
            unload_model = teardown,
        ),
    )
    monkeypatch.setattr(
        route,
        "get_inference_backend",
        lambda: SimpleNamespace(
            get_loading_model = lambda: None,
            active_model_name = None,
            models = {},
            unload_model = lambda path: None,
        ),
    )


def test_a_slow_gguf_unload_still_pads(route, monkeypatch):
    """POST /unload of a resident GGUF: the 160s path /unload is padded for."""
    from models.inference import UnloadRequest

    chunks: list[bytes] = []
    teardown = _SlowSyncTeardown(chunks)
    _stub_gguf_unload(route, monkeypatch, teardown = teardown)

    async def run():
        response = await route.unload_model(UnloadRequest(model_path = "org/A-GGUF"), "tester")
        if not isinstance(response, StreamingResponse):
            return response
        await _collect_into(response, chunks)
        return response

    response = asyncio.run(run())
    assert isinstance(
        response, StreamingResponse
    ), "an unload whose teardown blocks the loop can never be padded"
    teardown.assert_padded_off_the_loop("the manual GGUF teardown")
    assert json.loads(b"".join(chunks)) == {"status": "unloaded", "model": "org/A-GGUF"}


def test_every_gguf_teardown_on_a_padded_route_is_off_loop():
    """Fence: a new bare ``unload_model()`` in /load or /unload silently un-pads it."""
    import ast

    src = (_backend_root / "routes" / "inference.py").read_text(encoding = "utf-8")
    tree = ast.parse(src)
    bare: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef) or node.name not in (
            "_load_model_impl",
            "_unload_model_impl",
        ):
            continue
        for call in ast.walk(node):
            # A bare call: attribute .unload_model(...) rather than a to_thread arg.
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "unload_model"
            ):
                bare.append(call.lineno)
    assert bare == [], (
        f"unload_model called on the event loop at routes/inference.py:{bare}; "
        "wrap it in asyncio.to_thread or the padding never gets to run"
    )


# ── Non-browser clients ───────────────────────────────────────────────────────


def test_a_python_client_can_recognise_the_late_failure(route):
    """The CLI is not a browser: it treats any 200 as success, so it must know the same
    `_deferred_error` key and fields or a late OOM reads as a successful load."""

    async def slow_boom():
        await asyncio.sleep(0.2)
        raise HTTPException(status_code = 507, detail = "CUDA out of memory")

    async def run():
        return b"".join(await _collect(await route._tunnel_safe_json(slow_boom(), label = "t")))

    payload = json.loads(asyncio.run(run()))[route._DEFERRED_ERROR_KEY]
    assert payload == {"status_code": 507, "detail": "CUDA out of memory"}

    # Read as text, not imported: the backend test env need not import the CLI.
    cli = (_repo_root / "unsloth_cli" / "_inference.py").read_text(encoding = "utf-8")
    assert f'_DEFERRED_ERROR_KEY = "{route._DEFERRED_ERROR_KEY}"' in cli
    assert 'deferred.get("status_code")' in cli
    assert 'deferred.get("detail")' in cli
    # unsloth_cli/tests/test_inference_chat.py asserts it actually raises.
    assert "def raise_for_deferred_error(" in cli


def test_both_clients_reject_a_truncated_padded_body():
    """A proxy killing a padded response after the 200 committed leaves the measured
    200 with an EMPTY body. Both clients must call that a failure, else the same reply
    means "loaded" to one and "failed" to the other. Behaviour is tested in their own
    suites (test_inference_chat.py, padded-response.test.ts); this only pins that
    neither side can drop the check.
    """
    cli = (_repo_root / "unsloth_cli" / "_inference.py").read_text(encoding = "utf-8")
    assert "def require_completed_padded_body(" in cli
    assert "require_completed_padded_body(url, raise_for_deferred_error(url, body))" in cli

    web = (
        _repo_root
        / "studio"
        / "frontend"
        / "src"
        / "features"
        / "chat"
        / "api"
        / "padded-response.ts"
    ).read_text(encoding = "utf-8")
    assert "export function assertCompletedPaddedBody(" in web
    # Scoped to /load and /unload: shared parseJsonOrThrow serves ~30 endpoints,
    # some legitimately with no body.
    chat_api = (
        _repo_root / "studio" / "frontend" / "src" / "features" / "chat" / "api" / "chat-api.ts"
    ).read_text(encoding = "utf-8")
    assert re.findall(r'parseJsonOrThrow<[^>]*>\(\s*response,\s*"([^"]+)"', chat_api) == [
        "Model load",
        "Model unload",
    ]


def test_the_pad_interval_stays_inside_the_proxy_window():
    """The tunnel kills a connection ~100s after the last byte, so leave margin.

    Reads the source, not the module, so the fixture's fast values cannot mask a
    bad default.
    """
    src = (_backend_root / "routes" / "inference.py").read_text(encoding = "utf-8")
    after = float(src.split("_TUNNEL_KEEPALIVE_AFTER_S = ")[1].split("\n")[0])
    every = float(src.split("_TUNNEL_KEEPALIVE_EVERY_S = ")[1].split("\n")[0])
    assert after + every < 90.0, "first pad must land well before the ~100s cutoff"
    assert every < 90.0, "every subsequent gap resets the timer, so it must too"
