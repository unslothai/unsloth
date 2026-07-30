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

Two consequences the tests below pin down. The padded reply is a StreamingResponse,
so anything that does not drain a body must not go through the route: in-process
callers await ``load_model_gated`` instead. And a client that reads the body but
treats any 200 as success still has to be taught the in-band failure key.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

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
    """/load whose work outruns the keepalive timer. Returns the recorder of
    model paths the load actually finished for."""
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
    """preview awaits load_model_gated, so a slow load blocks it until the model
    is really resident. Awaiting the route instead would hand back a
    StreamingResponse that nothing in-process drains, and the preview chat would
    run against the previous model."""
    result = asyncio.run(route.load_model_gated(_load_request(), None, "admin"))

    assert not isinstance(result, StreamingResponse)
    assert result == {"status": "loaded", "model": "unsloth/Kimi-K3-GGUF"}
    # Returned only after the load finished, not at the 15s mark.
    assert slow_load == ["unsloth/Kimi-K3-GGUF"]


def test_an_in_process_caller_sees_a_late_failure(route, monkeypatch):
    """The same load failing late raises for a direct caller, instead of becoming
    a 200 whose body carries `_deferred_error`."""

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
    """``_tunnel_safe_json`` belongs to the route, not to the gated coroutine: a
    future in-process caller of the latter must not inherit a StreamingResponse."""
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
    """End to end through the real in-process caller: /p must not start generating
    while the checkpoint is still loading."""
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
    # The load had finished before the chat ran; pre-fix this list was empty
    # because the padded StreamingResponse returned at the keepalive threshold.
    assert loaded_when_chat_started == [[str(checkpoint)]]
    assert not preview._preview_lock.locked()


# ── Non-browser clients ───────────────────────────────────────────────────────


def test_a_python_client_can_recognise_the_late_failure(route):
    """The browser learned `_deferred_error` (chat-api.ts); the CLI is not a
    browser and treats any 200 as success, so it has to know the same key and
    fields or a late OOM is reported as a successful load."""

    async def slow_boom():
        await asyncio.sleep(0.2)
        raise HTTPException(status_code = 507, detail = "CUDA out of memory")

    async def run():
        return b"".join(await _collect(await route._tunnel_safe_json(slow_boom(), label = "t")))

    payload = json.loads(asyncio.run(run()))[route._DEFERRED_ERROR_KEY]
    assert payload == {"status_code": 507, "detail": "CUDA out of memory"}

    # The shared CLI helper every unsloth_cli load path funnels through. Read as
    # text, not imported: the backend test env need not import the CLI.
    cli = (_repo_root / "unsloth_cli" / "_inference.py").read_text(encoding = "utf-8")
    assert f'_DEFERRED_ERROR_KEY = "{route._DEFERRED_ERROR_KEY}"' in cli
    assert 'deferred.get("status_code")' in cli
    assert 'deferred.get("detail")' in cli
    # unsloth_cli/tests/test_inference_chat.py asserts it actually raises.
    assert "def raise_for_deferred_error(" in cli


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
