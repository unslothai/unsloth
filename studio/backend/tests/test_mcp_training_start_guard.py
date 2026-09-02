# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""End-to-end coverage for the MCP `start_training` inference guard.

PR #9434 made the MCP tool call POST /training/start with via_api_key = True, so
an MCP agent can no longer unload the chat model out from under a live stream.
The PR shipped only a forwarding assertion; these tests drive the real route (and
the real MCP tool) against a simulated in-flight inference request.
"""

import asyncio
import importlib.util
import threading
from pathlib import Path

import pytest
from fastapi import HTTPException

import core.inference.llama_keepwarm as keepwarm
from core.training.training import TrainingBackend
from models.training import TrainingStartRequest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _load_training_route(name: str):
    spec = importlib.util.spec_from_file_location(
        name,
        _BACKEND_ROOT / "routes" / "training.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(**overrides):
    payload = {
        "model_name": "unsloth/test",
        "training_type": "LoRA/QLoRA",
        "format_type": "alpaca",
    }
    payload.update(overrides)
    return payload


def _arm(
    monkeypatch,
    route,
    *,
    inflight = 0,
    video = False,
):
    """Point the route at a fresh backend and a controllable inference count."""
    backend = TrainingBackend()
    monkeypatch.setattr(route, "get_training_backend", lambda: backend)
    monkeypatch.setattr(
        keepwarm,
        "other_inference_request_count",
        lambda current_request_counted = True, **_: inflight,
    )
    monkeypatch.setattr(route, "_background_video_generation_active", lambda: video)
    return backend


async def _call(route, config):
    return await route.start_training(
        TrainingStartRequest.model_validate(config),
        current_subject = "mcp",
        via_api_key = True,
    )


# --------------------------------------------------------------------------------------
# The guard itself
# --------------------------------------------------------------------------------------


def test_live_chat_stream_refuses_the_mcp_start(monkeypatch):
    route = _load_training_route("training_route_guard_stream_test")
    _arm(monkeypatch, route, inflight = 1)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(_call(route, _config()))

    assert excinfo.value.status_code == 409
    assert "inference request is in progress" in excinfo.value.detail


def test_background_video_generation_also_refuses_the_mcp_start(monkeypatch):
    """Wider than the PR title: a background clip blocks MCP training too."""
    route = _load_training_route("training_route_guard_video_test")
    _arm(monkeypatch, route, inflight = 0, video = True)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(_call(route, _config()))

    assert excinfo.value.status_code == 409


def test_idle_backend_lets_the_mcp_start_through_the_guard(monkeypatch):
    """No inference in flight: the 409 must not fire (the regression question)."""
    route = _load_training_route("training_route_guard_idle_test")
    _arm(monkeypatch, route, inflight = 0)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(_call(route, _config()))

    # Validation past the guard still rejects the fake model, but never with the
    # guard's 409/"inference request is in progress".
    assert not (
        excinfo.value.status_code == 409
        and "inference request is in progress" in str(excinfo.value.detail)
    )


def test_stream_finishing_then_retrying_starts(monkeypatch):
    route = _load_training_route("training_route_guard_retry_test")
    counter = {"n": 1}
    backend = TrainingBackend()
    monkeypatch.setattr(route, "get_training_backend", lambda: backend)
    monkeypatch.setattr(
        keepwarm,
        "other_inference_request_count",
        lambda current_request_counted = True, **_: counter["n"],
    )
    monkeypatch.setattr(route, "_background_video_generation_active", lambda: False)

    with pytest.raises(HTTPException) as first:
        asyncio.run(_call(route, _config()))
    assert first.value.status_code == 409

    counter["n"] = 0
    with pytest.raises(HTTPException) as second:
        asyncio.run(_call(route, _config()))
    assert "inference request is in progress" not in str(second.value.detail)


# --------------------------------------------------------------------------------------
# Counting boundary cases (does the guard over-count and block valid MCP training?)
# --------------------------------------------------------------------------------------


def test_the_mcp_call_does_not_count_itself():
    """/mcp is not an inference path, so the MCP request is never tracked."""
    assert keepwarm._is_inference_path("/mcp") is False
    assert keepwarm._is_inference_path("/mcp/") is False


def test_idle_but_warm_model_counts_as_zero(monkeypatch):
    monkeypatch.setattr(keepwarm, "_inflight", 0)
    monkeypatch.setattr(keepwarm, "_pending", 0)
    assert keepwarm.other_inference_request_count(current_request_counted = False) == 0


def test_a_completed_request_is_reaped_by_the_middleware_finally():
    """A stream that ends (or raises) must not leave the count positive forever."""

    async def drive(explode):
        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200})
            if explode:
                raise RuntimeError("client vanished mid-stream")
            await send({"type": "http.response.body", "body": b"", "more_body": False})

        middleware = keepwarm.LlamaKeepWarmMiddleware(app)
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"authorization", b"Bearer sk-unsloth-x")],
        }
        try:
            await middleware(scope, None, lambda message: asyncio.sleep(0))
        except RuntimeError:
            pass

    before = keepwarm.other_inference_request_count(current_request_counted = False)
    asyncio.run(drive(False))
    assert keepwarm.other_inference_request_count(current_request_counted = False) == before
    asyncio.run(drive(True))
    assert keepwarm.other_inference_request_count(current_request_counted = False) == before


def test_an_untracked_external_provider_request_does_not_block_training():
    scope = {}
    before = keepwarm.other_inference_request_count(current_request_counted = False)
    keepwarm._note_pending()
    keepwarm._note_start()
    assert keepwarm.other_inference_request_count(current_request_counted = False) == before + 1
    keepwarm.untrack_current_request(scope)
    assert keepwarm.other_inference_request_count(current_request_counted = False) == before


def test_a_pending_waiter_counts_as_active(monkeypatch):
    """include_pending defaults True, so a queued chat also refuses training."""
    before = keepwarm.other_inference_request_count(current_request_counted = False)
    keepwarm._note_pending()
    try:
        assert keepwarm.other_inference_request_count(current_request_counted = False) == before + 1
    finally:
        keepwarm._note_unpending()


# --------------------------------------------------------------------------------------
# FINDING 1: the 409 poisons a caller-supplied start_request_id
# --------------------------------------------------------------------------------------


def test_a_guard_409_does_not_poison_the_supplied_start_request_id(monkeypatch):
    """The refusal is transient, so it must not resolve the idempotency key."""
    route = _load_training_route("training_route_guard_sticky_test")
    counter = {"n": 1}
    backend = TrainingBackend()
    monkeypatch.setattr(route, "get_training_backend", lambda: backend)
    monkeypatch.setattr(
        keepwarm,
        "other_inference_request_count",
        lambda current_request_counted = True, **_: counter["n"],
    )
    monkeypatch.setattr(route, "_background_video_generation_active", lambda: False)

    config = _config(start_request_id = "agent-retry-1")

    with pytest.raises(HTTPException) as first:
        asyncio.run(_call(route, config))
    assert first.value.status_code == 409

    # The guard runs before the reservation, so no permanent record is written.
    assert backend.get_start_request("agent-retry-1") is None

    # Stream is over. The agent retries with the same idempotency key and the
    # start is admitted (it fails later on the fake model, never on the guard).
    counter["n"] = 0
    with pytest.raises(HTTPException) as second:
        asyncio.run(_call(route, config))
    assert "inference request is in progress" not in str(second.value.detail)


def test_a_fresh_start_request_id_recovers(monkeypatch):
    """The workaround: a new id per attempt is not poisoned."""
    route = _load_training_route("training_route_guard_fresh_id_test")
    counter = {"n": 1}
    backend = TrainingBackend()
    monkeypatch.setattr(route, "get_training_backend", lambda: backend)
    monkeypatch.setattr(
        keepwarm,
        "other_inference_request_count",
        lambda current_request_counted = True, **_: counter["n"],
    )
    monkeypatch.setattr(route, "_background_video_generation_active", lambda: False)

    with pytest.raises(HTTPException):
        asyncio.run(_call(route, _config(start_request_id = "attempt-1")))

    counter["n"] = 0
    with pytest.raises(HTTPException) as second:
        asyncio.run(_call(route, _config(start_request_id = "attempt-2")))
    assert "inference request is in progress" not in str(second.value.detail)


def test_a_resolved_start_request_id_still_replays_under_the_guard(monkeypatch):
    """The transient guard must not swallow the idempotent replay.

    An agent that retries an ACCEPTED start (its first response was lost) while an
    unrelated inference request is in flight has to hear "your job is queued", not a
    fresh 409 telling it the start never happened."""
    route = _load_training_route("training_route_guard_replay_test")
    backend = _arm(monkeypatch, route, inflight = 1)

    backend.reserve_start_request("agent-accepted", "job-accepted")
    backend.resolve_start_request(
        "agent-accepted",
        state = "accepted",
        message = "Training started",
    )

    response = asyncio.run(_call(route, _config(start_request_id = "agent-accepted")))

    assert response.status == "queued"
    assert response.job_id == "job-accepted"
    assert "inference request is in progress" not in str(response.message)


def test_a_cancelled_start_request_id_replays_and_keeps_its_tombstone(monkeypatch):
    """A retry blocked by the guard must still refresh the cancellation tombstone.

    Otherwise the tombstone expires mid-inference and the next retry reserves the id
    afresh and spawns the very run the user cancelled."""
    import time

    from core.training import training as training_module

    route = _load_training_route("training_route_guard_tombstone_test")
    backend = _arm(monkeypatch, route, inflight = 1)

    outcome, cancelled = backend.cancel_start_request("agent-cancelled")
    assert outcome == "cancelled"

    # Wind the tombstone to the brink of its TTL: without a refresh the next retry
    # would find nothing and start the job.
    backend._start_cancel_tombstones[backend._start_key("agent-cancelled")] = (
        time.monotonic() + 0.5,
        cancelled,
    )

    response = asyncio.run(_call(route, _config(start_request_id = "agent-cancelled")))

    assert response.status == "error"
    assert response.error_code == training_module._START_CANCELLED_ERROR_CODE
    assert "inference request is in progress" not in str(response.message)

    expires_at, _ = backend._start_cancel_tombstones[backend._start_key("agent-cancelled")]
    assert expires_at > time.monotonic() + (training_module._START_CANCEL_TOMBSTONE_TTL_S / 2)


def test_an_unknown_start_request_id_is_still_refused_without_a_record(monkeypatch):
    """The replay lookup must not resurrect the poisoning bug this PR exists to fix."""
    route = _load_training_route("training_route_guard_replay_fresh_test")
    backend = _arm(monkeypatch, route, inflight = 1)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(_call(route, _config(start_request_id = "agent-never-seen")))

    assert excinfo.value.status_code == 409
    assert backend.get_start_request("agent-never-seen") is None
    assert backend.peek_start_request("agent-never-seen") is None


# --------------------------------------------------------------------------------------
# FINDING 2: what the MCP client sees on the 409
# --------------------------------------------------------------------------------------


def test_the_mcp_tool_surfaces_the_409_as_a_tool_error_not_a_dict(monkeypatch):
    """stop_training/get_training_status return dicts; a refused start raises."""
    import mcp_server
    import routes.training as training_routes

    monkeypatch.setattr(
        training_routes,
        "get_training_backend",
        lambda: TrainingBackend(),
    )
    monkeypatch.setattr(
        keepwarm,
        "other_inference_request_count",
        lambda current_request_counted = True, **_: 1,
    )
    monkeypatch.setattr(training_routes, "_background_video_generation_active", lambda: False)

    server = mcp_server.create_studio_mcp()

    # The tool body raises rather than returning the {"status": ...} dict that
    # stop_training / get_training_status return.
    # Public fastmcp API only. The private equivalents (_get_tool/_call_tool_mcp)
    # work too, but _call_tool_mcp was dropped in fastmcp 4.0.0 while the pin is an
    # open ">=3.0.2", so a private call here breaks the suite on an upstream major
    # that the product itself is unaffected by. get_tool/call_tool are present with
    # these same signatures on both 3.0.2 (the floor) and 4.x.
    async def run_tool_body():
        tool = await server.get_tool("start_training")
        assert tool is not None, "start_training must be registered on the studio MCP server"
        return await tool.fn(config = _config())

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(run_tool_body())
    assert excinfo.value.status_code == 409

    from fastmcp.exceptions import ToolError

    # call_tool, not the private _call_tool_mcp fastmcp 4 dropped: requirements
    # floor it at 3.0.2 and never cap it, so a test may only use the public API.
    with pytest.raises(ToolError) as tool_error:
        asyncio.run(server.call_tool("start_training", {"config": _config()}))

    # mask_error_details defaults False, so the 409 detail survives to the client
    # (as the text of an isError CallToolResult, not a JSON-RPC protocol error).
    # Asserted through the surfaced message rather than the server's private
    # _mask_error_details flag: the detail reaching the caller is the property that
    # matters, and it stays true however that flag is spelled upstream.
    assert "inference request is in progress" in str(tool_error.value)
    assert "Error calling tool 'start_training'" in str(tool_error.value)


# --------------------------------------------------------------------------------------
# Concurrency
# --------------------------------------------------------------------------------------


def test_two_concurrent_mcp_starts_during_a_stream_both_refuse(monkeypatch):
    route = _load_training_route("training_route_guard_concurrent_test")
    _arm(monkeypatch, route, inflight = 1)

    async def both():
        return await asyncio.gather(
            _call(route, _config()),
            _call(route, _config()),
            return_exceptions = True,
        )

    results = asyncio.run(both())
    assert all(isinstance(r, HTTPException) and r.status_code == 409 for r in results)


def test_guard_path_uses_no_platform_specific_apis():
    """The guard is pure Python: threading + a counter, no fork/signal/posix."""
    source = (_BACKEND_ROOT / "routes" / "training.py").read_text(encoding = "utf-8")
    guard = source[source.index("if via_api_key is True:") :][:800]
    for banned in ("os.fork", "signal.", "SIGKILL", "winreg", "msvcrt"):
        assert banned not in guard
    assert threading is not None
