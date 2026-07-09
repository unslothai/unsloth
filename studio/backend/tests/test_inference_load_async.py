# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Async model load: POST /load with async_load=true returns immediately and
finishes the real load in a background task, surfacing failures via
GET /status's load_error field.

No GPU or llama-server: _load_model_impl is mocked to a bare asyncio.sleep,
mirroring tests/test_openai_auto_switch.py.
"""

import asyncio

import pytest
from fastapi import HTTPException

import routes.inference as inference_route
from models.inference import LoadAcceptedResponse, LoadRequest, LoadResponse


@pytest.fixture(autouse = True)
def _reset_async_load_state(monkeypatch):
    monkeypatch.setattr(inference_route, "_last_async_load_error", None)
    monkeypatch.setattr(inference_route, "_async_load_generation", 0)
    monkeypatch.setattr(inference_route, "_accepted_async_load_model", None)
    monkeypatch.setattr(inference_route, "_active_async_load_task", None)
    inference_route._background_tasks.clear()
    yield
    inference_route._background_tasks.clear()


def _request(model_path = "unsloth/A-GGUF", async_load = True) -> LoadRequest:
    return LoadRequest(model_path = model_path, async_load = async_load)


def _run(coro):
    return asyncio.run(coro)


def test_async_load_returns_immediately(monkeypatch):
    """The route must not block on the slow load: it returns as soon as the
    background task is scheduled, well before the mocked load finishes."""

    async def _slow_load(request, fastapi_request, current_subject):
        await asyncio.sleep(0.2)
        return LoadResponse(
            status = "loaded", model = request.model_path, display_name = "A", inference = {}
        )

    monkeypatch.setattr(inference_route, "_load_model_impl", _slow_load)
    async def _scenario():
        start = asyncio.get_event_loop().time()
        result = await inference_route.load_model(_request(), object(), "tester")
        elapsed = asyncio.get_event_loop().time() - start
        return result, elapsed

    result, elapsed = _run(_scenario())
    assert isinstance(result, LoadAcceptedResponse)
    assert result.status == "loading"
    assert result.model == "unsloth/A-GGUF"
    assert elapsed < 0.1


def test_async_load_success_leaves_error_none(monkeypatch):
    """A background load that succeeds must not populate load_error."""

    async def _quick_load(request, fastapi_request, current_subject):
        await asyncio.sleep(0.02)
        return LoadResponse(
            status = "loaded", model = request.model_path, display_name = "A", inference = {}
        )

    monkeypatch.setattr(inference_route, "_load_model_impl", _quick_load)
    async def _scenario():
        await inference_route.load_model(_request(), object(), "tester")
        await asyncio.sleep(0.1)

    _run(_scenario())
    assert inference_route._last_async_load_error is None


def test_async_load_failure_surfaces_via_last_async_load_error(monkeypatch):
    """A background load that raises must record its message so GET /status can
    surface it via load_error."""

    async def _failing_load(request, fastapi_request, current_subject):
        await asyncio.sleep(0.02)
        raise HTTPException(status_code = 400, detail = "boom: unsupported model")

    monkeypatch.setattr(inference_route, "_load_model_impl", _failing_load)
    async def _scenario():
        await inference_route.load_model(_request(), object(), "tester")
        await asyncio.sleep(0.1)

    _run(_scenario())
    assert inference_route._last_async_load_error == "boom: unsupported model"


def test_async_load_clears_previous_error_before_scheduling(monkeypatch):
    """Starting a new async load must clear a stale error synchronously, before
    the background task even runs -- otherwise a poller could read a load_error
    left over from an unrelated, earlier failed load."""

    async def _quick_load(request, fastapi_request, current_subject):
        await asyncio.sleep(0.2)
        return LoadResponse(
            status = "loaded", model = request.model_path, display_name = "A", inference = {}
        )

    monkeypatch.setattr(inference_route, "_load_model_impl", _quick_load)
    monkeypatch.setattr(inference_route, "_last_async_load_error", "stale error from a prior load")

    async def _scenario():
        result = await inference_route.load_model(_request(), object(), "tester")
        # Checked immediately after the route returns, before the background
        # task (sleeping 0.2s) has had a chance to run.
        return result

    _run(_scenario())
    assert inference_route._last_async_load_error is None


def test_async_load_status_tracks_accepted_model_before_backend_loading(monkeypatch):
    """The accepted model should appear as loading immediately, before the
    backend-specific loading set is populated."""

    entered_load = asyncio.Event()
    release_load = asyncio.Event()

    async def _slow_load(request, fastapi_request, current_subject):
        entered_load.set()
        await release_load.wait()
        return LoadResponse(
            status = "loaded", model = request.model_path, display_name = "A", inference = {}
        )

    monkeypatch.setattr(inference_route, "_load_model_impl", _slow_load)

    async def _scenario():
        result = await inference_route.load_model(_request("unsloth/Pending-GGUF"), object(), "tester")
        assert isinstance(result, LoadAcceptedResponse)
        assert "unsloth/Pending-GGUF" in inference_route._async_loading_models()
        release_load.set()
        await asyncio.wait_for(entered_load.wait(), timeout = 1)
        await asyncio.sleep(0.05)

    _run(_scenario())


def test_async_load_rejects_second_accepted_load_while_one_is_active(monkeypatch):
    """Only one async load is accepted at a time, so double-clicks/retries do
    not build an unbounded queue behind the lifecycle gate."""

    entered_load = asyncio.Event()
    release_load = asyncio.Event()

    async def _slow_load(request, fastapi_request, current_subject):
        entered_load.set()
        await release_load.wait()
        return LoadResponse(
            status = "loaded", model = request.model_path, display_name = "A", inference = {}
        )

    monkeypatch.setattr(inference_route, "_load_model_impl", _slow_load)

    async def _scenario():
        first = await inference_route.load_model(_request("unsloth/A-GGUF"), object(), "tester")
        assert isinstance(first, LoadAcceptedResponse)
        await asyncio.wait_for(entered_load.wait(), timeout = 1)
        with pytest.raises(HTTPException) as exc:
            await inference_route.load_model(_request("unsloth/B-GGUF"), object(), "tester")
        assert exc.value.status_code == 409
        assert len(inference_route._background_tasks) == 1
        release_load.set()
        await asyncio.sleep(0.05)

    _run(_scenario())


def test_older_async_failure_cannot_overwrite_newer_accepted_load(monkeypatch):
    """A newer async load cannot be accepted while an older one is active, so
    the older task cannot later write a failure against that newer accepted load."""

    entered_load = asyncio.Event()

    async def _failing_load(request, fastapi_request, current_subject):
        entered_load.set()
        await asyncio.sleep(0.05)
        raise HTTPException(status_code = 400, detail = "A failed after B attempted")

    monkeypatch.setattr(inference_route, "_load_model_impl", _failing_load)

    async def _scenario():
        first = await inference_route.load_model(_request("unsloth/A-GGUF"), object(), "tester")
        assert isinstance(first, LoadAcceptedResponse)
        await asyncio.wait_for(entered_load.wait(), timeout = 1)
        with pytest.raises(HTTPException) as exc:
            await inference_route.load_model(_request("unsloth/B-GGUF"), object(), "tester")
        assert exc.value.status_code == 409
        await asyncio.sleep(0.1)

    _run(_scenario())
    assert inference_route._last_async_load_error == "A failed after B attempted"


def test_overlapping_loads_success_clears_stale_error(monkeypatch):
    """Race condition: load A fails (sets error), then load B succeeds.
    The success path must clear the error so /status doesn't return stale
    failure info from load A."""

    call_count = 0

    async def _alternating_load(request, fastapi_request, current_subject):
        nonlocal call_count
        call_count += 1
        seq = call_count
        if seq == 1:
            await asyncio.sleep(0.02)
            raise HTTPException(status_code = 400, detail = "A failed")
        await asyncio.sleep(0.02)
        return LoadResponse(
            status = "loaded", model = request.model_path, display_name = "B", inference = {}
        )

    monkeypatch.setattr(inference_route, "_load_model_impl", _alternating_load)
    monkeypatch.setattr(inference_route, "_last_async_load_error", None)

    async def _scenario():
        # Fire load A (will fail)
        await inference_route.load_model(_request(), object(), "tester")
        await asyncio.sleep(0.1)
        assert inference_route._last_async_load_error == "A failed"

        # Fire load B (will succeed) -- acceptance clears the error, but the
        # important part is that the *background success* also clears it.
        await inference_route.load_model(_request(), object(), "tester")
        await asyncio.sleep(0.1)

    _run(_scenario())
    assert inference_route._last_async_load_error is None


def test_sync_load_unaffected_returns_load_response_directly(monkeypatch):
    """async_load=false (the default) must keep behaving synchronously: no
    background task, and the caller gets the real LoadResponse directly."""

    calls = []

    async def _load(request, fastapi_request, current_subject):
        calls.append(request)
        return LoadResponse(
            status = "loaded", model = request.model_path, display_name = "A", inference = {}
        )

    monkeypatch.setattr(inference_route, "_load_model_impl", _load)

    result = _run(inference_route.load_model(_request(async_load = False), object(), "tester"))
    assert isinstance(result, LoadResponse)
    assert result.model == "unsloth/A-GGUF"
    assert len(calls) == 1
