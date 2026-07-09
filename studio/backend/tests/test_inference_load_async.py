# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Async model load: POST /load with async_load=true returns immediately and
finishes the real load in a background task, surfacing failures via
GET /status's load_error field.

No GPU or llama-server: _load_model_impl is mocked to a bare asyncio.sleep,
mirroring tests/test_openai_auto_switch.py.
"""

import asyncio
import time

import pytest
from fastapi import HTTPException

import core.inference.llama_keepwarm as keepwarm
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
    """The route must return as soon as the background task is scheduled."""

    gate_entered = asyncio.Event()

    async def _slow_load(request, fastapi_request, current_subject):
        time.sleep(0.2)
        await asyncio.sleep(0.2)
        return LoadResponse(
            status = "loaded", model = request.model_path, display_name = "A", inference = {}
        )

    monkeypatch.setattr(inference_route, "_load_model_impl", _slow_load)
    monkeypatch.setattr(
        keepwarm, "acquire_inference_lifecycle_gate_nowait", lambda: gate_entered.set() or True
    )
    monkeypatch.setattr(keepwarm, "release_inference_lifecycle_gate", lambda: None)

    async def _scenario():
        start = asyncio.get_event_loop().time()
        result = await inference_route.load_model(_request(), object(), "tester")
        elapsed = asyncio.get_event_loop().time() - start
        return result, elapsed

    result, elapsed = _run(_scenario())
    assert isinstance(result, LoadAcceptedResponse)
    assert result.status == "loading"
    assert result.model == "unsloth/A-GGUF"
    assert gate_entered.is_set()
    assert elapsed < 0.1


def test_async_load_success_leaves_error_none(monkeypatch):
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
    async def _quick_load(request, fastapi_request, current_subject):
        await asyncio.sleep(0.2)
        return LoadResponse(
            status = "loaded", model = request.model_path, display_name = "A", inference = {}
        )

    monkeypatch.setattr(inference_route, "_load_model_impl", _quick_load)
    monkeypatch.setattr(inference_route, "_last_async_load_error", "stale error from a prior load")

    async def _scenario():
        return await inference_route.load_model(_request(), object(), "tester")

    _run(_scenario())
    assert inference_route._last_async_load_error is None


def test_async_load_rejects_second_load_while_pending(monkeypatch):
    release = asyncio.Event()
    calls = []

    async def _slow_load(request, fastapi_request, current_subject):
        calls.append(request.model_path)
        await release.wait()
        return LoadResponse(
            status = "loaded", model = request.model_path, display_name = "A", inference = {}
        )

    monkeypatch.setattr(inference_route, "_load_model_impl", _slow_load)

    async def _scenario():
        first = await inference_route.load_model(_request("unsloth/A-GGUF"), object(), "tester")
        with pytest.raises(HTTPException) as exc_info:
            await inference_route.load_model(_request("unsloth/B-GGUF"), object(), "tester")
        release.set()
        await inference_route._active_async_load_task
        return first, exc_info.value

    first, exc = _run(_scenario())
    assert isinstance(first, LoadAcceptedResponse)
    assert exc.status_code == 409
    assert "unsloth/A-GGUF" in exc.detail
    assert calls == ["unsloth/A-GGUF"]


def test_sync_load_rejected_while_async_load_pending(monkeypatch):
    release = asyncio.Event()

    async def _slow_load(request, fastapi_request, current_subject):
        await release.wait()
        return LoadResponse(
            status = "loaded", model = request.model_path, display_name = "A", inference = {}
        )

    monkeypatch.setattr(inference_route, "_load_model_impl", _slow_load)

    async def _scenario():
        await inference_route.load_model(_request("unsloth/A-GGUF"), object(), "tester")
        with pytest.raises(HTTPException) as exc_info:
            await inference_route.load_model(
                _request("unsloth/B-GGUF", async_load = False), object(), "tester"
            )
        release.set()
        await inference_route._active_async_load_task
        return exc_info.value

    exc = _run(_scenario())
    assert exc.status_code == 409
    assert "unsloth/A-GGUF" in exc.detail


def test_async_load_rejects_when_lifecycle_gate_is_busy(monkeypatch):
    calls = []

    async def _load(request, fastapi_request, current_subject):
        calls.append(request.model_path)
        return LoadResponse(
            status = "loaded", model = request.model_path, display_name = "A", inference = {}
        )

    monkeypatch.setattr(inference_route, "_load_model_impl", _load)
    monkeypatch.setattr(keepwarm, "acquire_inference_lifecycle_gate_nowait", lambda: False)

    async def _scenario():
        with pytest.raises(HTTPException) as exc_info:
            await inference_route.load_model(_request("unsloth/A-GGUF"), object(), "tester")
        return exc_info.value

    exc = _run(_scenario())
    assert exc.status_code == 409
    assert calls == []
    assert inference_route._accepted_async_load_model is None


def test_async_load_releases_lifecycle_gate_when_cancelled_before_start(monkeypatch):
    calls = []
    releases = []

    async def _load(request, fastapi_request, current_subject):
        calls.append(request.model_path)
        return LoadResponse(
            status = "loaded", model = request.model_path, display_name = "A", inference = {}
        )

    monkeypatch.setattr(inference_route, "_load_model_impl", _load)
    monkeypatch.setattr(keepwarm, "acquire_inference_lifecycle_gate_nowait", lambda: True)
    monkeypatch.setattr(
        keepwarm, "release_inference_lifecycle_gate", lambda: releases.append("released")
    )

    async def _scenario():
        await inference_route.load_model(_request("unsloth/A-GGUF"), object(), "tester")
        task = inference_route._active_async_load_task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0)

    _run(_scenario())
    assert calls == []
    assert releases == ["released"]
    assert inference_route._active_async_load_task is None
    assert inference_route._accepted_async_load_model is None


def test_async_load_status_reports_pending_model_immediately(monkeypatch):
    release = asyncio.Event()

    class _LlamaBackend:
        is_loaded = False

        @staticmethod
        def _find_llama_server_binary():
            raise RuntimeError("no llama-server in test")

    class _Backend:
        active_model_name = None
        models = {}
        loading_models = set()

    async def _slow_load(request, fastapi_request, current_subject):
        await release.wait()
        return LoadResponse(
            status = "loaded", model = request.model_path, display_name = "A", inference = {}
        )

    monkeypatch.setattr(inference_route, "_load_model_impl", _slow_load)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _LlamaBackend())
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: _Backend())
    monkeypatch.setattr(
        inference_route,
        "_detect_safetensors_features",
        lambda backend, chat_template: {
            "supports_reasoning": False,
            "reasoning_style": "enable_thinking",
            "reasoning_effort_levels": [],
            "reasoning_always_on": False,
            "supports_preserve_thinking": False,
            "supports_tools": False,
        },
    )
    monkeypatch.setattr(inference_route, "_resolve_loaded_trust_remote_code", lambda *args: False)

    async def _scenario():
        accepted = await inference_route.load_model(_request("unsloth/A-GGUF"), object(), "tester")
        status = await inference_route.get_status("tester")
        release.set()
        await inference_route._active_async_load_task
        return accepted, status

    accepted, status = _run(_scenario())
    assert isinstance(accepted, LoadAcceptedResponse)
    assert status.loading == ["unsloth/A-GGUF"]
    assert status.load_error is None


def test_async_load_status_keeps_previous_gguf_loaded_while_pending(monkeypatch):
    release = asyncio.Event()

    class _LlamaBackend:
        is_loaded = True
        model_identifier = "unsloth/Previous-GGUF"
        is_vision = False
        is_diffusion = False
        hf_variant = None
        supports_reasoning = False
        reasoning_style = "enable_thinking"
        reasoning_effort_levels = []
        reasoning_always_on = False
        supports_preserve_thinking = False
        supports_tools = False
        chat_template = None
        context_length = None
        max_context_length = None
        native_context_length = None
        cache_type_kv = None
        chat_template_override = None
        requested_spec_mode = None
        spec_draft_n_max = None
        tensor_parallel = False
        spec_fallback_reason = None

        @staticmethod
        def _find_llama_server_binary():
            raise RuntimeError("no llama-server in test")

    async def _slow_load(request, fastapi_request, current_subject):
        await release.wait()
        return LoadResponse(
            status = "loaded", model = request.model_path, display_name = "A", inference = {}
        )

    monkeypatch.setattr(inference_route, "_load_model_impl", _slow_load)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _LlamaBackend())
    monkeypatch.setattr(inference_route, "display_label_for_native_path", lambda model_id: model_id)
    monkeypatch.setattr(inference_route, "load_inference_config", lambda model_id: None)
    monkeypatch.setattr(
        inference_route, "resolve_effective_chat_template_override", lambda **kwargs: None
    )

    async def _scenario():
        accepted = await inference_route.load_model(
            _request("unsloth/Next-GGUF"), object(), "tester"
        )
        status = await inference_route.get_status("tester")
        release.set()
        await inference_route._active_async_load_task
        return accepted, status

    accepted, status = _run(_scenario())
    assert isinstance(accepted, LoadAcceptedResponse)
    assert status.loaded == ["unsloth/Previous-GGUF"]
    assert status.loading == ["unsloth/Next-GGUF"]


def test_overlapping_loads_success_clears_stale_error(monkeypatch):
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

    async def _scenario():
        await inference_route.load_model(_request(), object(), "tester")
        await asyncio.sleep(0.1)
        assert inference_route._last_async_load_error == "A failed"

        await inference_route.load_model(_request(), object(), "tester")
        await asyncio.sleep(0.1)

    _run(_scenario())
    assert inference_route._last_async_load_error is None


def test_sync_load_success_clears_stale_async_error(monkeypatch):
    async def _load(request, fastapi_request, current_subject):
        return LoadResponse(
            status = "loaded", model = request.model_path, display_name = "A", inference = {}
        )

    monkeypatch.setattr(inference_route, "_load_model_impl", _load)
    monkeypatch.setattr(inference_route, "_last_async_load_error", "stale async failure")

    result = _run(inference_route.load_model(_request(async_load = False), object(), "tester"))
    assert isinstance(result, LoadResponse)
    assert inference_route._last_async_load_error is None


def test_sync_load_unaffected_returns_load_response_directly(monkeypatch):
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
