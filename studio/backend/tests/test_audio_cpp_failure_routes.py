# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A classified failure reaches the client as {"detail": {"code", "message", ...}}; anything
unclassified keeps today's status and text."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import routes.inference as ri
from core.inference import audio_cpp_failures as af
from core.inference import audio_cpp_server as srv
from models.inference import ChatCompletionRequest, LoadRequest
from .test_audio_cpp_failures import _model

_MISSING = "The audio runtime is not installed. Run `unsloth studio update` to install it."


def _oom() -> af.AudioCppFailure:
    return af.classify_audio_cpp_failure("cudaMalloc failed: out of memory", backend = "cuda")


def _load(side_effect) -> HTTPException:
    request = LoadRequest(model_path = "audio-cpp/audio.cpp-gguf/M-GGUF")
    with (
        patch.object(
            ri,
            "_resolve_model_identifier_for_request",
            return_value = (request.model_path,) * 2 + (False,),
        ),
        patch.object(ri, "resolve_effective_chat_template_override", return_value = None),
        patch.object(ri, "get_inference_backend", return_value = MagicMock(active_model_name = None)),
        patch.object(ri, "get_llama_cpp_backend", return_value = MagicMock()),
        patch.object(ri.ModelConfig, "from_identifier", side_effect = side_effect),
    ):
        with pytest.raises(HTTPException) as raised:
            asyncio.run(ri.load_model(request, MagicMock(), current_subject = "test-user"))
    return raised.value


def test_load_detail_is_structured_when_classified_else_500_prefix():
    error = _load(af.AudioCppFailureError("Failed to load model", _oom()))
    assert error.status_code == 503 and error.detail["code"] == "out_of_memory"
    assert "ran out of memory" in error.detail["message"]
    error = _load(RuntimeError("totally different failure"))
    assert error.status_code == 500 and error.detail.startswith("Failed to load model: ")


def test_deferred_load_error_keeps_dict_detail(monkeypatch):
    monkeypatch.setattr(ri, "_TUNNEL_KEEPALIVE_AFTER_S", 0.05)
    monkeypatch.setattr(ri, "_TUNNEL_KEEPALIVE_EVERY_S", 0.02)
    detail = _oom().detail()

    async def slow_boom():
        await asyncio.sleep(0.2)
        raise HTTPException(status_code = 503, detail = detail)

    async def run():
        response = await ri._tunnel_safe_json(slow_boom(), label = "t")
        return b"".join([chunk async for chunk in response.body_iterator])

    body = json.loads(asyncio.run(run()))
    assert body["_deferred_error"] == {"status_code": 503, "detail": detail}


def test_v1_path_maps_detail_code_to_openai_error_code():
    from utils.api_errors import install_api_error_handlers

    app = FastAPI()
    install_api_error_handlers(app)

    @app.post("/v1/audio/speech")
    async def speech():
        raise af.problem_http_exception(_MISSING)

    response = TestClient(app).post("/v1/audio/speech")
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "runtime_missing" and "isn't installed" in error["message"]


def _preflight(
    problem,
    *,
    availability = None,
    prepared = None,
):
    import utils.hardware as hardware

    model = _model()
    config = SimpleNamespace(identifier = model.id, audio_type = "audiocpp_tts", audio_cpp = model)
    request = SimpleNamespace(gpu_ids = None, hf_token = None, max_seq_length = 2048, audio_device = None)
    placement = ri._LoadPlacement(None, None, False, False)
    with (
        patch.object(srv, "model_runtime_problem", lambda _model: problem),
        patch.object(ri, "_native_audio_cpu_load", lambda *_a: False),
        patch.object(ri, "_native_audio_post_handoff_free_gb", lambda: availability),
        patch("utils.hardware.get_device", return_value = hardware.DeviceType.CUDA),
        patch.object(hardware, "IS_ROCM", False),
        patch("core.inference.native_audio.native_audio_security_targets", return_value = [model.id]),
        patch("utils.hardware.prepare_gpu_selection", return_value = prepared or ([0], {})),
    ):
        return asyncio.run(ri._preflight_native_audio_placement(config, request, placement))


@pytest.mark.parametrize(
    "problem, status, detail",
    [
        (_MISSING, 503, "runtime_missing"),
        ("The audio runtime is being updated. Try again in a moment.", 501, None),
    ],
)
def test_preflight_problem_is_503_with_code_else_501_string(problem, status, detail):
    with pytest.raises(HTTPException) as raised:
        _preflight(problem)
    assert raised.value.status_code == status
    assert (
        raised.value.detail == problem if detail is None else raised.value.detail["code"] == detail
    )


def test_placement_refusal_lists_variants_that_fit(monkeypatch):
    monkeypatch.setattr(af, "offload_options", lambda model: [])
    availability = ri._NativeAudioAvailability({0: 7.0, 1: 5.0}, {0: 0.0, 1: 0.0})
    with pytest.raises(HTTPException) as raised:
        _preflight(
            None, availability = availability, prepared = ([0, 1], {"selection_mode": "fallback_all"})
        )
    detail = raised.value.detail
    assert ri.audio_cpp_resident_gb is af.estimate_resident_gb
    assert raised.value.status_code == 400 and detail["code"] == "out_of_memory"
    assert [row["label"] for row in detail["smaller_variants"]] == ["Q2", "Q1"]
    assert detail["message"].startswith("No single GPU has enough free memory for Model M.")


def test_generation_crash_oom_maps_to_out_of_memory(monkeypatch):
    tail = "ggml_gallocr_reserve_n: failed to allocate CUDA0 buffer of size 9663676416"
    crash = af.attach_failure(
        srv.AudioCppUnavailableError("The audio runtime stopped while serving the request."),
        af.classify_audio_cpp_failure(tail, backend = "cuda"),
    )

    model = "audio-cpp/audio.cpp-gguf/M-GGUF"
    backend = SimpleNamespace(
        active_model_name = model,
        models = {model: {"is_audio": True, "audio_type": "audiocpp_tts"}},
        generate_audio_response = MagicMock(side_effect = crash),
    )
    monkeypatch.setattr(
        ri, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False, _is_audio = False)
    )
    monkeypatch.setattr(ri, "get_inference_backend", lambda: backend)
    monkeypatch.setattr(ri, "_maybe_auto_switch_model", AsyncMock(return_value = None))
    payload = ChatCompletionRequest(model = model, messages = [{"role": "user", "content": "hello"}])
    with pytest.raises(HTTPException) as raised:
        asyncio.run(ri._generate_tts_wav("hello", payload, request = None, current_subject = "t"))
    assert raised.value.status_code == 503 and raised.value.detail["code"] == "out_of_memory"


def test_oom_retry_and_monitor_read_the_structured_detail(monkeypatch):
    oom = HTTPException(status_code = 503, detail = _oom().detail())
    assert ri._ran_out_of_memory(oom)
    other = af.classify_audio_cpp_failure("model path does not exist: /m", backend = "cuda")
    assert not ri._ran_out_of_memory(HTTPException(status_code = 422, detail = other.detail()))
    # Out of CPU RAM: unloading kept GPU models cannot make room, so no eviction retry.
    cpu = af.classify_audio_cpp_failure(
        "failed to allocate CPU buffer of size 9663676416", backend = "cpu"
    )
    assert cpu.extras == {"device": "cpu"}
    assert not ri._ran_out_of_memory(HTTPException(status_code = 503, detail = cpu.detail()))
    monitor = MagicMock()
    monkeypatch.setattr(ri, "api_monitor", monitor)
    monkeypatch.setattr(ri, "_request_used_api_key", lambda request: False)
    request = SimpleNamespace(
        state = SimpleNamespace(), url = SimpleNamespace(path = "/x"), method = "POST"
    )

    async def run():
        async with ri._monitored_media_request(request, model = "m", prompt = "", subject = None):
            raise oom

    with pytest.raises(HTTPException):
        asyncio.run(run())
    assert monitor.fail.call_args.args[1] == oom.detail["message"]
