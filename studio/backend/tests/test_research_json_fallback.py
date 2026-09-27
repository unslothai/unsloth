# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Research may request JSON through prompts when local guided decoding is unavailable."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from auth.authentication import get_current_subject
from core import research_runs
from core.inference.api_monitor import ApiMonitor
from models.inference import ChatCompletionRequest, ChatMessage
from routes import inference as inference_route
from state import tool_policy
from utils.api_errors import install_api_error_handlers
from .test_sf_client_tools_passthrough import _ScriptedBackend, _fixed, _install


_NO_GRAMMAR_ENGINE = (
    "response_format needs the llama.cpp grammar engine; load a GGUF model to use it."
)
# Same code and param, different cause. The real-route cases keep these strings honest.
_AUDIO_REFUSAL_MESSAGE = (
    "response_format cannot be honored by an audio reply; send the request to a text model "
    "to use guided decoding."
)
_TOOL_LOOP_REFUSAL_MESSAGE = (
    "response_format is not supported with Unsloth tool execution; send the request without "
    "enable_tools to use guided decoding."
)


def _refusal(message = _NO_GRAMMAR_ENGINE):
    return {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "code": "unsupported_parameter",
            "param": "response_format",
        }
    }


_REFUSAL = _refusal()


@pytest.fixture
def research_call(monkeypatch):
    supervisor = research_runs.ResearchSupervisor(
        SimpleNamespace(state = SimpleNamespace(server_port = 1))
    )

    async def noop(*args, **kwargs):
        pass

    monkeypatch.setattr(supervisor, "_check_active", noop)
    monkeypatch.setattr(supervisor, "_note_phase", noop)
    monkeypatch.setattr(research_runs, "_loaded_context_length", lambda *args: 8192)
    monkeypatch.setattr(
        research_runs.auth_storage, "create_api_key", lambda **kwargs: ("test", {"id": 1})
    )
    revoked = []
    monkeypatch.setattr(research_runs.auth_storage, "revoke_internal_api_key", revoked.append)
    run = {
        "id": "json-fallback",
        "ownerSubject": "test-user",
        "config": {"model": "sf-model", "budgets": {"modelTimeoutSeconds": 10}},
    }
    real_client = httpx.AsyncClient

    def install_transport(transport):
        monkeypatch.setattr(
            research_runs.httpx,
            "AsyncClient",
            lambda **kwargs: real_client(transport = transport, **kwargs),
        )

    def complete(**kwargs):
        return asyncio.run(
            supervisor._stream_completion(
                run,
                [{"role": "user", "content": "Return only JSON."}],
                json_mode = kwargs.pop("json_mode", True),
                report_progress = False,
                **kwargs,
            )
        )

    return SimpleNamespace(
        supervisor = supervisor,
        run = run,
        install = install_transport,
        complete = complete,
        revoked = revoked,
    )


@pytest.mark.parametrize("forced_tools", [False, True], ids = ["default-tools", "forced-tools"])
@pytest.mark.parametrize("is_mlx", [True, False], ids = ["mlx", "transformers"])
@pytest.mark.parametrize("phase", ["planning", "decision", "synthesis_audit"])
def test_local_json_research_recovers_through_the_real_route(
    monkeypatch, research_call, is_mlx, phase, forced_tools
):
    backend = _ScriptedBackend(_fixed('{"ok": true}'))
    backend.models[backend.active_model_name]["is_mlx"] = is_mlx
    _install(monkeypatch, backend, supports_tools = forced_tools)
    if forced_tools:
        monkeypatch.setattr(tool_policy, "_tool_policy", True)
    app = FastAPI()
    app.include_router(inference_route.router, prefix = "/v1")
    install_api_error_handlers(app)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    sent = []
    statuses = []

    class RecordingTransport(httpx.ASGITransport):
        async def handle_async_request(self, request):
            sent.append(json.loads(request.content))
            response = await super().handle_async_request(request)
            statuses.append(response.status_code)
            return response

    research_call.install(RecordingTransport(app = app))
    report, _, finish, _ = research_call.complete(phase = phase, max_tokens = 32)
    assert json.loads(report) == {"ok": True}
    assert finish == "stop"
    assert statuses == [400, 200]
    assert len(backend.calls) == 1
    assert not backend.calls[0]["tools"]
    first, second = sent
    assert first.pop("response_format") == {"type": "json_object"}
    assert first == second
    assert second["tool_choice"] == "none"
    assert second["enabled_tools"] == []
    assert research_call.revoked == [1]


def _completion():
    chunk = {"choices": [{"delta": {"content": '{"ok": true}'}, "finish_reason": "stop"}]}
    return httpx.Response(200, text = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")


@pytest.mark.parametrize("provider", [False, True], ids = ["local", "provider"])
def test_supported_json_mode_keeps_the_format(research_call, provider):
    if provider:
        research_call.run["config"]["inferenceRequest"] = {
            "providerType": "openai",
            "providerId": "connection",
            "externalModel": "model",
        }
    sent = []

    def serve(request):
        sent.append(json.loads(request.content))
        return _completion()

    research_call.install(httpx.MockTransport(serve))
    assert json.loads(research_call.complete()[0]) == {"ok": True}
    assert len(sent) == 1
    assert sent[0]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    "status,body,provider,json_mode",
    [
        (400, {"error": {"code": "unsupported_parameter", "param": "temperature"}}, False, True),
        (
            400,
            {"error": {"code": "context_length_exceeded", "param": "response_format"}},
            False,
            True,
        ),
        (400, {"error": "response_format unsupported"}, False, True),
        (400, [], False, True),
        (400, "not JSON", False, True),
        (422, _REFUSAL, False, True),
        (400, _REFUSAL, True, True),
        (400, _REFUSAL, False, False),
        (400, _refusal(_AUDIO_REFUSAL_MESSAGE), False, True),
        (400, _refusal(_TOOL_LOOP_REFUSAL_MESSAGE), False, True),
        (400, _refusal(""), False, True),
        (400, _refusal(None), False, True),
    ],
    ids = [
        "other-param",
        "other-code",
        "string-error",
        "list-body",
        "not-json",
        "wrong-status",
        "external-provider",
        "no-json-mode",
        "audio-reply",
        "unsloth-tool-loop",
        "empty-message",
        "null-message",
    ],
)
def test_unrelated_errors_and_provider_contracts_are_not_retried(
    research_call, status, body, provider, json_mode
):
    if provider:
        research_call.run["config"]["inferenceRequest"] = {
            "providerType": "openai",
            "providerId": "connection",
            "externalModel": "model",
        }
    sent = []

    def serve(request):
        sent.append(request)
        return httpx.Response(status, content = body if isinstance(body, str) else json.dumps(body))

    research_call.install(httpx.MockTransport(serve))
    with pytest.raises(httpx.HTTPStatusError) as caught:
        research_call.complete(json_mode = json_mode)
    assert caught.value.response.status_code == status
    assert len(sent) == 1
    assert research_call.revoked == [1]


def test_an_audio_model_is_refused_rather_than_re_sent(monkeypatch, research_call):
    """Audio refusal must remain distinct from unavailable grammar support."""
    backend = _ScriptedBackend(_fixed('{"ok": true}'))
    backend.models[backend.active_model_name].update(is_audio = True, audio_type = "tts")
    _install(monkeypatch, backend)
    app = FastAPI()
    app.include_router(inference_route.router, prefix = "/v1")
    install_api_error_handlers(app)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    statuses = []

    class RecordingTransport(httpx.ASGITransport):
        async def handle_async_request(self, request):
            response = await super().handle_async_request(request)
            statuses.append(response.status_code)
            return response

    research_call.install(RecordingTransport(app = app))
    with pytest.raises(httpx.HTTPStatusError) as caught:
        research_call.complete(phase = "planning", max_tokens = 32)
    assert caught.value.response.status_code == 400
    assert statuses == [400], "the guided-decoding refusal is the answer, not a retry"
    assert not backend.calls
    assert research_call.revoked == [1]


@pytest.mark.parametrize("requested_audio", [False, True])
def test_fallback_uses_the_refused_request_not_an_intervening_model(
    monkeypatch, research_call, requested_audio
):
    backend = _ScriptedBackend(_fixed('{"ok": true}'))
    backend.models["sf-model"].update(is_audio = requested_audio, audio_type = "tts")
    backend.models["intervening"] = {"is_audio": not requested_audio, "audio_type": "tts"}
    _install(monkeypatch, backend)
    monkeypatch.setattr(research_runs, "_peek_inference_backend", lambda: backend)
    app = FastAPI()
    app.include_router(inference_route.router, prefix = "/v1")
    install_api_error_handlers(app)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    statuses = []
    audio_calls = []

    async def audio(*args, **kwargs):
        audio_calls.append(True)
        return JSONResponse({"unexpected_audio": True})

    monkeypatch.setattr(inference_route, "generate_audio", audio)

    class SwitchingTransport(httpx.ASGITransport):
        async def handle_async_request(self, request):
            # Model loading is scripted; requests and the refusal use the real route.
            backend.active_model_name = "sf-model"
            response = await super().handle_async_request(request)
            statuses.append(response.status_code)
            if len(statuses) == 1:
                backend.active_model_name = "intervening"
            return response

    research_call.install(SwitchingTransport(app = app))
    if requested_audio:
        with pytest.raises(httpx.HTTPStatusError):
            research_call.complete()
        assert statuses == [400]
    else:
        assert json.loads(research_call.complete()[0]) == {"ok": True}
        assert statuses == [400, 200]
    assert audio_calls == []
    assert research_call.revoked == [1]


@pytest.mark.parametrize("gguf", [False, True], ids = ["non-gguf", "gguf"])
def test_text_requirement_survives_a_manual_switch_before_resend(monkeypatch, research_call, gguf):
    backend = _ScriptedBackend(_fixed('{"ok": true}'))
    backend.models["audio-model"] = {"is_audio": True, "audio_type": "tts"}
    _install(monkeypatch, backend)
    monkeypatch.setattr(research_runs, "_peek_inference_backend", lambda: backend)
    app = FastAPI()
    app.include_router(inference_route.router, prefix = "/v1")
    install_api_error_handlers(app)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    audio_calls = []
    statuses = []

    async def switch_after_check(*args):
        backend.active_model_name = "audio-model"
        if gguf:
            monkeypatch.setattr(
                inference_route,
                "get_llama_cpp_backend",
                lambda: SimpleNamespace(is_loaded = True, _is_audio = True, context_length = 8192),
            )

    async def audio(*args, **kwargs):
        audio_calls.append(True)
        return JSONResponse({"unexpected_audio": True})

    monkeypatch.setattr(research_call.supervisor, "_check_active", switch_after_check)
    monkeypatch.setattr(inference_route, "generate_audio", audio)

    class RecordingTransport(httpx.ASGITransport):
        async def handle_async_request(self, request):
            response = await super().handle_async_request(request)
            statuses.append(response.status_code)
            return response

    research_call.install(RecordingTransport(app = app))
    with pytest.raises(httpx.HTTPStatusError) as caught:
        research_call.complete()
    assert "text output" in caught.value.response.text
    assert statuses == [400, 400]
    assert audio_calls == []
    assert research_call.revoked == [1]


def test_ordinary_audio_request_retains_audio_dispatch(monkeypatch):
    from .test_sf_client_tools_passthrough import _call, _request

    backend = _ScriptedBackend(_fixed("unused"))
    backend.models["sf-model"].update(is_audio = True, audio_type = "tts")
    audio_calls = []

    async def audio(*args, **kwargs):
        audio_calls.append(True)
        return JSONResponse({"audio": "scripted"})

    monkeypatch.setattr(inference_route, "generate_audio", audio)
    response = _call(_request(), monkeypatch, backend)
    assert response.status_code == 200
    assert audio_calls == [True]


@pytest.mark.parametrize("gguf", [True, False], ids = ["gguf", "non-gguf"])
def test_a_request_without_headers_still_reaches_audio(monkeypatch, gguf):
    """The durable-run producer builds its own request without headers, so reading the
    opt-out off request.headers turned every audio reply there into an AttributeError."""
    backend = _ScriptedBackend(_fixed("unused"))
    backend.models["sf-model"].update(is_audio = True, audio_type = "tts")
    audio_calls = []

    async def audio(*args, **kwargs):
        audio_calls.append(True)
        return JSONResponse({"audio": "scripted"})

    monkeypatch.setattr(inference_route, "generate_audio", audio)
    monkeypatch.setattr(inference_route, "api_monitor", ApiMonitor(max_entries = 4))
    monkeypatch.setattr(
        inference_route,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(
            is_loaded = gguf,
            _is_audio = gguf,
            supports_tools = False,
            is_vision = False,
            model_identifier = "gguf-tts",
            context_length = 2048,
        ),
    )
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: backend)
    monkeypatch.setattr(
        inference_route, "_detect_safetensors_features", lambda *a, **k: {"supports_tools": False}
    )
    headerless = SimpleNamespace(
        state = SimpleNamespace(),
        url = SimpleNamespace(path = "/v1/chat/completions"),
        method = "POST",
    )
    asyncio.run(
        inference_route.openai_chat_completions(
            ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "say hello")],
            ),
            request = headerless,
            current_subject = "test",
        )
    )
    assert audio_calls == [True]
    assert inference_route._text_output_required(headerless) is False


def test_format_fallback_is_attempted_only_once(research_call):
    sent = []

    def serve(request):
        sent.append(json.loads(request.content))
        return httpx.Response(400, json = _REFUSAL)

    research_call.install(httpx.MockTransport(serve))
    with pytest.raises(httpx.HTTPStatusError):
        research_call.complete()
    assert len(sent) == 2
    assert "response_format" in sent[0]
    assert "response_format" not in sent[1]


def test_cancellation_before_fallback_releases_response_and_key(monkeypatch, research_call):
    responses = []

    def serve(request):
        response = httpx.Response(400, json = _REFUSAL)
        responses.append(response)
        return response

    async def cancelled(*args):
        raise research_runs.RunCancelled()

    monkeypatch.setattr(research_call.supervisor, "_check_active", cancelled)
    research_call.install(httpx.MockTransport(serve))
    with pytest.raises(research_runs.RunCancelled):
        research_call.complete()
    assert len(responses) == 1
    assert responses[0].is_closed
    assert research_call.revoked == [1]


@pytest.mark.parametrize(
    "output",
    [
        '{"title": "Plan", "steps": [{"title": "Check docs", "query": "MLX JSON support"}]}',
        "not JSON",
        '{"title": "Plan", "steps": []}',
    ],
)
def test_planning_after_fallback_still_validates_before_saving(monkeypatch, research_call, output):
    run = research_call.run
    run.update(threadId = "thread", userMessageId = "message")
    run["config"]["budgets"]["maxSteps"] = 3
    monkeypatch.setattr(
        research_runs, "_research_question_context", lambda *args: ("Research MLX JSON", "[]")
    )
    saved = []

    def save_plan(run_id, plan, *args):
        saved.append(plan)
        return {"plan": plan}

    monkeypatch.setattr(research_runs.db, "set_plan", save_plan)
    monkeypatch.setattr(research_runs.db, "append_worker_event", lambda *args: 1)
    sent = []

    def serve(request):
        payload = json.loads(request.content)
        sent.append(payload)
        if len(sent) == 1:
            return httpx.Response(400, json = _REFUSAL)
        chunk = {"choices": [{"delta": {"content": output}, "finish_reason": "stop"}]}
        return httpx.Response(200, text = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")

    research_call.install(httpx.MockTransport(serve))
    if "Check docs" in output:
        asyncio.run(research_call.supervisor._plan(run))
        assert saved == [json.loads(output)]
    else:
        with pytest.raises(ValueError, match = "Planner"):
            asyncio.run(research_call.supervisor._plan(run))
        assert saved == []
    assert len(sent) == 2
    assert "Return only strict JSON" in sent[1]["messages"][0]["content"]
    assert "response_format" not in sent[1]


def test_json_fallback_does_not_restart_the_total_timeout(research_call):
    # 600ms of scheduling slack: a 0.4s/0.25s pairing leaves 150ms, which flakes under -n 4.
    research_call.run["config"]["budgets"]["modelTimeoutSeconds"] = 1.6
    sent = []

    async def serve(request):
        sent.append(json.loads(request.content))
        await asyncio.sleep(1.0)
        return httpx.Response(400, json = _REFUSAL) if len(sent) == 1 else _completion()

    research_call.install(httpx.MockTransport(serve))
    with pytest.raises(research_runs.ModelWallClockTimeout):
        research_call.complete()
    assert len(sent) == 2
    assert "response_format" not in sent[1]
    assert research_call.revoked == [1]


def test_json_fallback_does_not_refund_transport_retries(monkeypatch, research_call):
    statuses = iter([500, 400, 500, 500])
    sent = []
    delays = []
    real_sleep = asyncio.sleep

    async def sleep(delay):
        delays.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(research_runs.asyncio, "sleep", sleep)

    def serve(request):
        sent.append(json.loads(request.content))
        status = next(statuses)
        return httpx.Response(status, json = _REFUSAL if status == 400 else {"error": "server"})

    research_call.install(httpx.MockTransport(serve))
    with pytest.raises(httpx.HTTPStatusError) as caught:
        research_call.complete()
    assert caught.value.response.status_code == 500
    assert len(sent) == 4
    assert ["response_format" in body for body in sent] == [True, True, False, False]
    assert delays == [1, 2]


def test_json_fallback_never_replays_a_started_generation(research_call):
    sent = []

    def serve(request):
        sent.append(request)
        chunk = {"choices": [{"delta": {"content": "partial"}}]}
        error = {"error": {**_REFUSAL["error"], "message": "late format error"}}
        return httpx.Response(
            200, text = f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(error)}\n\ndata: [DONE]\n\n"
        )

    research_call.install(httpx.MockTransport(serve))
    with pytest.raises(RuntimeError, match = "late format error"):
        research_call.complete()
    assert len(sent) == 1
    assert research_call.revoked == [1]
