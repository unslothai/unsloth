# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Research may request JSON through prompts when local guided decoding is unavailable."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from auth.authentication import get_current_subject
from core import research_runs
from routes import inference as inference_route
from utils.api_errors import install_api_error_handlers
from .test_sf_client_tools_passthrough import _ScriptedBackend, _fixed, _install


_REFUSAL = {"error": {"code": "unsupported_parameter", "param": "response_format"}}


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


@pytest.mark.parametrize("is_mlx", [True, False], ids = ["mlx", "transformers"])
@pytest.mark.parametrize("phase", ["planning", "decision", "synthesis_audit"])
def test_local_json_research_recovers_through_the_real_route(
    monkeypatch, research_call, is_mlx, phase
):
    backend = _ScriptedBackend(_fixed('{"ok": true}'))
    backend.models[backend.active_model_name]["is_mlx"] = is_mlx
    _install(monkeypatch, backend, supports_tools = False)
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
