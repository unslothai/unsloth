# SPDX-License-Identifier: AGPL-3.0-only

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from routes import inference
from utils.api_errors import install_api_error_handlers


@pytest.fixture
def responses_client():
    app = FastAPI()
    app.include_router(inference.router, prefix = "/v1")
    app.dependency_overrides[get_current_subject] = lambda: "test-owner"
    install_api_error_handlers(app)

    @app.middleware("http")
    async def skip_monitor(request: Request, call_next):
        request.state.skip_api_monitor = True
        return await call_next(request)

    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "history_param,history_reference",
    [
        ("previous_response_id", "resp_previous"),
        ("previous_response_id", ""),
        ("conversation", "conv_previous"),
        ("conversation", {"id": "conv_previous"}),
    ],
)
def test_history_reference_is_rejected_before_any_processing(
    monkeypatch, responses_client, stream, history_param, history_reference
):
    normalise = Mock(side_effect = AssertionError("normalization reached"))
    switch = AsyncMock(side_effect = AssertionError("model switch reached"))
    generate = AsyncMock(side_effect = AssertionError("generation reached"))
    monkeypatch.setattr(inference, "_normalise_responses_input", normalise)
    monkeypatch.setattr(inference, "_maybe_auto_switch_model", switch)
    monkeypatch.setattr(inference, "_responses_stream", generate)
    monkeypatch.setattr(inference, "_responses_non_streaming", generate)

    response = responses_client.post(
        "/v1/responses",
        json = {
            "model": "different/model",
            "input": "What was my project code?",
            history_param: history_reference,
            "stream": stream,
        },
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "unsupported_parameter"
    assert error["param"] == history_param
    assert history_param in error["message"]
    assert "full conversation history" in error["message"]
    assert "input" in error["message"]
    normalise.assert_not_called()
    switch.assert_not_called()
    generate.assert_not_called()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "previous",
    [
        {},
        {"previous_response_id": None},
        {"conversation": None},
        {"previous_response_id": None, "conversation": None},
    ],
)
def test_full_history_without_history_reference_preserves_dispatch(
    monkeypatch, responses_client, stream, previous
):
    switch = AsyncMock()
    messages_seen = []

    async def generate(payload, messages, *args):
        messages_seen.extend((message.role, message.content) for message in messages)
        return JSONResponse({"id": "resp_control", "object": "response"})

    monkeypatch.setattr(inference, "_maybe_auto_switch_model", switch)
    monkeypatch.setattr(inference, "_responses_stream", generate)
    monkeypatch.setattr(inference, "_responses_non_streaming", generate)
    history = [
        {"role": "user", "content": "My project code is AZURE-317."},
        {"role": "assistant", "content": "Understood."},
        {"role": "user", "content": "What was my project code?"},
    ]

    response = responses_client.post(
        "/v1/responses",
        json = {
            "input": history,
            "stream": stream,
            **previous,
        },
    )

    assert response.status_code == 200
    assert messages_seen == [(item["role"], item["content"]) for item in history]
    assert switch.await_count == int(stream)
