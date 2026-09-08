# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""
Endpoint-level cover for the /v1/responses message-attachment refusal.

The unit tests in test_responses_tool_passthrough.py call
``_normalise_responses_input`` directly. These drive the real router instead, because
three things only the route can answer:

- the refusal has to arrive as a JSON 400, not as a 500 out of request handling;
- a streaming request has to be refused *before* the SSE stream opens, or the client
  gets a broken stream with a 200 already on the wire;
- the refusal has to land before the model switch, or an unservable request evicts the
  resident model on its way to being rejected.

No running server, no GPU, no model.
"""

import os
import sys

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

import pytest


def _route_client():
    """The real inference router, with only the auth dependency stubbed."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from auth.authentication import get_current_subject
    import routes.inference as inference_route

    app = FastAPI()
    # main.py mounts this router at /v1 for the OpenAI-compatible surface.
    app.include_router(inference_route.router, prefix = "/v1")
    app.dependency_overrides[get_current_subject] = lambda: "test"
    return TestClient(app, raise_server_exceptions = False)


def _body(part, *, role = "user", stream = False):
    message = {
        "role": role,
        "content": [{"type": "input_text", "text": "what does this say?"}, part],
    }
    body = {"model": "default", "input": [message]}
    if stream:
        body["stream"] = True
    return body


_REFUSED_PARTS = [
    ({"type": "input_file", "file_data": "data:application/pdf;base64,AA", "filename": "r.pdf"},
     "input_file"),
    ({"type": "input_file", "file_id": "file_abc"}, "input_file"),
    ({"type": "input_file", "file_url": "https://example.com/d.pdf"}, "input_file"),
    ({"type": "input_image", "file_id": "file_abc"}, "file_id"),
    ({"type": "input_image"}, "require an image_url string"),
    ({"type": "input_image", "image_url": "https://example.com/a.png", "detail": "medium"},
     "auto, low, high, or original"),
    ({"type": "input_audio", "input_audio": {"data": "AA", "format": "wav"}}, "input_audio"),
    ({"type": "input_something_new", "value": 1}, "input_something_new"),
]


def _part_id(case):
    part, _ = case
    return f"{part['type']}-{'+'.join(k for k in part if k != 'type') or 'bare'}"


@pytest.mark.parametrize("case", _REFUSED_PARTS, ids = _part_id)
def test_the_route_answers_json_400_not_a_500(case):
    part, needle = case
    with _route_client() as client:
        response = client.post("/v1/responses", json = _body(part))

    assert response.status_code == 400, response.text
    error = response.json()["detail"]["error"]
    assert needle in error["message"]
    assert error["code"] == "unsupported_parameter"
    assert error["param"] == "input"


@pytest.mark.parametrize("case", _REFUSED_PARTS, ids = _part_id)
def test_a_streaming_request_is_refused_before_the_stream_opens(case):
    """A 200 text/event-stream carrying an error frame is a broken stream, not a refusal."""
    part, needle = case
    with _route_client() as client:
        response = client.post("/v1/responses", json = _body(part, stream = True))

    assert response.status_code == 400, response.text
    assert "text/event-stream" not in response.headers.get("content-type", "")
    assert needle in response.json()["detail"]["error"]["message"]


@pytest.mark.parametrize("role", ["system", "developer", "assistant"])
def test_the_route_refuses_an_attachment_on_a_non_user_turn(role):
    part = {"type": "input_file", "filename": "r.pdf"}
    with _route_client() as client:
        response = client.post("/v1/responses", json = _body(part, role = role))

    assert response.status_code == 400, response.text
    assert "input_file" in response.json()["detail"]["error"]["message"]


@pytest.mark.parametrize("role", ["system", "developer", "assistant"])
def test_the_route_refuses_a_servable_image_on_a_role_that_flattens(role):
    # These roles are flattened to a string, so the image was dropped and the turn answered.
    part = {"type": "input_image", "image_url": "https://example.com/a.png"}
    with _route_client() as client:
        response = client.post("/v1/responses", json = _body(part, role = role))

    assert response.status_code == 400, response.text
    assert "only supported on user messages" in response.json()["detail"]["error"]["message"]


@pytest.mark.parametrize("case", _REFUSED_PARTS, ids = _part_id)
def test_the_refusal_lands_before_the_model_switch(case, monkeypatch):
    """The route holds this for its other refusals: a 400 must not evict the model."""
    import routes.inference as inference_route

    switched = []

    async def _record(*args, **kwargs):
        switched.append(args)
        raise AssertionError("model switch reached on an unservable request")

    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _record)

    part, _ = case
    with _route_client() as client:
        response = client.post("/v1/responses", json = _body(part))

    assert response.status_code == 400, response.text
    assert switched == []


@pytest.mark.parametrize(
    "part",
    [
        {"type": "input_image", "image_url": "https://example.com/a.png"},
        {"type": "input_image", "image_url": "https://example.com/a.png", "detail": "original"},
        {"type": "input_image", "image_url": "https://example.com/a.png", "file_id": "file_abc"},
        {"type": "input_text", "text": "second line"},
    ],
    ids = ["url", "url+detail", "url+file_id", "text"],
)
def test_a_servable_turn_is_not_refused(part):
    """The refusal must not reach past the shapes it owns.

    No model here, so a servable turn stops at the load check below; reaching it is the
    assertion, since only an accepted normalisation gets that far.
    """
    with _route_client() as client:
        response = client.post("/v1/responses", json = _body(part))

    assert "No model loaded" in response.text, response.text
    assert "unsupported_parameter" not in response.text
