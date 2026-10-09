# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json

import httpx
import pytest
from fastmcp.exceptions import ToolError

from studio_mcp.errors import raise_for_payload, raise_for_route, raise_for_status_field
from studio_mcp.forward import ndjson_last

from .mcp_harness import DEFERRED, openai_error

SENTINEL = "/srv/mcp-sentinel/models/secret.gguf"
WIN_SENTINEL = "C:\\mcp-sentinel\\models\\secret.gguf"


def _response(
    status,
    body = None,
    *,
    headers = None,
    content = None,
):
    if content is None:
        content = json.dumps(body).encode()
    return httpx.Response(
        status, content = content, headers = {"content-type": "application/json", **(headers or {})}
    )


def _v1(status, message, **fields):
    # The OpenAI-style envelope the /v1 routes answer with.
    sent = openai_error(message, status, **fields)
    return httpx.Response(status, content = sent.body, headers = sent.raw_headers)


def _message(resp, **kwargs):
    with pytest.raises(ToolError) as caught:
        raise_for_route(resp, **kwargs)
    message = str(caught.value)
    assert "mcp-sentinel" not in message
    return message


GPU_BUSY_MESSAGE = f"Another account is generating on the resident model {SENTINEL}"

CASES = [
    pytest.param(
        _response(404, {"detail": f"Model not found {SENTINEL}"}),
        ["Model not found", "(HTTP 404)"],
        id = "api-detail",
    ),
    pytest.param(
        _response(409, {"detail": {"message": f"Busy loading {SENTINEL}", "code": "x"}}),
        ["Busy loading", "(HTTP 409)"],
        id = "api-detail-dict",
    ),
    pytest.param(
        _v1(404, f"No model {SENTINEL}", type = "not_found_error", code = "model_not_found"),
        ["No model", "(model_not_found)", "(HTTP 404)"],
        id = "v1-envelope",
    ),
    pytest.param(
        _response(
            409,
            {"error": "gpu_busy", "message": GPU_BUSY_MESSAGE, "retry_after": 7},
            headers = {"Retry-After": "7"},
        ),
        ["GPU busy: Another account is generating", "Retry after 7 s."],
        id = "gpu-busy-unwrapped",
    ),
    pytest.param(
        _v1(
            409,
            GPU_BUSY_MESSAGE,
            type = "conflict_error",
            param = "model",
            code = "gpu_busy",
            headers = {"Retry-After": "4"},
        ),
        ["GPU busy: Another account is generating", "Retry after 4 s."],
        id = "gpu-busy-v1-envelope",
    ),
    pytest.param(
        _response(
            409, {"detail": {"error": "gpu_busy", "message": GPU_BUSY_MESSAGE, "retry_after": 9}}
        ),
        ["GPU busy: Another account is generating", "Retry after 9 s."],
        id = "gpu-busy-detail-wrapped",
    ),
    pytest.param(
        _v1(400, "messages: Field required", param = "messages"),
        ["Invalid arguments: messages: Field required"],
        id = "v1-validation-400",
    ),
    pytest.param(
        _response(
            422,
            {
                "detail": [
                    {
                        "loc": ["body", "max_tokens"],
                        "msg": "Input should be a valid integer",
                        "type": "int_parsing",
                    },
                    {"loc": ["query", "name"], "msg": f"bad {SENTINEL}", "type": "value_error"},
                ]
            },
        ),
        ["Invalid arguments: max_tokens: Input should be a valid integer; query.name: bad"],
        id = "api-validation-422",
    ),
    pytest.param(
        _response(
            503,
            {
                "detail": {
                    "error_type": "model_loading",
                    "message": f"Decision model is loading {SENTINEL}",
                }
            },
            headers = {"Retry-After": "30"},
        ),
        ["Decision model is loading", "(HTTP 503)", "Retry after 30 s."],
        id = "systemone-503-retry-after",
    ),
    pytest.param(
        _response(
            400,
            {"detail": f"Not enough memory for 4096x4096 {SENTINEL}"},
            headers = {"X-Unsloth-Refusal": "memory-estimate"},
        ),
        ["Not enough memory for 4096x4096", "Pass allow_oversized=true to try anyway."],
        id = "image-memory-refusal",
    ),
    pytest.param(
        _response(499, {"detail": "Client cancelled"}), ["Client cancelled", "(HTTP 499)"], id = "499"
    ),
    pytest.param(
        _response(501, {"detail": f"STT runtime missing {WIN_SENTINEL}"}),
        ["STT runtime missing", "(HTTP 501)"],
        id = "501-windows-path",
    ),
    pytest.param(
        _response(502, content = f"Bad gateway {SENTINEL}".encode()),
        ["Bad gateway", "(HTTP 502)"],
        id = "plain-text",
    ),
    pytest.param(
        _response(401, {"detail": "Invalid or expired API key"}),
        ["Invalid or expired API key", "(HTTP 401)"],
        id = "401",
    ),
    pytest.param(
        _response(403, {"detail": "Only the installation owner can do this"}),
        ["Only the installation owner can do this", "(HTTP 403)"],
        id = "403",
    ),
]


@pytest.mark.parametrize("resp,expected", CASES)
def test_route_errors_become_scrubbed_tool_errors(resp, expected):
    message = _message(resp)
    for text in expected:
        assert text in message


@pytest.mark.parametrize("deferred", DEFERRED)
def test_a_padded_200_with_a_deferred_error_is_an_error(deferred):
    sent = {**deferred, "detail": f"{deferred['detail']} {SENTINEL}"}
    body = b" " * 64 + json.dumps({"_deferred_error": sent}).encode()
    message = _message(_response(200, content = body))
    assert deferred["detail"] in message
    assert f"(HTTP {deferred['status_code']})" in message


def test_a_padded_200_success_parses():
    body = b"   \n " + json.dumps({"status": "loaded", "model": "m"}).encode()
    assert raise_for_route(_response(200, content = body)) == {"status": "loaded", "model": "m"}


def test_an_ndjson_error_line_is_an_error():
    body = (
        json.dumps({"type": "progress", "fraction": 0.5})
        + "\n"
        + json.dumps({"type": "error", "message": f"Decoding failed {SENTINEL}"})
        + "\n"
    ).encode()
    resp = _response(200, content = body)
    message = _message(resp, payload = ndjson_last(resp.content))
    assert "Decoding failed" in message


def test_an_ndjson_result_line_passes():
    body = (
        json.dumps({"type": "progress"}) + "\n" + json.dumps({"type": "result", "text": "hi"})
    ).encode()
    resp = _response(200, content = body)
    assert raise_for_route(resp, payload = ndjson_last(resp.content)) == {
        "type": "result",
        "text": "hi",
    }


def test_train_start_status_error_is_an_error():
    payload = {
        "job_id": "",
        "status": "error",
        "message": f"Training is already in progress {SENTINEL}",
        "error": "Training already active",
    }
    assert raise_for_route(_response(200, payload)) == payload
    with pytest.raises(ToolError) as caught:
        raise_for_status_field(payload)
    assert "Training is already in progress" in str(caught.value)
    assert "mcp-sentinel" not in str(caught.value)
    assert (
        raise_for_status_field({"job_id": "j", "status": "queued", "message": "ok"})["job_id"]
        == "j"
    )


def test_a_non_json_success_body_returns_none():
    assert raise_for_route(httpx.Response(200, content = b"\x89PNG\r\n")) is None
    assert raise_for_payload(None) is None


def test_tool_guidance_survives_the_scrub():
    hint = "Load one with load_model(kind='image') first, then retry POST /v1/images/generations."
    resp = _response(409, {"detail": f"No image model is loaded at {SENTINEL}"})
    message = _message(resp, hints = {409: hint})
    assert message == f"No image model is loaded at <path> (HTTP 409) {hint}"
    # A status the hints do not name gets none.
    assert hint not in _message(_response(500, {"detail": "boom"}), hints = {409: hint})


def test_gpu_busy_keeps_its_retry_hint_instead_of_tool_guidance():
    resp = _response(
        409,
        {"error": "gpu_busy", "message": "Another account is generating.", "retry_after": 3},
        headers = {"Retry-After": "3"},
    )
    message = _message(resp, hints = {409: "Download the model first."})
    assert message == "GPU busy: Another account is generating. Retry after 3 s."


def test_a_routes_own_call_this_route_hint_is_dropped():
    from studio_mcp.errors import tool_error

    error = tool_error(
        "No model loaded. Call POST /inference/load first.", "Load a chat model first."
    )
    assert str(error) == "No model loaded. Load a chat model first."
    # A path in a sentence that is not a route hint is still scrubbed, not dropped.
    assert str(tool_error("Saved to /srv/mcp-sentinel/x.png")) == "Saved to <path>"
