# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json

import httpx
import pytest
from fastmcp.exceptions import ToolError

from studio_mcp.errors import raise_for_payload, raise_for_route, raise_for_status_field, tool_error
from studio_mcp.forward import ndjson_last

from .mcp_harness import DEFERRED, openai_error

SENTINEL = "/srv/mcp-sentinel/models/secret.gguf"
WIN_SENTINEL = "C:\\mcp-sentinel\\models\\secret.gguf"
BUSY = f"Another account is generating on the resident model {SENTINEL}"
BUSY_OUT = "GPU busy: Another account is generating"


def _response(
    status,
    body = None,
    *,
    headers = None,
    content = None,
):
    content = json.dumps(body).encode() if content is None else content
    return httpx.Response(
        status, content = content, headers = {"content-type": "application/json", **(headers or {})}
    )


def _v1(status, message, **fields):
    sent = openai_error(message, status, **fields)
    return httpx.Response(status, content = sent.body, headers = sent.raw_headers)


def _route(resp, **kwargs):
    # An NDJSON stream's caller passes its last line as the payload.
    if resp.content.startswith(b'{"type"'):
        kwargs["payload"] = ndjson_last(resp.content)
    return raise_for_route(resp, **kwargs)


def _message(resp, **kwargs):
    with pytest.raises(ToolError) as caught:
        _route(resp, **kwargs)
    message = str(caught.value)
    assert "mcp-sentinel" not in message
    return message


def _ndjson(*lines):
    return _response(200, content = "\n".join(json.dumps(line) for line in lines).encode())


# A plain {"detail": str} at each status the routes send.
DETAIL_CASES = [
    pytest.param(_response(status, {"detail": f"{text} {path}"}), [text, f"(HTTP {status})"], id = id)
    for status, text, path, id in [
        (404, "Model not found", SENTINEL, "api-detail"),
        (499, "Client cancelled", "", "499"),
        (501, "STT runtime missing", WIN_SENTINEL, "501-windows-path"),
        (401, "Invalid or expired API key", "", "401"),
        (403, "Only the installation owner can do this", "", "403"),
    ]
]
VALIDATION_422 = {
    "detail": [
        {"loc": ["body", "max_tokens"], "msg": "Input should be a valid integer", "type": "x"},
        {"loc": ["query", "name"], "msg": f"bad {SENTINEL}", "type": "value_error"},
    ]
}
CASES = DETAIL_CASES + [
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
            {"error": "gpu_busy", "message": BUSY, "retry_after": 7},
            headers = {"Retry-After": "7"},
        ),
        [BUSY_OUT, "Retry after 7 s."],
        id = "gpu-busy-unwrapped",
    ),
    pytest.param(
        _v1(409, BUSY, type = "conflict_error", code = "gpu_busy", headers = {"Retry-After": "4"}),
        [BUSY_OUT, "Retry after 4 s."],
        id = "gpu-busy-v1-envelope",
    ),
    pytest.param(
        _response(409, {"detail": {"error": "gpu_busy", "message": BUSY, "retry_after": 9}}),
        [BUSY_OUT, "Retry after 9 s."],
        id = "gpu-busy-detail-wrapped",
    ),
    pytest.param(
        _v1(400, "messages: Field required", param = "messages"),
        ["Invalid arguments: messages: Field required"],
        id = "v1-validation-400",
    ),
    pytest.param(
        _response(422, VALIDATION_422),
        ["Invalid arguments: max_tokens: Input should be a valid integer; query.name: bad"],
        id = "api-validation-422",
    ),
    pytest.param(
        _response(
            503,
            {"detail": {"error_type": "model_loading", "message": f"Loading {SENTINEL}"}},
            headers = {"Retry-After": "30"},
        ),
        ["Loading", "(HTTP 503)", "Retry after 30 s."],
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
        _response(502, content = f"Bad gateway {SENTINEL}".encode()),
        ["Bad gateway", "(HTTP 502)"],
        id = "plain-text",
    ),
    pytest.param(
        _ndjson({"type": "progress"}, {"type": "error", "message": f"Decoding failed {SENTINEL}"}),
        ["Decoding failed"],
        id = "ndjson-error-line",
    ),
    *[
        pytest.param(
            _response(
                200,
                content = b" " * 64
                + json.dumps(
                    {"_deferred_error": {**d, "detail": f"{d['detail']} {SENTINEL}"}}
                ).encode(),
            ),
            [d["detail"], f"(HTTP {d['status_code']})"],
            id = f"padded-200-deferred-{d['status_code']}",
        )
        for d in DEFERRED
    ],
]


@pytest.mark.parametrize("resp,expected", CASES)
def test_route_errors_become_scrubbed_tool_errors(resp, expected):
    message = _message(resp)
    for text in expected:
        assert text in message


@pytest.mark.parametrize(
    "resp,expected",
    [
        (_response(200, content = b"   \n " + b'{"status": "loaded"}'), {"status": "loaded"}),
        (
            _ndjson({"type": "progress"}, {"type": "result", "text": "hi"}),
            {"type": "result", "text": "hi"},
        ),
        (httpx.Response(200, content = b"\x89PNG\r\n"), None),
    ],
    ids = ["padded-200", "ndjson-result", "non-json"],
)
def test_a_successful_body_parses(resp, expected):
    assert _route(resp) == expected
    assert raise_for_payload(None) is None


def test_train_start_status_error_is_an_error():
    payload = {"job_id": "", "status": "error", "message": f"Training is in progress {SENTINEL}"}
    assert raise_for_route(_response(200, payload)) == payload
    with pytest.raises(ToolError) as caught:
        raise_for_status_field(payload)
    assert str(caught.value).startswith("Training is in progress")
    assert "mcp-sentinel" not in str(caught.value)
    assert raise_for_status_field({"job_id": "j", "status": "queued"})["job_id"] == "j"


def test_tool_guidance_survives_the_scrub():
    hint = "Load one with load_model(kind='image') first, then retry POST /v1/images/generations."
    message = _message(_response(409, {"detail": f"No model at {SENTINEL}"}), hints = {409: hint})
    assert message == f"No model at <path> (HTTP 409) {hint}"
    assert hint not in _message(_response(500, {"detail": "boom"}), hints = {409: hint})
    # gpu_busy keeps its retry hint instead of the tool's guidance.
    busy = _response(
        409,
        {"error": "gpu_busy", "message": "Another account is generating.", "retry_after": 3},
        headers = {"Retry-After": "3"},
    )
    assert _message(busy, hints = {409: "x"}) == f"{BUSY_OUT}. Retry after 3 s."


def test_a_routes_own_call_this_route_hint_is_dropped():
    error = tool_error("No model loaded. Call POST /inference/load first.", "Load a chat model.")
    assert str(error) == "No model loaded. Load a chat model."
    # A path outside a route hint is scrubbed, not dropped.
    assert str(tool_error("Saved to /srv/mcp-sentinel/x.png")) == "Saved to <path>"
