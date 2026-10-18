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
        ["[model_not_found] No model", "(HTTP 404)"],
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
        _response(409, {"detail": {"error": "download_busy", "retry_after": 1}}),
        [
            "[download_busy retry_after=1] Another speech-to-text download is running. Retry after 1 s."
        ],
        id = "download-busy-detail",
    ),
    pytest.param(
        _v1(400, "messages: Field required", param = "messages"),
        ["Invalid arguments: messages: Field required"],
        id = "v1-validation-400",
    ),
    pytest.param(
        _v1(
            400,
            "response_format is not supported.",
            param = "response_format",
            code = "unsupported_parameter",
        ),
        ["[unsupported_parameter] Invalid arguments: response_format is not supported."],
        id = "v1-validation-400-keeps-its-code",
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
    message = _message(busy, hints = {409: "x"})
    assert message.startswith("[gpu_busy retry_after=3] ")
    assert message.endswith(f"] {BUSY_OUT}. Retry after 3 s.")


def test_a_routes_own_call_this_route_hint_is_dropped():
    error = tool_error("No model loaded. Call POST /inference/load first.", "Load a chat model.")
    assert str(error) == "No model loaded. Load a chat model."
    # A path outside a route hint is scrubbed, not dropped.
    assert str(tool_error("Saved to /srv/mcp-sentinel/x.png")) == "Saved to <path>"


# The Keep multiple models loaded refusal, as the owner and a managed account see it.
OWNER_ALWAYS = (
    "Loading this model would unload the chat model unsloth/Qwen3-0.6B, and Keep multiple models"
    " loaded is set to Always. Unload that model first, or choose When they fit."
)
MANAGED_ALWAYS = (
    "Loading this model would unload the chat model unsloth/Qwen3-0.6B, which Unsloth Studio is"
    " set to keep loaded. Unload that model first, or ask the Unsloth Studio owner."
)
ALWAYS_TAG = "[multi_model_always kind=chat] "
TOOL_HINT = "Wait for the load to finish, then retry."


def _padded(deferred):
    return _response(200, content = b" " * 64 + json.dumps({"_deferred_error": deferred}).encode())


def _assert_always(message, tag = ALWAYS_TAG):
    assert message.startswith(tag), message
    assert "unload_model(kind='llm')" in message
    assert "retrying won't help" in message
    assert "Retry after" not in message
    assert "force_cancel_active" not in message
    assert TOOL_HINT not in message


@pytest.mark.parametrize(
    "resp",
    [
        _padded({"status_code": 409, "detail": OWNER_ALWAYS}),
        _response(
            409,
            {"detail": OWNER_ALWAYS},
            headers = {"X-Unsloth-Refusal": "multi-model-always", "Retry-After": "5"},
        ),
        _response(409, {"detail": MANAGED_ALWAYS}),
        _v1(409, OWNER_ALWAYS, type = "conflict_error", param = "model", code = "multi_model_always"),
        # A route that copies the headers into the padded body; the copy is not the marker text.
        _padded(
            {
                "status_code": 409,
                "detail": "Loading this model would unload the chat model x.",
                "headers": {"X-UNSLOTH-REFUSAL": "multi-model-always"},
            }
        ),
        _response(
            409, {"detail": {"code": "multi_model_always", "kind": "chat", "message": "No."}}
        ),
    ],
    ids = ["padded-deferred", "header", "managed-phrase", "v1-code", "deferred-headers", "code"],
)
def test_a_multi_model_always_refusal_is_tagged_and_names_unload_model(resp):
    _assert_always(_message(resp, hints = {409: TOOL_HINT}))


def test_a_multi_model_always_refusal_of_an_unknown_kind_keeps_a_bare_tag():
    resp = _response(409, {"detail": "Keep multiple models loaded is set to Always."})
    message = _message(resp, hints = {409: TOOL_HINT})
    assert message.startswith("[multi_model_always] ")
    assert message.endswith(
        "Call unload_model(kind=…) for the model it names to free the GPU; retrying won't help."
    )


def test_always_wording_on_another_status_is_not_a_refusal():
    message = _message(_response(500, {"detail": "Keep multiple models loaded is set to Always."}))
    assert not message.startswith("[")
    assert "unload_model" not in message


@pytest.mark.parametrize(
    "resp",
    [
        _response(409, {"error": "gpu_busy", "message": BUSY, "retry_after": 12}),
        _response(409, {"detail": {"error": "gpu_busy", "message": BUSY, "retry_after": 12}}),
        _v1(409, BUSY, type = "conflict_error", code = "gpu_busy", headers = {"Retry-After": "12"}),
    ],
    ids = ["top-level", "detail-wrapped", "v1-envelope"],
)
def test_gpu_busy_is_tagged_with_its_retry_after(resp):
    message = _message(resp, hints = {409: TOOL_HINT})
    assert message.startswith("[gpu_busy retry_after=12] GPU busy: Another account")
    assert message.endswith("Retry after 12 s.")
    assert TOOL_HINT not in message


def test_gpu_busy_without_a_number_keeps_a_bare_tag():
    message = _message(_response(409, {"error": "gpu_busy", "message": "Busy."}))
    assert message.startswith("[gpu_busy] GPU busy: Busy.")


def test_a_detail_code_becomes_the_tag():
    resp = _response(500, {"detail": {"code": "out_of_memory", "message": f"OOM at {SENTINEL}"}})
    message = _message(resp)
    assert message.startswith("[out_of_memory] OOM at <path>")
    assert message.endswith("(HTTP 500)")
    # Not a plain identifier: the message stays, the tag goes.
    bad = _message(_response(500, {"detail": {"code": "Bad Code!", "message": "boom"}}))
    assert bad.startswith("boom")


def test_a_plain_string_detail_has_no_tag():
    message = _message(_response(409, {"detail": "Another model is loading"}))
    assert not message.startswith("[")
    assert message.startswith("Another model is loading (HTTP 409)")


ACTIVE = {
    "error": "active_generations",
    "message": (
        "Unloading would stop 1 chat that is still generating. Stop them first, or retry with"
        " force_cancel_active."
    ),
    "running": 1,
    "thread_ids": ["thread-secret"],
}


@pytest.mark.parametrize(
    "resp",
    [_response(409, {"detail": ACTIVE}), _padded({"status_code": 409, "detail": ACTIVE})],
    ids = ["direct", "padded"],
)
def test_running_chats_tell_the_agent_to_wait_or_cancel_its_own(resp):
    message = _message(resp, hints = {409: TOOL_HINT})
    assert message.startswith("[active_generations running=1] Unloading would stop 1 chat")
    assert "Stop them first. (HTTP 409)" in message
    assert message.endswith("cancel(kind='chat', id=<its cancel_id>).")
    assert "Wait for the running chats to finish" in message
    assert "thread" not in message
    assert "force_cancel_active" not in message
    assert TOOL_HINT not in message


def _at(status, body, path):
    return httpx.Response(
        status,
        content = json.dumps(body).encode(),
        headers = {"content-type": "application/json"},
        request = httpx.Request("POST", f"http://x{path}"),
    )


@pytest.mark.parametrize(
    "resp,kind",
    [
        (_at(400, {"detail": "No model loaded."}, "/v1/audio/run"), "audio"),
        (_at(400, {"detail": "No model loaded."}, "/v1/audio/transcriptions"), "stt"),
        (_at(409, {"detail": "No model loaded"}, "/api/inference/audio/stt/load"), "stt"),
        (_at(400, {"detail": "No model loaded."}, "/v1/images/generations"), "image"),
        (_at(400, {"detail": "No model loaded."}, "/v1/chat/completions"), "chat"),
        (_at(400, {"detail": "No model loaded."}, "/api/inference/unload"), "chat"),
        (_response(409, {"detail": "No diffusion model is loaded."}), "image"),
        (_response(503, {"detail": "No video model loaded. Load a video model first."}), "video"),
        (_v1(400, "No model loaded. Call POST /inference/load first."), "chat"),
    ],
    ids = [
        "audio-path",
        "stt-v1-path",
        "stt-api-path",
        "image-path",
        "chat-path",
        "unload-path",
        "diffusion-wording",
        "video-wording",
        "no-request",
    ],
)
def test_not_loaded_names_the_kind(resp, kind):
    assert _message(resp).startswith(f"[not_loaded kind={kind}] No ")


def test_a_deferred_not_loaded_uses_the_route_path():
    padded = httpx.Response(
        200,
        content = json.dumps(
            {"_deferred_error": {"status_code": 400, "detail": "No model loaded."}}
        ).encode(),
        request = httpx.Request("POST", "http://x/api/inference/audio/load"),
    )
    assert _message(padded).startswith("[not_loaded kind=audio] ")


NOISE = [f"load_tensors: layer {i} assigned to device CUDA0" for i in range(300)]
ERRORS = [
    "llama_model_load: error loading model: tensor 'blk.0' has wrong shape",
    "llama_init_from_model: failed to load model",
    "main: exiting due to model loading error",
]
HEAD = "llama-server failed to start (exit code 1)."
LLAMA_FAILURE = (
    f"{HEAD}\n\nllama-server output:\n"
    + "\n".join(NOISE[:150] + ERRORS[:1] + NOISE[150:] + ERRORS[1:])
    + "\n\nFull log: /srv/mcp-sentinel/logs/llama.log"
)


@pytest.mark.parametrize(
    "error",
    [
        lambda: tool_error(LLAMA_FAILURE),
        lambda: _caught(_response(500, {"detail": LLAMA_FAILURE})),
    ],
    ids = ["tool-error", "route"],
)
def test_llama_server_output_is_cut_to_its_error_lines(error):
    message = str(error())
    assert message.startswith(HEAD)
    assert message.removesuffix(" (HTTP 500)").endswith(ERRORS[-1])
    assert len(message) <= 2000
    assert "load_tensors" not in message
    assert "Full log" not in message
    for line in ERRORS:
        assert line in message
    # Briefing an already brief message changes nothing.
    assert str(tool_error(message)) == message


def _caught(resp):
    with pytest.raises(ToolError) as caught:
        raise_for_route(resp)
    return caught.value


def test_a_tag_that_is_not_a_plain_code_is_dropped():
    assert str(tool_error("boom", tag = "Bad Tag")) == "boom"
    assert str(tool_error("boom", "Try again.", tag = "out_of_memory")).startswith(
        "[out_of_memory] boom Try again."
    )
