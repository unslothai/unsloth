# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json

import httpx
import pytest
from fastmcp.exceptions import ToolError

from studio_mcp.errors import audio_code, raise_for_audio

from .mcp_harness import openai_error

OOM = "Kokoro ran out of memory. Free GPU memory, or set the audio device to CPU."
VARIANTS = [{"variant": "Q4_K_M", "label": "Q4_K_M", "size_bytes": 1}, {"variant": "Q3_K_M"}]


def _api(status, detail, **headers):
    content = json.dumps({"detail": detail}).encode()
    return httpx.Response(status, content = content, headers = headers)


def _v1(status, message, **fields):
    sent = openai_error(message, status, **fields)
    return httpx.Response(status, content = sent.body, headers = sent.raw_headers)


def _message(resp, **kwargs):
    with pytest.raises(ToolError) as caught:
        raise_for_audio(resp, **kwargs)
    return str(caught.value)


def test_out_of_memory_names_cpu_and_the_smaller_variants():
    detail = {"code": "out_of_memory", "message": OOM, "smaller_variants": VARIANTS}
    message = _message(_api(503, detail))
    assert message.startswith("[out_of_memory] Kokoro ran out of memory.")
    assert "device='cpu'" in message
    assert message.endswith("load_model(variant=...): Q4_K_M, Q3_K_M.")


def test_the_v1_envelope_code_gets_the_tag_and_the_generic_hint():
    message = _message(_v1(503, "Out of memory.", type = "server_error", code = "out_of_memory"))
    assert message.startswith("[out_of_memory] Out of memory.")
    assert message.endswith("Load the audio model again with device='cpu'.")
    # The envelope's "(out_of_memory)" suffix is dropped; the tag names the code once.
    assert message.count("out_of_memory") == 1


def test_gpu_unavailable_points_at_the_cpu():
    detail = {"code": "gpu_unavailable", "message": "The audio runtime could not use the GPU."}
    message = _message(_api(503, detail))
    assert message.startswith("[gpu_unavailable] ")
    assert message.endswith("Load the audio model again with device='cpu'.")


def test_runtime_stopped_says_retry_once():
    detail = {"code": "runtime_stopped", "message": "The audio runtime stopped: exit 9."}
    message = _message(_api(500, detail))
    assert message.startswith("[runtime_stopped] The audio runtime stopped")
    assert "Retry once" in message


def test_an_already_tagged_message_is_not_tagged_twice():
    detail = {"code": "out_of_memory", "message": "[out_of_memory] Out of memory."}
    message = _message(_api(503, detail))
    assert message.startswith("[out_of_memory] Out of memory.")
    assert message.count("[out_of_memory]") == 1


def test_a_gpu_busy_answer_keeps_its_retry_and_gets_no_audio_tag():
    resp = _v1(409, "Busy.", code = "gpu_busy", headers = {"Retry-After": "5"})
    assert _message(resp) == "GPU busy: Busy. Retry after 5 s."


@pytest.mark.parametrize(
    "resp",
    [
        _api(400, "No model loaded."),
        _api(500, {"code": "something_else", "message": "Broke."}),
        _v1(400, "No model loaded."),
    ],
)
def test_other_errors_are_unchanged(resp):
    message = _message(resp, hints = {400: "Load one."})
    assert not message.startswith("[")
    assert audio_code(json.loads(resp.content)) is None


def test_a_success_is_returned_as_is():
    resp = httpx.Response(200, content = b'{"clips": []}')
    assert raise_for_audio(resp) == {"clips": []}
