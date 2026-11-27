# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import urllib.error
import urllib.request

import pytest

from unsloth_cli.antigravity import (
    _chat_request,
    _gemini_stream,
    antigravity_endpoint,
    session_settings,
)


def test_tool_results_follow_call_ids_even_when_returned_in_reverse_order():
    turns = [
        {
            "role": "model",
            "parts": [
                {"functionCall": {"name": "read_file", "id": key, "args": {"Path": key}}}
                for key in ("one", "two")
            ],
        },
        {
            "role": "user",
            "parts": [
                {"functionResponse": {"name": "read_file", "id": key, "response": {"output": key}}}
                for key in ("two", "one")
            ],
        },
    ]
    req = _chat_request({"contents": turns}, "local-model", {})
    msg, second, first = req["messages"]
    assert [c["id"] for c in msg["tool_calls"]] == ["one", "two"]
    assert [second["tool_call_id"], first["tool_call_id"]] == ["two", "one"]
    assert json.loads(first["content"])["output"] == "one"


def test_stream_combines_tool_argument_fragments_and_preserves_usage():
    def frame(delta, finish = None):
        payload = {"choices": [{"delta": delta, "finish_reason": finish}]}
        return b"data: " + json.dumps(payload).encode()

    events = list(
        _gemini_stream(
            [
                frame({"content": "Opening file."}),
                frame(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "f1",
                                "function": {"name": "read_file", "arguments": '{"Path":'},
                            }
                        ]
                    }
                ),
                frame(
                    {"tool_calls": [{"index": 0, "function": {"arguments": '"demo.txt"}'}}]},
                    "tool_calls",
                ),
                b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3,"total_tokens":15}}',
                b"data: [DONE]",
            ]
        )
    )
    assert events[0]["candidates"][0]["content"]["parts"] == [{"text": "Opening file."}]
    last = events[-1]["candidates"][0]
    assert last["finishReason"] == "STOP"
    call = last["content"]["parts"][0]["functionCall"]
    assert call == {"id": "f1", "name": "read_file", "args": {"Path": "demo.txt"}}
    assert events[-1]["usageMetadata"]["totalTokenCount"] == 15


def test_interrupted_stream_is_an_error():
    with pytest.raises(ValueError, match = "before inference finished"):
        list(_gemini_stream([b'data: {"choices":[{"delta":{"content":"partial"}}]}']))


def test_endpoint_requires_session_credential_and_closes_after_launch():
    with antigravity_endpoint("http://127.0.0.1:1", "backend-key", "local-model", {}) as (
        url,
        token,
    ):
        req = urllib.request.Request(
            url + "/v1beta/models/unsloth:streamGenerateContent", data = b"{}"
        )
        # Even the backend key is not a session credential.
        req.add_header("x-goog-api-key", "backend-key")
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(req, timeout = 3)
        assert error.value.code == 401
        req.add_header("x-goog-api-key", token)
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(req, timeout = 3)
        assert error.value.code == 400
    with pytest.raises(urllib.error.URLError):
        urllib.request.urlopen(req, timeout = 3)


def test_images_from_agy_tool_results_are_user_inputs_after_all_parallel_results():
    image = {"inlineData": {"mimeType": "image/png", "data": "YWJj"}}
    turns = [
        {
            "role": "model",
            "parts": [
                {"functionCall": {"id": key, "name": "read_file", "args": {"Path": key}}}
                for key in ("photo", "notes")
            ],
        },
        # Stock agy sends results in model turns, including embedded tool media.
        {
            "role": "model",
            "parts": [
                {
                    "functionResponse": {
                        "id": "photo",
                        "name": "read_file",
                        "response": {"output": "image/png"},
                        "parts": [image],
                    }
                }
            ],
        },
        {
            "role": "model",
            "parts": [
                {
                    "functionResponse": {
                        "id": "notes",
                        "name": "read_file",
                        "response": {"output": "notes"},
                    }
                }
            ],
        },
    ]
    messages = _chat_request({"contents": turns}, "local-model", {})["messages"]
    assert [m["role"] for m in messages] == ["assistant", "tool", "tool", "user"]
    assert messages[-1]["content"] == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,YWJj"}}
    ]


def test_forwarded_settings_preserve_prompt_and_use_last_file(tmp_path):
    path = tmp_path / "custom settings.json"
    path.write_text('{"notifications":false}', encoding = "utf-8-sig")
    settings, args = session_settings(
        [
            "--settings={}",
            "-p",
            "--settings",
            "--continue",
            "--settings",
            str(path),
            "--",
            "--settings=literal",
        ]
    )
    assert settings == {"notifications": False}
    assert args == ["-p", "--settings", "--continue", "--", "--settings=literal"]


@pytest.mark.parametrize(
    "args", [["--settings"], ["--settings={broken"], ["--settings", "missing-file"]]
)
def test_invalid_forwarded_settings_fail_before_launch(args, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match = "--settings"):
        session_settings(args)


def test_midstream_failure_does_not_emit_success_after_partial_text():
    frames = iter(
        _gemini_stream(
            [
                b'data: {"choices":[{"delta":{"content":"partial"}}]}',
                b'data: {"error":{"message":"inference interrupted"}}',
                b"data: [DONE]",
            ]
        )
    )
    assert next(frames)["candidates"][0]["content"]["parts"] == [{"text": "partial"}]
    with pytest.raises(ValueError, match = "inference interrupted"):
        next(frames)
