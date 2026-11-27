# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import http.client
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from unsloth_cli.antigravity import (
    _chat_request,
    _gemini_stream,
    antigravity_endpoint,
    session_settings,
)


def test_endpoint_sends_the_cli_user_agent_to_inference():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            received.append(self.headers["User-Agent"])
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'data: {"choices":[{"finish_reason":"stop"}]}\n\n')

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as backend:
        thread = threading.Thread(target = backend.serve_forever)
        thread.start()
        try:
            base = f"http://127.0.0.1:{backend.server_port}"
            with antigravity_endpoint(base, "backend-key", "test-model", {}) as (url, token):
                req = urllib.request.Request(
                    url + "/v1beta/models/unsloth:streamGenerateContent",
                    data = b'{"contents":[{"parts":[{"text":"hello"}]}]}',
                    headers = {"x-goog-api-key": token},
                )
                with urllib.request.urlopen(req, timeout = 3) as response:
                    assert b'"finishReason": "STOP"' in response.read()
        finally:
            backend.shutdown()
            thread.join()

    assert received == ["unsloth-cli"]


def test_endpoint_rejects_another_model_instead_of_using_the_resident_model():
    with antigravity_endpoint("http://127.0.0.1:1", "backend-key", "local-model", {}) as (
        url,
        token,
    ):
        req = urllib.request.Request(
            url + "/v1beta/models/gemini-3.8-flash:streamGenerateContent",
            data = b'{"contents":[{"parts":[{"text":"hello"}]}]}',
            headers = {"x-goog-api-key": token},
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(req, timeout = 3)
        assert error.value.code == 404


def test_endpoint_reports_truncated_http_chunks_after_partial_text():
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            text = b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            self.wfile.write(f"{len(text):x}\r\n".encode() + text + b"\r\n")
            self.wfile.write(b"28\r\ndata: {")
            self.close_connection = True

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as backend:
        thread = threading.Thread(target = backend.serve_forever)
        thread.start()
        try:
            base = f"http://127.0.0.1:{backend.server_port}"
            with antigravity_endpoint(base, "backend-key", "test-model", {}) as (url, token):
                req = urllib.request.Request(
                    url + "/v1beta/models/unsloth:streamGenerateContent",
                    data = b'{"contents":[{"parts":[{"text":"hello"}]}]}',
                    headers = {"x-goog-api-key": token},
                )
                with urllib.request.urlopen(req, timeout = 3) as response:
                    frames = [
                        json.loads(line[6:]) for line in response if line.startswith(b"data: ")
                    ]
        finally:
            backend.shutdown()
            thread.join()

    assert frames[0]["candidates"][0]["content"]["parts"] == [{"text": "partial"}]
    assert frames[1]["error"]["code"] == 502


@pytest.mark.parametrize("stream_started", [False, True], ids = ["prefill", "buffered-tool-call"])
def test_endpoint_closes_upstream_when_client_cancels(stream_started):
    upstream_ready = threading.Event()
    upstream_cancelled = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            if stream_started:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(
                    b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                    b'"function":{"name":"read_file","arguments":"{"}}]}}]}\n\n'
                )
                self.wfile.flush()
            upstream_ready.set()
            self.connection.settimeout(3)
            try:
                if not self.connection.recv(1):
                    upstream_cancelled.set()
            except ConnectionResetError:
                upstream_cancelled.set()
            except TimeoutError:
                pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as backend:
        thread = threading.Thread(target = backend.serve_forever)
        thread.start()
        try:
            base = f"http://127.0.0.1:{backend.server_port}"
            with antigravity_endpoint(base, "backend-key", "test-model", {}) as (url, token):
                client = http.client.HTTPConnection(url.removeprefix("http://"), timeout = 3)
                try:
                    client.request(
                        "POST",
                        "/v1beta/models/unsloth:streamGenerateContent",
                        body = b'{"contents":[{"parts":[{"text":"hello"}]}]}',
                        headers = {"x-goog-api-key": token},
                    )
                    assert upstream_ready.wait(3), "the backend never received the request"
                finally:
                    client.close()
                assert upstream_cancelled.wait(2), "the backend connection stayed open after cancel"
        finally:
            backend.shutdown()
            thread.join()


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


@pytest.mark.parametrize("supports_tools", [False, True])
def test_chat_request_only_sends_tools_to_capable_models(supports_tools):
    body = {
        "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
        "tools": [{"functionDeclarations": [{"name": "read_file"}]}],
    }
    request = _chat_request(body, "local-model", {}, supports_tools = supports_tools)
    assert ("tools" in request) is supports_tools
    if supports_tools:
        assert request["tools"][0]["function"]["name"] == "read_file"


def test_chat_request_maps_thinking_level_and_honors_explicit_engine_override():
    body = {
        "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
        "generationConfig": {"thinkingConfig": {"thinkingLevel": "LOW"}},
    }
    assert _chat_request(body, "local-model", {})["reasoning_effort"] == "low"
    assert (
        _chat_request(body, "local-model", {"reasoning_effort": "high"})["reasoning_effort"]
        == "high"
    )
    assert "reasoning_effort" not in _chat_request(body, "local-model", {"enable_thinking": False})


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
                b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3,'
                b'"total_tokens":15,"prompt_tokens_details":{"cached_tokens":8}}}',
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
    assert events[-1]["usageMetadata"]["cachedContentTokenCount"] == 8


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
