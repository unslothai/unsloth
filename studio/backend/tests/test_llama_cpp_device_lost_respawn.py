# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A llama-server whose GPU device was lost is restarted instead of failing forever (#11453).

Vulkan reports VK_ERROR_DEVICE_LOST as an exception llama-server catches, so the process
stays up and fails every later request, either as a 500 before the stream opens or as an
in-band SSE error chunk after the 200. The dead-process respawn never fired, and the chat
stayed stuck until a manual eject and reload.
"""

from __future__ import annotations

import http.server
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.llama_cpp import LlamaCppBackend
from core.inference.stream_errors import stream_error_from_chunk

# The reporter's exact llama-server error.
DEVICE_LOST_ERROR = {
    "error": {
        "code": 500,
        "message": "got exception: vk::Queue::submit: ErrorDeviceLost",
        "type": "server_error",
    }
}
DEVICE_LOST_BODY = json.dumps(DEVICE_LOST_ERROR)
OTHER_500_BODY = json.dumps(
    {"error": {"code": 500, "message": "got exception: bad request", "type": "server_error"}}
)
_PROGRESS = (
    'data: {"choices":[{"index":0,"delta":{}}],'
    '"prompt_progress":{"total":8,"cache":0,"processed":0,"time_ms":0}}\n\n'
)
# What llama-server sends when the loss happens after the 200 (Studio asks for
# return_progress, so the headers and a progress event go out before the decode).
SSE_DEVICE_LOST = _PROGRESS + f"data: {DEVICE_LOST_BODY}\n\n"
SSE_KV_STARVATION = (
    _PROGRESS + 'data: {"error":{"code":500,"message":"Context size has been exceeded.",'
    '"type":"server_error"}}\n\n'
)
SSE_PARTIAL_THEN_DEVICE_LOST = (
    'data: {"choices":[{"index":0,"delta":{"content":"Partial"}}]}\n\n'
    f"data: {DEVICE_LOST_BODY}\n\n"
)
SSE_OK = 'data: {"choices":[{"index":0,"delta":{"content":"Recovered."}}]}\n\n' "data: [DONE]\n\n"


def _serve(status: int, body: str, content_type: str):
    hits = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            hits.append(self.path)
            data = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    return server, hits


def _child():
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])


def _close_port_when_exited(child, server):
    # A real llama-server's port closes with it, so the next request gets ConnectError.
    closed = threading.Event()

    def watch():
        child.wait()
        server.shutdown()
        server.server_close()
        closed.set()

    threading.Thread(target = watch, daemon = True).start()
    return closed


@pytest.fixture
def harness():
    servers, children = [], []

    def make(
        broken_body: str,
        *,
        status: int = 500,
        reload_ok: bool = True,
    ):
        content_type = "application/json" if status != 200 else "text/event-stream"
        broken, broken_hits = _serve(status, broken_body, content_type)
        healthy, healthy_hits = _serve(200, SSE_OK, "text/event-stream")
        servers.extend([broken, healthy])
        backend = LlamaCppBackend()
        first = _child()
        children.append(first)
        backend.port_closed = _close_port_when_exited(first, broken)
        backend._process = first
        backend._port = broken.server_address[1]
        backend._healthy = True
        backend._model_identifier = "unsloth/test-GGUF"
        backend._last_load_intent = object()
        loads = []

        def fake_load_model(intent):
            loads.append(intent)
            if not reload_ok:
                return False
            replacement = _child()
            children.append(replacement)
            backend._process = replacement
            backend._port = healthy.server_address[1]
            backend._healthy = True
            return True

        backend.load_model = fake_load_model
        return backend, first, loads, broken_hits, healthy_hits

    yield make
    for child in children:
        if child.poll() is None:
            child.kill()
            child.wait()
    for server in servers:
        if server.socket.fileno() != -1:
            server.shutdown()
        server.server_close()


def _chat(backend):
    return list(
        backend.generate_chat_completion(
            [{"role": "user", "content": "Write space invaders in HTML"}],
            max_tokens = 16,
        )
    )


def _text(events):
    return "".join(str(e) for e in events)


@pytest.mark.parametrize(
    "status, body", [(500, DEVICE_LOST_BODY), (200, SSE_DEVICE_LOST)], ids = ["http500", "sse"]
)
def test_device_lost_restarts_the_server_and_retries_the_chat(harness, status, body):
    backend, first, loads, broken_hits, healthy_hits = harness(body, status = status)

    events = _chat(backend)

    assert len(loads) == 1
    assert first.poll() is not None
    assert backend._process is not first and backend._process.poll() is None
    assert len(broken_hits) == 1 and len(healthy_hits) == 1
    assert "Recovered." in _text(events)


def test_device_lost_after_output_fails_once_then_the_next_turn_recovers(harness):
    backend, first, loads, broken_hits, healthy_hits = harness(
        SSE_PARTIAL_THEN_DEVICE_LOST, status = 200
    )

    with pytest.raises(RuntimeError) as info:
        _chat(backend)
    # Output was already streamed, so this turn is not replayed; the server is retired.
    assert "ErrorDeviceLost" in str(info.value)
    assert first.poll() is not None and loads == []
    assert backend.port_closed.wait(5)

    events = _chat(backend)

    assert len(loads) == 1 and "Recovered." in _text(events)
    assert len(broken_hits) == 1 and len(healthy_hits) == 1


def test_device_lost_restarts_the_server_on_the_tool_stream_path(harness):
    backend, first, loads, broken_hits, healthy_hits = harness(DEVICE_LOST_BODY)
    respawned = []
    payload = {"messages": [{"role": "user", "content": "hi"}], "stream": True}

    with backend._open_chat_stream_with_respawn_retry(
        payload, None, on_respawn = lambda: respawned.append(True)
    ) as (response, _deadline):
        assert response.status_code == 200

    assert len(loads) == 1 and respawned == [True]
    assert first.poll() is not None
    assert len(broken_hits) == 1 and len(healthy_hits) == 1


def test_device_lost_mid_stream_on_the_tool_path_retires_the_server(harness):
    backend, first, loads, _broken_hits, _healthy_hits = harness(SSE_DEVICE_LOST, status = 200)
    payload = {"messages": [{"role": "user", "content": "hi"}], "stream": True}

    with pytest.raises(RuntimeError, match = "ErrorDeviceLost"):
        with backend._open_chat_stream_with_respawn_retry(payload, None):
            # What the tool loops do with an in-band error chunk.
            raise stream_error_from_chunk(DEVICE_LOST_ERROR)

    assert first.poll() is not None and loads == []
    assert backend.port_closed.wait(5)
    with backend._open_chat_stream_with_respawn_retry(payload, None) as (response, _deadline):
        assert response.status_code == 200
    assert len(loads) == 1


@pytest.mark.parametrize(
    "status, body", [(500, OTHER_500_BODY), (200, SSE_KV_STARVATION)], ids = ["http500", "sse"]
)
def test_other_server_errors_do_not_restart_the_server(harness, status, body):
    backend, first, loads, broken_hits, _healthy_hits = harness(body, status = status)

    with pytest.raises(RuntimeError) as info:
        _chat(backend)

    if status == 500:
        assert str(info.value) == f"llama-server returned 500: {OTHER_500_BODY}"
    assert loads == []
    assert first.poll() is None and backend._process is first
    assert len(broken_hits) == 1


def test_a_failed_restart_reraises_the_original_error(harness):
    backend, _first, loads, broken_hits, healthy_hits = harness(DEVICE_LOST_BODY, reload_ok = False)

    with pytest.raises(RuntimeError) as info:
        _chat(backend)

    assert str(info.value) == f"llama-server returned 500: {DEVICE_LOST_BODY}"
    assert len(loads) == 1
    assert len(broken_hits) == 1 and healthy_hits == []


def test_a_late_caller_does_not_kill_the_replacement(harness):
    backend, first, loads, _broken_hits, _healthy_hits = harness(DEVICE_LOST_BODY)
    _chat(backend)
    replacement = backend._process
    assert replacement is not first and len(loads) == 1

    # A second request that also failed on the lost server reports it after the reload.
    backend._retire_device_lost_server(first)

    assert backend._process is replacement and replacement.poll() is None
    assert len(loads) == 1


def _logging_child(line: str):
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"import sys, time; print({line!r}, flush = True); time.sleep(120)",
        ],
        stdout = subprocess.PIPE,
        text = True,
    )


@pytest.mark.parametrize(
    "line, retired",
    [
        (
            "0.01.000.000 E slot iterate: id  0 | got exception: vk::Queue::submit: ErrorDeviceLost",
            True,
        ),
        (
            "0.01.000.000 I slot launch_slot_: id  0 | prompt: what does ErrorDeviceLost mean?",
            False,
        ),
        ("what does vk::Queue::submit: ErrorDeviceLost mean?", False),
    ],
    ids = ["server_error", "info_line", "unprefixed_request_dump"],
)
def test_a_device_loss_in_the_server_log_retires_the_server(monkeypatch, line, retired):
    # The passthrough endpoints forward llama-server's error untouched, so the
    # server's own error log is what retires the server for them.
    import core.inference.llama_cpp as llama_cpp

    monkeypatch.setattr(llama_cpp, "_DEVICE_LOST_LOG_GRACE_S", 0.0)
    child = _logging_child(line)
    try:
        backend = LlamaCppBackend()
        backend._process = child
        backend._model_identifier = "unsloth/test-GGUF"
        drain = threading.Thread(target = backend._drain_stdout, daemon = True)
        drain.start()
        try:
            child.wait(timeout = 5)
        except subprocess.TimeoutExpired:
            pass
        assert (child.poll() is not None) is retired
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()


def test_retiring_disarms_the_mtp_crash_watchdog(harness):
    backend, first, _loads, _broken_hits, _healthy_hits = harness(DEVICE_LOST_BODY)
    stop = threading.Event()
    backend._mtp_watchdog_stop = stop

    backend._retire_device_lost_server(first)

    assert stop.is_set() and first.poll() is not None
