# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""audiocpp_server is started again on a new port when another process binds its port first.

The port is reserved, released, then bound by the server, so a parallel test or app can take it in
between. Seen in CI: the fake server died with "OSError: [Errno 98] Address already in use" and the
transcription test failed. Here the port really is taken, by a listener bound as the reservation closes.
"""

import socket
import sys
import textwrap
import types
from pathlib import Path

import pytest

from core.inference import audio_cpp_server as srv
from core.inference import lemonade_server

FAKE_SERVER = textwrap.dedent(
    r"""
    import json, os, socketserver, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer

    cfg = json.load(open(sys.argv[sys.argv.index("--config") + 1], encoding="utf-8"))
    model_id = cfg["models"][0]["id"]
    with open(os.environ["FAKE_STARTS"], "a", encoding="utf-8") as log:
        log.write(f"{cfg['port']}\n")
    if os.environ.get("FAKE_BROKEN"):
        sys.exit("error: tensor 'blk.0.attn_q.weight' has the wrong shape")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            payload = {"status": "ok"} if self.path == "/health" else {"data": [{"id": model_id}]}
            body = json.dumps(payload).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)

    class S(HTTPServer):
        allow_reuse_address = False  # SO_REUSEADDR on Windows would let it share a taken port
        def server_bind(self):  # skips the getfqdn reverse lookup, slow on macOS runners
            socketserver.TCPServer.server_bind(self)
            self.server_name, self.server_port = "127.0.0.1", self.server_address[1]

    S(("127.0.0.1", cfg["port"]), H).serve_forever()
    """
)

MODEL = types.SimpleNamespace(
    id = "fake/model", family = "qwen3_asr", server_task = "asr", model_options = {}
)


class _Thief:
    """A reservation whose close() hands the port straight to another listener."""

    def __init__(self, reservation, port, taken):
        self._reservation, self._port, self._taken = reservation, port, taken

    def close(self):
        self._reservation.close()
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        blocker.bind(("127.0.0.1", self._port))
        blocker.listen(1)
        self._taken.append(blocker)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER, encoding = "utf-8")
    binary = tmp_path / srv.BINARY_NAME
    binary.write_bytes(b"")
    starts = tmp_path / "starts.txt"
    starts.touch()
    real_popen = srv.subprocess.Popen

    def launch(command, *args, **kwargs):
        if command and command[0] == str(binary):
            command = [sys.executable, str(script), *command[1:]]
        return real_popen(command, *args, **kwargs)

    monkeypatch.setattr(srv.subprocess, "Popen", launch)
    monkeypatch.setattr(srv, "ensure_binary", lambda: str(binary))
    monkeypatch.setattr(srv, "model_runtime_problem", lambda model, binary = None: None)
    monkeypatch.setattr(srv, "select_backend", lambda binary, force_cpu: "cpu")
    monkeypatch.setenv("FAKE_STARTS", str(starts))
    monkeypatch.delenv("FAKE_BROKEN", raising = False)

    taken: list[socket.socket] = []
    state = types.SimpleNamespace(steal = 0, taken = taken, starts = starts, dirs = [])
    real_reserve, real_mkdtemp = srv._reserve_free_port, srv.tempfile.mkdtemp

    def reserve():
        reservation, port = real_reserve()
        if state.steal:
            state.steal -= 1
            return _Thief(reservation, port, taken), port
        return reservation, port

    def mkdtemp(*args, **kwargs):
        state.dirs.append(Path(real_mkdtemp(*args, **kwargs)))
        return str(state.dirs[-1])

    monkeypatch.setattr(srv, "_reserve_free_port", reserve)
    monkeypatch.setattr(srv.tempfile, "mkdtemp", mkdtemp)
    yield state
    for blocker in taken:
        blocker.close()


def _ports(state):
    return [int(line) for line in state.starts.read_text("utf-8").split()]


def test_a_port_taken_before_the_server_binds_it_is_retried_on_a_new_one(fake):
    fake.steal = 1
    server = srv.AudioCppServer.start(MODEL, "/models/fake.gguf")
    try:
        assert server.alive()
        first, second = _ports(fake)
        assert first == fake.taken[0].getsockname()[1]
        assert server.port == second != first
        assert not fake.dirs[0].exists(), "the failed attempt left its config dir behind"
    finally:
        server.stop()


def test_a_port_lost_on_every_attempt_fails_after_three_with_the_runtime_output(fake):
    fake.steal = 99
    with pytest.raises(srv.AudioCppUnavailableError, match = "exited before becoming ready") as err:
        srv.AudioCppServer.start(MODEL, "/models/fake.gguf")
    assert len(_ports(fake)) == srv._START_ATTEMPTS == 3
    assert srv._PORT_TAKEN.search(str(err.value)), str(err.value)
    assert not any(d.exists() for d in fake.dirs)


def test_any_other_early_exit_fails_at_once(fake, monkeypatch):
    monkeypatch.setenv("FAKE_BROKEN", "1")
    with pytest.raises(srv.AudioCppUnavailableError, match = "wrong shape"):
        srv.AudioCppServer.start(MODEL, "/models/fake.gguf")
    assert len(_ports(fake)) == 1


def test_a_cancelled_start_is_not_retried(fake):
    fake.steal = 99
    cancel = srv.threading.Event()
    real_wait = srv.AudioCppServer._wait_until_ready

    def wait_then_cancel(self, cancel_event):
        try:
            real_wait(self, cancel_event)
        finally:
            cancel.set()

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(srv.AudioCppServer, "_wait_until_ready", wait_then_cancel)
        with pytest.raises(srv.AudioCppUnavailableError):
            srv.AudioCppServer.start(MODEL, "/models/fake.gguf", cancel_event = cancel)
    assert len(_ports(fake)) == 1


@pytest.mark.parametrize(
    "line, taken",
    [
        ("OSError: [Errno 98] Address already in use", True),
        ("bind: EADDRINUSE", True),
        ("OSError: [Errno 48] Address already in use", True),
        (
            "OSError: [WinError 10048] Only one usage of each socket address is normally permitted",
            True,
        ),
        ("error: failed to load model; listening on port 10048", False),
        ("error: tensor has the wrong shape", False),
    ],
)
def test_the_collision_pattern_matches_lemonade_and_only_port_collisions(line, taken):
    assert srv._PORT_TAKEN.pattern == lemonade_server._PORT_TAKEN.pattern
    assert srv._PORT_TAKEN.flags == lemonade_server._PORT_TAKEN.flags
    assert bool(srv._PORT_TAKEN.search(line)) is taken
