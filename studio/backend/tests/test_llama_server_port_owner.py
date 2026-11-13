# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A /health 200 only counts when our llama-server child holds the port.

_find_free_port releases its socket before llama-server binds, so another local user can take
the port and answer /health; chat traffic and the per-launch bearer would then go to them.
"""

from __future__ import annotations

import http.server
import subprocess
import sys
import threading
import types as _types
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
sys.modules.setdefault("structlog", _types.ModuleType("structlog"))

psutil = pytest.importorskip("psutil")

import core.inference.llama_cpp as llama_cpp  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

_CHILD_SERVER = r"""
import http.server, sys
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"ok"}')
    def log_message(self, *a): pass
s = http.server.HTTPServer(("127.0.0.1", 0), H)
print(s.server_address[1], flush=True)
s.serve_forever()
"""


class _Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def log_message(self, *a):
        pass


def _urllib_get(
    url,
    timeout = 2.0,
    **_kw,
):
    try:
        with urllib.request.urlopen(url, timeout = timeout) as r:
            return SimpleNamespace(status_code = r.status)
    except urllib.error.HTTPError as e:
        return SimpleNamespace(status_code = e.code)
    except OSError as e:
        raise llama_cpp.httpx.ConnectError(str(e))


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.setattr(llama_cpp.httpx, "get", _urllib_get)


@pytest.fixture
def foreign_listener():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Health)
    t = threading.Thread(target = srv.serve_forever, daemon = True)
    t.start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def idle_child():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    yield p
    p.kill()
    p.wait()


def _backend(port: int, process) -> LlamaCppBackend:
    b = LlamaCppBackend.__new__(LlamaCppBackend)
    b._port = port
    b._stdout_thread = None
    b._stdout_lines = []
    b._process = process
    return b


def test_foreign_listener_on_our_port_is_not_healthy(probe, foreign_listener, idle_child):
    b = _backend(foreign_listener, idle_child)
    assert b._wait_for_health(timeout = 3.0, interval = 0.05) is False


def test_hidden_owner_with_our_child_not_listening_is_not_healthy(
    probe, foreign_listener, idle_child, monkeypatch
):
    # The Linux/macOS case: another user's socket shows no pid.
    monkeypatch.setattr(llama_cpp, "_socket_owner_is_always_named", lambda: False)
    real = psutil.net_connections

    def hidden(kind = "inet"):
        return [c._replace(pid = None) for c in real(kind = kind)]

    monkeypatch.setattr(psutil, "net_connections", hidden)
    b = _backend(foreign_listener, idle_child)
    assert b._wait_for_health(timeout = 3.0, interval = 0.05) is False


def test_our_own_child_listening_is_healthy(probe):
    p = subprocess.Popen([sys.executable, "-c", _CHILD_SERVER], stdout = subprocess.PIPE, text = True)
    try:
        port = int(p.stdout.readline())
        b = _backend(port, p)
        assert b._wait_for_health(timeout = 10.0, interval = 0.05) is True
    finally:
        p.kill()
        p.wait()


def test_unknowable_ownership_keeps_todays_behaviour(
    probe, foreign_listener, idle_child, monkeypatch
):
    def denied(*_a, **_kw):
        raise psutil.AccessDenied()

    monkeypatch.setattr(psutil.Process, "net_connections", denied, raising = False)
    monkeypatch.setattr(psutil.Process, "connections", denied, raising = False)
    monkeypatch.setattr(psutil, "net_connections", denied)
    b = _backend(foreign_listener, idle_child)
    assert b._wait_for_health(timeout = 3.0, interval = 0.05) is True


def test_a_descendant_holding_the_port_is_ours(probe):
    # Wrapped launches (e.g. the diffusion shim) listen from a grandchild, not the direct child.
    launcher = (
        "import subprocess, sys\n"
        "p = subprocess.Popen([sys.executable, '-c', sys.argv[1]], stdout = sys.stdout)\n"
        "p.wait()\n"
    )
    p = subprocess.Popen(
        [sys.executable, "-c", launcher, _CHILD_SERVER], stdout = subprocess.PIPE, text = True
    )
    try:
        port = int(p.stdout.readline())
        b = _backend(port, p)
        assert b._wait_for_health(timeout = 10.0, interval = 0.05) is True
    finally:
        for child in psutil.Process(p.pid).children(recursive = True):
            child.kill()
        p.kill()
        p.wait()


def test_an_exited_child_is_never_the_server_that_answered(foreign_listener):
    # A child that could not bind (the port was taken) exits; the 200 then came from the squatter.
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    b = _backend(foreign_listener, p)
    assert b._health_listener_is_ours(p, b._port) is False


def test_a_foreign_listener_listed_before_ours_does_not_refuse_us(monkeypatch):
    # Own tree unreadable, so the system-wide scan decides; its order must not matter.
    p = subprocess.Popen([sys.executable, "-c", _CHILD_SERVER], stdout = subprocess.PIPE, text = True)
    try:
        port = int(p.stdout.readline())

        def denied(*_a, **_kw):
            raise psutil.AccessDenied()

        def conn(pid, ip):
            return SimpleNamespace(
                laddr = SimpleNamespace(ip = ip, port = port), status = psutil.CONN_LISTEN, pid = pid
            )

        monkeypatch.setattr(psutil.Process, "net_connections", denied, raising = False)
        monkeypatch.setattr(psutil.Process, "connections", denied, raising = False)
        monkeypatch.setattr(
            psutil, "net_connections", lambda kind = "inet": [conn(1, "::"), conn(p.pid, "127.0.0.1")]
        )
        b = _backend(port, p)
        assert b._health_listener_is_ours(p, b._port) is True
    finally:
        p.kill()
        p.wait()


# ── Windows: psutil can misdescribe our own live listener (#15022) ──


def _row(
    port,
    *,
    pid,
    status,
    ip = "127.0.0.1",
    raddr = (),
):
    return SimpleNamespace(
        laddr = SimpleNamespace(ip = ip, port = port),
        raddr = raddr,
        status = status,
        pid = pid,
        type = llama_cpp.socket.SOCK_STREAM,
    )


def _tables(
    monkeypatch,
    *,
    own,
    everyone,
    windows = True,
):
    monkeypatch.setattr(llama_cpp, "_socket_owner_is_always_named", lambda: windows)
    monkeypatch.setattr(
        psutil.Process, "net_connections", lambda self, kind = "inet": own, raising = False
    )
    monkeypatch.setattr(psutil, "net_connections", lambda kind = "inet": everyone)


@pytest.mark.parametrize("windows", [True, False])
def test_our_child_reported_with_status_none_is_ours_on_windows(idle_child, monkeypatch, windows):
    port = 50855
    none = _row(port, pid = idle_child.pid, status = psutil.CONN_NONE)
    _tables(monkeypatch, own = [none], everyone = [none], windows = windows)

    assert LlamaCppBackend._health_listener_is_ours(idle_child, port) is windows


@pytest.mark.parametrize("windows", [True, False])
def test_a_listener_owned_by_pid_0_is_unknown_on_windows(idle_child, monkeypatch, windows):
    port = 54185
    _tables(
        monkeypatch,
        own = [],
        everyone = [_row(port, pid = 0, status = psutil.CONN_LISTEN)],
        windows = windows,
    )

    assert LlamaCppBackend._health_listener_is_ours(idle_child, port) is windows


def test_a_connected_socket_on_our_port_is_not_a_listener(idle_child, monkeypatch):
    port = 57043
    peer = SimpleNamespace(ip = "127.0.0.1", port = 60000)
    accepted = _row(port, pid = idle_child.pid, status = psutil.CONN_NONE, raddr = peer)
    _tables(monkeypatch, own = [accepted], everyone = [_row(port, pid = 4242, status = "LISTEN")])

    assert LlamaCppBackend._health_listener_is_ours(idle_child, port) is False


def test_a_named_foreign_listener_still_refuses_on_windows(idle_child, monkeypatch):
    port = 52792
    _tables(
        monkeypatch,
        own = [],
        everyone = [
            _row(port, pid = 0, status = psutil.CONN_LISTEN),
            _row(port, pid = 4242, status = psutil.CONN_LISTEN),
        ],
    )

    assert LlamaCppBackend._health_listener_is_ours(idle_child, port) is False


def test_a_refused_listener_is_not_reported_as_a_bad_gguf(probe, foreign_listener, idle_child):
    b = _backend(foreign_listener, idle_child)
    assert b._wait_for_health(timeout = 3.0, interval = 0.05) is False

    message = LlamaCppBackend._classify_llama_start_failure(
        "\n".join(b._stdout_lines), "model.gguf", None
    )

    assert "could not confirm that the listener" in message
    assert "enough memory" not in message
