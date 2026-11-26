# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The vLLM / SGLang readiness poll scans port ownership only once something accepts on the port."""

import os
import socket
import sys
from contextlib import nullcontext

import pytest

from core.inference import llama_cpp, managed_engine
from core.inference.llama_cpp import LlamaCppBackend


class _Response:
    def __init__(self, status_code):
        self.status_code = status_code


def _engine(
    monkeypatch,
    tmp_path,
    *,
    listen,
    owned,
    status = 200,
    limit = 1.5,
):
    """Engine whose child never binds. ``owned``: None = real ownership check, else successive answers."""
    events, errors, holder = [], [], []
    # vLLM / SGLang write a per-attempt log under the Studio home; keep it out of the real one.
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setattr(managed_engine, "engine_lease", lambda _: nullcontext())
    monkeypatch.setattr(
        managed_engine, "installed", lambda _: {"host": "wsl", "path": str(tmp_path)}
    )
    monkeypatch.setattr(managed_engine, "STARTUP_LIMIT_S", limit)
    engine = managed_engine.ManagedEngine("vllm")

    def launch(*args):
        if listen:
            server = socket.socket()
            server.bind(("127.0.0.1", args[-1]))
            server.listen()
            holder.append(server)
        # Ends when stop() closes its stdin, like the WSL guest runner. The real
        # _wsl_command always hands back the wsl.exe environment as a dict, which
        # _start_once extends in place (WSL_UTF8 when it captures engine logs).
        return [sys.executable, "-c", "import sys; sys.stdin.read()"], dict(os.environ)

    monkeypatch.setattr(engine, "_wsl_command", launch)

    real = LlamaCppBackend._health_listener_is_ours
    answers = iter(owned or ())

    def ours(
        process,
        port,
        bind_host = None,
    ):
        answer = real(process, port, bind_host) if owned is None else next(answers)
        events.append(("own", answer))
        return answer

    monkeypatch.setattr(LlamaCppBackend, "_health_listener_is_ours", staticmethod(ours))

    class Client:
        def __init__(self, **_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def get(
            self,
            url,
            headers = None,
        ):
            events.append(("get", (headers or {}).get("Authorization")))
            return _Response(status)

    monkeypatch.setattr(managed_engine.httpx, "Client", Client)
    error = llama_cpp.logger.error
    monkeypatch.setattr(
        llama_cpp.logger,
        "error",
        lambda message, *a, **k: (errors.append(message % a), error(message, *a, **k)),
    )
    return engine, events, errors, holder


def _close(holder):
    for server in holder:
        server.close()


def test_no_ownership_scan_or_error_while_nothing_listens(monkeypatch, tmp_path):
    engine, events, errors, holder = _engine(monkeypatch, tmp_path, listen = False, owned = None)
    with pytest.raises(RuntimeError, match = "Engine startup timed out"):
        engine.start("model", 2048, [0], {})
    assert events == []
    assert errors == []
    assert engine.process is None


def test_ownership_runs_every_poll_before_the_keyed_probe(monkeypatch, tmp_path):
    engine, events, _, holder = _engine(monkeypatch, tmp_path, listen = True, owned = [False, True])
    try:
        engine.start("model", 2048, [0], {})
        assert engine.phase == "ready"
    finally:
        engine.stop()
        _close(holder)
    assert [kind for kind, _ in events] == ["own", "own", "get"]
    assert events[1] == ("own", True)


def test_ownership_check_runs_again_before_each_repeated_probe(monkeypatch, tmp_path):
    engine, events, _, holder = _engine(
        monkeypatch, tmp_path, listen = True, owned = [True] * 100, status = 503, limit = 1.0
    )
    try:
        with pytest.raises(RuntimeError, match = "Engine startup timed out"):
            engine.start("model", 2048, [0], {})
    finally:
        _close(holder)
    kinds = [kind for kind, _ in events]
    assert kinds.count("get") >= 2
    assert kinds == ["own", "get"] * (len(kinds) // 2)


@pytest.mark.skipif(sys.platform == "win32", reason = "real owner lookup differs on Windows")
def test_a_listener_not_owned_by_our_child_is_still_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(llama_cpp, "_socket_owner_is_always_named", lambda: False)
    engine, events, errors, holder = _engine(
        monkeypatch, tmp_path, listen = True, owned = None, limit = 1.0
    )
    try:
        with pytest.raises(RuntimeError, match = "Engine startup timed out"):
            engine.start("model", 2048, [0], {})
    finally:
        _close(holder)
    assert events and all(event == ("own", False) for event in events)
    assert errors and all("refusing to use it" in message for message in errors)


@pytest.mark.parametrize("windows,ready", [(True, True), (False, False)])
def test_windows_unnamed_owner_rule_is_unchanged(monkeypatch, tmp_path, windows, ready):
    """A listener whose owner psutil cannot name: Windows still probes it, Linux/macOS still refuse."""
    import psutil

    monkeypatch.setattr(llama_cpp, "_socket_owner_is_always_named", lambda: windows)
    monkeypatch.setattr(psutil, "net_connections", lambda kind = "inet": [])
    engine, events, errors, holder = _engine(
        monkeypatch, tmp_path, listen = True, owned = None, limit = 1.0
    )
    try:
        if ready:
            engine.start("model", 2048, [0], {})
            assert engine.phase == "ready"
        else:
            with pytest.raises(RuntimeError, match = "Engine startup timed out"):
                engine.start("model", 2048, [0], {})
    finally:
        engine.stop()
        _close(holder)
    kinds = [kind for kind, _ in events]
    assert kinds[0] == "own"
    assert ("get" in kinds) is ready
    assert bool(errors) is not ready


def test_port_accepts_refuses_a_bound_socket_that_is_not_listening_yet():
    # vLLM's create_server_socket binds before the model load and listens only after it.
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
        assert not managed_engine._port_accepts(f"http://127.0.0.1:{port}")


def test_port_accepts_sends_nothing_and_follows_the_url_host():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        assert managed_engine._port_accepts(f"http://127.0.0.1:{port}")
        peer, _ = server.accept()
        with peer:
            peer.settimeout(2)
            assert peer.recv(1) == b""
    assert not managed_engine._port_accepts(f"http://127.0.0.1:{port}")


def test_port_accepts_ipv6_loopback():
    if not socket.has_ipv6:
        pytest.skip(reason = "host Python built without IPv6 support")
    with socket.socket(socket.AF_INET6) as server:
        try:
            server.bind(("::1", 0))
        except OSError:
            pytest.skip(
                reason = "host has no ::1 loopback (IPv6 disabled in the kernel or container)"
            )
        server.listen()
        port = server.getsockname()[1]
        assert managed_engine._port_accepts(f"http://[::1]:{port}")
        with socket.socket() as other:
            try:
                other.bind(("127.0.0.1", port))
            except OSError:
                return
        assert not managed_engine._port_accepts(f"http://127.0.0.1:{port}")
