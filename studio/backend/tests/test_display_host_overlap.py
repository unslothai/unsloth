# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The wildcard-bind display host resolves in a thread that overlaps the app import, with the same answer."""

import ast
import socket
import threading
import time
import urllib.request
from pathlib import Path

import pytest

import run

_RUN = Path(run.__file__).resolve()


def _run_server_src() -> str:
    tree = ast.parse(_RUN.read_text(encoding = "utf-8"))
    fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run_server"
    )
    return ast.unparse(fn)


def test_resolution_starts_before_the_app_import_and_is_joined_before_the_banner():
    src = _run_server_src()
    started = src.index("display_host_result = _start_display_host_resolution(host)")
    joined = src.index("display_host = display_host_result()")
    assert started < src.index("from main import app") < joined
    assert joined < src.index("_install_uvicorn_startup_log_rewrite(host)")
    assert joined < src.index("_emit_startup_output(")
    # No second, serial resolution left behind.
    assert "_display_host_for_bind(" not in src


class _UDP:
    def connect(self, addr):
        pass

    def getsockname(self):
        return ("10.1.2.3", 0)

    def close(self):
        pass


@pytest.fixture
def slow_public_ip(monkeypatch):
    """GCE metadata fails, the public service answers only once released."""
    release = threading.Event()
    seen = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"203.0.113.7\n"

    def _urlopen(req, timeout = None):
        url = req if isinstance(req, str) else req.full_url
        seen.append(url)
        if "metadata.google.internal" in url:
            raise OSError("not on GCE")
        assert release.wait(10)
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(socket, "socket", lambda *a, **k: _UDP())
    monkeypatch.delenv(run.DISABLE_PUBLIC_CHECK_ENV, raising = False)
    return release, seen


def test_wildcard_resolution_runs_in_the_background_with_the_same_answer(slow_public_ip):
    release, seen = slow_public_ip
    result = run._start_display_host_resolution("0.0.0.0")
    # start returned while the thread sits blocked in the public lookup: the caller imports meanwhile.
    deadline = time.monotonic() + 10
    while "https://ifconfig.me" not in seen and time.monotonic() < deadline:
        time.sleep(0.01)
    assert seen[:2] == [
        "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip",
        "https://ifconfig.me",
    ]
    release.set()
    background = result()
    assert background == "203.0.113.7"
    assert background == run._display_host_for_bind("0.0.0.0")


@pytest.mark.parametrize("disable_public", [False, True])
def test_failing_lookups_give_the_same_fallback_as_a_direct_call(monkeypatch, disable_public):
    def _urlopen(*_a, **_k):
        raise TimeoutError("timed out")

    def _no_route(*_a, **_k):
        raise OSError("network unreachable")

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(socket, "socket", _no_route)
    if disable_public:
        monkeypatch.setenv(run.DISABLE_PUBLIC_CHECK_ENV, "1")
    else:
        monkeypatch.delenv(run.DISABLE_PUBLIC_CHECK_ENV, raising = False)
    direct = run._display_host_for_bind("0.0.0.0")
    assert direct == "0.0.0.0"
    assert run._start_display_host_resolution("0.0.0.0")() == direct


def test_an_exception_reaches_the_caller_at_join(monkeypatch):
    class _Boom(RuntimeError):
        pass

    def _raise(_host):
        raise _Boom("resolution failed")

    monkeypatch.setattr(run, "_display_host_for_bind", _raise)
    result = run._start_display_host_resolution("0.0.0.0")
    with pytest.raises(_Boom):
        result()


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost", "192.168.1.20"])
def test_non_wildcard_binds_never_resolve_or_start_a_thread(monkeypatch, host):
    monkeypatch.setattr(
        run, "_display_host_for_bind", lambda *_a: pytest.fail("resolved a non-wildcard bind")
    )
    monkeypatch.setattr(
        run, "_resolve_external_ip", lambda: pytest.fail("looked up the external IP")
    )
    monkeypatch.setattr(
        threading.Thread, "start", lambda self: pytest.fail("started a resolution thread")
    )
    assert run._start_display_host_resolution(host)() == host
