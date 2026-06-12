# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the on-demand RAG dependency installer.

Real pip is never invoked: ``subprocess.run`` / ``find_spec`` / the sqlite-vec
import are monkeypatched so the install-to-available transition, idempotency,
the sticky-failure guard and pip command construction are checked in isolation.
Module globals are saved and restored per test.
"""

import subprocess
import sys
import time
import types

import pytest

from core.rag import deps
from storage import rag_db


@pytest.fixture(autouse = True)
def _reset_state():
    saved = (rag_db.RAG_AVAILABLE, rag_db.sqlite_vec, deps._installing, deps._error)
    had_module = "sqlite_vec" in sys.modules
    saved_module = sys.modules.get("sqlite_vec")
    yield
    rag_db.RAG_AVAILABLE, rag_db.sqlite_vec, deps._installing, deps._error = saved
    if had_module:
        sys.modules["sqlite_vec"] = saved_module
    else:
        sys.modules.pop("sqlite_vec", None)


def _recording_thread(counter):
    class _Thread:
        def __init__(self, *args, **kwargs):
            self.target = kwargs.get("target")
            self.args = kwargs.get("args", ())

        def start(self):
            counter["n"] += 1

    return _Thread


# missing_packages


def test_missing_packages_none_when_all_present(monkeypatch):
    monkeypatch.setattr(deps.importlib.util, "find_spec", lambda name: object())
    assert deps.missing_packages() == []


def test_missing_packages_reports_absent_spec(monkeypatch):
    monkeypatch.setattr(
        deps.importlib.util,
        "find_spec",
        lambda name: None if name == "docx" else object(),
    )
    assert deps.missing_packages() == ["python-docx==1.2.0"]


# status


def test_status_shape(monkeypatch):
    monkeypatch.setattr(deps.importlib.util, "find_spec", lambda name: object())
    rag_db.RAG_AVAILABLE = True
    st = deps.status()
    assert set(st) == {"available", "installing", "missing", "error"}
    assert st["available"] is True
    assert st["installing"] is False
    assert st["missing"] == []
    assert st["error"] is None


# pip command construction


def test_pip_install_cmd_prefers_uv(monkeypatch):
    monkeypatch.setattr(deps.shutil, "which", lambda name: "/usr/bin/uv")
    assert deps._pip_install_cmd("x==1") == [
        "uv", "pip", "install", "--python", sys.executable, "x==1",
    ]


def test_pip_install_cmd_falls_back_to_pip(monkeypatch):
    monkeypatch.setattr(deps.shutil, "which", lambda name: None)
    assert deps._pip_install_cmd("x==1") == [
        sys.executable, "-m", "pip", "install", "x==1",
    ]


# _install body


def test_install_success_enables_rag(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout = "ok")

    monkeypatch.setattr(deps.subprocess, "run", fake_run)

    def fake_refresh():
        rag_db.RAG_AVAILABLE = True
        return True

    monkeypatch.setattr(rag_db, "refresh_rag_available", fake_refresh)
    rag_db.RAG_AVAILABLE, deps._installing, deps._error = False, True, None

    deps._install(["x==1"])

    assert "x==1" in seen["cmd"]
    assert deps._installing is False
    assert deps._error is None
    assert rag_db.RAG_AVAILABLE is True


def test_install_failure_records_error(monkeypatch):
    monkeypatch.setattr(
        deps.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout = "boom"),
    )
    rag_db.RAG_AVAILABLE, deps._installing, deps._error = False, True, None

    deps._install(["x==1"])

    assert deps._installing is False
    assert deps._error and "pip install failed" in deps._error


# ensure_async gating


def test_ensure_async_noop_when_available(monkeypatch):
    spawned = {"n": 0}
    monkeypatch.setattr(deps.threading, "Thread", _recording_thread(spawned))
    rag_db.RAG_AVAILABLE = True
    deps.ensure_async()
    assert spawned["n"] == 0


def test_ensure_async_error_is_sticky_without_force(monkeypatch):
    spawned = {"n": 0}
    monkeypatch.setattr(deps.threading, "Thread", _recording_thread(spawned))
    monkeypatch.setattr(deps, "missing_packages", lambda: ["x==1"])
    rag_db.RAG_AVAILABLE, deps._installing, deps._error = False, False, "prev"

    deps.ensure_async()
    assert spawned["n"] == 0  # not retried automatically

    deps.ensure_async(force = True)
    assert spawned["n"] == 1  # forced retry spawns and clears the error
    assert deps._installing is True
    assert deps._error is None


def test_ensure_async_installs_and_enables(monkeypatch):
    monkeypatch.setattr(deps, "missing_packages", lambda: ["x==1"])
    monkeypatch.setattr(
        deps.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout = "ok"),
    )

    def fake_refresh():
        rag_db.RAG_AVAILABLE = True
        return True

    monkeypatch.setattr(rag_db, "refresh_rag_available", fake_refresh)
    rag_db.RAG_AVAILABLE, deps._installing, deps._error = False, False, None

    deps.ensure_async()
    deadline = time.time() + 5
    while deps.status()["installing"] and time.time() < deadline:
        time.sleep(0.02)

    assert rag_db.RAG_AVAILABLE is True
    assert deps._installing is False
    assert deps._error is None


# refresh_rag_available re-check


def test_refresh_rag_available_noop_when_true():
    rag_db.RAG_AVAILABLE = True
    assert rag_db.refresh_rag_available() is True


def test_refresh_rag_available_rechecks_after_install():
    fake = types.ModuleType("sqlite_vec")
    sys.modules["sqlite_vec"] = fake
    rag_db.RAG_AVAILABLE, rag_db.sqlite_vec = False, None

    assert rag_db.refresh_rag_available() is True
    assert rag_db.RAG_AVAILABLE is True
    assert rag_db.sqlite_vec is fake
