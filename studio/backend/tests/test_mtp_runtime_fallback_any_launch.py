# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MTP crash fallback on every MTP launch, and its reload surviving its thread (#13253)."""

from __future__ import annotations

import inspect
import os
import re
import sys
import threading
import time
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend  # noqa: E402
from utils import process_lifetime  # noqa: E402


def test_runtime_fallback_armed_without_tensor_parallel():
    src = inspect.getsource(LlamaCppBackend.load_model)
    m = re.search(r"_mtp_active_for_launched_server = bool\(([^)]*)\)", src)
    assert m, "load_model must record whether the launched server runs MTP"
    assert m.group(1).strip() == "_spec_requested_mtp"
    assert "self._mtp_runtime_fallback_active = _mtp_active_for_launched_server" in src


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "PDEATHSIG is Linux-only")
def test_llama_server_started_from_a_short_thread_outlives_it(tmp_path, monkeypatch):
    if not process_lifetime._pdeathsig_available():
        pytest.skip(reason = "PR_SET_PDEATHSIG blocked here (seccomp), so the hazard cannot occur")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    b = LlamaCppBackend()
    started = {}

    def _reload_thread():
        with b._lock:
            started["ok"] = b._start_llama_process(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                dict(os.environ),
                child_gpu_physical_ids = None,
            )

    t = threading.Thread(target = _reload_thread, name = "mtp-crash-reload")
    t.start()
    t.join()
    try:
        assert started["ok"] is True
        time.sleep(1.0)
        assert b._process.poll() is None, "child died with the thread that spawned it"
    finally:
        b._kill_process()


class _AliveProcess:
    returncode = None

    def poll(self):
        return None


class _DeadProcess:
    returncode = -6

    def poll(self):
        return -6

    def terminate(self):
        pass

    def kill(self):
        pass

    def wait(self, timeout = None):
        return -6


def _armed_backend(proc, extra_args = None):
    b = LlamaCppBackend()
    b._mtp_runtime_fallback_active = True
    b._process = proc
    b._last_load_intent = GgufLoadIntent(
        model_identifier = "unsloth/gemma-4-E2B-it-GGUF",
        speculative_type = "auto",
        extra_args = extra_args,
    )
    return b


def test_transient_error_on_a_live_server_keeps_the_normal_retry(monkeypatch):
    b = _armed_backend(_AliveProcess())
    monkeypatch.setattr(b, "_server_socket_is_open", lambda *a, **k: True)
    monkeypatch.setattr(b, "load_model", lambda intent: pytest.fail("no reload for a live server"))
    assert b._maybe_recover_from_mtp_crash(RuntimeError("reset")) is False
    assert b._mtp_runtime_fallback_in_progress is False


def test_recovery_with_extras_owned_mtp_matches_an_identical_apply(monkeypatch):
    extras = ("--spec-type", "draft-mtp")
    b = _armed_backend(_DeadProcess(), extra_args = extras)
    loads = []
    monkeypatch.setattr(b, "load_model", lambda intent: loads.append(intent) or True)
    assert b._maybe_recover_from_mtp_crash(RuntimeError("peer closed")) is True
    deadline = time.monotonic() + 5
    while (not loads or b._mtp_runtime_fallback_in_progress) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert loads and loads[0].speculative_type == "off"
    assert "draft-mtp" not in (loads[0].extra_args or ())
    assert b._spec_fallback_reason == "runtime_error"
    # _runtime_matches_intent treats any other value as a new request and relaunches MTP.
    assert b._requested_spec_mode is None
