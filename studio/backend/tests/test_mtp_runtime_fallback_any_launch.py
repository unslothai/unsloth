# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Mid-generation MTP crash recovery for every MTP launch (#13253).

A Gemma 4 drafter on Apple Silicon passed /health and then aborted llama-server on
every decode. The runtime fallback (reload without speculative decoding) was armed
only under tensor parallelism, so the respawn replayed the same MTP launch and every
request failed. The reload also runs on a short-lived thread, which on Linux killed
the replacement through PR_SET_PDEATHSIG as soon as that thread returned.
"""

from __future__ import annotations

import inspect
import os
import re
import signal
import subprocess
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

from core.inference import llama_cpp as llama_cpp_module  # noqa: E402
from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend  # noqa: E402
from utils.process_lifetime import child_popen_kwargs, spawn_on_lifetime_thread  # noqa: E402


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


def test_runtime_fallback_armed_without_tensor_parallel():
    src = inspect.getsource(LlamaCppBackend.load_model)
    m = re.search(r"_mtp_active_for_launched_server = bool\(([^)]*)\)", src)
    assert m, "load_model must record whether the launched server runs MTP"
    assert m.group(1).strip() == "_spec_requested_mtp"
    assert "self._mtp_runtime_fallback_active = _mtp_active_for_launched_server" in src


def test_decode_probe_stays_tensor_only():
    # Widening recovery must not add a load-time decode to every MTP load.
    src = inspect.getsource(LlamaCppBackend.load_model)
    probe = src.find("and not self._probe_mtp_decode()")
    assert probe != -1
    assert "self._tensor_parallel" in src[max(0, probe - 300) : probe]


def test_single_gpu_mtp_crash_reloads_without_mtp(monkeypatch):
    b = LlamaCppBackend()
    b._tensor_parallel = False
    b._speculative_type = "draft-mtp"
    b._mtp_runtime_fallback_active = True
    b._process = _DeadProcess()
    b._last_load_intent = GgufLoadIntent(
        model_identifier = "unsloth/gemma-4-E2B-it-GGUF",
        mtp_draft_path = "/models/mtp-gemma-4-E2B-it.gguf",
        speculative_type = "auto",
    )
    done = threading.Event()
    captured = {}

    def _fake_load_model(intent):
        captured.update(vars(intent))
        done.set()
        return True

    monkeypatch.setattr(b, "load_model", _fake_load_model)
    assert b._maybe_recover_from_mtp_crash(RuntimeError("peer closed")) is True
    assert done.wait(timeout = 5)
    assert captured["speculative_type"] == "off"
    deadline = time.monotonic() + 2
    while b._mtp_runtime_fallback_in_progress and time.monotonic() < deadline:
        time.sleep(0.02)
    assert b._spec_fallback_reason == "runtime_error"
    assert b._requested_spec_mode == "auto"


def test_llama_server_spawns_use_the_lifetime_thread():
    src = inspect.getsource(llama_cpp_module)
    guards = [m.end() for m in re.finditer(r"not starting llama-server", src)]
    assert len(guards) >= 2, "expected both llama-server Popen sites"
    for at in guards:
        spawn = src.find("_spawned = ", at)
        assert "_spawn_on_lifetime_thread(" in src[spawn : spawn + 80], src[spawn : spawn + 80]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "PDEATHSIG is Linux-only")
def test_lifetime_thread_keeps_a_child_spawned_from_a_short_thread_alive():
    def _spawn():
        return subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            **child_popen_kwargs(),
        )

    procs = {}

    def _worker(key, via_lifetime):
        procs[key] = spawn_on_lifetime_thread(_spawn) if via_lifetime else _spawn()

    try:
        for key, via in (("direct", False), ("lifetime", True)):
            t = threading.Thread(target = _worker, args = (key, via))
            t.start()
            t.join()
        time.sleep(1.0)
        # The control proves the hazard is real on this host.
        assert procs["direct"].poll() is not None
        assert procs["lifetime"].poll() is None
    finally:
        for p in procs.values():
            if p.poll() is None:
                os.kill(p.pid, signal.SIGKILL)
                p.wait(timeout = 5)
