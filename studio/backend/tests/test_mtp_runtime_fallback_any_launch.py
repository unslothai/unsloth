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

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402
from utils import process_lifetime  # noqa: E402


def test_runtime_fallback_armed_without_tensor_parallel():
    src = inspect.getsource(LlamaCppBackend.load_model)
    m = re.search(r"_mtp_active_for_launched_server = bool\(([^)]*)\)", src)
    assert m, "load_model must record whether the launched server runs MTP"
    assert m.group(1).strip() == "_spec_requested_mtp"
    assert "self._mtp_runtime_fallback_active = _mtp_active_for_launched_server" in src


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "PDEATHSIG is Linux-only")
def test_llama_server_started_from_a_short_thread_outlives_it(tmp_path, monkeypatch):
    # The MTP-crash reload spawns from a thread that exits right after the load.
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
