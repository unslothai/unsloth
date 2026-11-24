# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Mid-generation speculative recovery for every draft mode, and its reload surviving its thread (#13253, #19239)."""

from __future__ import annotations

import inspect
import os
import re
import struct
import subprocess
import sys
import threading
import time
import types as _types
from pathlib import Path
from unittest.mock import patch

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend  # noqa: E402
from utils import process_lifetime  # noqa: E402

_REAL_POPEN = subprocess.Popen


def test_runtime_fallback_armed_without_tensor_parallel():
    src = inspect.getsource(LlamaCppBackend.load_model)
    m = re.search(r"_mtp_active_for_launched_server = bool\(\s*([^)]*?)\s*\)", src, re.DOTALL)
    assert m, "load_model must record whether the launched server runs a draft speculative mode"
    armed = re.sub(r"\s+", " ", m.group(1)).strip()
    assert "_spec_requested_mtp" in armed
    assert "_spec_requested_dspark" in armed
    assert "_spec_requested_dflash" in armed
    assert "_spec_requested_eagle3" in armed
    assert "self._mtp_runtime_fallback_active = _mtp_active_for_launched_server" in src


def _write_gguf(path: Path) -> Path:
    def string(value: str) -> bytes:
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    metadata = string("general.architecture") + struct.pack("<I", 8) + string("qwen35")
    path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 1) + metadata)
    return path


def _load(tmp_path, monkeypatch, *, dspark: bool):
    caps = dict(LlamaCppBackend.probe_server_capabilities("/nonexistent/llama-server"))
    caps.update(
        found = True,
        supports_dspark = True,
        supports_ngram_mod = True,
        mtp_probe_inconclusive = False,
        spec_draft_n_max_flag = "--spec-draft-n-max",
        help_probe_ok = True,
        flags = {
            "--no-spec-draft-backend-sampling": "",
            "--spec-draft-backend-sampling": "",
        },
    )
    monkeypatch.setattr(
        LlamaCppBackend, "probe_server_capabilities", classmethod(lambda cls, b = None: caps)
    )
    backend = LlamaCppBackend()
    backend._get_gpu_memory = lambda _binary = None, **_kw: []
    backend._get_gpu_free_memory = lambda _binary = None, **_kw: []
    backend._read_gguf_metadata = lambda _path: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _path: 1024
    backend._mmproj_vram_bytes = lambda _path: 0
    backend._resolve_launch_mmproj_path = lambda **kwargs: None
    backend._apu_ram_shortfall_message = lambda *a, **k: None
    backend._amd_apu_wants_unified_memory = lambda *a, **k: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: False
    backend._wait_for_health = lambda timeout, **_kw: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _detected: True

    draft = None
    if dspark:
        draft = tmp_path / "dspark-model-F16.gguf"
        draft.write_bytes(b"GGUF")

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)
        return type(
            "Process",
            (),
            {
                "pid": 123,
                "stdout": (),
                "poll": lambda self: None,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: 0,
                "kill": lambda self: None,
            },
        )()

    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        backend.load_model(
            GgufLoadIntent(
                gguf_path = str(_write_gguf(tmp_path / "model.gguf")),
                dspark_draft_path = str(draft) if draft is not None else None,
                model_identifier = "unsloth/Inkling-2-Mini-GGUF",
                speculative_type = "dspark" if dspark else "off",
                n_ctx = 4096,
            )
        )
    return backend


def test_load_model_arms_runtime_fallback_for_draft_dspark(tmp_path, monkeypatch):
    backend = _load(tmp_path, monkeypatch, dspark = True)
    assert backend._speculative_type == "draft-dspark"
    assert backend._mtp_runtime_fallback_active is True


def test_load_model_leaves_runtime_fallback_off_without_a_drafter(tmp_path, monkeypatch):
    backend = _load(tmp_path, monkeypatch, dspark = False)
    assert backend._speculative_type in (None, "default", "none", "off") or (
        backend._speculative_type is not None
        and "draft-" not in str(backend._speculative_type)
        and "mtp" not in str(backend._speculative_type)
    )
    assert backend._mtp_runtime_fallback_active is False


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
    assert b._requested_spec_mode is None
