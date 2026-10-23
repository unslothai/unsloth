# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""#14906: the native-Linux system ROCm prepend is dropped only when it hides the GPU."""

from __future__ import annotations

import os
import stat
import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("structlog")
sys.modules.setdefault("structlog", _structlog_stub)
if not hasattr(sys.modules["structlog"], "get_logger"):
    sys.modules["structlog"].get_logger = _structlog_stub.get_logger

from core.inference import llama_cpp as llama_module  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason = "runs a POSIX shell stub")

_GPU = "Available devices:\n  ROCm0: AMD Radeon RX 7900 XTX (24560 MiB, 24000 MiB free)\n"
_NONE = "Available devices:\n  (none)\n"


def _make_server(
    tmp_path,
    *,
    mixed,
    bundled,
    mixed_rc = 0,
):
    """A llama-server stand-in whose --list-devices answer depends on LD_LIBRARY_PATH."""
    sysrocm = tmp_path / "opt" / "rocm" / "lib"
    sysrocm.mkdir(parents = True)
    binary = tmp_path / "llama.cpp" / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    calls = tmp_path / "calls"
    _write_server(binary, sysrocm, calls, mixed = mixed, bundled = bundled, mixed_rc = mixed_rc)
    return binary, str(sysrocm), calls


def _write_server(
    path,
    sysrocm,
    calls,
    *,
    mixed,
    bundled,
    mixed_rc = 0,
):
    path.write_text(
        "#!/bin/sh\n"
        f'echo x >> "{calls}"\n'
        f'case ":$LD_LIBRARY_PATH:" in *":{sysrocm}:"*)\n'
        f"  printf '%s' '{mixed}'; exit {mixed_rc};;\n"
        "esac\n"
        f"printf '%s' '{bundled}'\n"
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _probe_count(calls: Path) -> int:
    return len(calls.read_text().splitlines()) if calls.exists() else 0


@pytest.fixture
def linux_host(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("LD_LIBRARY_PATH", raising = False)
    monkeypatch.setattr(llama_module, "_wsl_system_rocm_lib_dirs", lambda: [])
    monkeypatch.setattr(LlamaCppBackend, "_bundle_only_rocm_dirs", {})
    monkeypatch.setattr(LlamaCppBackend, "_system_rocm_probed", {})


def _use_system_rocm(monkeypatch, sysrocm):
    monkeypatch.setattr(llama_module, "_native_linux_system_rocm_lib_dirs", lambda _d: [sysrocm])


def _ld(env):
    return env["LD_LIBRARY_PATH"].split(":")


def test_prepend_hiding_the_gpu_falls_back_to_the_bundle(monkeypatch, tmp_path, linux_host):
    binary, sysrocm, calls = _make_server(tmp_path, mixed = _NONE, bundled = _GPU)
    _use_system_rocm(monkeypatch, sysrocm)
    env = LlamaCppBackend._llama_server_env_for_binary(str(binary))
    assert sysrocm not in _ld(env)
    assert _ld(env)[0] == str(binary.parent)
    assert LlamaCppBackend._prefers_bundle_only_rocm(str(binary))
    # A caller that checked the latch before the first probe finished still gets the bundle.
    assert LlamaCppBackend._system_rocm_hides_gpu(str(binary), [sysrocm])
    # Latched: later children (STT sidecar, device probes) neither prepend nor re-probe.
    assert _probe_count(calls) == 2
    from core.inference.stt_mtmd_sidecar import _llama_server_child_env

    assert sysrocm not in _ld(_llama_server_child_env(str(binary)))
    assert _probe_count(calls) == 2
    # The device enumeration Studio plans with now sees the GPU too.
    assert LlamaCppBackend._run_list_devices(str(binary)) == ["ROCm0"]


def test_prepend_that_shows_the_gpu_is_kept_with_one_probe(monkeypatch, tmp_path, linux_host):
    # #7208: the bundle is the broken side; the mix lists the GPU, so nothing changes.
    binary, sysrocm, calls = _make_server(tmp_path, mixed = _GPU, bundled = _NONE)
    _use_system_rocm(monkeypatch, sysrocm)
    for _ in range(3):
        env = LlamaCppBackend._llama_server_env_for_binary(str(binary))
        assert _ld(env)[0] == sysrocm
    assert _probe_count(calls) == 1
    assert not LlamaCppBackend._prefers_bundle_only_rocm(str(binary))


@pytest.mark.parametrize(
    "mixed, bundled, mixed_rc",
    [
        (_NONE, _NONE, 0),  # no GPU either way (masked, CPU host): today's env
        ("garbage", _GPU, 0),  # no answer from the mix is not evidence
        (_NONE, _GPU, 1),  # mix probe crashed: load_model's crash retry owns that
    ],
)
def test_no_proof_keeps_the_prepend(monkeypatch, tmp_path, linux_host, mixed, bundled, mixed_rc):
    binary, sysrocm, _calls = _make_server(
        tmp_path, mixed = mixed, bundled = bundled, mixed_rc = mixed_rc
    )
    _use_system_rocm(monkeypatch, sysrocm)
    env = LlamaCppBackend._llama_server_env_for_binary(str(binary))
    assert _ld(env)[0] == sysrocm
    assert not LlamaCppBackend._prefers_bundle_only_rocm(str(binary))


def test_unexecutable_binary_keeps_the_prepend(monkeypatch, tmp_path, linux_host):
    binary = tmp_path / "llama.cpp" / "build" / "bin" / "llama-server"
    binary.parent.mkdir(parents = True)
    binary.write_text("")
    _use_system_rocm(monkeypatch, "/opt/rocm/lib")
    env = LlamaCppBackend._llama_server_env_for_binary(str(binary))
    assert _ld(env)[0] == "/opt/rocm/lib"


def test_no_system_rocm_runs_no_probe(monkeypatch, tmp_path, linux_host):
    binary, _sysrocm, calls = _make_server(tmp_path, mixed = _NONE, bundled = _GPU)
    monkeypatch.setattr(llama_module, "_native_linux_system_rocm_lib_dirs", lambda _d: [])
    LlamaCppBackend._llama_server_env_for_binary(str(binary))
    assert _probe_count(calls) == 0


def test_a_replaced_runtime_is_probed_again(monkeypatch, tmp_path, linux_host):
    binary, sysrocm, calls = _make_server(tmp_path, mixed = _GPU, bundled = _NONE)
    _use_system_rocm(monkeypatch, sysrocm)
    LlamaCppBackend._llama_server_env_for_binary(str(binary))
    assert _probe_count(calls) == 1
    # The updater swaps a build in place; one whose bundle works must get the fallback.
    replacement = binary.with_name("new-llama-server")
    _write_server(replacement, sysrocm, calls, mixed = _NONE, bundled = _GPU)
    os.replace(replacement, binary)
    env = LlamaCppBackend._llama_server_env_for_binary(str(binary))
    assert sysrocm not in _ld(env)
    assert _probe_count(calls) == 3
