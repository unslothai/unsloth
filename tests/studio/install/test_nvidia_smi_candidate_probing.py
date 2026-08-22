# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""_has_usable_nvidia_gpu must keep probing after an unusable nvidia-smi.

A stale or driverless nvidia-smi on PATH exits non-zero and lists no GPU.
Treating the first executable found as the answer reports a mixed AMD+NVIDIA
Windows host as NVIDIA-free, which routes it into _ensure_rocm_torch() and
replaces a working CUDA stack with ROCm wheels. install.ps1 and setup.ps1 both
gate their fixed-location fallback on the GPU check failing rather than on the
PATH lookup missing; this pins the Python helper to the same rule.

The stubs are real executables run through the real subprocess call, so the
test exercises the actual control flow rather than a mocked return value.
"""

import importlib.util
import os
import pathlib
import sys
import types

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_STUDIO = _REPO_ROOT / "studio"
_SRC = _STUDIO / "install_python_stack.py"

_STALE = 'echo "No devices were found"; exit 9'
_WORKING = 'echo "GPU 0: NVIDIA H100 (UUID: GPU-abc)"; exit 0'


def _load_module():
    # install_python_stack imports backend.utils.wheel_utils, which resolves
    # only with studio/ on sys.path. That is how the installer invokes it.
    if str(_STUDIO) not in sys.path:
        sys.path.insert(0, str(_STUDIO))
    spec = importlib.util.spec_from_file_location("_ips_probe_under_test", _SRC)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_ips_probe_under_test"] = module
    spec.loader.exec_module(module)
    return module


def _write_stub(path: pathlib.Path, body: str) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    # Not /usr/bin/env: PATH is narrowed to the stub directory below, so env
    # would not find an interpreter.
    path.write_text("#!/bin/bash\n" + body + "\n")
    path.chmod(0o755)


@pytest.fixture
def probe(tmp_path, monkeypatch):
    """Run _has_usable_nvidia_gpu as if on Windows, with stubbed nvidia-smi."""

    def _run(
        path_smi: str | None,
        fixed_smi: str | None,
        cuda_visible_devices: str | None = None,
    ) -> bool:
        path_dir = tmp_path / "pathbin"
        path_dir.mkdir(exist_ok = True)
        if path_smi is not None:
            _write_stub(path_dir / "nvidia-smi", path_smi)
        program_files = tmp_path / "ProgramFiles"
        if fixed_smi is not None:
            _write_stub(
                program_files / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
                fixed_smi,
            )
        monkeypatch.setenv("PATH", str(path_dir))
        monkeypatch.setenv("ProgramFiles", str(program_files))
        monkeypatch.setenv("SystemRoot", str(tmp_path / "Windows"))
        if cuda_visible_devices is None:
            monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
        else:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cuda_visible_devices)
        module = _load_module()
        monkeypatch.setattr(module, "IS_WINDOWS", True)
        # A real NVIDIA host has /proc/driver/nvidia/gpus, and the helper's
        # Linux-only fallback would then answer True for every case and mask
        # what the Windows path did. Present as win32 to isolate it.
        monkeypatch.setattr(module, "sys", types.SimpleNamespace(platform = "win32"))
        return module._has_usable_nvidia_gpu()

    return _run


def test_stale_path_nvidia_smi_still_reaches_the_fixed_locations(probe):
    # The regression: a driverless nvidia-smi on PATH used to end the search.
    assert probe(_STALE, _WORKING) is True


def test_absent_path_nvidia_smi_reaches_the_fixed_locations(probe):
    assert probe(None, _WORKING) is True


def test_working_path_nvidia_smi_is_enough(probe):
    assert probe(_WORKING, None) is True


def test_no_nvidia_smi_anywhere_reports_no_gpu(probe):
    assert probe(None, None) is False


def test_stale_everywhere_reports_no_gpu(probe):
    # Every candidate answering "no GPU" must stay False, or an AMD-only host
    # with a leftover nvidia-smi would be denied the ROCm wheels.
    assert probe(_STALE, _STALE) is False


@pytest.mark.parametrize("hidden", ["", "-1", "  "])
def test_cuda_visible_devices_hidden_wins_over_a_working_probe(probe, hidden):
    assert probe(_WORKING, _WORKING, cuda_visible_devices = hidden) is False


def test_cuda_visible_devices_listing_a_device_does_not_block_detection(probe):
    assert probe(_WORKING, None, cuda_visible_devices = "0") is True
