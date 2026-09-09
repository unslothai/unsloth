# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""_has_usable_nvidia_gpu must keep probing after an unusable nvidia-smi.

A stale nvidia-smi exits non-zero listing no GPU; stopping there makes a mixed
AMD+NVIDIA Windows host look NVIDIA-free and swaps its CUDA stack for ROCm.
install.ps1 / setup.ps1 gate the fallback on the GPU check failing, not on the
PATH lookup missing. Stubs are real executables run via real subprocess.
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
    # backend.utils.wheel_utils resolves only with studio/ on sys.path.
    if str(_STUDIO) not in sys.path:
        sys.path.insert(0, str(_STUDIO))
    spec = importlib.util.spec_from_file_location("_ips_probe_under_test", _SRC)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_ips_probe_under_test"] = module
    spec.loader.exec_module(module)
    return module


def _write_stub(path: pathlib.Path, body: str) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    # Not /usr/bin/env: PATH is narrowed to the stub directory below.
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
        # Pose as win32 so the Linux /proc fallback cannot answer True for us.
        monkeypatch.setattr(module, "sys", types.SimpleNamespace(platform = "win32"))
        return module._has_usable_nvidia_gpu()

    return _run


def test_stale_path_nvidia_smi_still_reaches_the_fixed_locations(probe):
    assert probe(_STALE, _WORKING) is True


def test_absent_path_nvidia_smi_reaches_the_fixed_locations(probe):
    assert probe(None, _WORKING) is True


def test_working_path_nvidia_smi_is_enough(probe):
    assert probe(_WORKING, None) is True


def test_no_nvidia_smi_anywhere_reports_no_gpu(probe):
    assert probe(None, None) is False


def test_stale_everywhere_reports_no_gpu(probe):
    # Must stay False, or an AMD-only host with a leftover nvidia-smi loses ROCm.
    assert probe(_STALE, _STALE) is False


@pytest.mark.parametrize("hidden", ["", "-1", "  "])
def test_cuda_visible_devices_hidden_wins_over_a_working_probe(probe, hidden):
    assert probe(_WORKING, _WORKING, cuda_visible_devices = hidden) is False


def test_cuda_visible_devices_listing_a_device_does_not_block_detection(probe):
    assert probe(_WORKING, None, cuda_visible_devices = "0") is True
