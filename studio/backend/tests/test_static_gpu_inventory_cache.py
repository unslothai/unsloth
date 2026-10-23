# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static AMD answers (gfx target, HIP id map) are kept until hardware re-detection; live reads stay live."""

import subprocess

import pytest

from utils.hardware import amd, gpu_query
from utils.hardware import hardware as hw


class _Result:
    def __init__(self, stdout, returncode = 0):
        self.stdout, self.stderr, self.returncode = stdout, "", returncode


_MASKS = (
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "HSA_OVERRIDE_GFX_VERSION",
)


@pytest.fixture
def rocm_cli(monkeypatch):
    calls = []

    def run(argv, *a, **k):
        calls.append(list(argv))
        if argv[0] == "rocminfo":
            return _Result("  Name:                    gfx1100\n")
        if argv[0] == "amd-smi" and argv[1:3] == ["list", "-e"]:
            return _Result('[{"gpu": 0, "hip_id": 0}]')
        if argv[0] == "amd-smi" and argv[1] == "metric":
            return _Result(
                '[{"gpu": 0, "mem_usage": {"total_vram": {"value": 24560, "unit": "MB"},'
                ' "used_vram": {"value": 100, "unit": "MB"}}}]'
            )
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(
        "shutil.which", lambda name, *a, **k: f"/opt/rocm/bin/{name}" if name in ("rocminfo", "amd-smi") else None
    )
    monkeypatch.delenv("UNSLOTH_GPU_QUERY_CACHE", raising = False)
    for name in _MASKS:
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(hw, "_gfx_candidates_cache", None, raising = False)
    return calls


def _count(calls, *prefix):
    return sum(1 for argv in calls if argv[: len(prefix)] == list(prefix))


def test_rocminfo_runs_once_across_inventory_passes(rocm_cli):
    for _ in range(10):
        assert hw._linux_amd_gfx_candidates() == ["gfx1100"]
    assert _count(rocm_cli, "rocminfo") == 1


def test_hardware_redetection_reads_the_gfx_target_again(rocm_cli):
    hw._linux_amd_gfx_candidates()
    hw.invalidate_detection()
    hw._linux_amd_gfx_candidates()
    assert _count(rocm_cli, "rocminfo") == 2


@pytest.mark.parametrize("name", ["ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION"])
def test_a_changed_mask_or_override_reads_the_gfx_target_again(rocm_cli, monkeypatch, name):
    hw._linux_amd_gfx_candidates()
    monkeypatch.setenv(name, "1" if "VISIBLE" in name else "11.0.0")
    hw._linux_amd_gfx_candidates()
    hw._linux_amd_gfx_candidates()
    assert _count(rocm_cli, "rocminfo") == 2


def test_a_rocm_install_landing_mid_session_is_seen(rocm_cli, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name, *a, **k: None)
    hw._linux_amd_gfx_candidates()
    monkeypatch.setattr("shutil.which", lambda name, *a, **k: f"/opt/rocm/bin/{name}")
    assert hw._linux_amd_gfx_candidates() == ["gfx1100"]
    assert _count(rocm_cli, "rocminfo") == 2


def test_a_timed_out_probe_is_not_kept(rocm_cli, monkeypatch):
    def hung(argv, *a, **k):
        rocm_cli.append(list(argv))
        raise subprocess.TimeoutExpired(argv, 10)

    monkeypatch.setattr(subprocess, "run", hung)
    assert hw._linux_amd_gfx_candidates() == []
    assert hw._linux_amd_gfx_candidates() == []
    assert _count(rocm_cli, "rocminfo") == 2


def test_a_failing_probe_is_not_kept(rocm_cli, monkeypatch):
    def failing(argv, *a, **k):
        rocm_cli.append(list(argv))
        return _Result("", returncode = 1)

    monkeypatch.setattr(subprocess, "run", failing)
    assert hw._linux_amd_gfx_candidates() == []
    assert hw._linux_amd_gfx_candidates() == []
    assert _count(rocm_cli, "rocminfo") == 2


def test_the_cache_switch_turns_it_off(rocm_cli, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GPU_QUERY_CACHE", "0")
    for _ in range(3):
        hw._linux_amd_gfx_candidates()
    assert _count(rocm_cli, "rocminfo") == 3


def test_hardware_redetection_reads_the_hip_id_map_again(rocm_cli):
    assert amd.get_hip_id_by_gpu_index() == {0: 0}
    assert amd.get_hip_id_by_gpu_index() == {0: 0}
    assert _count(rocm_cli, "amd-smi", "list", "-e") == 1
    gpu_query.invalidate_static("re-seated card")
    assert amd.get_hip_id_by_gpu_index() == {0: 0}
    assert _count(rocm_cli, "amd-smi", "list", "-e") == 2


def test_live_vram_reads_still_run_every_time(rocm_cli):
    hw._linux_amd_gfx_candidates()
    amd.get_hip_id_by_gpu_index()
    for _ in range(3):
        assert amd.get_gpu_vram_mib()
    assert _count(rocm_cli, "amd-smi", "metric") == 3
