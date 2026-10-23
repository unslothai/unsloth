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


def test_an_amd_card_attached_beside_another_reads_the_gfx_target_again(rocm_cli, monkeypatch):
    topology = [(("0000:03:00.0", "0x744c"),)]
    monkeypatch.setattr(hw, "_linux_amd_pci_topology", lambda *a, **k: topology[0])
    hw._linux_amd_gfx_candidates()
    hw._linux_amd_gfx_candidates()
    assert _count(rocm_cli, "rocminfo") == 1
    topology[0] = (("0000:03:00.0", "0x744c"), ("0000:0c:00.0", "0x7550"))
    hw._linux_amd_gfx_candidates()
    assert _count(rocm_cli, "rocminfo") == 2


def test_amd_pci_topology_reads_sysfs(tmp_path):
    def card(name, vendor, device, slot):
        dev = tmp_path / "pci" / slot
        dev.mkdir(parents = True)
        (dev / "vendor").write_text(vendor + "\n")
        (dev / "device").write_text(device + "\n")
        (tmp_path / "drm" / name).mkdir(parents = True)
        (tmp_path / "drm" / name / "device").symlink_to(dev)

    card("card0", "0x1002", "0x744C", "0000:03:00.0")
    card("card1", "0x10de", "0x2684", "0000:01:00.0")
    (tmp_path / "drm" / "card0-DP-1").mkdir()
    assert hw._linux_amd_pci_topology(str(tmp_path / "drm")) == (("0000:03:00.0", "0x744c"),)


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


def test_a_fallback_answer_after_rocminfo_failed_is_not_kept(rocm_cli, monkeypatch):
    def rocminfo_hangs(argv, *a, **k):
        rocm_cli.append(list(argv))
        if argv[0] == "rocminfo":
            raise subprocess.TimeoutExpired(argv, 10)
        return _Result("ASIC: gfx1100\n")

    monkeypatch.setattr(subprocess, "run", rocminfo_hangs)
    assert hw._linux_amd_gfx_candidates() == ["gfx1100"]
    assert hw._linux_amd_gfx_candidates() == ["gfx1100"]
    assert _count(rocm_cli, "rocminfo") == 2


def test_a_present_but_unstartable_rocminfo_is_not_kept(rocm_cli, monkeypatch):
    def no_interpreter(argv, *a, **k):
        rocm_cli.append(list(argv))
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(subprocess, "run", no_interpreter)
    hw._linux_amd_gfx_candidates()
    hw._linux_amd_gfx_candidates()
    assert _count(rocm_cli, "rocminfo") == 2


def test_an_absent_rocm_userspace_is_kept(rocm_cli, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name, *a, **k: None)

    def absent(argv, *a, **k):
        rocm_cli.append(list(argv))
        raise FileNotFoundError(argv[0])

    monkeypatch.setattr(subprocess, "run", absent)
    hw._linux_amd_gfx_candidates()
    hw._linux_amd_gfx_candidates()
    assert _count(rocm_cli, "rocminfo") == 1


def test_a_numbered_per_device_override_reads_the_gfx_target_again(rocm_cli, monkeypatch):
    hw._linux_amd_gfx_candidates()
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION_1", "11.0.0")
    hw._linux_amd_gfx_candidates()
    assert _count(rocm_cli, "rocminfo") == 2


def test_forced_hardware_detection_reads_the_gfx_target_again(rocm_cli, monkeypatch):
    hw._linux_amd_gfx_candidates()
    # detect_hardware() rebinds the module's detection globals: restore them after the test.
    for name in list(vars(hw)):
        if name.lstrip("_").isupper():
            monkeypatch.setattr(hw, name, getattr(hw, name))
    monkeypatch.setattr(hw, "_has_torch", lambda: False)
    try:
        hw.detect_hardware()
    except Exception:
        pass
    hw._linux_amd_gfx_candidates()
    assert _count(rocm_cli, "rocminfo") >= 2


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
