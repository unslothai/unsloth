# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Static AMD answers (gfx target, HIP id map) are kept until hardware re-detection; live reads stay live."""

import subprocess
import sys

import pytest

from utils.hardware import amd, gpu_query
from utils.hardware import hardware as hw


class _Result:
    def __init__(
        self,
        stdout,
        returncode = 0,
    ):
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
        "shutil.which",
        lambda name, *a, **k: f"/opt/rocm/bin/{name}" if name in ("rocminfo", "amd-smi") else None,
    )
    monkeypatch.delenv("UNSLOTH_GPU_QUERY_CACHE", raising = False)
    for name in _MASKS:
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(hw, "_gfx_candidates_cache", None, raising = False)
    # Other files can leave amd-smi latched off (failures, a non-Linux platform stub): start clean.
    monkeypatch.setattr(amd, "_amd_smi_disabled", False)
    monkeypatch.setattr(amd, "_amd_smi_consecutive_failures", 0)
    monkeypatch.setattr(amd, "_amd_smi_allowed", lambda *a, **k: True)
    monkeypatch.setattr(amd, "_hip_id_map_cache", None)
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


@pytest.mark.parametrize(
    "name", ["ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION"]
)
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


@pytest.mark.skipif(
    sys.platform == "win32", reason = "sysfs is Linux only; PCI addresses contain ':'"
)
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


def test_forced_hardware_detection_reads_the_hip_id_map_again(rocm_cli, monkeypatch):
    monkeypatch.setattr(amd, "_hip_id_map_cache", None)
    amd.get_hip_id_by_gpu_index()
    for name in list(vars(hw)):
        if name.lstrip("_").isupper():
            monkeypatch.setattr(hw, name, getattr(hw, name))
    monkeypatch.setattr(hw, "_has_torch", lambda: False)
    try:
        hw.detect_hardware()
    except Exception:
        pass
    amd.get_hip_id_by_gpu_index()
    assert _count(rocm_cli, "amd-smi", "list", "-e") >= 2


def test_partial_output_from_a_failing_probe_is_used_not_kept(rocm_cli, monkeypatch):
    def partial(argv, *a, **k):
        rocm_cli.append(list(argv))
        return _Result("  Name:                    gfx1100\n", returncode = 1)

    monkeypatch.setattr(subprocess, "run", partial)
    assert hw._linux_amd_gfx_candidates() == ["gfx1100"]
    assert hw._linux_amd_gfx_candidates() == ["gfx1100"]
    assert _count(rocm_cli, "rocminfo") == 2


def test_a_probe_running_across_a_forced_detection_cannot_republish(rocm_cli, monkeypatch):
    import threading

    entered, release = threading.Event(), threading.Event()
    real_probe = hw._probe_linux_amd_gfx_candidates

    def slow_probe():
        entered.set()
        release.wait(10)
        return ["gfx1030"], True

    monkeypatch.setattr(hw, "_probe_linux_amd_gfx_candidates", slow_probe)
    reader = threading.Thread(target = hw._linux_amd_gfx_candidates)
    reader.start()
    assert entered.wait(10)
    # What forced detection does to this cache, while the old probe is still running.
    monkeypatch.setattr(hw, "_gfx_detect_epoch", hw._gfx_detect_epoch + 1)
    monkeypatch.setattr(hw, "_gfx_candidates_cache", None)
    release.set()
    reader.join(10)
    monkeypatch.setattr(hw, "_probe_linux_amd_gfx_candidates", real_probe)
    assert hw._linux_amd_gfx_candidates() == ["gfx1100"]
    assert _count(rocm_cli, "rocminfo") == 1


def test_an_empty_answer_is_asked_again_after_its_window(rocm_cli, monkeypatch):
    now = [1000.0]
    answers = ["  Name:                    CPU\n", "  Name:                    gfx1100\n"]

    def run(argv, *a, **k):
        rocm_cli.append(list(argv))
        if argv[0] == "rocminfo":
            return _Result(answers[0] if _count(rocm_cli, "rocminfo") == 1 else answers[1])
        return _Result("")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(hw.time, "monotonic", lambda: now[0])
    assert hw._linux_amd_gfx_candidates() == []
    now[0] += hw._GFX_EMPTY_TTL_S - 1
    assert hw._linux_amd_gfx_candidates() == []
    assert _count(rocm_cli, "rocminfo") == 1
    now[0] += 2
    assert hw._linux_amd_gfx_candidates() == ["gfx1100"]
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


def test_an_amd_card_attached_beside_another_reads_the_hip_id_map_again(rocm_cli, monkeypatch):
    topology = [(("0000:03:00.0", "0x744c"),)]
    monkeypatch.setattr(hw, "_linux_amd_pci_topology", lambda *a, **k: topology[0])
    monkeypatch.setattr(amd, "_hip_id_map_cache", None)
    amd.get_hip_id_by_gpu_index()
    amd.get_hip_id_by_gpu_index()
    assert _count(rocm_cli, "amd-smi", "list", "-e") == 1
    topology[0] = (("0000:03:00.0", "0x744c"), ("0000:0c:00.0", "0x7550"))
    amd.get_hip_id_by_gpu_index()
    assert _count(rocm_cli, "amd-smi", "list", "-e") == 2


def test_an_unreadable_topology_keeps_the_hip_id_map(rocm_cli, monkeypatch):
    topology = [(("0000:03:00.0", "0x744c"),)]
    monkeypatch.setattr(hw, "_linux_amd_pci_topology", lambda *a, **k: topology[0])
    monkeypatch.setattr(amd, "_hip_id_map_cache", None)
    amd.get_hip_id_by_gpu_index()
    topology[0] = None
    amd.get_hip_id_by_gpu_index()
    assert _count(rocm_cli, "amd-smi", "list", "-e") == 1


def test_a_topology_that_becomes_readable_sets_the_hip_id_map_baseline(rocm_cli, monkeypatch):
    topology = [None]
    monkeypatch.setattr(hw, "_linux_amd_pci_topology", lambda *a, **k: topology[0])
    monkeypatch.setattr(amd, "_hip_id_map_cache", None)
    amd.get_hip_id_by_gpu_index()
    topology[0] = (("0000:03:00.0", "0x744c"),)
    amd.get_hip_id_by_gpu_index()
    amd.get_hip_id_by_gpu_index()
    assert _count(rocm_cli, "amd-smi", "list", "-e") == 2
    topology[0] = (("0000:03:00.0", "0x744c"), ("0000:0c:00.0", "0x7550"))
    amd.get_hip_id_by_gpu_index()
    assert _count(rocm_cli, "amd-smi", "list", "-e") == 3


def test_an_unreadable_topology_probes_the_gfx_target_every_pass(rocm_cli, monkeypatch):
    monkeypatch.setattr(hw, "_linux_amd_pci_topology", lambda *a, **k: None)
    hw._linux_amd_gfx_candidates()
    hw._linux_amd_gfx_candidates()
    assert _count(rocm_cli, "rocminfo") == 2


def test_no_drm_class_is_no_cards(tmp_path):
    assert hw._linux_amd_pci_topology(str(tmp_path / "missing")) == ()


def test_an_invalidation_during_a_hip_id_map_read_is_not_overwritten(rocm_cli, monkeypatch):
    import threading

    monkeypatch.setattr(amd, "_hip_id_map_cache", None)
    entered, release = threading.Event(), threading.Event()

    def slow_read():
        entered.set()
        release.wait(10)
        return {0: 0}

    monkeypatch.setattr(amd, "_read_hip_id_by_gpu_index", slow_read)
    reader = threading.Thread(target = amd.get_hip_id_by_gpu_index)
    reader.start()
    assert entered.wait(10)
    clearer = threading.Thread(target = amd.invalidate_hip_id_map)
    clearer.start()
    clearer.join(0.3)  # an unlocked clear would finish here, before the read lands
    release.set()
    reader.join(10)
    clearer.join(10)
    assert amd._hip_id_map_cache is None


def test_live_vram_reads_still_run_every_time(rocm_cli):
    hw._linux_amd_gfx_candidates()
    amd.get_hip_id_by_gpu_index()
    for _ in range(3):
        assert amd.get_gpu_vram_mib()
    assert _count(rocm_cli, "amd-smi", "metric") == 3
