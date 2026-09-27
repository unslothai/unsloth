# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A slow or failing GPU probe must not turn a detected GPU into "No GPU detected"."""

from __future__ import annotations

import time

import pytest

import utils.hardware.hardware as hw
from utils.hardware import nvidia

B200 = {"index": 0, "name": "NVIDIA B200", "memory_total_gb": 178.35}


@pytest.fixture(autouse = True)
def _cuda_host(monkeypatch):
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: [0])
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": "0", "numeric_ids": [0], "supports_explicit_gpu_ids": True},
    )
    monkeypatch.setattr(hw, "_repair_smi_visible_devices", lambda devices, ids: True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(hw, "_last_good_visible_info", {})
    monkeypatch.setattr(hw, "_torch_get_device_inventory", lambda idx: [])
    monkeypatch.setattr(hw, "_torch_get_physical_gpu_count", lambda: None)


def _smi(monkeypatch, rows):
    """rows: a list (nvidia-smi answered), None (timed out / failed) or NVIDIA_SMI_ABSENT."""
    monkeypatch.setattr(nvidia, "_query_gpu_inventory", lambda caller: rows)


def test_failed_probe_after_a_good_read_keeps_the_gpu(monkeypatch):
    _smi(monkeypatch, [dict(B200)])
    good = hw.get_backend_visible_gpu_info()
    assert good["available"] and [d["name"] for d in good["devices"]] == ["NVIDIA B200"]
    assert "stale" not in good

    _smi(monkeypatch, None)  # timeout / non-zero exit
    kept = hw.get_backend_visible_gpu_info()
    assert kept["available"] is True and kept["stale"] is True
    assert [d["name"] for d in kept["devices"]] == ["NVIDIA B200"]
    assert kept["devices"][0]["memory_total_gb"] == 178.35
    for key in ("probe_failed", "smi_absent", "_confirmed_empty"):
        assert key not in kept

    _smi(monkeypatch, [dict(B200)])  # answers again: fresh, no longer stale
    assert "stale" not in hw.get_backend_visible_gpu_info()


def test_failed_probe_with_no_prior_read_claims_no_gpu(monkeypatch):
    _smi(monkeypatch, None)
    info = hw.get_backend_visible_gpu_info()
    assert info["available"] is False and info["devices"] == [] and "stale" not in info


def test_nvidia_smi_answering_zero_rows_reports_none(monkeypatch):
    _smi(monkeypatch, [dict(B200)])
    assert hw.get_backend_visible_gpu_info()["available"]
    _smi(monkeypatch, [])  # the driver answered: no card under this mask
    info = hw.get_backend_visible_gpu_info()
    assert info["available"] is False and info["devices"] == [] and "stale" not in info
    _smi(monkeypatch, None)  # and a later failure does not resurrect it
    assert hw.get_backend_visible_gpu_info()["available"] is False


@pytest.mark.parametrize(
    "mask",
    ["CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "ZE_AFFINITY_MASK"],
)
def test_a_different_mask_does_not_inherit_the_inventory(monkeypatch, mask):
    _smi(monkeypatch, [dict(B200)])
    assert hw.get_backend_visible_gpu_info()["available"]
    monkeypatch.setenv(mask, "7")
    _smi(monkeypatch, None)
    assert hw.get_backend_visible_gpu_info()["available"] is False


def test_nvidia_marks_failure_and_absence_apart(monkeypatch):
    _smi(monkeypatch, None)
    assert nvidia.get_backend_visible_gpu_info([0], "0")["probe_failed"] is True
    _smi(monkeypatch, nvidia.NVIDIA_SMI_ABSENT)
    absent = nvidia.get_backend_visible_gpu_info([0], "0")
    assert absent["smi_absent"] is True and "probe_failed" not in absent
    _smi(monkeypatch, [])
    empty = nvidia.get_backend_visible_gpu_info([0], "0")
    assert empty["available"] is False and "probe_failed" not in empty and "smi_absent" not in empty


def test_an_older_probe_does_not_bring_back_a_gpu_a_newer_one_ruled_out(monkeypatch):
    import threading

    release = threading.Event()
    found = {"available": True, "devices": [dict(B200)], "index_kind": "physical"}
    empty = {"available": False, "devices": [], "index_kind": "physical"}

    def slow_probe(device):
        release.wait(5)
        return dict(found, devices = [dict(B200)])

    monkeypatch.setattr(hw, "_probe_backend_visible_gpu_info", slow_probe)
    older = threading.Thread(target = hw.get_backend_visible_gpu_info)
    older.start()
    time.sleep(0.2)
    monkeypatch.setattr(
        hw, "_probe_backend_visible_gpu_info", lambda d: dict(empty, _confirmed_empty = True)
    )
    assert hw.get_backend_visible_gpu_info()["available"] is False
    release.set()
    older.join(5)
    monkeypatch.setattr(
        hw, "_probe_backend_visible_gpu_info", lambda d: dict(empty, probe_failed = True)
    )
    assert hw.get_backend_visible_gpu_info()["available"] is False
