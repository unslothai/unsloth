# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""WSL nvidia-smi off PATH on every query, Core Ultra iGPU PCI ids, and masks cut at an invalid index; all faked."""

from __future__ import annotations

import subprocess

import pytest

import utils.hardware.hardware as hw
from utils.hardware import nvidia

_UUID = "GPU-11111111-2222-3333-4444-555555555555"


def _fake_smi(where, calls):
    """run_nvidia_smi that only answers when spawned as one of ``where``, the way a missing binary raises."""

    def run(argv, **_kwargs):
        calls.append(argv[0])
        if argv[0] not in where:
            raise FileNotFoundError(argv[0])
        args = " ".join(argv[1:])
        if args == "-L":
            out = "GPU 0: NVIDIA GeForce RTX 4090 (UUID: %s)\n" % _UUID
        elif "index,uuid" in args and "utilization" not in args:
            out = "0, %s\n" % _UUID
        elif args.startswith("--query-gpu=index,"):
            out = "0, 37, 51, 1024, 24564, 80.5, 450.0\n"
        else:
            out = "37, 51, 1024, 24564, 80.5, 450.0\n"
        return subprocess.CompletedProcess(argv, 0, stdout = out, stderr = "")

    return run


def _host(monkeypatch, *, system, which, wsl_file):
    monkeypatch.setattr(nvidia.platform, "system", lambda: system)
    monkeypatch.setattr(nvidia.shutil, "which", lambda _name: which)
    monkeypatch.setattr(
        nvidia.os.path, "isfile", lambda p: wsl_file and p == nvidia._WSL_NVIDIA_SMI
    )
    nvidia._uuid_mask_cache.clear()


def _queries():
    return {
        "count": nvidia.get_physical_gpu_count(),
        "primary": nvidia.get_primary_gpu_utilization().get("available"),
        "visible": len(nvidia.get_visible_gpu_utilization([0], "0")["devices"]),
        "uuid": nvidia._query_uuid_mask(_UUID),
    }


def test_wsl_nvidia_smi_off_path_answers_every_query(monkeypatch):
    calls = []
    _host(monkeypatch, system = "Linux", which = None, wsl_file = True)
    monkeypatch.setattr(
        nvidia.gpu_query, "run_nvidia_smi", _fake_smi({nvidia._WSL_NVIDIA_SMI}, calls)
    )
    assert _queries() == {"count": 1, "primary": True, "visible": 1, "uuid": [0]}
    assert set(calls) == {nvidia._WSL_NVIDIA_SMI}


def test_nvidia_smi_on_path_is_still_the_one_spawned(monkeypatch):
    calls = []
    _host(monkeypatch, system = "Linux", which = "/usr/bin/nvidia-smi", wsl_file = True)
    # Both spellings reach the same binary through PATH.
    monkeypatch.setattr(
        nvidia.gpu_query, "run_nvidia_smi", _fake_smi({"nvidia-smi", "/usr/bin/nvidia-smi"}, calls)
    )
    assert _queries() == {"count": 1, "primary": True, "visible": 1, "uuid": [0]}


@pytest.mark.parametrize("system", ["Linux", "Windows"])
def test_no_nvidia_smi_anywhere_still_reports_unavailable(monkeypatch, system):
    calls = []
    _host(monkeypatch, system = system, which = None, wsl_file = False)
    monkeypatch.delenv("ProgramFiles", raising = False)
    monkeypatch.delenv("SystemRoot", raising = False)
    monkeypatch.setattr(nvidia.gpu_query, "run_nvidia_smi", _fake_smi(set(), calls))
    assert _queries() == {"count": None, "primary": False, "visible": 0, "uuid": None}
    assert set(calls) == {"nvidia-smi"}


@pytest.mark.parametrize(
    ("device_id", "xpu_class"),
    [
        ("0x7d55", True),  # Core Ultra (Meteor Lake-H) Arc Graphics
        ("0x7dd1", True),  # Core Ultra 200H (Arrow Lake-H)
        ("0x64a0", True),  # Core Ultra 200V (Lunar Lake) Arc 140V
        ("0xb080", True),  # Core Ultra Series 3 (Panther Lake)
        ("0x56a0", True),  # Arc A770: unchanged
        ("0xe20b", True),  # Arc B580: unchanged
        ("0x7d45", False),  # Meteor Lake-U Intel Graphics: not on the PyTorch XPU list
        ("0x7d67", False),  # Arrow Lake-S desktop iGPU
        ("0x9a49", False),  # Tiger Lake Iris Xe
        ("0x46a6", False),  # Alder Lake iGPU
        ("0x7d60", False),  # Meteor Lake-M, absent from Intel compute-runtime
    ],
)
def test_core_ultra_arc_igpus_are_xpu_class(tmp_path, device_id, xpu_class):
    (tmp_path / "device").write_text(device_id + "\n", encoding = "utf-8")
    assert hw._intel_pci_device_is_xpu_class(str(tmp_path)) is xpu_class


def test_a_core_ultra_record_establishes_a_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(hw, "_expected_xpu_flavor_was_chosen", lambda: False)
    monkeypatch.setattr(hw, "_torch_reports_an_xpu_runtime", lambda: False)
    monkeypatch.setattr(hw, "_vendors_masked_off", lambda: set())
    (tmp_path / "device").write_text("0x64a0\n", encoding = "utf-8")
    lnl = [
        {
            "vendor": "intel",
            "name": None,
            "index": 0,
            "xpu_class": hw._intel_pci_device_is_xpu_class(str(tmp_path)),
        }
    ]
    assert hw._devices_that_can_establish_a_mismatch(lnl) == lnl


@pytest.mark.parametrize(
    ("mask", "numeric_ids"),
    [
        ("0,2,-1,1", [0, 2]),  # NVIDIA's documented example
        ("-1,0", []),
        ("1,0,abc,2", [1, 0]),
        ("0,abc", [0]),
        ("0,1gpu2,2", [0, 1, 2]),  # strtoul prefix, as torch parses it
        ("1,0,1", []),  # a repeated ordinal empties the set
        ("0,,1", [0]),  # an empty index ends the list
        ("1,,1", [1]),  # ...before a later repeat is seen
        ("0,1,", [0, 1]),
        # Unchanged:
        ("-1", []),
        ("", []),
        ("1,0", [1, 0]),
        ("2", [2]),
        (_UUID, None),
        ("0," + _UUID, None),
        ("abc", None),
    ],
)
def test_a_cuda_mask_stops_at_the_first_invalid_index(monkeypatch, mask, numeric_ids):
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(hw, "IS_ROCM", False)
    for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER"):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    spec = hw._get_parent_visible_gpu_spec()
    assert spec["numeric_ids"] == numeric_ids
    assert spec["supports_explicit_gpu_ids"] is (numeric_ids is not None)
