# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Automatic GPU placement probes: failures are cached, comma names parse, MIG discovery is shared."""

import subprocess

from utils.hardware import gpu_query, hardware as hw, nvidia


def test_failed_bus_id_join_is_not_respawned(monkeypatch):
    calls = []

    def _rows():
        calls.append(1)
        return None

    monkeypatch.setattr(nvidia, "get_cuda_ordinal_smi_rows", _rows)
    monkeypatch.setattr(hw, "_cuda_ordinal_rows_cache", {})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    for _ in range(3):
        assert hw._smi_rows_for_cuda_ordinals([0, 1]) is None
    assert len(calls) == 1
    # Hardware re-detection retires the negative answer.
    monkeypatch.setattr(gpu_query, "_static_gen", gpu_query._static_gen + 1)
    hw._smi_rows_for_cuda_ordinals([0, 1])
    assert len(calls) == 2


def _smi(stdout, returncode = 0, calls = None):
    def _run(argv, **_kw):
        if calls is not None:
            calls.append(list(argv[1:]))
        return subprocess.CompletedProcess(argv, returncode, stdout, "")

    return _run


def test_card_identity_keeps_a_name_with_commas(monkeypatch):
    out = (
        "0, NVIDIA Example GPU, 2-slot, 81559 MiB, 9.0, 1980 MHz\n"
        "1, NVIDIA Example GPU, 2-slot, 81559 MiB, 9.0, 1980 MHz\n"
    )
    monkeypatch.setattr(gpu_query, "run_nvidia_smi", _smi(out))
    ids = nvidia.get_card_identities()
    assert ids[0] == ("NVIDIA Example GPU, 2-slot", "81559 MiB", "9.0", "1980 MHz")
    assert ids[0] == ids[1]


def test_card_identity_plain_names_unchanged(monkeypatch):
    out = "0, NVIDIA H100 80GB HBM3, 81559 MiB, 9.0, 1980 MHz\n1, NVIDIA A100-SXM4-40GB, 40960 MiB, 8.0, 1410 MHz\n"
    monkeypatch.setattr(gpu_query, "run_nvidia_smi", _smi(out))
    assert nvidia.get_card_identities() == {
        0: ("NVIDIA H100 80GB HBM3", "81559 MiB", "9.0", "1980 MHz"),
        1: ("NVIDIA A100-SXM4-40GB", "40960 MiB", "8.0", "1410 MHz"),
    }


def test_failed_mig_listing_backs_off(monkeypatch):
    calls = []
    monkeypatch.setattr(gpu_query, "run_nvidia_smi", _smi("", returncode = 9, calls = calls))
    monkeypatch.setattr(nvidia, "_mig_entry", None)
    for _ in range(3):
        assert hw._nvidia_mig_gpu_indices_or_none() is None
    assert sum(1 for c in calls if c == ["-L"]) == 1


def test_mig_listing_answer(monkeypatch):
    out = "GPU 0: NVIDIA H100 (UUID: GPU-0)\nGPU 1: NVIDIA H100 (UUID: GPU-1)\n  MIG 3g.40gb     Device  0: (UUID: MIG-1)\n"
    monkeypatch.setattr(gpu_query, "run_nvidia_smi", _smi(out))
    monkeypatch.setattr(gpu_query, "enabled", lambda: False)
    monkeypatch.setattr(nvidia, "_mig_entry", None)
    assert hw._nvidia_mig_gpu_indices_or_none() == {1}
