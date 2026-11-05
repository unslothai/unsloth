# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""A blank CUDA_VISIBLE_DEVICES hides every GPU, so the NVML memory probe answers no rows without a child."""

from __future__ import annotations

import json

import pytest

from core.inference import llama_cpp as mod
from core.inference.llama_cpp import LlamaCppBackend


def _payload():
    rows = [
        {
            "index": str(i),
            "uuid": f"GPU-{i:04d}",
            "name": "NVIDIA test",
            "compute_cap": "8.9",
            "memory_total_mib": "24576",
            "memory_free_mib": str(8000 + i),
        }
        for i in range(2)
    ]
    rows.append({**rows[0], "mig": True, "uuid": "MIG-0000"})
    return {
        "source": "nvml",
        "cuda_driver_version": [13, 0],
        "driver_version": "580.65.06",
        "devices": rows,
    }


@pytest.fixture
def probe(tmp_path, monkeypatch):
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(_payload()), encoding = "utf-8")
    calls = tmp_path / "calls"
    script = tmp_path / "nvidia_probe.py"
    script.write_text(
        f"import sys; open({str(calls)!r}, 'a').write('x');"
        f" sys.stdout.write(open({str(payload_path)!r}).read())\n",
        encoding = "utf-8",
    )
    monkeypatch.setenv("UNSLOTH_NVIDIA_PROBE", str(script))
    monkeypatch.delenv("UNSLOTH_NVIDIA_LIBRARY_PROBE", raising = False)
    monkeypatch.setattr(mod.sys, "platform", "linux")
    monkeypatch.setattr(LlamaCppBackend, "_NVML_ROWS", [])
    return lambda: len(calls.read_text()) if calls.exists() else 0


@pytest.mark.parametrize("mask", ["", " ", "\t", " \n "])
def test_a_blank_mask_answers_no_rows_without_a_child(monkeypatch, probe, mask):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    for _ in range(3):
        assert LlamaCppBackend._get_gpu_memory_nvml() == []
    assert probe() == 0
    assert LlamaCppBackend._child_visibility_for([0, 1]) == "0,1"


@pytest.mark.parametrize(
    "mask, expected",
    [
        (None, [(0, 8000, 24576), (1, 8001, 24576)]),
        ("1", [(1, 8001, 24576)]),
        ("-1", []),
        (",0", []),
        ("  ,  ", []),
        ("GPU-0001", [(1, 8001, 24576)]),
    ],
)
def test_any_other_mask_still_runs_the_probe(monkeypatch, probe, mask, expected):
    if mask is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    assert LlamaCppBackend._get_gpu_memory_nvml() == expected
    assert probe() == 1
    assert LlamaCppBackend._NVML_ROWS == _payload()["devices"]


def test_the_probe_is_never_reached_on_macos_or_when_disabled(monkeypatch, probe):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    monkeypatch.setattr(mod.sys, "platform", "darwin")
    assert LlamaCppBackend._get_gpu_memory_nvml() == []
    monkeypatch.setattr(mod.sys, "platform", "linux")
    monkeypatch.setenv("UNSLOTH_NVIDIA_LIBRARY_PROBE", "0")
    assert LlamaCppBackend._get_gpu_memory_nvml() == []
    assert probe() == 0
