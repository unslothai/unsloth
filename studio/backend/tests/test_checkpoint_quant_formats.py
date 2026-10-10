# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Hub's Recommended filter lists FP8 / NVFP4 checkpoints only where every card runs them
natively. Hermetic: nvidia-smi is stubbed."""

from __future__ import annotations

import subprocess
import types

import pytest
from utils.hardware import nvidia


def _smi(
    monkeypatch,
    stdout: str = "",
    returncode: int = 0,
    raises = None,
) -> list:
    calls: list = []

    def fake_run(argv, **_kwargs):
        calls.append(argv)
        if raises is not None:
            raise raises
        return types.SimpleNamespace(returncode = returncode, stdout = stdout)

    monkeypatch.setattr(nvidia.gpu_query, "run_nvidia_smi", fake_run)
    return calls


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("8.6\n", []),  # RTX 3090
        ("8.9\n", ["fp8"]),  # RTX 4090
        ("9.0\n", ["fp8"]),  # H100
        ("10.0\n", ["fp8", "nvfp4"]),  # B200
        ("12.0\n", ["fp8", "nvfp4"]),  # RTX 5090
        ("12.0\n8.9\n", ["fp8"]),  # mixed host answers for the least capable card
        ("12.0\n8.0\n", []),
        ("12.0\n[N/A]\n", []),  # an unreadable card counts as none
        ("", []),
    ],
)
def test_formats_follow_the_least_capable_card(monkeypatch, stdout, expected):
    calls = _smi(monkeypatch, stdout)
    assert nvidia.get_checkpoint_quant_formats() == expected
    assert "--query-gpu=compute_cap" in calls[0]


def test_no_nvidia_smi_or_a_failed_query_lists_nothing(monkeypatch):
    _smi(monkeypatch, raises = FileNotFoundError("nvidia-smi"))
    assert nvidia.get_checkpoint_quant_formats() == []
    _smi(monkeypatch, raises = subprocess.TimeoutExpired("nvidia-smi", 5))
    assert nvidia.get_checkpoint_quant_formats() == []
    _smi(monkeypatch, stdout = "9.0\n", returncode = 6)
    assert nvidia.get_checkpoint_quant_formats() == []


def test_compute_cap_is_a_cached_static_query():
    assert (
        nvidia.gpu_query.classify(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
        )
        == nvidia.gpu_query.STATIC
    )
