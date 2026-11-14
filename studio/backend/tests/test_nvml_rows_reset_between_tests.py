# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""The NVML inventory a probe test leaves behind does not pin the next test's launch.

``LlamaCppBackend._NVML_ROWS`` is the last NVML inventory, class state that
``_child_visibility_for`` maps indices through. A MIG probe test in
test_gpu_memory_nvml_fallback.py left its slices there, and placement tests that later ran on
the same xdist worker with CUDA_VISIBLE_DEVICES unset launched on ``MIG-dddd4444-0`` instead
of ``0``. The two tests below run in order on one worker: the first leaves rows behind the way
a real probe does, the second fails unless conftest forgot them in between.
"""

from __future__ import annotations

import pytest

from core.inference.llama_cpp import LlamaCppBackend

pytestmark = pytest.mark.xdist_group("nvml_rows_reset_between_tests")


def _mig_row(uuid, free):
    return {
        "index": "0",
        "uuid": uuid,
        "mig": "1",
        "memory_total_mib": "20480",
        "memory_free_mib": str(free),
    }


def test_a_probe_leaves_its_mig_inventory_behind(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    # Assigned, not monkeypatched: _get_gpu_memory writes the class attribute the same way.
    LlamaCppBackend._NVML_ROWS = [
        {
            "index": "0",
            "uuid": "GPU-aaaa1111-0",
            "memory_total_mib": "81920",
            "memory_free_mib": "60000",
        },
        _mig_row("MIG-cccc3333-0", 9000),
        _mig_row("MIG-dddd4444-0", 4000),
    ]
    assert LlamaCppBackend._child_visibility_for([0]) == "MIG-dddd4444-0"


def test_the_next_test_starts_without_that_inventory(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    assert LlamaCppBackend._NVML_ROWS == []
    assert LlamaCppBackend._child_visibility_for([0]) == "0"
