# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""tests/conftest.py gives each xdist worker one share of the CPUs for torch.

Without it every worker's torch pool is as wide as the machine, and CPU training tests
thrash each other on CI's 4 vCPU runners (test_decision_model's Clef training went past
the 330 s timeout). Only meaningful under xdist, so a serial run skips it.
"""

import os

import pytest


def test_each_xdist_worker_gets_one_cpu_share():
    workers = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT") or 0)
    if workers <= 1:
        pytest.skip("not running under pytest-xdist")
    import torch

    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    share = max(1, cpus // workers)
    assert os.environ.get("OMP_NUM_THREADS"), "conftest did not set OMP_NUM_THREADS"
    if os.environ["OMP_NUM_THREADS"] == str(share):
        assert torch.get_num_threads() == share
