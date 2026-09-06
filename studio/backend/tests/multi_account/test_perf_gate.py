# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Prove the performance gate and I/O counter fail for actual extra work."""

import copy
import importlib.util
import os
import sqlite3
from contextlib import closing

import pytest

from .perf_utils import PROBE


def load_script(name):
    path = PROBE.with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("account_perf_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("endpoint,metric", [
    ("status", "p50_ms"), ("status", "p95_ms"),
    ("history", "p50_ms"), ("history", "p95_ms"),
])
def test_gate_rejects_each_percentile_above_five_percent(endpoint, metric):
    module = load_script("compare")
    baseline = {name: {"p50_ms": 10.0, "p95_ms": 20.0} for name in ("status", "history")}
    results = {"base": baseline, "head": copy.deepcopy(baseline)}
    results["head"][endpoint][metric] = baseline[endpoint][metric] * 1.05
    assert module.regressions(results) == []
    results["head"][endpoint][metric] += 0.0001
    assert len(module.regressions(results)) == 1


def test_percentiles_use_all_samples():
    module = load_script("probe")
    assert module.summarize(list(range(1, 101))) == {"samples": 100, "p50_ms": 50.5, "p95_ms": 95}


def test_io_counter_observes_real_connections_queries_and_mkdirs(tmp_path):
    module = load_script("probe")

    def extra_work():
        os.mkdir(tmp_path / "new-directory")
        with closing(sqlite3.connect(tmp_path / "extra.db")) as conn:
            assert conn.execute("SELECT 1").fetchone() == (1,)

    assert module.measure_cost(extra_work) == {
        "connections": 1, "queries": 1, "statements": 1,
        "mkdir_calls": 1, "directories_created": 1,
    }


def test_io_counter_distinguishes_existing_directory_attempts(tmp_path):
    module = load_script("probe")
    result = module.measure_cost(lambda: tmp_path.mkdir(exist_ok = True))
    assert result["mkdir_calls"] == 1
    assert result["directories_created"] == 0
