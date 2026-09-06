# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Compare actual warm-path I/O against pre-contract code in an isolated process."""

import shutil
import uuid

import pytest

from .perf_utils import REPO, SCRATCH, baseline_ref, materialize_revision, run_probe


@pytest.fixture(scope = "module")
def measured_costs():
    ref = baseline_ref()
    if ref is None:
        pytest.skip("no pre-account baseline commit is reachable from this clone")
    scratch = SCRATCH / uuid.uuid4().hex
    scratch.mkdir(parents = True)
    try:
        baseline = materialize_revision(ref, scratch / "baseline")
        yield (
            run_probe(baseline, scratch / "baseline-home", mode = "cost"),
            run_probe(REPO / "studio/backend", scratch / "head-home", mode = "cost"),
        )
    finally:
        shutil.rmtree(scratch, ignore_errors = True)


@pytest.mark.parametrize("operation,connections,queries,mkdir_calls", [
    ("status", 3, 3, 3), ("authenticated_get", 1, 1, 1), ("workspace_1000", 0, 0, 0),
])
def test_one_account_adds_no_hot_path_io(measured_costs, operation, connections, queries, mkdir_calls):
    baseline, head = measured_costs
    measured = baseline[operation]
    assert measured["connections"] == connections
    assert measured["queries"] == queries
    assert measured["mkdir_calls"] == mkdir_calls
    assert measured["directories_created"] == 0
    assert head[operation] == measured, {"baseline": measured, "head": head[operation]}
