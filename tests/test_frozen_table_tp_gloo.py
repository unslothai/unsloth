# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Frozen lookup tables under tensor parallelism on 2 gloo ranks, every UNSLOTH_FROZEN_TABLE mode."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys  # noqa: E402

import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _frozen_table_parallel as ftp  # noqa: E402

_why = ftp.available()
pytestmark = pytest.mark.skipif(_why is not None, reason = _why or "")


@pytest.fixture(scope = "module")
def ckpt(tmp_path_factory):
    return ftp.make_checkpoint(str(tmp_path_factory.mktemp("qwen4exp_table")))


@pytest.mark.parametrize("table_mode", ["gpu", "ram", "disk"])
def test_tp_tables(ckpt, table_mode):
    res = ftp.run(2, mode = "tp", table_mode = table_mode, path = ckpt)
    assert ftp.violations(res, "tp", table_mode, ckpt) == []
    assert ftp.check_ids(res, ckpt, table_mode, "tp") == []
    a, b = res[0]["calls"], res[1]["calls"]
    assert len(a) == len(b)
    for (_, ids0, rows0), (_, ids1, rows1) in zip(a, b):
        assert ftp.torch.equal(ids0, ids1) and ftp.torch.equal(rows0, rows1)


def test_tp_catches_wrong_rows_on_one_rank(ckpt):
    res = ftp.run(2, mode = "tp", table_mode = "disk", path = ckpt, fault = "rows")
    assert any(
        v.startswith("rank 1") and "rows differ" in v
        for v in ftp.violations(res, "tp", "disk", ckpt)
    )


@pytest.fixture(scope = "module")
def big(tmp_path_factory):
    return ftp.big_checkpoints(tmp_path_factory)


@pytest.mark.parametrize("table_mode", ["gpu", "ram", "disk"])
def test_tp_table_memory(big, table_mode):
    with_table, without_table = big
    a = ftp.run(2, mode = "tp", table_mode = table_mode, path = with_table, steps = 1)
    b = ftp.run(2, mode = "tp", table_mode = table_mode, path = without_table, steps = 1)
    assert ftp.memory_violations(a, b, table_mode, with_table) == []


@pytest.mark.parametrize("table_mode", ["ram", "disk"])
def test_tp_sharded_load_never_reads_the_table(ckpt, table_mode):
    res = ftp.run(2, mode = "tp", table_mode = table_mode, path = ckpt, sharded_load = True)
    refused = [r["load_refused"] for r in res if "load_refused" in r]
    if refused:
        # Sharded load refuses this MoE model; future sharded MoE loads will run the checks.
        assert len(refused) == len(res) and all(
            "experts" in m or "mixture-of-experts" in m for m in refused
        ), refused
        return
    assert ftp.violations(res, "tp", table_mode, ckpt) == []
    assert ftp.check_ids(res, ckpt, table_mode, "tp") == []
