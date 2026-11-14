# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Frozen lookup tables (Qwen4Exp PLE n-gram table, in layer 1) under pipeline parallelism, 2 gloo stages, every
UNSLOTH_FROZEN_TABLE mode: only the stage owning layer 1 holds and reads the table (the other stage has no table
module and no open file), each micro-batch looks up its own rows (GPipe and 1F1B, also with checkpoint recompute,
whose memo must not hand one micro-batch's rows to another), the loader never reads a ram / disk table and the
adapter save holds no table key. See _frozen_table_parallel.py."""

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


def _pipelining():
    try:
        ftp.load_module("pp_native")._check_torch_pipelining()
    except RuntimeError as exc:
        pytest.skip(str(exc))


CELLS = {
    "gpu_gpipe": dict(table_mode = "gpu"),
    "ram_gpipe": dict(table_mode = "ram"),
    "disk_gpipe": dict(table_mode = "disk"),
    "disk_1f1b_recompute": dict(table_mode = "disk", schedule = "1f1b", gc = True),
    "ram_1f1b_recompute": dict(table_mode = "ram", schedule = "1f1b", gc = True),
}


@pytest.mark.parametrize("cell", sorted(CELLS))
def test_pp_tables(ckpt, cell):
    _pipelining()
    kw = CELLS[cell]
    res = ftp.run(2, mode = "pp", path = ckpt, **kw)
    assert [r["owns_table"] for r in res] == [True, False]
    assert ftp.violations(res, "pp", kw["table_mode"], ckpt) == []
    assert ftp.check_ids(res, ckpt, kw["table_mode"], "pp") == []
    if kw.get("gc"):
        # Forward + checkpoint recompute per micro-batch and step: every one checked against the file rows above.
        # The memo keeps one lookup, so under 1F1B a recompute (F0 F1 B0 B1) re-reads its own rows.
        assert len(res[0]["calls"]) == 2 * 2 * 2, len(res[0]["calls"])


def test_pp_catches_a_memo_that_crosses_micro_batches(ckpt):
    _pipelining()
    res = ftp.run(
        2, mode = "pp", path = ckpt, table_mode = "disk", schedule = "1f1b", gc = True, fault = "memo"
    )
    assert any("rows differ" in v for v in ftp.violations(res, "pp", "disk", ckpt))


def test_pp_catches_a_table_kept_on_the_other_stage(ckpt):
    _pipelining()
    res = ftp.run(2, mode = "pp", path = ckpt, table_mode = "disk", fault = "leak_table")
    assert any(
        v.startswith("rank 1") and "holds it" in v for v in ftp.violations(res, "pp", "disk", ckpt)
    )


@pytest.fixture(scope = "module")
def big(tmp_path_factory):
    return ftp.big_checkpoints(tmp_path_factory)


@pytest.mark.parametrize("table_mode", ["gpu", "ram", "disk"])
def test_pp_table_memory(big, table_mode):
    with_table, without_table = big
    _pipelining()
    a = ftp.run(2, mode = "pp", table_mode = table_mode, path = with_table, steps = 1)
    b = ftp.run(2, mode = "pp", table_mode = table_mode, path = without_table, steps = 1)
    assert ftp.memory_violations(a, b, table_mode, with_table) == []


@pytest.mark.parametrize("table_mode", ["ram", "disk"])
def test_pp_sharded_load_never_reads_the_table(ckpt, table_mode):
    _pipelining()
    res = ftp.run(2, mode = "pp", table_mode = table_mode, path = ckpt, sharded_load = True)
    refused = [r["load_refused"] for r in res if "load_refused" in r]
    if refused:
        # Sharded (meta-device) loading refuses this MoE model before reading anything: the full load (the tests
        # above, which prove the table is never read) is the only path. A future sharded MoE load runs the checks.
        assert len(refused) == len(res) and all(
            "experts" in m or "mixture-of-experts" in m for m in refused
        ), refused
        return
    assert ftp.violations(res, "pp", table_mode, ckpt) == []
    assert ftp.check_ids(res, ckpt, table_mode, "pp") == []
