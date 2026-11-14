# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Frozen lookup tables (Qwen4Exp PLE n-gram table) under native FSDP2, 2 gloo ranks, the shard layout (frozen
weights sharded), every UNSLOTH_FROZEN_TABLE mode. The table is never sharded, all-gathered or reduced (gpu: kept
whole by ignored_params; ram / disk: no parameter at all), the loader never reads it (ram / disk), each rank's
lookups are its own rows of the file, and the adapter save holds no table key. See _frozen_table_parallel.py."""

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
def test_fsdp2_tables(ckpt, table_mode):
    res = ftp.run(2, mode = "fsdp2", table_mode = table_mode, path = ckpt)
    assert ftp.violations(res, "fsdp2", table_mode, ckpt) == []
    assert ftp.check_ids(res, ckpt, table_mode, "fsdp2") == []
    if table_mode in ("ram", "disk"):
        # Torch alone fails on the table's checkpoint names (what Unsloth's DCP wrappers prevent); a full save
        # still writes the table, as zero-copy views read from its own files on the writer.
        assert all(r["raw_dcp_error"] == "AttributeError" for r in res)
        assert res[0]["full_save_table_keys"] > 0
    # Each data parallel rank looked up its own rows: different batches, different ids.
    assert not ftp.torch.equal(res[0]["calls"][0][1], res[1]["calls"][0][1])


def test_fsdp2_would_shard_a_gpu_table_without_the_ignore(ckpt):
    # Flip: without keeping the table out of sharding, the shard layout makes it a DTensor.
    res = ftp.run(2, mode = "fsdp2", table_mode = "gpu", path = ckpt, fault = "shard_table")
    assert any("DTensor" in v for v in ftp.violations(res, "fsdp2", "gpu", ckpt))


def test_fsdp2_catches_wrong_rows(ckpt):
    res = ftp.run(2, mode = "fsdp2", table_mode = "disk", path = ckpt, fault = "rows")
    assert any("rows differ" in v for v in ftp.violations(res, "fsdp2", "disk", ckpt))


@pytest.fixture(scope = "module")
def big(tmp_path_factory):
    return ftp.big_checkpoints(tmp_path_factory)


@pytest.mark.parametrize("table_mode", ["gpu", "ram", "disk"])
def test_fsdp2_table_memory(big, table_mode):
    with_table, without_table = big
    a = ftp.run(2, mode = "fsdp2", table_mode = table_mode, path = with_table, steps = 1)
    b = ftp.run(2, mode = "fsdp2", table_mode = table_mode, path = without_table, steps = 1)
    assert ftp.memory_violations(a, b, table_mode, with_table) == []


@pytest.mark.parametrize("table_mode", ["ram", "disk"])
def test_fsdp2_sharded_load_never_reads_the_table(ckpt, table_mode):
    res = ftp.run(2, mode = "fsdp2", table_mode = table_mode, path = ckpt, sharded_load = True)
    refused = [r["load_refused"] for r in res if "load_refused" in r]
    if refused:
        # Sharded (meta-device) loading refuses this MoE model before reading anything: the full load (the tests
        # above, which prove the table is never read) is the only path. A future sharded MoE load runs the checks.
        assert len(refused) == len(res) and all(
            "experts" in m or "mixture-of-experts" in m for m in refused
        ), refused
        return
    assert ftp.violations(res, "fsdp2", table_mode, ckpt) == []
    assert ftp.check_ids(res, ckpt, table_mode, "fsdp2") == []
