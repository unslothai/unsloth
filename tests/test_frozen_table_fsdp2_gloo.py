# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Frozen lookup tables under native FSDP2 on 2 gloo ranks, every UNSLOTH_FROZEN_TABLE mode."""

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


def _fake_table_model():
    nn = ftp.torch.nn

    class UnslothFrozenTableEmbedding(
        nn.Module
    ):  # what fsdp2_native recognises: name + a bound store
        frozen_table_store = object()

        def _save_to_state_dict(self, destination, prefix, keep_vars):
            destination[prefix + "shard_0.weight"] = ftp.torch.zeros(1)

    model = nn.Module()
    model.proj = nn.Linear(2, 2)
    model.table = UnslothFrozenTableEmbedding()
    return model


@pytest.mark.parametrize(
    "rank, writer, cpu_offload, keys",
    [
        (0, 0, True, {"proj", "table.shard_0.weight"}),
        (1, 0, True, set()),  # not the writer: no table-only state dict
        (2, 2, True, {"proj", "table.shard_0.weight"}),  # a sub-mesh writer gets the tables too
        (3, 2, True, set()),
        (
            1,
            0,
            False,
            {"proj", "table.shard_0.weight"},
        ),  # no cpu_offload: the full state on every rank
    ],
)
def test_full_export_puts_the_tables_where_the_full_state_is(
    monkeypatch, rank, writer, cpu_offload, keys
):
    """native_full_state_dict places the frozen tables where the full state dict goes."""
    import torch.distributed as dist
    import torch.distributed.checkpoint.state_dict as dsd

    native = ftp.load_module("fsdp2_native")
    fsave = ftp.load_module("fsdp2_save")

    def get_model_state_dict(model, options):
        full = options.full_state_dict and (rank == 0 or not options.cpu_offload)
        return {"proj": ftp.torch.zeros(2)} if full or not options.full_state_dict else {}

    monkeypatch.setattr(dsd, "get_model_state_dict", get_model_state_dict)
    monkeypatch.setattr(fsave, "mesh_writer_rank", lambda model: writer)
    monkeypatch.setattr(native, "_gather_to_writer", lambda state, w: state if rank == w else {})
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda group = None: rank)
    state = native.native_full_state_dict(_fake_table_model(), cpu_offload = cpu_offload)
    assert set(state) == keys


def test_full_export_table_keys_drop_compile_wrappers():
    """A compiled model (or table owner) has `_orig_mod.` in its module names; the state dict keys do not."""
    native = ftp.load_module("fsdp2_native")
    outer = ftp.torch.nn.Module()
    outer._orig_mod = _fake_table_model()
    assert set(native.full_frozen_table_state(outer)) == {"table.shard_0.weight"}


@pytest.mark.parametrize("mode", ["fsdp2", "tp"])
def test_setup_alone_keeps_tables_out_of_dcp_without_bitsandbytes(ckpt, mode):
    """16-bit LoRA FSDP2 / TP without bitsandbytes or DCP install: state dicts hold no table key."""
    res = ftp.run(
        2, mode = mode, table_mode = "ram", path = ckpt, steps = 1, no_bitsandbytes = True, install_dcp = False
    )
    for r in res:
        assert "dcp_full_error" not in r and "dcp_adapter_error" not in r, r
        assert r["dcp_full_table_keys"] == 0 and r["dcp_adapter_table_keys"] == 0
        assert r["raw_dcp_error"] == "AttributeError"
