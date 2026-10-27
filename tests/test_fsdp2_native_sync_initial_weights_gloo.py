# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 starts every rank from the same trainable weights, on a gloo CPU world of 2.

DDP broadcasts rank 0's parameters when it wraps the model. fully_shard keeps each rank's own slice,
so an adapter initialised without a common seed on each rank (FastSentenceTransformer.get_peft_model
ignored random_state) was sharded as a mix of both ranks' values: rank 0 held its half of the LoRA
matrices, rank 1 its own, and the step-0 gradient norm under FSDP2 was 12% off one process (DDP: 2%).
shard_model now checks a checksum per trainable tensor and broadcasts the first rank's values of the
ones that differ."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port, _full  # noqa: E402
from test_fsdp2_native_trainer_args_gloo import _lora  # noqa: E402


def _worker(rank, world, port, out, case):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
    )
    import datetime
    import io
    from contextlib import redirect_stdout

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 120)
    )
    try:
        native = _load_native()
        # Same base weights everywhere; the adapters differ per rank unless the case says otherwise.
        seed = 1 + rank if case == "differ" else 1
        model = _lora(seed)
        first = _lora(1)  # rank 0's starting adapters, built locally for the comparison
        if case == "nan":
            # Identical non-finite weights on every rank (tiny Llama4 ships some) are not a mismatch.
            for m in (model, first):
                next(p for n, p in m.named_parameters() if p.requires_grad).data.view(-1)[0] = (
                    float("nan")
                )
        if case in ("permuted", "nan_elsewhere") and rank == 1:
            # Same sum and L1 norm as rank 0, different contents: the elements reversed.
            w = next(p for n, p in model.named_parameters() if p.requires_grad).data.view(-1)
            w.copy_(w.flip(0))
        if case == "nan_elsewhere":
            # A shared NaN must not hide a difference in the other elements.
            for m in (model, first):
                next(p for n, p in m.named_parameters() if p.requires_grad).data.view(-1)[0] = (
                    float("nan")
                )
        want = {n: p.detach().clone() for n, p in first.named_parameters() if p.requires_grad}
        buf = io.StringIO()
        with redirect_stdout(buf):
            native.shard_model(model, native.FSDP2Config(param_dtype = torch.float32, device = "cpu"))
        got = _full(model, trainable_only = True)
        res = {
            "n": len(want),
            "max_diff": max((got[n] - want[n]).nan_to_num(0.0).abs().max().item() for n in want),
            "message": "differed across ranks" in buf.getvalue(),
        }
        out.put((rank, res))
    except Exception as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        dist.destroy_process_group()


def _run(case):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    mp.start_processes(
        _worker, args = (2, _free_port(), out, case), nprocs = 2, join = True, start_method = "spawn"
    )
    results = {}
    while not out.empty():
        rank, r = out.get(timeout = 5)
        results[rank] = r
    assert sorted(results) == [0, 1], results
    for rank, r in results.items():
        print("SYNC_RESULT", case, rank, r)
        assert "crash" not in r, r["crash"]
    return results


def test_ranks_with_different_adapters_start_from_the_first_ranks():
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    results = _run("differ")
    for rank, r in results.items():
        assert r["n"] > 0, r
        assert r["max_diff"] == 0.0, (rank, r)  # rank 0's adapters, bit for bit, on both ranks
    assert results[0]["message"] and not results[1]["message"], results  # said once, on rank 0


def test_ranks_that_already_agree_are_left_alone():
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    for rank, r in _run("same").items():
        assert r["max_diff"] == 0.0, (rank, r)
        assert not r["message"], (rank, r)


def test_identical_non_finite_weights_are_not_a_mismatch():
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    for rank, r in _run("nan").items():
        assert r["max_diff"] == 0.0, (rank, r)
        assert not r["message"], (rank, r)


def test_a_permuted_adapter_is_not_mistaken_for_the_same():
    for rank, r in _run("permuted").items():
        assert r["max_diff"] == 0.0, (rank, r)
        assert r["message"] or rank != 0, (rank, r)


def test_a_shared_nan_does_not_hide_other_differences():
    for rank, r in _run("nan_elsewhere").items():
        assert r["max_diff"] == 0.0, (rank, r)
        assert r["message"] or rank != 0, (rank, r)
