# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""bfloat16 stochastic rounding on DTensor weights (FSDP2 / TP shards), world 2 (gloo on CPU, NCCL on 2 GPUs).

Before, every DTensor weight was skipped, so FSDP2 / TP bfloat16 full finetuning silently fell back to
round-to-nearest and lost every update below half a bfloat16 spacing. The update and the rounding are
elementwise, so each rank now rounds its local shard. Noise: the seed also mixes in the shard's global start
(0 for a plain weight, so the first shard draws exactly what one process draws for those elements), so
different shards draw independent noise and replicas of one shard round identically."""

import hashlib
import importlib.util
import os
import sys
import traceback
from pathlib import Path

import pytest
import torch

ROOT = str(Path(__file__).resolve().parents[1])
STEPS = 8
LR = 1e-3  # under half the bfloat16 spacing below 1.0 (2**-9): round-to-nearest never moves a 1.0 weight


def _init(
    rank,
    world,
    port,
    backend = "gloo",
):
    import datetime

    import torch.distributed as dist

    os.environ.update(
        MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port), RANK = str(rank), WORLD_SIZE = str(world)
    )
    dist.init_process_group(
        backend, rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 120)
    )
    return dist


def _spawn(worker, *args):
    import torch.multiprocessing as mp

    sys.path.insert(0, ROOT)
    from tests._rendezvous_port import free_port

    out = mp.get_context("spawn").Queue()
    mp.start_processes(
        worker, args = (2, free_port(), out, *args), nprocs = 2, join = True, start_method = "spawn"
    )
    results = []
    while not out.empty():
        results.append(out.get(timeout = 5))
    return results


def _load(root):
    path = os.path.join(root, "unsloth/optimizers/bf16_stochastic_rounding.py")
    spec = importlib.util.spec_from_file_location("_test_bf16_sr_dtensor", path)
    sr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sr)
    assert os.path.realpath(sr.__file__).startswith(os.path.realpath(root)), sr.__file__
    return sr


def _digest(t):
    return hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]


def _inputs(device):
    g = torch.Generator().manual_seed(1)
    half = torch.randn(32, 128, generator = g)
    # Both row halves (the two Shard(0) shards) see the same gradient: identical inputs, so only the noise differs.
    return {
        "row": (torch.ones(64, 128), torch.cat([half, half])),
        "col": (torch.ones(64, 128), torch.randn(64, 128, generator = g)),
        "rep": (torch.ones(128), torch.randn(128, generator = g)),
    }, device


def _dtensor_worker(rank, world, port, out, root, device, fused, sr_env):
    try:
        os.environ["UNSLOTH_BF16_SR_FUSED"] = fused
        if sr_env is None:
            os.environ.pop("UNSLOTH_BF16_STOCHASTIC_ROUNDING", None)
        else:
            os.environ["UNSLOTH_BF16_STOCHASTIC_ROUNDING"] = sr_env
        if device == "cuda":
            torch.cuda.set_device(rank)  # DTensor collectives on CUDA need NCCL: one GPU per rank
        dist = _init(rank, world, port, "nccl" if device == "cuda" else "gloo")
        from torch.distributed.device_mesh import init_device_mesh

        try:
            from torch.distributed.tensor import Replicate, Shard, distribute_tensor
        except ImportError:  # torch < 2.5
            from torch.distributed._tensor import Replicate, Shard, distribute_tensor
        sr = _load(root)
        mesh = init_device_mesh(device, (world,))
        data, _ = _inputs(device)
        placements = {"row": [Shard(0)], "col": [Shard(1)], "rep": [Replicate()]}

        def bf16(t):
            return t.to(device = device, dtype = torch.bfloat16)

        params = {
            k: torch.nn.Parameter(distribute_tensor(bf16(w), mesh, placements[k]))
            for k, (w, _) in data.items()
        }
        grads = {k: distribute_tensor(bf16(g), mesh, placements[k]) for k, (_, g) in data.items()}
        opt = torch.optim.AdamW(list(params.values()), lr = LR, weight_decay = 0.0)
        covered = sr.enable_bf16_stochastic_rounding(opt, seed = 7, verbose = False)

        # One process, plain bfloat16 weights, same seed: what this rounding does without sharding.
        single = {k: torch.nn.Parameter(bf16(w)) for k, (w, _) in data.items()}
        single_opt = torch.optim.AdamW(list(single.values()), lr = LR, weight_decay = 0.0)
        sr.enable_bf16_stochastic_rounding(single_opt, seed = 7, verbose = False)
        # float32 AdamW: the exact update the rounding brackets.
        exact = {k: torch.nn.Parameter(w.to(device).float()) for k, (w, _) in data.items()}
        exact_opt = torch.optim.AdamW(list(exact.values()), lr = LR, weight_decay = 0.0)

        result = {"covered": covered, "rank": rank}
        # Seed index = position in the param groups (resume-stable), plus where this rank's shard starts.
        order = [opt.__dict__.get("_unsloth_sr_index", {}).get(id(p)) for p in params.values()]
        result["index_order"] = order
        result["seeds"] = [sr._seed_for(opt, p, 7, 1) for p in params.values()] if covered else []
        result["single_seeds"] = (
            [sr._seed_for(single_opt, p, 7, 1) for p in single.values()] if covered else []
        )
        # A graph saved before the step must refuse to backpropagate through the stepped weight.
        stale = (params["row"] * params["row"]).sum()
        for step in range(STEPS):
            for k in params:
                params[k].grad = grads[k]
                single[k].grad = bf16(data[k][1])
                exact[k].grad = bf16(data[k][1]).float()
            opt.step()
            single_opt.step()
            exact_opt.step()
            if step == 0:
                try:
                    stale.backward()
                    result["stale_graph_refused"] = False
                except RuntimeError as exc:
                    result["stale_graph_refused"] = "modified by an inplace operation" in str(exc)
                first = {}
                for k, p in params.items():
                    full = p.full_tensor().float()
                    # The two bfloat16 neighbours of the float32 result (weights stay positive).
                    bits = exact[k].detach().float().contiguous().view(torch.int32) & -65536
                    lo, hi = bits.view(torch.float32), (bits + 65536).view(torch.float32)
                    first[k] = {
                        "bracketed": bool(((full == lo) | (full == hi)).all().item()),
                        "moved": float((full != 1).float().mean().item()),
                    }
                row_local = params["row"]._local_tensor.float()
                first["row_rank0_equals_single"] = (
                    bool(torch.equal(row_local, single["row"].detach()[:32].float()))
                    if rank == 0
                    else None
                )
                result["first"] = first
                result["row_local_digest"] = _digest(row_local)
        result["rep_digest"] = _digest(params["rep"]._local_tensor)
        # Signed along the descent direction: Adam on a constant gradient moves each weight about LR per step.
        sign = {k: -torch.sign(g).to(device) for k, (_, g) in data.items()}

        def descent(k, w):
            return float(((w.float() - 1) * sign[k]).mean().item())

        result["mean_change"] = {k: descent(k, p.full_tensor()) for k, p in params.items()}
        result["exact_mean_change"] = {k: descent(k, p.detach()) for k, p in exact.items()}
        result["single_mean_change"] = {k: descent(k, p.detach()) for k, p in single.items()}
        result["dtensor_rounds"] = opt.__dict__.get("_unsloth_sr_dtensor_rounds", 0)
        result["fused_steps"] = opt.__dict__.get("_unsloth_sr_fused_steps", 0)
        result["dtype_kept"] = all(
            p.dtype == torch.bfloat16 and p._local_tensor.dtype == torch.bfloat16
            for p in params.values()
        )
        result["state_dtensor"] = all(
            type(opt.state[p]["exp_avg"]).__name__ == "DTensor" for p in params.values()
        )
        gathered = [None] * world
        dist.all_gather_object(gathered, result)
        if rank == 0:
            out.put({"ranks": gathered})
        dist.destroy_process_group()
    except Exception:
        out.put({"error": traceback.format_exc()[-3000:]})


def _module_worker(rank, world, port, out, root, layout):
    """Real FSDP2 (fully_shard) / TP (parallelize_module) on CPU: every bfloat16 weight is a DTensor."""
    try:
        os.environ.pop("UNSLOTH_BF16_STOCHASTIC_ROUNDING", None)
        dist = _init(rank, world, port)
        from torch.distributed.device_mesh import init_device_mesh

        sr = _load(root)
        mesh = init_device_mesh("cpu", (world,))
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(128, 64), torch.nn.Linear(64, 128))
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(1.0)
        model = model.to(torch.bfloat16)
        if layout == "fsdp2":
            try:
                from torch.distributed.fsdp import fully_shard
            except ImportError:
                from torch.distributed._composable.fsdp import fully_shard
            for layer in model:
                fully_shard(layer, mesh = mesh)
            fully_shard(model, mesh = mesh)
        else:
            from torch.distributed.tensor.parallel import (
                ColwiseParallel,
                RowwiseParallel,
                parallelize_module,
            )
            parallelize_module(model, mesh, {"0": ColwiseParallel(), "1": RowwiseParallel()})
        params = list(model.parameters())
        opt = torch.optim.AdamW(params, lr = LR, weight_decay = 0.0)
        covered = sr.enable_bf16_stochastic_rounding(opt, seed = 7, verbose = False)
        g = torch.Generator().manual_seed(100)  # same data on both ranks (TP needs it)
        for step in range(STEPS):
            x = torch.randn(8, 128, generator = g).to(torch.bfloat16)
            try:
                model(x).float().pow(2).mean().backward()
            except AttributeError as exc:
                # Older torch's FSDP2 needs accelerator streams (no CPU support); every rank fails alike.
                if step == 0 and layout == "fsdp2" and "record_event" in str(exc):
                    if rank == 0:
                        out.put({"skip": f"FSDP2 on CPU unsupported by torch {torch.__version__}"})
                    dist.destroy_process_group()
                    return
                raise
            opt.step()
            opt.zero_grad(set_to_none = True)
        moved = []
        for p in params:
            full = p.full_tensor() if hasattr(p, "full_tensor") else p
            moved.append(float((full.float() != 1).float().mean().item()))
        out_r = {
            "covered": covered,
            "n_params": len(params),
            "dtensor": [type(p).__name__ == "DTensor" for p in params],
            "dtensor_rounds": opt.__dict__.get("_unsloth_sr_dtensor_rounds", 0),
            "moved": moved,
        }
        gathered = [None] * world
        dist.all_gather_object(gathered, out_r)
        if rank == 0:
            out.put({"ranks": gathered})
        dist.destroy_process_group()
    except Exception:
        out.put({"error": traceback.format_exc()[-3000:]})


def _run(worker, *args):
    results = _spawn(worker, ROOT, *args)
    assert len(results) == 1 and "error" not in results[0], results
    if "skip" in results[0]:
        pytest.skip(results[0]["skip"])
    return results[0]["ranks"]


CASES = [
    pytest.param("cpu", "1", id = "cpu-float32-route"),
    pytest.param("cuda", "1", id = "cuda-one-pass", marks = pytest.mark.gpu),
    pytest.param("cuda", "0", id = "cuda-float32-route", marks = pytest.mark.gpu),
]


@pytest.mark.parametrize("device, fused", CASES)
def test_dtensor_shards_round_stochastically(device, fused):
    if device == "cuda":
        if torch.cuda.device_count() < 2:
            pytest.skip("requires 2 CUDA devices (NCCL, one per rank)")
        if not torch.distributed.is_nccl_available():
            pytest.skip("requires NCCL")
        pytest.importorskip("triton")
    ranks = _run(_dtensor_worker, device, fused, None)
    r0, r1 = ranks
    for r in ranks:
        # Engaged on all three DTensor weights, every step (one-pass kernel or float32 route).
        assert r["covered"] == 3, r
        assert r["dtensor_rounds"] == 3 * STEPS, r
        if device == "cuda" and fused == "1":
            assert r["fused_steps"] == 3 * STEPS, r
        assert r["dtype_kept"] and r["state_dtensor"], r
        assert r["stale_graph_refused"], r
        for k, f in r["first"].items():
            if isinstance(f, dict):
                # Each element is one of the two bfloat16 neighbours of the exact float32 AdamW result.
                assert f["bracketed"], (k, f)
                # Round-to-nearest leaves every 1.0 weight in place; rounding stochastically moves about
                # LR / 2**-8 of them on the first step.
                assert 0.15 < f["moved"] < 0.4, (k, f)
        for k, change in r["mean_change"].items():
            want = r["exact_mean_change"][k]
            # Unbiased: the mean move tracks float32 AdamW (about STEPS * LR) as one process's does.
            assert want > 0.5 * STEPS * LR, (k, want)
            assert abs(change - want) < 0.25 * abs(want), (k, change, want)
            assert abs(r["single_mean_change"][k] - want) < 0.25 * abs(want), (
                k,
                r["single_mean_change"],
            )
    # Rank 0's shard starts at element 0: it draws exactly what one process draws there.
    if device == "cpu" or fused == "0":
        if device == "cpu":
            assert r0["first"]["row_rank0_equals_single"], r0["first"]
    # Indices follow the param groups on every rank. The shard starting at element 0 and the replicated
    # weight seed exactly as one process does; rank 1's shards of `row` / `col` seed differently.
    assert r0["index_order"] == r1["index_order"] == [0, 1, 2], (
        r0["index_order"],
        r1["index_order"],
    )
    assert r0["seeds"] == r0["single_seeds"], r0["seeds"]
    assert r1["seeds"][2] == r1["single_seeds"][2]
    assert r1["seeds"][0] != r0["seeds"][0] and r1["seeds"][1] != r0["seeds"][1]
    # Two shards with identical inputs draw different noise; replicas stay bitwise equal.
    assert r0["row_local_digest"] != r1["row_local_digest"]
    assert r0["rep_digest"] == r1["rep_digest"]


def test_opt_out_keeps_round_to_nearest_on_dtensors():
    ranks = _run(_dtensor_worker, "cpu", "1", "0")
    for r in ranks:
        assert r["covered"] == 0 and r["dtensor_rounds"] == 0, r
        assert r["stale_graph_refused"], r
        assert all(f["moved"] == 0.0 for f in r["first"].values() if isinstance(f, dict)), r
        assert all(change == 0.0 for change in r["mean_change"].values()), r
        assert all(change > 0.5 * STEPS * LR for change in r["exact_mean_change"].values()), r


@pytest.mark.parametrize("layout", ["fsdp2", "tp"])
def test_fsdp2_and_tp_modules_keep_small_updates(layout):
    ranks = _run(_module_worker, layout)
    for r in ranks:
        assert all(r["dtensor"]), r
        assert r["covered"] == r["n_params"], r
        assert r["dtensor_rounds"] > 0, r
        # Every weight started at 1.0 and moves by about LR per step: only stochastic rounding keeps that.
        assert all(m > 0 for m in r["moved"]), r
