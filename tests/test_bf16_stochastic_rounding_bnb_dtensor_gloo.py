# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""bfloat16 stochastic rounding for bitsandbytes AdamW on DTensor weights (FSDP2 / TP shards), world 2.

bitsandbytes Adam / AdamW step each DTensor's local shard (models/fsdp2_bnb_optim.py). Before, the rounding
skipped every DTensor weight on that route, so a bfloat16 full finetune under FSDP2 or TP with the default
8-bit AdamW rounded each update to nearest and lost every one below half a bfloat16 spacing. Now the shard
steps with the same rounding a plain weight gets, seeded by the DTensor (param index + shard start)."""

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


def _init(rank, world, port, backend):
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


def _load(root, relative, name):
    path = os.path.join(root, relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert os.path.realpath(module.__file__).startswith(os.path.realpath(root)), module.__file__
    return module


def _worker(rank, world, port, out, root, device, layout, sr_env, sr_first):
    try:
        if sr_env is None:
            os.environ.pop("UNSLOTH_BF16_STOCHASTIC_ROUNDING", None)
        else:
            os.environ["UNSLOTH_BF16_STOCHASTIC_ROUNDING"] = sr_env
        if device == "cuda":
            torch.cuda.set_device(rank)
        dist = _init(rank, world, port, "nccl" if device == "cuda" else "gloo")
        import bitsandbytes as bnb
        from torch.distributed.device_mesh import init_device_mesh

        fsdp2_bnb_optim = _load(root, "unsloth/models/fsdp2_bnb_optim.py", "_test_fsdp2_bnb_optim")
        sr = _load(root, "unsloth/optimizers/bf16_stochastic_rounding.py", "_test_bf16_sr_bnb")
        mesh = init_device_mesh(device, (world,))
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(128, 64), torch.nn.Linear(64, 128))
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(1.0)
        model = model.to(device = device, dtype = torch.bfloat16)
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
        # 8-bit blockwise state for the weights, 32-bit for the small biases (bitsandbytes' min_8bit_size).
        opt = bnb.optim.AdamW8bit(params, lr = LR, weight_decay = 0.0, min_8bit_size = 4096)
        if sr_first:
            # A Trainer may build the scheduler (which enables the rounding) before FSDP2 / TP localize.
            covered = sr.enable_bf16_stochastic_rounding(opt, seed = 7, verbose = False)
            assert fsdp2_bnb_optim.localize_bnb_optimizer(opt)
        else:
            assert fsdp2_bnb_optim.localize_bnb_optimizer(opt)
            covered = sr.enable_bf16_stochastic_rounding(opt, seed = 7, verbose = False)
        g = torch.Generator().manual_seed(100)  # same data on both ranks (TP needs it)
        for step in range(STEPS):
            x = torch.randn(8, 128, generator = g).to(device = device, dtype = torch.bfloat16)
            try:
                model(x).float().pow(2).mean().backward()
            except AttributeError as exc:
                if step == 0 and layout == "fsdp2" and "record_event" in str(exc):
                    if rank == 0:
                        out.put({"skip": f"FSDP2 on CPU unsupported by torch {torch.__version__}"})
                    dist.destroy_process_group()
                    return
                raise
            opt.step()
            opt.zero_grad(set_to_none = True)
        moved, mean_change = [], []
        for p in params:
            full = p.full_tensor() if hasattr(p, "full_tensor") else p
            moved.append(float((full.float() != 1).float().mean().item()))
            mean_change.append(float((full.float() - 1).abs().mean().item()))
        state_kinds = sorted(
            {str(s["state1"].dtype) for s in opt.state.values() if torch.is_tensor(s.get("state1"))}
        )
        result = {
            "covered": covered,
            "n_params": len(params),
            "dtensor": [type(p).__name__ == "DTensor" for p in params],
            "dtype_kept": all(p.dtype == torch.bfloat16 for p in params),
            "dtensor_rounds": opt.__dict__.get("_unsloth_sr_dtensor_rounds", 0),
            "bnb_fused_steps": opt.__dict__.get("_unsloth_sr_bnb_fused_steps", 0),
            "moved": moved,
            "mean_change": mean_change,
            "state_kinds": state_kinds,
            "steps": sorted({int(s.get("step", 0)) for s in opt.state.values()}),
        }
        gathered = [None] * world
        dist.all_gather_object(gathered, result)
        if rank == 0:
            out.put({"ranks": gathered})
        dist.destroy_process_group()
    except Exception:
        out.put({"error": traceback.format_exc()[-3000:]})


def _run(
    device,
    layout,
    sr_env = None,
    sr_first = False,
):
    pytest.importorskip("bitsandbytes")
    if device == "cpu":
        local = _load(ROOT, "unsloth/models/fsdp2_bnb_optim.py", "_test_fsdp2_bnb_optim")
        if not local.bnb_steps_cpu_tensors():
            pytest.skip("this bitsandbytes cannot step CPU tensors")
    else:
        if torch.cuda.device_count() < 2:
            pytest.skip("requires 2 CUDA devices (NCCL, one per rank)")
        if not torch.distributed.is_nccl_available():
            pytest.skip("requires NCCL")
    results = _spawn(_worker, ROOT, device, layout, sr_env, sr_first)
    assert len(results) == 1 and "error" not in results[0], results
    if "skip" in results[0]:
        pytest.skip(results[0]["skip"])
    return results[0]["ranks"]


CASES = [
    pytest.param("cpu", "fsdp2", False, id = "cpu-fsdp2"),
    pytest.param("cpu", "tp", False, id = "cpu-tp"),
    pytest.param("cpu", "fsdp2", True, id = "cpu-fsdp2-rounding-enabled-before-localize"),
    pytest.param("cuda", "fsdp2", False, id = "cuda-fsdp2", marks = pytest.mark.gpu),
    pytest.param("cuda", "tp", False, id = "cuda-tp", marks = pytest.mark.gpu),
]


@pytest.mark.parametrize("device, layout, sr_first", CASES)
def test_bnb_adamw_rounds_dtensor_shards_stochastically(device, layout, sr_first):
    ranks = _run(device, layout, sr_first = sr_first)
    for r in ranks:
        assert all(r["dtensor"]) and r["dtype_kept"], r
        assert r["covered"] == r["n_params"], r
        assert r["steps"] == [STEPS], r
        # Both bitsandbytes state widths took part (8-bit weights, 32-bit biases).
        assert r["state_kinds"] == ["torch.float32", "torch.uint8"], r
        assert r["dtensor_rounds"] + r["bnb_fused_steps"] > 0, r
        if device == "cuda":
            assert r["bnb_fused_steps"] > 0, r
        # Every weight started at 1.0 and Adam moves it by about LR per step: round-to-nearest keeps all of
        # them at 1.0, stochastic rounding moves most of them and keeps the mean move near STEPS * LR.
        assert all(m > 0.5 for m in r["moved"]), r
        assert all(0.3 * STEPS * LR < c < 2 * STEPS * LR for c in r["mean_change"]), r


def test_opt_out_keeps_round_to_nearest_for_bnb_dtensors():
    ranks = _run("cpu", "tp", "0")
    for r in ranks:
        assert r["covered"] == 0 and r["dtensor_rounds"] == 0, r
        assert all(m == 0.0 for m in r["moved"]), r
