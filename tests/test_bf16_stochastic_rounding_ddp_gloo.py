# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""bfloat16 stochastic rounding under DDP (gloo, world 2): gradients stay synced and replicas stay bitwise equal.

Stepping a weight through a float32 copy must never change the weight's .data dtype: that resets its
AccumulateGrad node, which holds DDP's allreduce hook, so from the next backward the ranks would train on
their own gradients and drift apart."""

import hashlib
import importlib.util
import os
import sys
import traceback
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from real_accelerator import has_real_cuda  # noqa: E402

STEPS = 4
ROOT = str(Path(__file__).resolve().parents[1])


def _init(rank, world, port):
    # Not the shared gloo helpers: importing those hides the GPU from spawned ranks, and the CUDA cases need it.
    import datetime

    import torch.distributed as dist

    os.environ.update(
        MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port), RANK = str(rank), WORLD_SIZE = str(world)
    )
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 120)
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


def _digest(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _worker(rank, world, port, out, root, device, optim, fused, fused_bnb):
    try:
        os.environ["UNSLOTH_BF16_SR_FUSED"] = fused
        os.environ["UNSLOTH_BF16_SR_FUSED_BNB"] = fused_bnb
        os.environ.pop("UNSLOTH_BF16_STOCHASTIC_ROUNDING", None)
        dist = _init(rank, world, port)
        path = os.path.join(root, "unsloth/optimizers/bf16_stochastic_rounding.py")
        spec = importlib.util.spec_from_file_location("_test_bf16_sr_ddp", path)
        sr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sr)
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(16, 64), torch.nn.Tanh(), torch.nn.Linear(64, 300)
        ).to(device = device, dtype = torch.bfloat16)
        params = list(model.parameters())
        ddp = torch.nn.parallel.DistributedDataParallel(model)
        if optim == "adamw8bit":
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(params, lr = 1e-3, weight_decay = 0.01, min_8bit_size = 256)
        else:
            opt = torch.optim.AdamW(params, lr = 1e-3, weight_decay = 0.01)
        assert sr.enable_bf16_stochastic_rounding(opt, seed = 7, verbose = False) == len(params)
        g = torch.Generator().manual_seed(100 + rank)  # different data per rank
        steps = []
        for _ in range(STEPS):
            x = torch.randn(8, 16, generator = g).to(device = device, dtype = torch.bfloat16)
            ddp(x).float().pow(2).mean().backward()
            grads = _digest(p.grad for p in params)
            opt.step()
            opt.zero_grad(set_to_none = True)
            steps.append((grads, _digest(params), all(p.dtype == torch.bfloat16 for p in params)))
        gathered = [None] * world
        dist.all_gather_object(gathered, steps)
        if rank == 0:
            out.put(
                {
                    "steps": gathered,
                    "fused": opt.__dict__.get("_unsloth_sr_fused_steps", 0),
                    "rounds": opt.__dict__.get("_unsloth_sr_triton_rounds", 0),
                    "bnb_fused": opt.__dict__.get("_unsloth_sr_bnb_fused_steps", 0),
                }
            )
        dist.destroy_process_group()
    except Exception:
        out.put({"error": traceback.format_exc()[-3000:]})


# (device, optim, FUSED, FUSED_BNB, one-pass AdamW steps, round-kernel rounds, one-kernel bnb steps)
CASES = [
    pytest.param("cpu", "adamw", "1", "1", 0, 0, 0, id = "cpu-adamw-float32-route"),
    pytest.param(
        "cuda", "adamw", "1", "1", STEPS * 4, 0, 0, id = "cuda-adamw-one-pass", marks = pytest.mark.gpu
    ),
    pytest.param(
        "cuda", "adamw", "0", "1", 0, 0, 0, id = "cuda-adamw-float32-route", marks = pytest.mark.gpu
    ),
    pytest.param(
        "cuda",
        "adamw8bit",
        "1",
        "1",
        0,
        0,
        STEPS * 4,
        id = "cuda-bitsandbytes",
        marks = pytest.mark.gpu,
    ),
    pytest.param(
        "cuda",
        "adamw8bit",
        "1",
        "0",
        0,
        STEPS * 4,
        0,
        id = "cuda-bitsandbytes-float32-copy",
        marks = pytest.mark.gpu,
    ),
]


@pytest.mark.parametrize("device, optim, fused, fused_bnb, fused_steps, rounds, bnb_fused", CASES)
def test_ddp_replicas_stay_synced_with_stochastic_rounding(
    device, optim, fused, fused_bnb, fused_steps, rounds, bnb_fused
):
    if device == "cuda":
        if not has_real_cuda():
            pytest.skip("requires CUDA (both ranks share one GPU over gloo)")
        if sys.platform == "win32":
            # Windows gloo crashes a spawned rank on GPU-tensor collectives before any assertion
            # (0xC0000005 on ROCm, base and head alike); the CPU case covers the fix there.
            pytest.skip("gloo collectives on GPU tensors are not supported on Windows")
        pytest.importorskip("triton")
    if optim == "adamw8bit":
        pytest.importorskip("bitsandbytes")
    results = _spawn(_worker, ROOT, device, optim, fused, fused_bnb)
    assert len(results) == 1 and "error" not in results[0], results
    r = results[0]
    ranks = r["steps"]
    for step, (a, b) in enumerate(zip(*ranks)):
        assert a[2] and b[2], f"weights left bfloat16 at step {step}"
        assert a[0] == b[0], f"gradients not synced at step {step}: {a[0]} vs {b[0]}"
        assert a[1] == b[1], f"replicas differ after step {step}"
    assert r["fused"] == fused_steps and r["rounds"] == rounds and r["bnb_fused"] == bnb_fused, r
