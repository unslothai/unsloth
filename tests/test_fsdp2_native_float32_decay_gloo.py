# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Float32 masters + bf16 gather: GDN A_log / dt_bias must reach the forward in float32, as on one GPU.
Two gloo ranks on CPU vs one process, tiny Qwen3-Next."""

import json
import os
import sys

import pytest
import torch

if __import__("multiprocessing").parent_process() is not None or getattr(
    __import__("multiprocessing").current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _batch, _build, _free_port, _load_native  # noqa: E402

REPO_ID = "hf-internal-testing/tiny-random-Qwen3NextForCausalLM"
DECAY = ("A_log", "dt_bias")
STEPS = 5


def _model():
    m = _build(REPO_ID, "AutoModelForCausalLM")
    m.config.torch_dtype = torch.bfloat16  # float32 masters, bf16 compute: the gather dtype is bf16
    with torch.no_grad():  # distinct, non-round decay values (bf16 rounds them visibly)
        for name, p in m.named_parameters():
            if name.endswith(DECAY):
                g = torch.Generator().manual_seed(len(name))
                p.copy_(torch.empty_like(p).uniform_(0.1, 2.5, generator = g))
    return m.train()


def _decay_seen(model, masters):
    seen, drift = {}, {"max": 0.0}

    def hook(module, args):
        for attr in DECAY:
            p = getattr(module, attr, None)
            if p is None:
                continue
            seen.setdefault(attr, set()).add(str(p.dtype))
            master = masters.pop((module._decay_name, attr), None)
            if master is not None:
                d = (p.detach().float() - master).abs().max().item()
                drift["max"] = max(drift["max"], d)

    handles = []
    for name, m in model.named_modules():
        if isinstance(getattr(m, "A_log", None), torch.Tensor):
            m._decay_name = name.split("layers.", 1)[-1]
            handles.append(m.register_forward_pre_hook(hook))
    return seen, drift, handles


def _masters(model):
    return {
        (name.split("layers.", 1)[-1], attr): getattr(m, attr).detach().float().clone()
        for name, m in model.named_modules()
        if isinstance(getattr(m, "A_log", None), torch.Tensor)
        for attr in DECAY
        if isinstance(getattr(m, attr, None), torch.Tensor)
    }


def _train(
    model,
    step_fn,
    foreach = None,
):
    # The sharded run: a plain user-built foreach AdamW over model.parameters(), no Unsloth group split.
    opt = torch.optim.AdamW(model.parameters(), lr = 1e-3, foreach = foreach)
    losses, decay_grads = [], None
    for step in range(STEPS):
        opt.zero_grad()
        with torch.autocast("cpu", dtype = torch.bfloat16):
            loss = model(**_batch(model, seed = 1 + step)).loss
        loss.backward()
        if decay_grads is None:
            decay_grads = step_fn(model)
        opt.step()
        losses.append(loss.item())
    return losses, decay_grads


def _grads(model):
    from torch.distributed.tensor import DTensor

    out = {}
    for name, p in model.named_parameters():
        if name.endswith(DECAY) and p.grad is not None:
            g = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
            out[name] = g.detach().float().clone()
    return out


def _worker(rank, world, port, out):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        UNSLOTH_FSDP2 = "1",
    )
    import torch.distributed as dist

    dist.init_process_group("gloo", rank = rank, world_size = world)
    try:
        native = _load_native()
        ref, model = _model(), _model()
        masters = _masters(ref)
        ref_losses, ref_grads = _train(ref, _grads)
        native.shard_model(model, native.FSDP2Config())
        seen, drift, handles = _decay_seen(model, masters)
        from torch.distributed.tensor import DTensor

        losses, grads = _train(model, _grads, foreach = True)
        for h in handles:
            h.remove()
        decay = [p for n, p in model.named_parameters() if n.endswith(DECAY)]
        local = torch.cat([p.to_local().detach().flatten() for p in decay])
        copies = [torch.zeros_like(local) for _ in range(world)]
        dist.all_gather(copies, local)
        grad_err = max(
            ((grads[n] - r).abs().max() / (r.abs().max() + 1e-12)).item()
            for n, r in ref_grads.items()
        )
        if rank == 0:
            with open(out, "w", encoding = "utf-8") as f:
                json.dump(
                    {
                        "seen": {k: sorted(v) for k, v in seen.items()},
                        "drift": drift["max"],
                        "unchecked": len(masters),
                        "loss_err": max(abs(a - b) for a, b in zip(losses, ref_losses)),
                        "grad_err": grad_err,
                        "n_decay": len(ref_grads),
                        "all_dtensor": all(isinstance(p, DTensor) for p in model.parameters()),
                        "decay_layout": sorted(
                            {
                                f"{p.dtype}:{all(pl.is_replicate() for pl in p.placements)}"
                                for p in decay
                            }
                        ),
                        "replica_diff": (copies[0] - copies[1]).abs().max().item(),
                        "decay_names_kept": sorted(
                            n.split("layers.", 1)[-1]
                            for n, _ in model.named_parameters()
                            if n.endswith(DECAY)
                        )
                        == sorted(n.split("layers.", 1)[-1] for n in ref_grads),
                    },
                    f,
                )
    finally:
        dist.destroy_process_group()


def test_full_ft_float32_decay_params_gathered_as_float32(tmp_path):
    pytest.importorskip("transformers")
    import torch.multiprocessing as mp

    out = str(tmp_path / "decay.json")
    mp.spawn(_worker, args = (2, _free_port(), out), nprocs = 2, join = True)
    r = json.load(open(out, encoding = "utf-8"))
    print(r)
    assert r["n_decay"] > 0, r
    assert r["seen"] == {"A_log": ["torch.float32"], "dt_bias": ["torch.float32"]}, r
    # Kept as float32 Replicate DTensors: a plain foreach AdamW over model.parameters() steps, and the
    # averaged gradients keep the replicas identical.
    assert r["all_dtensor"] and r["decay_layout"] == ["torch.float32:True"], r
    assert r["replica_diff"] == 0.0 and r["decay_names_kept"], r
    assert r["unchecked"] == 0 and r["drift"] == 0.0, r
    assert r["loss_err"] < 2e-3 and r["grad_err"] < 5e-2, r
