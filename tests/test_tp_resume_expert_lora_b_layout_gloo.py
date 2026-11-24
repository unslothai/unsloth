# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""2 gloo ranks, fused Llama 4 expert LoRA sharded by expert index: a per-rank Trainer checkpoint resumes bitwise
under the same UNSLOTH_MOE_LORA_B_LAYOUT and is refused on every rank under the other one."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only

import sys  # noqa: E402
import tempfile  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tp_native_gloo import _free_port  # noqa: E402
from test_tp_native_oracle import tpn  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYOUT = "UNSLOTH_MOE_LORA_B_LAYOUT"


def _set_layout(layout):
    if layout == "rank_major":
        os.environ.pop(LAYOUT, None)
    else:
        os.environ[LAYOUT] = layout


def _trainer(model, optimizer):
    args = types.SimpleNamespace(should_save = False)
    return types.SimpleNamespace(model = model, optimizer = optimizer, lr_scheduler = None, args = args)


def _sharded(rank, world, mesh):
    from test_tp_save_expert_factor_swap_gloo import _build

    model = _build("legacy")
    shard = tpn._shard_moe_experts(model.base_model.model, mesh, rank, world)
    return model, shard


def _experts(model, x):
    index = torch.arange(len(x)).view(-1, 1) % 4
    return model.base_model.model.experts(x, index, torch.ones(len(x), 1))


def _step(model, optimizer, x):
    optimizer.zero_grad()
    _experts(model, x).square().sum().backward()
    optimizer.step()


def _case(rank, world, tmp, saved, resumed):
    try:
        import importlib.util

        # spawned ranks skip conftest's device spoof, which unsloth_zoo needs at import
        spec = importlib.util.spec_from_file_location(
            "_tp_conftest", os.path.join(REPO, "tests", "conftest.py")
        )
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
        import unsloth_zoo.temporary_patches.moe_lora_factor_swap  # noqa: F401
        from transformers.models.llama4.modeling_llama4 import Llama4TextExperts  # noqa: F401
    except Exception as e:
        return {"skip": repr(e)}
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))
    torch.manual_seed(1)
    x = torch.randn(12, 16)

    _set_layout(saved)
    model, shard = _sharded(rank, world, mesh)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 1e-2)
    _step(model, opt, x)
    tpn._tp_save_optimizer_and_scheduler(_trainer(model, opt), tmp)
    _step(model, opt, x)
    with torch.no_grad():
        expected = _experts(model, x)
    out = {"local_experts": (shard.lo, shard.hi)}

    _set_layout(resumed)
    model, _ = _sharded(rank, world, mesh)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.zero_()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 1e-2)
    trainer = _trainer(model, opt)
    try:
        tpn._tp_load_from_checkpoint(trainer, tmp)
        tpn._tp_load_optimizer_and_scheduler(trainer, tmp)
    except RuntimeError as e:
        out["error"] = str(e)
        return out
    _step(model, opt, x)
    with torch.no_grad():
        out["diff"] = (_experts(model, x) - expected).abs().max().item()
    return out


def _worker(rank, world, port, tmp, saved, resumed):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
    )
    import datetime

    import torch.distributed as dist

    torch.set_num_threads(2)
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 120)
    )
    try:
        res = _case(rank, world, tmp, saved, resumed)
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _spawn(
    saved,
    resumed,
    world = 2,
):
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, tmp, saved, resumed))
            for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(300)
        assert [p.exitcode for p in procs] == [0] * world
        results = [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]
    if "skip" in results[0]:
        pytest.skip(
            reason = f"unsloth_zoo without the fused Llama 4 expert LoRA: {results[0]['skip']}"
        )
    return results


@pytest.mark.parametrize("saved", ["grouped_by_expert", "rank_major"])
@pytest.mark.parametrize("resumed", ["grouped_by_expert", "rank_major"])
def test_tp_resume_across_expert_lora_b_layouts(saved, resumed):
    results = _spawn(saved, resumed)
    assert [res["local_experts"] for res in results] == [(0, 2), (2, 4)]
    for rank, res in enumerate(results):
        if saved == resumed:
            assert "error" not in res, (rank, res["error"])
            assert res["diff"] == 0.0, (rank, res["diff"])
        else:
            assert "diff" not in res, (rank, res)
            assert f"{LAYOUT}={saved}" in res["error"], (rank, res["error"])
            assert f"reads them as {resumed}" in res["error"], (rank, res["error"])
