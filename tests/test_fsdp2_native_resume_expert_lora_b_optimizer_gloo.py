# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 resume of a grouped_by_expert expert LoRA (override unset) repacks lora_B to rank_major;
the restored Adam moments must move with it, so the resumed next step equals the uninterrupted one."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only

import sys  # noqa: E402
import tempfile  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_resume_expert_factor_swap_gloo import _build, _expert_deltas  # noqa: E402
from test_tp_native_gloo import _free_port  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORMATS = ("full", "dcp")
LR = 0.05


def _targets(model, step):
    generator = torch.Generator().manual_seed(100 + step)
    deltas = _expert_deltas(_full(model), rank_major = False)
    return {
        name: torch.randn(d.shape, generator = generator, dtype = d.dtype) for name, d in deltas.items()
    }


def _full(model, detach = True):
    from torch.distributed.tensor import DTensor

    out = {}
    for name, p in model.named_parameters():
        if "lora_" in name:
            full = p.full_tensor() if isinstance(p, DTensor) else p
            out[name] = full.detach().clone() if detach else full
    return out


def _step(model, optimizer, targets, rank_major):
    """One Adam step on a loss of the per-expert deltas: the same objective in either lora_B layout."""
    optimizer.zero_grad()
    deltas = _expert_deltas(_full(model, detach = False), rank_major = rank_major)
    loss = sum((deltas[name] * targets[name]).sum() for name in targets)
    loss.backward()
    optimizer.step()


def _shard(model, world):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from unsloth.models import fsdp2_native

    fully_shard(model, mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("dp",)))
    setattr(model, fsdp2_native._NATIVE_FLAG, True)
    return model


def _case(rank, world, tmp):
    try:
        import importlib.util

        # spawned ranks skip conftest's device spoof, which unsloth_zoo needs at import
        spec = importlib.util.spec_from_file_location(
            "_tp_conftest", os.path.join(REPO, "tests", "conftest.py")
        )
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
        import unsloth_zoo.temporary_patches.moe_lora_factor_swap  # noqa: F401
        from unsloth_zoo.temporary_patches.moe_utils import (  # noqa: F401
            convert_legacy_lora_b,
            repack_optimizer_state_for_legacy_lora_b,
        )
        from unsloth.models import fsdp2_native
    except Exception as e:
        return {"skip": repr(e)}
    import torch.distributed as dist

    # Trained (and saved) under the grouped_by_expert override.
    os.environ["UNSLOTH_MOE_LORA_B_LAYOUT"] = "grouped_by_expert"
    checkpoint = os.path.join(tmp, "checkpoint")
    optimizer_pt = os.path.join(tmp, "optimizer.pt")
    optimizer_dcp = os.path.join(tmp, "optimizer_dcp")
    trained = _shard(_build("legacy", seed = 1), world)
    optimizer = torch.optim.Adam([p for p in trained.parameters() if p.requires_grad], lr = LR)
    _step(trained, optimizer, _targets(trained, 1), rank_major = False)
    _step(trained, optimizer, _targets(trained, 2), rank_major = False)
    saved = _full(trained)
    if rank == 0:
        plain = _build("legacy", seed = 1)
        with torch.no_grad():
            for name, p in plain.named_parameters():
                if name in saved:
                    p.copy_(saved[name])
        plain.save_pretrained(checkpoint)
    fsdp2_native.native_save_optimizer(trained, optimizer, optimizer_pt)
    fsdp2_native.native_save_optimizer_sharded(trained, optimizer, optimizer_dcp)
    dist.barrier()
    targets = _targets(trained, 3)
    _step(trained, optimizer, targets, rank_major = False)
    want = _expert_deltas(_full(trained), rank_major = False)

    # Resumed with the override unset: lora_B is repacked to rank_major on load.
    os.environ.pop("UNSLOTH_MOE_LORA_B_LAYOUT")
    got = {}
    for fmt in FORMATS:
        model = _shard(_build("swap", seed = 2), world)
        try:
            fsdp2_native.native_load_model_checkpoint(model, checkpoint)
            resumed = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr = LR)
            fsdp2_native.native_load_optimizer(
                model, resumed, optimizer_dcp if fmt == "dcp" else optimizer_pt
            )
        except Exception as e:
            return {"error": f"{fmt}: {type(e).__name__}: {e}"}
        loaded = _expert_deltas(_full(model), rank_major = True)
        _step(model, resumed, targets, rank_major = True)
        got[fmt] = {
            "loaded": loaded,
            "stepped": _expert_deltas(_full(model), rank_major = True),
            "markers": sum(hasattr(p, "_unsloth_lora_b_repacked") for p in model.parameters()),
        }
    return {"saved": _expert_deltas(saved, rank_major = False), "want": want, "got": got}


def _worker(rank, world, port, tmp):
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
        torch.save(_case(rank, world, tmp), os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        dist.destroy_process_group()


def test_native_resume_repacks_adam_state_with_grouped_by_expert_lora_b():
    ctx = multiprocessing.get_context("spawn")
    port, world = _free_port(), 2
    with tempfile.TemporaryDirectory() as tmp:
        procs = [ctx.Process(target = _worker, args = (r, world, port, tmp)) for r in range(world)]
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
        pytest.skip(reason = f"unsloth_zoo without the lora_B repack: {results[0]['skip']}")
    for rank, res in enumerate(results):
        assert "error" not in res, (rank, res["error"])
        assert res["want"]
        for fmt in FORMATS:
            got = res["got"][fmt]
            for name, want in res["want"].items():
                # The load itself already matched before the optimizer fix.
                torch.testing.assert_close(got["loaded"][name], res["saved"][name], rtol = 0, atol = 0)
                torch.testing.assert_close(
                    got["stepped"][name], want, rtol = 1e-6, atol = 1e-9, msg = f"{fmt} {name}"
                )
            assert got["markers"] == 0, (rank, fmt)  # consumed: a later load cannot permute twice
