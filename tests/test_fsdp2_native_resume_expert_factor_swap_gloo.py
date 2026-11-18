# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 resume of an expert LoRA adapter Zoo saved in vanilla PEFT's layout restores it exactly."""

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
from test_tp_native_gloo import _free_port  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODES = ("noswap", "swap", "legacy")


def _build(mode, seed):
    import peft
    from torch import nn
    from transformers.models.llama4.configuration_llama4 import Llama4TextConfig
    from transformers.models.llama4.modeling_llama4 import Llama4TextExperts
    import unsloth_zoo.temporary_patches.moe_lora_factor_swap as swap
    from unsloth_zoo.temporary_patches.llama4_moe import patch_llama4_moe

    patch_llama4_moe()
    swap.patch_expert_lora_factor_swap()
    Llama4TextExperts._unsloth_expert_lora_factors_swapped = mode != "noswap"
    cfg = Llama4TextConfig(hidden_size = 16, intermediate_size = 12, num_local_experts = 4)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = Llama4TextExperts(cfg)

    torch.manual_seed(0)
    model = Block()
    config = peft.LoraConfig(
        r = 2, target_modules = [], target_parameters = ["experts.gate_up_proj", "experts.down_proj"]
    )
    model = peft.get_peft_model(model, config)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.copy_(0.02 * torch.randn(p.shape, generator = generator))
    return model


def _case(rank, world, tmp, mode):
    try:
        import importlib.util

        # spawned ranks skip conftest's device spoof, which unsloth_zoo needs at import
        spec = importlib.util.spec_from_file_location(
            "_tp_conftest", os.path.join(REPO, "tests", "conftest.py")
        )
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
        import unsloth_zoo.temporary_patches.moe_lora_factor_swap  # noqa: F401
        from transformers.models.llama4.modeling_llama4 import Llama4TextExperts  # noqa: F401
        from unsloth.models import fsdp2_native
    except Exception as e:
        return {"skip": repr(e)}
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    if mode == "legacy":
        os.environ["UNSLOTH_MOE_LORA_B_LAYOUT"] = "grouped_by_expert"
    checkpoint = os.path.join(tmp, "checkpoint")
    trained = _build(mode, seed = 1)
    want = {n: p.detach().clone() for n, p in trained.named_parameters() if "lora_" in n}
    if rank == 0:
        trained.save_pretrained(checkpoint)
    dist.barrier()
    model = _build(mode, seed = 2)
    fully_shard(model, mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("dp",)))
    setattr(model, fsdp2_native._NATIVE_FLAG, True)
    try:
        fsdp2_native.native_load_model_checkpoint(model, checkpoint)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    got = {
        n: (p.full_tensor() if isinstance(p, DTensor) else p.detach())
        for n, p in model.named_parameters()
        if "lora_" in n
    }
    return {"want": want, "got": got}


def _worker(rank, world, port, tmp, mode):
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
        torch.save(_case(rank, world, tmp, mode), os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", MODES)
def test_native_resume_restores_expert_lora_saved_in_vanilla_layout(mode):
    ctx = multiprocessing.get_context("spawn")
    port, world = _free_port(), 2
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, tmp, mode)) for r in range(world)
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
        pytest.skip(reason = f"unsloth_zoo without the expert factor swap: {results[0]['skip']}")
    for rank, res in enumerate(results):
        assert "error" not in res, (rank, res["error"])
        assert sorted(res["got"]) == sorted(res["want"])
        for name, want in res["want"].items():
            torch.testing.assert_close(res["got"][name], want, rtol = 0, atol = 0, msg = name)
