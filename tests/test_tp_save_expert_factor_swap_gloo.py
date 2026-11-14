# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""save_tp_adapter on FSDP2 DTensor expert LoRA factors must match single-process save_pretrained."""

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
from test_tp_native_oracle import tpn  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODES = ("noswap", "swap", "legacy")


def _build(mode):
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
    for p in model.parameters():
        nn.init.normal_(p, std = 0.02)
    config = peft.LoraConfig(
        r = 2, target_modules = [], target_parameters = ["experts.gate_up_proj", "experts.down_proj"]
    )
    model = peft.get_peft_model(model, config)
    for name, p in model.named_parameters():
        if "lora_B" in name:
            nn.init.normal_(p, std = 0.02)
    return model


def _read(directory):
    from safetensors.torch import load_file
    return load_file(os.path.join(directory, "adapter_model.safetensors"))


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
    except Exception as e:
        return {"skip": repr(e)}
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    if mode == "legacy":
        os.environ["UNSLOTH_MOE_LORA_B_LAYOUT"] = "grouped_by_expert"
    reference = os.path.join(tmp, f"{mode}_single")
    if rank == 0:
        _build(mode).save_pretrained(reference)
    dist.barrier()
    model = _build(mode)
    fully_shard(model, mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("dp",)))
    out = {
        "dtensors": sum(
            tpn._is_dtensor(v) for k, v in model.state_dict(keep_vars = True).items() if "lora_" in k
        )
    }
    target = os.path.join(tmp, f"{mode}_tp")
    try:
        tpn.save_tp_adapter(model, target, save_embedding_layers = False)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    if rank == 0:
        out["single"] = _read(reference)
        out["tp"] = _read(target)
    return out


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


def _spawn(mode, world = 2):
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
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
    return results


@pytest.mark.parametrize("mode", MODES)
def test_tp_save_matches_single_process_after_expert_factor_swap(mode):
    results = _spawn(mode)
    for rank, res in enumerate(results):
        assert res["dtensors"] > 0, (rank, res)
        assert "error" not in res, (rank, res["error"])
    single, tp = results[0]["single"], results[0]["tp"]
    assert sorted(single) == sorted(tp)
    for key in single:
        assert single[key].shape == tp[key].shape, (key, single[key].shape, tp[key].shape)
        torch.testing.assert_close(tp[key], single[key], rtol = 0, atol = 0, msg = key)
    if mode == "swap":
        # converted (vanilla PEFT) layout; down_proj is the outer wrapper
        shapes = {k.split("model.experts.")[-1]: list(v.shape) for k, v in tp.items()}
        assert shapes == {
            "base_layer.lora_A.weight": [8, 24],
            "base_layer.lora_B.weight": [16, 8],
            "lora_A.weight": [8, 16],
            "lora_B.weight": [12, 8],
        }, shapes
