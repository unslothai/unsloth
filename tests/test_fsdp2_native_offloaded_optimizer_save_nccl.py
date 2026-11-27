# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 full optimizer save (fsdp2_native.native_save_optimizer, what the Trainer checkpoint calls) with
UNSLOTH_OFFLOAD_OPTIMIZER_STATE host-backed state, two NCCL ranks on two GPUs.

The offloader keeps each DTensor state's local shard in host memory under its device wrapper, so torch's full state
gather handed NCCL host tensors and the save failed. Now the saved state equals the state before offloading, the
state stays host-backed afterwards, the optimizer still steps, and the file loads back. gloo cannot show this (it
gathers host tensors), hence two GPUs."""

import multiprocessing
import os
import sys
import tempfile

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if (
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 2
    or not torch.distributed.is_available()
    or not torch.distributed.is_nccl_available()
):
    pytest.skip("needs two CUDA GPUs with NCCL", allow_module_level = True)
fsdp = pytest.importorskip("torch.distributed.fsdp")
if not hasattr(fsdp, "fully_shard"):
    pytest.skip("torch has no FSDP2 fully_shard", allow_module_level = True)
pytest.importorskip("peft")


def _state_devices(optimizer):
    devices = set()
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                devices.add(getattr(value, "_local_tensor", value).device.type)
    return devices


def _worker(rank, port, tmp):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = "2",
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = "2",
    )
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    try:
        sys.path.insert(0, REPO)
        from peft import LoraConfig, get_peft_model
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard
        from transformers import LlamaConfig, LlamaForCausalLM

        from unsloth.models.fsdp2_native import native_load_optimizer, native_save_optimizer
        from unsloth.optimizers.state_offload import offload_optimizer_state

        torch.manual_seed(0)
        config = LlamaConfig(
            vocab_size = 128,
            hidden_size = 64,
            intermediate_size = 128,
            num_hidden_layers = 2,
            num_attention_heads = 4,
            num_key_value_heads = 2,
        )
        model = get_peft_model(
            LlamaForCausalLM(config).cuda(),
            LoraConfig(r = 4, target_modules = "all-linear", init_lora_weights = False),
        )
        mesh = init_device_mesh("cuda", (2,))
        for layer in model.get_base_model().model.layers:
            fully_shard(layer, mesh = mesh)
        fully_shard(model, mesh = mesh)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 1e-3)
        ids = torch.randint(0, 128, (2, 8), device = "cuda")
        model(input_ids = ids, labels = ids).loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        want = native_save_optimizer(model, optimizer)  # before offloading: device state
        res = {"offloaded": offload_optimizer_state(optimizer) is not None}
        res["before"] = _state_devices(optimizer)
        path = os.path.join(tmp, "optimizer.pt")
        try:
            native_save_optimizer(model, optimizer, path)
            res["error"] = None
        except Exception as exc:
            res["error"] = f"{type(exc).__name__}: {exc}"
        res["after"] = _state_devices(optimizer)
        if res["error"] is None:
            if rank == 0:
                got = torch.load(path, weights_only = False)["state"]
                res["same"] = sorted(got) == sorted(want["state"]) and all(
                    torch.equal(got[k][n].cpu(), v.cpu())
                    for k, entry in want["state"].items()
                    for n, v in entry.items()
                    if isinstance(v, torch.Tensor)
                )
            model(input_ids = ids, labels = ids).loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            native_load_optimizer(model, optimizer, path)
            res["loaded"] = True
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        dist.destroy_process_group()


def test_native_full_optimizer_save_of_offloaded_state_over_nccl():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_tp_native_gloo import _free_port

    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [ctx.Process(target = _worker, args = (r, port, tmp)) for r in range(2)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(300)
        codes = [p.exitcode for p in procs]
        for p in procs:
            if p.is_alive():
                p.kill()
        assert codes == [0, 0], f"rank exit codes {codes}"
        results = [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False) for r in range(2)
        ]
    for res in results:
        assert res["offloaded"] and res["before"] == {"cpu"}, res
        assert res["error"] is None, res["error"]
        assert res["after"] == {"cpu"} and res["loaded"], res
    assert results[0]["same"], results[0]
