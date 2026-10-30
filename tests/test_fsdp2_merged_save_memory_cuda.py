# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Writer-GPU memory of an FSDP2 merged save of a sharded NF4 QLoRA (unsloth/models/fsdp2_save.py),
two NCCL ranks with a GPU each, or, with one visible GPU, one NCCL rank whose shards FSDP2 keeps on
the CPU (CPUOffloadPolicy), so every gathered tensor is a real allocation on the writer's GPU there
too. `python tests/test_fsdp2_merged_save_memory_cuda.py --out result.json` runs the second form.

The previous placement gathered every packed 4-bit weight onto rank 0's GPU and kept them there
until the write, so a merged save needed the whole quantized base on one card above its shards. The
merged saves now stage each packed weight on the CPU right after gathering it; merged_4bit brings
one layer back at a time to merge it. The peak above the sharded footprint must stay about one layer
(not the whole base), and the merged bytes must equal the previous placement's.
The module is loaded from its file (importing unsloth is not needed)."""

import importlib.util
import os
import warnings
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
N_LAYERS = 64
DIM = 2048
RANK = 8

if __name__ != "__main__":
    if not torch.cuda.is_available() or not torch.distributed.is_nccl_available():
        pytest.skip("needs a CUDA GPU with NCCL", allow_module_level = True)
    pytest.importorskip("bitsandbytes")
    pytest.importorskip("peft")
    fsdp_mod = pytest.importorskip("torch.distributed.fsdp")
    if not hasattr(fsdp_mod, "fully_shard"):
        pytest.skip("torch has no FSDP2 fully_shard", allow_module_level = True)


def _load(name):
    path = _ROOT / "unsloth" / "models" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_unsloth_{name}_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model(device):
    import bitsandbytes as bnb
    from peft import LoraConfig, get_peft_model

    class Block(torch.nn.Module):
        def __init__(self, seed):
            super().__init__()
            torch.manual_seed(seed)
            self.q = bnb.nn.Linear4bit(
                DIM,
                DIM,
                bias = False,
                compute_dtype = torch.bfloat16,
                quant_type = "nf4",
                quant_storage = torch.bfloat16,
            )
            self.q.weight = bnb.nn.Params4bit(
                torch.randn(DIM, DIM) * 0.02,
                requires_grad = False,
                quant_type = "nf4",
                quant_storage = torch.bfloat16,
                blocksize = 64,
                module = self.q,
            )

        def forward(self, x):
            return x + self.q(x)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block(i + 1) for i in range(N_LAYERS)])

    model = Model().to(device)  # quantizes on the GPU
    model.is_loaded_in_4bit = True
    torch.manual_seed(0)
    config = LoraConfig(r = RANK, lora_alpha = 16, target_modules = ["q"], init_lora_weights = False)
    return get_peft_model(model, config)


def _sharded(device, world_size, q):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard

    model = _model(device)
    q.prepare_fsdp2_qlora(model)
    mesh = init_device_mesh("cuda", (world_size,))
    # One rank: a shard is the whole tensor and gathering it aliases it, so keep the shards on the
    # CPU and every gather allocates on the GPU, as a gather of two GPU shards does.
    kwargs = {} if world_size > 1 else {"offload_policy": CPUOffloadPolicy()}
    for layer in model.base_model.model.layers:
        fully_shard(layer, mesh = mesh, **kwargs)
    fully_shard(model, mesh = mesh, **kwargs)
    return model


def _worker(rank, world_size, port, queue):
    try:
        torch.cuda.set_device(rank if torch.cuda.device_count() >= world_size else 0)
        os.environ.update(
            MASTER_ADDR = "127.0.0.1",
            MASTER_PORT = str(port),
            RANK = str(rank),
            WORLD_SIZE = str(world_size),
        )
        import torch.distributed as dist

        dist.init_process_group("nccl", rank = rank, world_size = world_size)
        warnings.filterwarnings("ignore")
        m, q = _load("fsdp2_save"), _load("fsdp2_qlora")
        device = torch.device("cuda", torch.cuda.current_device())
        packed_layer = DIM * DIM // 2
        out = {
            "packed_layer": packed_layer,
            "packed_total": packed_layer * N_LAYERS,
            # fp32 lora_A + lora_B of every layer: the merged_4bit gather keeps them on the GPU.
            "lora_total": N_LAYERS * 2 * RANK * DIM * 4,
        }
        for method in ("merged_4bit", "merged_16bit"):
            for mode in ("head", "streamed"):
                model = _sharded(device, world_size, q)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                base = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                if mode == "streamed":
                    options = m.merged_save_gather_options(model, method)
                elif method == "merged_4bit":
                    options = dict(offload_to_cpu = m.offload_all_but_4bit_merge_operands(model))
                else:
                    options = dict(offload_to_cpu = True)
                written = None
                with m.fsdp2_gathered(model, **options) as writer:
                    if writer and method == "merged_4bit":
                        if mode == "streamed":
                            m.merge_4bit_lora_layer_by_layer(model)
                        merged = model.base_model.merge_and_unload()
                        written = {
                            k: v.detach().reshape(-1).view(torch.uint8).cpu().clone()
                            for k, v in merged.state_dict().items()
                            if isinstance(v, torch.Tensor)
                        }
                    elif writer:
                        # merged_16bit reads the LoRA weights (the base comes from the 16-bit files).
                        written = {
                            k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()
                            if "lora_" in k
                        }
                    torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated(device) - base
                out[(method, mode)] = dict(peak_above_shards = peak, written = written)
                del model
        dist.destroy_process_group()
        if rank == 0:
            for method in ("merged_4bit", "merged_16bit"):
                a, b = out[(method, "head")]["written"], out[(method, "streamed")]["written"]
                out[(method, "same_bytes")] = sorted(a) == sorted(b) and all(
                    torch.equal(a[k], b[k]) for k in a
                )
                for mode in ("head", "streamed"):
                    out[(method, mode)].pop("written")
        queue.put((rank, out))
    except BaseException as error:
        import traceback
        queue.put((rank, {"error": f"{type(error).__name__}: {error}\n{traceback.format_exc()}"}))


def test_merged_saves_hold_one_packed_layer_on_the_writer_gpu():
    import torch.multiprocessing as mp
    from tests._rendezvous_port import free_port

    world_size = 2 if torch.cuda.device_count() >= 2 else 1
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = free_port()
    procs = [
        ctx.Process(target = _worker, args = (r, world_size, port, queue)) for r in range(world_size)
    ]
    for p in procs:
        p.start()
    results = dict(queue.get(timeout = 600) for _ in procs)
    for p in procs:
        p.join(timeout = 60)
    _check(results)


def _check(results):
    for r in results.values():
        assert "error" not in r, r.get("error")
    r0 = results[0]
    layer, total = r0["packed_layer"], r0["packed_total"]
    print({k: v for k, v in r0.items()})
    for method in ("merged_4bit", "merged_16bit"):
        assert r0[(method, "same_bytes")], method
        head = r0[(method, "head")]["peak_above_shards"]
        streamed = r0[(method, "streamed")]["peak_above_shards"]
        # The previous placement holds the whole gathered packed base.
        assert head >= total, (method, head, total)
    # Streaming holds one gathered layer, and merged_4bit one layer's merge temporaries on top (bf16
    # dequant, fp32 LoRA delta, the requantize copy: a few dozen packed layers' worth, whatever the
    # depth) next to the adapters, never the base.
    assert r0[("merged_16bit", "streamed")]["peak_above_shards"] <= 2 * layer, r0
    merged_4bit = r0[("merged_4bit", "streamed")]["peak_above_shards"]
    assert merged_4bit <= 40 * layer + r0["lora_total"], r0
    assert merged_4bit < total, r0


if __name__ == "__main__":
    import argparse
    import json
    import queue as queue_module
    import sys

    sys.path.insert(0, str(_ROOT))
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required = True)
    args = parser.parse_args()
    from tests._rendezvous_port import free_port

    results = queue_module.Queue()
    _worker(0, 1, free_port(), results)
    rank, result = results.get()
    printable = {str(k): v for k, v in result.items()}
    with open(args.out, "w") as f:
        json.dump(printable, f, indent = 1)
    print(json.dumps(printable))
    if rank == 0:
        _check({0: result})
        print("CHECK PASSED")
