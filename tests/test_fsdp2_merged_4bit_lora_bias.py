# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FSDP2 merged_4bit save with PEFT `lora_bias = True` on a biased 4-bit layer (Qwen2 q/k/v).

PEFT's bitsandbytes merge adds lora_B's bias into the base layer's bias, so the merged_4bit offload
filter (`offload_all_but_4bit_merge_operands`) must keep that base bias on the GPU with the adapters.
Before, it went to the CPU and `merge_and_unload` raised "Expected all tensors to be on the same device".

- the filter decision, on CPU (modules loaded from their files: importing unsloth needs a GPU);
- the real path on one GPU: bitsandbytes Linear4bit(bias=True) + PEFT lora_bias, FSDP2 fully_shard on a
  one-rank NCCL mesh, `fsdp2_gathered` with the filter, then `merge_and_unload` (what unsloth_zoo's
  merge_and_overwrite_lora runs for merged_4bit).
"""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    path = _ROOT / "unsloth" / "models" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_unsloth_{name}_lora_bias_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Linear4bit(torch.nn.Linear):
    """Stands in for bitsandbytes' Linear4bit (matched by class name)."""


def _lora_layer(quantized, lora_bias):
    layer = torch.nn.Module()
    layer.base_layer = (Linear4bit if quantized else torch.nn.Linear)(4, 4, bias = True)
    layer.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(4, 2, bias = False)})
    layer.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 4, bias = lora_bias)})
    return layer


def test_merged_4bit_filter_keeps_the_base_bias_a_lora_bias_merges_into():
    m = _load("fsdp2_save")
    model = torch.nn.Module()
    model.q4_lora_bias = _lora_layer(quantized = True, lora_bias = True)
    model.q4 = _lora_layer(quantized = True, lora_bias = False)
    model.plain_lora_bias = _lora_layer(quantized = False, lora_bias = True)
    params = dict(model.named_parameters())
    offload = m.offload_all_but_4bit_merge_operands(model)
    decide = lambda name: offload(name, params[name])
    # The merge operands of a 4-bit layer stay put, including the base bias lora_B's bias joins.
    assert not decide("q4_lora_bias.base_layer.bias")
    assert not decide("q4_lora_bias.lora_B.default.bias")
    assert not decide("q4_lora_bias.lora_A.default.weight")
    # Unchanged: without lora_bias the base bias is not a merge operand, and unquantized layers merge
    # on the CPU with everything else.
    assert decide("q4.base_layer.bias")
    assert decide("plain_lora_bias.base_layer.bias")
    assert decide("plain_lora_bias.lora_B.default.bias")
    assert decide("plain_lora_bias.base_layer.weight")


_GPU_SCRIPT = textwrap.dedent(
    r"""
    import importlib.util, json, os, sys, torch
    root = sys.argv[1]
    def load(name):
        spec = importlib.util.spec_from_file_location(f"_u_{name}", f"{root}/unsloth/models/{name}.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
    import bitsandbytes as bnb
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from peft import LoraConfig, get_peft_model
    os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = sys.argv[2], RANK = "0", WORLD_SIZE = "1")
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", rank = 0, world_size = 1)
    m = load("fsdp2_save"); q = load("fsdp2_qlora")

    def lin4(seed):
        torch.manual_seed(seed)
        layer = bnb.nn.Linear4bit(128, 128, bias = True, compute_dtype = torch.bfloat16,
                                  quant_type = "nf4", quant_storage = torch.bfloat16)
        layer.weight = bnb.nn.Params4bit(torch.randn(128, 128) * 0.05, requires_grad = False,
                                         quant_type = "nf4", quant_storage = torch.bfloat16,
                                         blocksize = 64, module = layer)
        layer.bias = torch.nn.Parameter(torch.randn(128) * 0.1, requires_grad = False)
        return layer

    class Block(torch.nn.Module):
        def __init__(self, s):
            super().__init__(); self.q_proj = lin4(s); self.o_proj = lin4(s + 1)
        def forward(self, x):
            return x + self.o_proj(torch.tanh(self.q_proj(x)))

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.layers = torch.nn.ModuleList([Block(10 * i) for i in range(2)])
        def forward(self, x):
            for layer in self.layers: x = layer(x)
            return x

    net = Net().cuda()
    net.is_loaded_in_4bit = True  # what transformers sets; PEFT picks its bitsandbytes LoRA layer on it
    model = get_peft_model(net, LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj", "o_proj"],
                                           lora_bias = True))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.normal_(0, 0.05)
    params = dict(model.named_parameters())
    expect = {
        name[: -len(".base_layer.bias")]: p.detach().float().cpu()
        + params[name.replace(".base_layer.bias", ".lora_B.default.bias")].detach().float().cpu()
        for name, p in params.items() if name.endswith(".base_layer.bias")
    }
    q.prepare_fsdp2_qlora(model)
    mesh = init_device_mesh("cuda", (1,))
    for block in model.base_model.model.layers:
        fully_shard(block, mesh = mesh)
    fully_shard(model, mesh = mesh)
    out = {"units": len(m.fsdp2_units(model))}
    offload = m.offload_all_but_4bit_merge_operands(model)
    try:
        with m.fsdp2_gathered(model, offload_to_cpu = offload) as writer:
            merged = model.base_model.merge_and_unload()
            got = {name: mod.bias.detach().float().cpu() for name, mod in merged.named_modules()
                   if isinstance(mod, bnb.nn.Linear4bit)}
            out["max_bias_err"] = max(
                (got[name.replace("base_model.model.", "")] - value).abs().max().item()
                for name, value in expect.items()
            )
            out["merged"] = True
    except RuntimeError as error:
        out["merged"] = False
        out["error"] = str(error).splitlines()[0]
    print("RESULT " + json.dumps(out))
    dist.destroy_process_group()
    """
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA GPU")
def test_fsdp2_merged_4bit_merges_lora_bias_into_a_sharded_base_bias(tmp_path):
    pytest.importorskip("bitsandbytes")
    pytest.importorskip("peft")
    fsdp = pytest.importorskip("torch.distributed.fsdp")
    if not hasattr(fsdp, "fully_shard") or not torch.distributed.is_nccl_available():
        pytest.skip("needs FSDP2 fully_shard and NCCL")
    from tests._rendezvous_port import free_port

    script = tmp_path / "fsdp2_lora_bias_merge.py"
    script.write_text(_GPU_SCRIPT)
    env = dict(os.environ)
    visible = env.get("CUDA_VISIBLE_DEVICES", "")
    if "," in visible:
        env["CUDA_VISIBLE_DEVICES"] = visible.split(",")[0]
    proc = subprocess.run(
        [sys.executable, str(script), str(_ROOT), str(free_port())],
        env = env,
        capture_output = True,
        text = True,
        timeout = 600,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    assert lines, proc.stdout[-2000:] + proc.stderr[-4000:]
    result = json.loads(lines[-1][len("RESULT ") :])
    assert result["units"] > 0, result
    assert result["merged"], result
    assert result["max_bias_err"] < 1e-2, result
