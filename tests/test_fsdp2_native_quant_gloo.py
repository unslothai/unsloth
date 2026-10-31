# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""bitsandbytes 4-bit weights under Unsloth's native FSDP2 wrap (fully_shard per block, then the root,
MixedPrecisionPolicy(param_dtype = compute dtype, reduce_dtype = float32)), two gloo ranks on CPU.

`fsdp2_native_quant.native_quant_context` around the fully_shard calls: a pre-quantized (uint8) 4-bit
weight is repacked as the compute dtype and sharded, the all-gathered weight is a Params4bit with its
QuantState and exact packed bytes inside the block's forward, the forward equals the unsharded model and
the trainable gradients equal the full-batch reference. Without it (control) the all-gathered weight is a
plain Parameter with no quant_state. A 16-bit model is left untouched. The modules are loaded from their files.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch

# Spawned gloo ranks re-import this module: keep them on the CPU (deterministic on GPU hosts; CUDA
# reads the mask at its lazy first use, after this).
if __import__("multiprocessing").parent_process() is not None or getattr(
    __import__("multiprocessing").current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

_ROOT = Path(__file__).resolve().parents[1]

fsdp_mod = pytest.importorskip("torch.distributed.fsdp")
if not hasattr(fsdp_mod, "fully_shard"):
    pytest.skip("torch has no FSDP2 fully_shard", allow_module_level = True)
if (
    sys.platform == "win32"
    or not torch.distributed.is_available()
    or not torch.distributed.is_gloo_available()
):
    pytest.skip("needs torch.distributed with gloo", allow_module_level = True)


def _load_models_package():
    """unsloth.models.fsdp2_native_quant and its relative imports, without importing unsloth."""
    pkg = types.ModuleType("_b_models")
    pkg.__path__ = [str(_ROOT / "unsloth" / "models")]
    sys.modules["_b_models"] = pkg
    out = {}
    for name in ("fsdp2_qlora", "fsdp2_save", "fsdp2_native_quant"):
        path = _ROOT / "unsloth" / "models" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"_b_models.{name}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"_b_models.{name}"] = module
        spec.loader.exec_module(module)
        out[name] = module
    return out


def _free_port():
    from tests._rendezvous_port import free_port
    return free_port()


N_BLOCKS = 3
HIDDEN = 128


def _linear4bit(bnb, seed, storage):
    torch.manual_seed(seed)
    layer = bnb.nn.Linear4bit(
        HIDDEN,
        HIDDEN,
        bias = False,
        compute_dtype = torch.bfloat16,
        quant_type = "nf4",
        quant_storage = storage,
    )
    layer.weight = bnb.nn.Params4bit(
        torch.randn(HIDDEN, HIDDEN) * 0.05,
        requires_grad = False,
        quant_type = "nf4",
        quant_storage = storage,
        blocksize = 64,
        module = layer,
    )
    return layer.to("cpu")


def _model(bnb, quantized = True):
    class Block(torch.nn.Module):
        def __init__(self, i):
            super().__init__()
            if quantized:
                # uint8: what a pre-quantized checkpoint loads as.
                self.base = _linear4bit(bnb, 10 + i, torch.uint8)
            else:
                torch.manual_seed(10 + i)
                self.base = torch.nn.Linear(HIDDEN, HIDDEN, bias = False, dtype = torch.bfloat16)
                self.base.requires_grad_(False)
            torch.manual_seed(100 + i)
            self.lora_A = torch.nn.Linear(HIDDEN, 8, bias = False, dtype = torch.bfloat16)
            self.lora_B = torch.nn.Linear(8, HIDDEN, bias = False, dtype = torch.bfloat16)
            torch.nn.init.normal_(self.lora_B.weight, std = 0.05)
            self.seen = []

        def forward(self, x):
            w = self.base.weight
            self.seen.append((type(w).__name__, getattr(w, "quant_state", None) is not None))
            return x + torch.tanh(self.base(x) + self.lora_B(self.lora_A(x)))

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block(i) for i in range(N_BLOCKS)])
            torch.manual_seed(7)
            self.norm = torch.nn.LayerNorm(HIDDEN, dtype = torch.bfloat16)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return self.norm(x)

    return Model()


def _shard(
    model,
    world_size,
    native_quant_context = None,
):
    import contextlib

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    mesh = init_device_mesh("cpu", (world_size,))
    mp = MixedPrecisionPolicy(param_dtype = torch.bfloat16, reduce_dtype = torch.float32)
    ctx = (
        native_quant_context(model, dtype = torch.bfloat16)
        if native_quant_context
        else contextlib.nullcontext()
    )
    with ctx as info:
        for layer in model.layers:
            fully_shard(layer, mesh = mesh, mp_policy = mp)
        fully_shard(model, mesh = mesh, mp_policy = mp)
    return info


def _case(rank, world_size, mods, case):
    bnb = pytest.importorskip("bitsandbytes")
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor

    nq = mods["fsdp2_native_quant"]
    quantized = case != "dense"
    model = _model(bnb, quantized)
    reference = _model(bnb, quantized)
    raw = [l.base.weight.data.view(torch.uint8).clone() for l in model.layers]
    xs = [
        torch.randn(4, HIDDEN, generator = torch.Generator().manual_seed(50 + r)).to(torch.bfloat16)
        for r in range(world_size)
    ]
    with torch.no_grad():
        ref_out = reference(xs[rank])
    ref_loss = sum(reference(x).float().square().mean() for x in xs) / world_size
    ref_loss.backward()
    before_ids = {id(p) for p in model.parameters()}

    out = {}
    if case == "control":
        try:
            _shard(model, world_size)
            with torch.no_grad():
                got = model(xs[rank])
            out["forward_equal"] = torch.equal(got, ref_out)
            out["seen"] = sorted({s for l in model.layers for s in l.seen})
        except Exception as error:
            out["forward_equal"] = f"{type(error).__name__}: {error}"[:300]
        return out

    info = _shard(model, world_size, nq.native_quant_context)
    report = nq.after_native_shard(model)
    out["info"] = (
        None if info is None else {k: info[k] for k in ("params4bit", "converted", "patched")}
    )
    out["storage"] = None if info is None else info["storage"]
    out["unsharded"] = report["unsharded"]
    out["params4bit_sharded"] = report["params4bit_sharded"]
    out["same_objects_dense"] = case == "dense" and all(
        isinstance(p, DTensor) for p in model.parameters()
    )
    for l in model.layers:
        l.seen.clear()
    with torch.no_grad():
        got = model(xs[rank])
    out["forward_equal"] = torch.equal(got, ref_out)
    out["seen"] = sorted({s for l in model.layers for s in l.seen})
    # Exact packed bytes on the all-gathered weight, inside the block's unshard.
    exact = []
    for l, r in zip(model.layers, raw):
        l.unshard()
        exact.append(torch.equal(l.base.weight.data.reshape(-1).view(torch.uint8), r.reshape(-1)))
        l.reshard()
    out["bytes_exact"] = all(exact)
    model(xs[rank]).float().square().mean().backward()
    diffs = []
    ref_named = dict(reference.named_parameters())
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        g = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
        diffs.append(float((g.float() - ref_named[name].grad.float()).abs().max()))
    out["grad_max_diff"] = max(diffs)
    out["all_dtensor_after"] = all(isinstance(p, DTensor) for p in model.parameters())
    return out


def _worker(rank, world_size, port, case, queue):
    try:
        os.environ.update(
            MASTER_ADDR = "127.0.0.1",
            MASTER_PORT = str(port),
            RANK = str(rank),
            WORLD_SIZE = str(world_size),
        )
        import torch.distributed as dist

        dist.init_process_group("gloo", rank = rank, world_size = world_size)
        mods = _load_models_package()
        result = _case(rank, world_size, mods, case)
        queue.put((rank, result))
        dist.destroy_process_group()
    except BaseException as error:
        import traceback
        queue.put((rank, {"error": f"{type(error).__name__}: {error}\n{traceback.format_exc()}"}))


def _run(case, world_size = 2):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target = _worker, args = (r, world_size, port, case, queue))
        for r in range(world_size)
    ]
    for p in procs:
        p.start()
    results = {}
    for _ in procs:
        rank, result = queue.get(timeout = 300)
        results[rank] = result
    for p in procs:
        p.join(timeout = 60)
    for r in results.values():
        assert "error" not in r, r.get("error")
    return results


def test_native_fsdp2_shards_4bit_and_keeps_quant_state():
    pytest.importorskip("bitsandbytes")
    results = _run("native")
    for r in results.values():
        assert r["info"] == {"params4bit": N_BLOCKS, "converted": N_BLOCKS, "patched": True}, r
        assert r["storage"] == ["bfloat16"], r
        assert r["unsharded"] == [], r
        assert r["params4bit_sharded"] == N_BLOCKS, r
        assert r["seen"] == [("Params4bit", True)], r
        assert r["bytes_exact"], r
        assert r["forward_equal"] is True, r
        assert r["grad_max_diff"] < 2e-2, r
        assert r["all_dtensor_after"], r


def test_without_native_quant_context_the_gathered_weight_is_not_4bit():
    # Control: the same fully_shard calls with no 4-bit handling. bitsandbytes' own Linear4bit forward
    # recovers (it re-reads the module's copy of the QuantState), but the all-gathered weight the block
    # sees is a plain Parameter without quant_state: what Unsloth's fused LoRA kernels and the zoo MoE
    # 4-bit paths read (they take it for a dense weight, or crash).
    pytest.importorskip("bitsandbytes")
    # torch < 2.12 cannot shard the uint8 weight at all ("only Tensors of floating point dtype can
    # require gradients"); later torch shards it but gathers a plain Parameter.
    results = _run("control")
    for r in results.values():
        raised = isinstance(r["forward_equal"], str) and "floating point" in r["forward_equal"]
        assert raised or r.get("seen") == [("Parameter", False)], r


def test_native_fsdp2_leaves_16bit_untouched():
    pytest.importorskip("bitsandbytes")
    results = _run("dense")
    for r in results.values():
        assert r["info"] is None, r
        assert r["unsharded"] == [] and r["params4bit_sharded"] == 0, r
        assert r["forward_equal"] is True and r["all_dtensor_after"], r
        assert r["grad_max_diff"] < 2e-2, r
