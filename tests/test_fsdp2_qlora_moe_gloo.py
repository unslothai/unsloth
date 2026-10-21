# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FSDP2 + bitsandbytes 4-bit MoE experts: the all-gathered weight must look like the Params4bit it
was sharded from. The unsloth-zoo 4-bit MoE paths pick their kernel with `isinstance(w, Params4bit)`
(ModuleList experts: moe_grouped_modulelist `_expert_weight` / `_lin_compute_dtype`; transformers v5
3D experts: moe_utils_bnb4bit `_is_bnb4bit_param`) and read a 3D stack's logical shape from
`w._original_shape`. FSDP2 rebuilds the gathered weight as a plain nn.Parameter, so without the
fsdp2_qlora wrapper those paths fed packed bytes to a dense matmul ("contraction dimension of mat_a
and mat_b must match", "size mismatch").

GPU-free: real bitsandbytes 4-bit weights (CPU backend) sharded over 2 gloo ranks by fully_shard.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = Path(
    os.environ.get("UNSLOTH_FSDP2_QLORA_PATH", _ROOT / "unsloth" / "models" / "fsdp2_qlora.py")
)

bnb = pytest.importorskip("bitsandbytes")
fully_shard_mod = pytest.importorskip("torch.distributed.fsdp")
if not hasattr(fully_shard_mod, "fully_shard"):
    pytest.skip("torch has no FSDP2 fully_shard", allow_module_level = True)
if not isinstance(getattr(bnb.nn, "Params4bit", None), type):
    pytest.skip("bitsandbytes without a Params4bit class", allow_module_level = True)

H, I, E = 64, 128, 4


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_unsloth_fsdp2_qlora_moe_under_test", _MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _free_port():
    from tests._rendezvous_port import free_port
    return free_port()


def _stack_params4bit(shape, seed, module):
    """A 3D expert stack quantized as one tensor, as zoo `_make_expert_params4bit` builds it."""
    from bitsandbytes.functional import quantize_4bit
    from bitsandbytes.nn import Params4bit

    torch.manual_seed(seed)
    dense = torch.randn(shape) * 0.05
    packed, state = quantize_4bit(dense, blocksize = 64, quant_type = "nf4")
    param = torch.Tensor._make_subclass(Params4bit, packed, False)
    param.quant_state = state
    param.blocksize = 64
    param.compress_statistics = False
    param.quant_type = "nf4"
    param.quant_storage = torch.uint8
    param.bnb_quantized = True
    param.module = module
    param._original_shape = torch.Size(shape)
    return param


def _linear4bit(in_features, out_features, seed):
    torch.manual_seed(seed)
    lin = bnb.nn.Linear4bit(
        in_features, out_features, bias = False, compute_dtype = torch.bfloat16, quant_type = "nf4"
    )
    lin.weight = bnb.nn.Params4bit(
        torch.randn(out_features, in_features) * 0.05,
        requires_grad = False,
        quant_type = "nf4",
        blocksize = 64,
        module = lin,
    )
    return lin.to("cpu")


def _is_4bit(w):
    from bitsandbytes.nn import Params4bit
    return isinstance(w, Params4bit) and getattr(w, "quant_state", None) is not None


def _dense(w, shape = None):
    """What the zoo 4-bit MoE readers do: dequantize a Params4bit to its logical shape, else use the
    tensor as a dense weight (which, for gathered packed bytes, is the crash this file guards)."""
    if _is_4bit(w):
        out = bnb.functional.dequantize_4bit(w.data, w.quant_state)
        shape = getattr(w, "_original_shape", None) or shape
        return (out.reshape(shape) if shape is not None else out).float()
    return w.float()


class _Experts3D(torch.nn.Module):
    """transformers v5 layout: one gate_up_proj [E, 2I, H] and one down_proj [E, H, I]."""

    def __init__(self, seed):
        super().__init__()
        self.gate_up_proj = _stack_params4bit((E, 2 * I, H), seed, self)
        self.down_proj = _stack_params4bit((E, H, I), seed + 1, self)
        self.seen = []

    def forward(self, x, expert):
        gu_w, dn_w = self.gate_up_proj, self.down_proj
        self.seen.append(
            dict(
                is_4bit = _is_4bit(gu_w) and _is_4bit(dn_w),
                shape = tuple(getattr(gu_w, "_original_shape", ()) or ()),
                has_module = getattr(gu_w, "module", None) is self,
                gathered_mark = getattr(gu_w, "_unsloth_fsdp2_gathered", False),
                # Params4bit.__torch_function__ reads .module and the quant attributes on chunk.
                chunk = len(torch.chunk(gu_w, 2)),
            )
        )
        gate_up = _dense(gu_w)[expert]
        down = _dense(dn_w)[expert]
        g, u = (x @ gate_up.t()).chunk(2, dim = -1)
        return (torch.nn.functional.silu(g) * u) @ down.t()


class _ExpertMLP(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.gate_proj = _linear4bit(H, I, seed)
        self.up_proj = _linear4bit(H, I, seed + 1)
        self.down_proj = _linear4bit(I, H, seed + 2)


class _ExpertsList(torch.nn.Module):
    """transformers 4.x layout: a ModuleList of Linear4bit experts, read by zoo's grouped MoE."""

    def __init__(self, seed):
        super().__init__()
        self.experts = torch.nn.ModuleList([_ExpertMLP(seed + 10 * e) for e in range(E)])
        self.seen = []

    def forward(self, x, expert):
        ex = self.experts[expert]
        ws = [ex.gate_proj.weight, ex.up_proj.weight, ex.down_proj.weight]
        self.seen.append(dict(is_4bit = all(_is_4bit(w) for w in ws)))
        g = x @ _dense(ws[0], (I, H)).t()
        u = x @ _dense(ws[1], (I, H)).t()
        return (torch.nn.functional.silu(g) * u) @ _dense(ws[2], (H, I)).t()


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = _Experts3D(100)
        self.listed = _ExpertsList(200)
        torch.manual_seed(7)
        self.adapter = torch.nn.Linear(H, H, bias = False)

    def forward(self, x):
        x = self.adapter(x)
        out = 0
        for e in range(E):
            out = out + self.stack(x, e) + self.listed(x, e)
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
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        dist.init_process_group("gloo", rank = rank, world_size = world_size)
        m = _load_module()
        model, reference = _Model(), _Model()
        torch.manual_seed(1234 + rank)
        x = torch.randn(3, H)
        ref_x = x.clone().requires_grad_(True)
        expected = reference(ref_x)
        expected.sum().backward()

        info = m.prepare_fsdp2_qlora(
            model, torch.bfloat16 if case == "bf16" else None, float_storage = case == "bf16"
        )
        mesh = init_device_mesh("cpu", (world_size,))
        # torch < 2.12 cannot build a uint8 sharded Parameter without the guard.
        with m.nonfloat_param_guard(model):
            fully_shard(model.stack, mesh = mesh)
            fully_shard(model.listed, mesh = mesh)
            fully_shard(model, mesh = mesh)
        sharded = [
            n
            for n, p in model.named_parameters()
            if "adapter" not in n and p.to_local().numel() * world_size == p.full_tensor().numel()
        ]
        model.stack.seen.clear()
        model.listed.seen.clear()
        x = x.clone().requires_grad_(True)
        out = model(x)
        out.sum().backward()
        # Outside forward the module holds FSDP2's sharded DTensor again, whatever the gathered class.
        from torch.distributed.tensor import DTensor

        resharded = isinstance(model.stack.gate_up_proj, DTensor)
        queue.put(
            (
                rank,
                dict(
                    params4bit = info["params4bit"],
                    sharded = len(sharded),
                    stack_seen = model.stack.seen,
                    list_seen = model.listed.seen,
                    max_diff = (out - expected).abs().max().item(),
                    # FSDP2's backward hooks may sum the per-expert dX contributions in another order.
                    grad_close = torch.allclose(x.grad, ref_x.grad, rtol = 1e-5, atol = 1e-6),
                    adapter_grad = model.adapter.weight.grad is not None,
                    resharded = resharded,
                ),
            )
        )
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
        rank, result = queue.get(timeout = 240)
        results[rank] = result
    for p in procs:
        p.join(timeout = 60)
    return results


_RUN_DIST = pytest.mark.skipif(
    sys.platform == "win32"
    or not torch.distributed.is_available()
    or not torch.distributed.is_gloo_available(),
    reason = "needs torch.distributed with gloo",
)


@_RUN_DIST
@pytest.mark.parametrize("case", ["bf16", "uint8"])
def test_gathered_4bit_experts_keep_params4bit_class_and_shape(case):
    m = _load_module()
    if case == "bf16" and (m._missing_fsdp2_anchors() or m._fsdp2_checks_frozen_dtypes()):
        pytest.skip("this torch keeps 4-bit weights uint8 under FSDP2")
    if case == "uint8" and m._fsdp2_checks_frozen_dtypes():
        pytest.skip("torch < 2.8 leaves uint8 weights out of the FSDP2 units")
    results = _run(case)
    for rank, r in results.items():
        assert "error" not in r, r.get("error")
        assert r["params4bit"] == 2 + 3 * E, r
        assert r["sharded"] == r["params4bit"], r
        assert r["stack_seen"] and r["list_seen"], r
        for seen in r["stack_seen"]:
            assert seen["is_4bit"], r
            assert seen["shape"] == (E, 2 * I, H), r
            assert seen["has_module"], r
            # unsloth-zoo re-keys its NF4 pointer tables per call for these (moe_ready_epoch).
            assert seen["gathered_mark"], r
            assert seen["chunk"] == 2, r
        assert all(seen["is_4bit"] for seen in r["list_seen"]), r
        # Same packed bytes, same dequantization: bit-exact vs the unsharded model in forward.
        assert r["max_diff"] == 0.0, r
        assert r["grad_close"], r
        assert r["adapter_grad"], r
        assert r["resharded"], r


def test_restore_param_class_leaves_other_tensors_alone():
    m = _load_module()
    from bitsandbytes.nn import Params4bit

    plain = torch.nn.Parameter(torch.zeros(4), requires_grad = False)
    ptr = plain.data_ptr()
    owner = torch.nn.Linear(1, 1)
    m._restore_param_class(plain, Params4bit, owner)
    assert type(plain) is Params4bit and plain.data_ptr() == ptr
    assert plain.module is owner
    # Not a plain Parameter (a DTensor or another subclass), or no recorded class: untouched.
    sub = torch.Tensor._make_subclass(Params4bit, torch.zeros(4), False)

    class Other(torch.nn.Parameter):
        pass

    other = Other(torch.zeros(4), requires_grad = False)
    m._restore_param_class(other, Params4bit)
    assert type(other) is Other
    m._restore_param_class(sub, Params4bit)
    assert type(sub) is Params4bit
    keep = torch.nn.Parameter(torch.zeros(4), requires_grad = False)
    m._restore_param_class(keep, None)
    assert type(keep) is torch.nn.Parameter
