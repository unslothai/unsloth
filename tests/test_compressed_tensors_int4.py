# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""compressed-tensors INT4 / INT8 kept packed: exact decode kernels and the default load route.

The kernels are checked against compressed-tensors' own ``decompress`` (bit for bit) and a dense
matmul; the route loads a tiny packed Llama (built by test_compressed_tensors_bnb) and compares
it with the same checkpoint decompressed to bf16 on disk.
"""

import copy
import itertools
import os
import sys

import pytest
import torch
from llama_patch_isolation import restore_llama_patches  # noqa: F401
from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

import unsloth  # noqa: F401
from unsloth.models.compressed_tensors_bnb import _transformers_supports_weight_converters

try:
    import inspect
    from compressed_tensors.quantization.lifecycle.forward import dequantize as _ct_dequantize

    HAS_CT = True
    # compressed-tensors 0.19 dropped GPTQ activation ordering (#840): its compressor ignores weight_g_idx.
    CT_HONOURS_G_IDX = "g_idx" in inspect.signature(_ct_dequantize).parameters
except Exception:
    HAS_CT = CT_HONOURS_G_IDX = False

HAS_CONVERTERS = _transformers_supports_weight_converters()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
needs_gpu = pytest.mark.skipif(not has_real_cuda(), reason = "needs a CUDA device")
needs_ct = pytest.mark.skipif(not HAS_CT, reason = "needs compressed-tensors")


def _packed_layer(
    out_f,
    in_f,
    bits,
    group_size,
    symmetric,
    actorder,
    scale_dtype,
    strategy = "group",
):
    from compressed_tensors.compressors import BaseCompressor
    from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs
    from compressed_tensors.quantization.utils import calculate_qparams

    args = QuantizationArgs(
        num_bits = bits,
        type = "int",
        symmetric = symmetric,
        strategy = strategy,
        group_size = group_size if strategy == "group" else None,
        actorder = "weight" if actorder else None,
    )
    if actorder and not CT_HONOURS_G_IDX:
        pytest.skip("this compressed-tensors cannot write or decompress activation-ordered weights")
    scheme = QuantizationScheme(targets = ["Linear"], weights = args)
    w = torch.randn(out_f, in_f) * 0.05
    gs = group_size if strategy == "group" else in_f
    g_idx = None
    if actorder:
        g_idx = (torch.randperm(in_f) // gs).to(torch.int32)
        grouped = w[:, torch.argsort(g_idx)].reshape(out_f, in_f // gs, gs)
    else:
        grouped = w.reshape(out_f, in_f // gs, gs)
    scale, zp = calculate_qparams(grouped.amin(-1), grouped.amax(-1), args)
    state = {"weight": w, "weight_scale": scale.to(scale_dtype)}
    if not symmetric:
        state["weight_zero_point"] = zp.to(torch.int8)
    if g_idx is not None:
        state["weight_g_idx"] = g_idx
    comp = BaseCompressor.get_value_from_registry("pack-quantized")
    packed = comp.compress(state, scheme)
    ref = comp.decompress(dict(packed), scheme)["weight"].to(torch.bfloat16)
    return {k: v.cuda() for k, v in packed.items()}, ref.cuda(), gs


CASES = [
    # bits, group, symmetric, actorder, scale dtype, strategy
    (4, 128, True, False, torch.bfloat16, "group"),
    (4, 64, False, False, torch.bfloat16, "group"),
    (4, 128, False, True, torch.bfloat16, "group"),
    (4, 32, True, False, torch.float32, "group"),
    (8, 128, True, False, torch.bfloat16, "group"),
    (8, 64, False, False, torch.float16, "group"),
    (2, 64, True, False, torch.bfloat16, "group"),
    (4, None, True, False, torch.bfloat16, "channel"),
]


@needs_gpu
@needs_ct
@pytest.mark.parametrize("bits,group,sym,actorder,scale_dtype,strategy", CASES)
def test_kernels_decode_bit_exact_and_multiply_like_dense(
    bits, group, sym, actorder, scale_dtype, strategy, monkeypatch
):
    from unsloth.kernels.int4_packed import (
        Int4QuantState,
        int4_dequantize,
        int4_dequantize_weight,
        int4_matmul,
        int4_matmul_t,
    )

    torch.manual_seed(0)
    out_f, in_f = 200, 512  # out_f not a multiple of the zero-point packing
    packed, ref, gs = _packed_layer(out_f, in_f, bits, group, sym, actorder, scale_dtype, strategy)
    qs = Int4QuantState(
        packed["weight_scale"],
        packed.get("weight_zero_point"),
        packed.get("weight_g_idx"),
        (out_f, in_f),
        bits,
        gs,
        torch.bfloat16,
    )
    W = packed["weight_packed"]
    assert torch.equal(int4_dequantize(W, qs), ref)
    assert torch.equal(int4_dequantize_weight(W.t(), qs), ref.t())
    import unsloth.kernels.int4_packed as ip

    for rows, gemv in itertools.product((1, 3, 5, 64), (True, False)):
        monkeypatch.setattr(ip, "GEMV_MAX_ROWS", 1 << 30 if gemv else 0)
        x = torch.randn(rows, in_f, device = "cuda", dtype = torch.bfloat16)
        y = int4_matmul(x, W, qs)
        want = x.float() @ ref.float().t()
        assert y.shape == (rows, out_f) and y.dtype == torch.bfloat16
        assert ((y.float() - want).norm() / want.norm()) < 5e-3
        dy = torch.randn(rows, out_f, device = "cuda", dtype = torch.bfloat16)
        dx = int4_matmul_t(dy, W, qs)
        want = dy.float() @ ref.float()
        assert ((dx.float() - want).norm() / want.norm()) < 5e-3


def _has_marlin():
    if not has_real_cuda():
        return False
    import unsloth.kernels.int4_packed as ip
    return bool(ip._marlin_api())


needs_sm80 = pytest.mark.skipif(
    not has_real_cuda() or torch.cuda.get_device_capability() < (8, 0),
    reason = "tensor-core layouts need sm_80+",
)


def _qs(
    ip,
    packed,
    shape,
    bits,
    gs,
    dtype = torch.bfloat16,
):
    return ip.Int4QuantState(
        packed["weight_scale"], packed.get("weight_zero_point"), None, shape, bits, gs, dtype
    )


REPACK_CASES = [
    (128, True, 256),
    (128, False, 512),
    (32, False, 256),
    (64, True, 1024),
    (256, False, 512),
]


@needs_gpu
@needs_ct
@needs_sm80
@pytest.mark.parametrize("layout", ["tinygemm", "marlin"])
@pytest.mark.parametrize("group,sym,out_f", REPACK_CASES)
def test_repacked_weights_dequantize_exactly_and_save_in_checkpoint_layout(
    layout, group, sym, out_f, monkeypatch
):
    import unsloth.kernels.int4_packed as ip

    if layout == "marlin" and (not _has_marlin() or group == 256):
        pytest.skip("needs vLLM's Marlin kernels (groups <= 128)")
    monkeypatch.setenv("UNSLOTH_INT4_LAYOUT", layout)
    torch.manual_seed(0)
    packed, ref, gs = _packed_layer(out_f, 1024, 4, group, sym, False, torch.bfloat16)
    W = packed["weight_packed"]
    checkpoint = W.clone()
    qs = _qs(ip, packed, (out_f, 1024), 4, gs)
    assert ip.int4_repack_(W, qs) == layout
    # Replaced in place (no second copy), same shape; training dequantizes the exact same weights; saves unchanged.
    assert W.shape == checkpoint.shape and not torch.equal(W, checkpoint)
    assert torch.equal(ip.int4_dequantize(W, qs), ref)
    assert torch.equal(ip.int4_unpack(W, qs), checkpoint)
    fast = ip._fast_rows(layout, torch.cuda.current_device())
    for rows in (1, 3, fast, fast + 1):
        x = torch.randn(rows, 1024, device = "cuda", dtype = torch.bfloat16)
        with torch.no_grad():
            y = ip.int4_matmul(x, W, qs)
        want = x.float() @ ref.float().t()
        assert ((y.float() - want).norm() / want.norm()) < 5e-3
    out = torch.empty(1, 1, out_f, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        got = ip.int4_matmul(x[:1].view(1, 1, 1024), W, qs, out = out)
        want = ip.int4_matmul(x[:1], W, qs)
    assert got.data_ptr() == out.data_ptr() and torch.equal(got.view(1, -1), want)
    # With grad enabled every row count decodes + cuBLAS: training numerics are those of the checkpoint layout.
    assert torch.equal(ip.int4_matmul(x, W, qs), x @ ref.t())


@needs_gpu
@needs_ct
@pytest.mark.parametrize(
    "why", ["8bit", "fp32_scale", "fp16_without_marlin", "channel", "kill_switch", "layout_changed"]
)
def test_layers_that_cannot_be_repacked_keep_the_checkpoint_layout(why, monkeypatch):
    import unsloth.kernels.int4_packed as ip

    monkeypatch.setattr(ip, "_MARLIN_API", False)
    if why == "kill_switch":
        monkeypatch.setenv("UNSLOTH_INT4_REPACK", "0")
    if why == "layout_changed":
        # A torch release that changes tinygemm's tile layout must fail the probe, not corrupt weights.
        TN, TK, nfast, bits = ip._LAYOUTS["tinygemm"]
        monkeypatch.setitem(ip._LAYOUTS, "tinygemm", (TN, TK, nfast, bits[::-1]))
        monkeypatch.setattr(ip, "_LAYOUT_OK", {})
    bits = 8 if why == "8bit" else 4
    scale_dtype = {"fp32_scale": torch.float32, "fp16_without_marlin": torch.float16}.get(
        why, torch.bfloat16
    )
    dtype = torch.float16 if why == "fp16_without_marlin" else torch.bfloat16
    strategy = "channel" if why == "channel" else "group"
    torch.manual_seed(0)
    packed, ref, gs = _packed_layer(256, 512, bits, 128, True, False, scale_dtype, strategy)
    W = packed["weight_packed"]
    checkpoint = W.clone()
    qs = _qs(ip, packed, (256, 512), bits, gs, dtype)
    assert ip.int4_repack_(W, qs) is None and qs.layout is None and torch.equal(W, checkpoint)
    x = torch.randn(2, 512, device = "cuda", dtype = dtype)
    with torch.no_grad():
        y = ip.int4_matmul(x, W, qs)
    want = x.float() @ ref.float().t()
    assert ((y.float() - want).norm() / want.norm()) < 5e-3


@needs_gpu
@needs_ct
@needs_sm80
def test_packed_linear_state_dict_round_trips_through_the_checkpoint_layout():
    from unsloth.models.compressed_tensors_int4 import (
        Int4PackedLinear,
        finalize_int4_packed_linears,
    )
    import unsloth.kernels.int4_packed as ip

    torch.manual_seed(0)
    packed, ref, gs = _packed_layer(256, 1024, 4, 128, False, False, torch.bfloat16)
    lin = Int4PackedLinear.__new__(Int4PackedLinear)
    torch.nn.Module.__init__(lin)
    lin.in_features, lin.out_features = 1024, 256
    for name, tensor in packed.items():
        lin.register_parameter(name, torch.nn.Parameter(tensor.clone(), requires_grad = False))
    lin.register_parameter("bias", None)
    lin._int4_bits, lin._int4_group_size = 4, gs
    model = torch.nn.Sequential(lin)
    finalize_int4_packed_linears(model, torch.bfloat16)
    assert lin.quant_state.layout is not None
    assert not torch.equal(lin.weight_packed, packed["weight_packed"])
    state = model.state_dict()
    assert torch.equal(state["0.weight_packed"], packed["weight_packed"])
    assert torch.equal(lin.dequantize_weight(), ref)
    # Training takes the exact dequantize + matmul whatever the layout; only inference uses the fused kernel.
    x = torch.randn(4, 1024, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    assert torch.equal(lin(x), x @ ref.t())
    # Loading checkpoint-layout words resets the layer to that layout; unrelated loads leave it alone.
    model.load_state_dict({}, strict = False)
    assert lin.quant_state.layout is not None
    model.load_state_dict(state)
    assert lin.quant_state.layout is None and torch.equal(lin.dequantize_weight(), ref)


def _repacked_linear():
    from unsloth.models.compressed_tensors_int4 import (
        Int4PackedLinear,
        finalize_int4_packed_linears,
    )

    torch.manual_seed(0)
    packed, ref, gs = _packed_layer(256, 1024, 4, 128, False, False, torch.bfloat16)
    lin = Int4PackedLinear.__new__(Int4PackedLinear)
    torch.nn.Module.__init__(lin)
    lin.in_features, lin.out_features = 1024, 256
    for name, tensor in packed.items():
        lin.register_parameter(name, torch.nn.Parameter(tensor.clone(), requires_grad = False))
    lin.register_parameter("bias", None)
    lin._int4_bits, lin._int4_group_size = 4, gs
    finalize_int4_packed_linears(torch.nn.Sequential(lin), torch.bfloat16)
    assert lin.quant_state.layout is not None
    return lin, packed, ref


@needs_gpu
@needs_ct
@needs_sm80
@pytest.mark.parametrize("cast", ["float", "half"])
def test_repacked_linear_infers_after_a_dtype_cast(cast):
    # The fused layout was picked for bf16; after a cast inference must take the exact dequantize path.
    lin, _, ref = _repacked_linear()
    getattr(lin, cast)()
    dtype = lin.quant_state.dtype
    x = torch.randn(2, 1024, device = "cuda", dtype = dtype)
    with torch.no_grad():
        y = lin(x)
    want = x.float() @ ref.float().t()
    assert y.dtype == dtype and ((y.float() - want).norm() / want.norm()) < 5e-3


@needs_gpu
@needs_ct
@needs_sm80
def test_adapter_saves_skip_unpacking_repacked_words(monkeypatch, tmp_path):
    from peft import LoraConfig, get_peft_model
    import unsloth.kernels.int4_packed as ip

    lin, packed, _ = _repacked_linear()

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = lin
            self.head = torch.nn.Linear(256, 8, device = "cuda", dtype = torch.bfloat16)

    model = get_peft_model(Tiny(), LoraConfig(r = 4, target_modules = ["head"]))
    calls = []
    real = ip.int4_unpack
    monkeypatch.setattr(ip, "int4_unpack", lambda *a: calls.append(1) or real(*a))
    model.save_pretrained(tmp_path)
    assert calls == [] and (tmp_path / "adapter_model.safetensors").exists()
    # A plain state_dict() still writes the checkpoint layout.
    assert torch.equal(model.state_dict()["base_model.model.proj.weight_packed"], packed["weight_packed"])
    assert calls == [1]


@needs_gpu
@needs_ct
def test_jit_launch_fallback_matches_the_compiled_launcher(monkeypatch):
    # Triton < 3.7 launchers reject CompiledKernel[grid](*runtime_args); the JIT launch must give the same bits.
    import unsloth.kernels.int4_packed as ip

    torch.manual_seed(0)
    packed, ref, gs = _packed_layer(200, 512, 4, 128, True, False, torch.bfloat16, "group")
    W = packed["weight_packed"]

    def run():
        qs = ip.Int4QuantState(
            packed["weight_scale"], None, None, (200, 512), 4, gs, torch.bfloat16
        )
        x = torch.randn(
            3,
            512,
            device = "cuda",
            dtype = torch.bfloat16,
            generator = torch.Generator("cuda").manual_seed(1),
        )
        return ip.int4_dequantize(W, qs), ip.int4_matmul(x, W, qs)

    fast = run()
    monkeypatch.setattr(ip, "_FAST_LAUNCH", False)
    jit = run()
    assert torch.equal(fast[0], ref) and torch.equal(jit[0], ref)
    assert torch.equal(fast[1], jit[1])


@needs_gpu
@pytest.mark.skipif(
    not (HAS_CT and HAS_CONVERTERS), reason = "needs compressed-tensors and the transformers 5 loader"
)
@pytest.mark.parametrize("variant", ["symmetric", "asymmetric", "actorder", "int8"])
@pytest.mark.usefixtures("restore_llama_patches")
def test_packed_route_keeps_the_checkpoint_weights_exactly(variant, tmp_path, monkeypatch):
    from unsloth import FastLanguageModel
    from unsloth.models.compressed_tensors_int4 import Int4PackedLinear
    from test_compressed_tensors_bnb import _tokenizer_free_load, _write_tiny_packed_llama

    monkeypatch.setenv("UNSLOTH_COMPRESSED_TENSORS_INT4", "packed")
    packed_dir, bf16_dir = _write_tiny_packed_llama(
        str(tmp_path),
        asymmetric = variant == "asymmetric",
        actorder = variant == "actorder",
        num_bits = 8 if variant == "int8" else 4,
    )
    for d in (packed_dir, bf16_dir):
        _tokenizer_free_load(d, str(tmp_path))
    kw = dict(max_seq_length = 64, dtype = torch.bfloat16)
    model_a, _ = FastLanguageModel.from_pretrained(packed_dir, load_in_4bit = True, **kw)
    model_b, _ = FastLanguageModel.from_pretrained(
        bf16_dir, load_in_4bit = False, load_in_16bit = True, **kw
    )
    dense = dict(model_b.named_modules())
    n = 0
    for name, m in model_a.named_modules():
        if isinstance(m, Int4PackedLinear):
            assert torch.equal(m.dequantize_weight(), dense[name].weight), name
            assert m.weight.dtype == torch.int32 and m.weight.quant_state is m.quant_state
            assert not any(p.requires_grad for p in m.parameters())
            n += 1
    assert n == 2 * 7, n
    ids = torch.randint(0, 256, (1, 16), device = "cuda:0")
    with torch.no_grad():
        a = model_a(input_ids = ids).logits.float()
        b = model_b(input_ids = ids).logits.float()
    assert (a - b).abs().max() < 1e-2 * b.abs().max()
    lin = next(m for m in model_a.modules() if isinstance(m, Int4PackedLinear))
    scale = lin.weight_scale.clone()
    lin.to(torch.float16)
    assert torch.equal(lin.weight_scale, scale) and lin.quant_state.dtype == torch.float16


@needs_gpu
@pytest.mark.skipif(
    not (HAS_CT and HAS_CONVERTERS), reason = "needs compressed-tensors and the transformers 5 loader"
)
@pytest.mark.usefixtures("restore_llama_patches")
def test_packed_route_lora_trains_merges_and_unmerges(tmp_path, monkeypatch):
    from unsloth import FastLanguageModel
    from unsloth.models.compressed_tensors_int4 import Int4PackedLinear
    from test_compressed_tensors_bnb import _tokenizer_free_load, _write_tiny_packed_llama

    monkeypatch.setenv("UNSLOTH_COMPRESSED_TENSORS_INT4", "packed")
    packed_dir, _ = _write_tiny_packed_llama(str(tmp_path), asymmetric = True)
    _tokenizer_free_load(packed_dir, str(tmp_path))
    model, _ = FastLanguageModel.from_pretrained(
        packed_dir, max_seq_length = 64, dtype = torch.bfloat16, load_in_4bit = True
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 4,
        lora_alpha = 8,
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model.train()
    ids = torch.randint(0, 256, (2, 16), device = "cuda:0")
    loss = model(input_ids = ids, labels = ids).loss
    loss.backward()
    grads = [p.grad for n, p in model.named_parameters() if "lora_B" in n]
    assert len(grads) == 14 and all(g is not None and torch.isfinite(g).all() for g in grads)
    assert all(g.abs().sum() > 0 for g in grads)
    layer = model.base_model.model.model.layers[0].self_attn.q_proj
    base = layer.get_base_layer()
    packed = base.weight_packed
    before = base.dequantize_weight()
    with torch.no_grad():
        layer.lora_B["default"].weight.normal_()
    layer.merge()
    assert type(layer.get_base_layer()) is torch.nn.Linear
    assert not torch.equal(layer.get_base_layer().weight, before)
    layer.unmerge()
    assert isinstance(layer.get_base_layer(), Int4PackedLinear)
    assert layer.get_base_layer().weight_packed is packed


@needs_ct
def test_adopt_swaps_plain_linears_and_leaves_routers_to_the_decompress_converter(tmp_path):
    import re

    from safetensors.torch import save_file
    from torch import nn
    from unsloth.models.compressed_tensors_bnb import (
        _PACKED_SUFFIXES,
        _build_quantization_config,
        adopt_int4_packed_linears,
    )
    from unsloth.models.compressed_tensors_int4 import Int4PackedLinear
    from test_compressed_tensors_bnb import _w4a16

    class Router(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(4, 64))

    with torch.device("meta"):
        model = nn.Module()
        model.proj = nn.Linear(64, 16, bias = False)
        model.gate = Router()
    tensors = {}
    for name, rows in (("proj", 16), ("gate", 4), ("mlp.experts.0.up", 8)):
        tensors[f"{name}.weight_packed"] = torch.zeros(rows, 8, dtype = torch.int32)
        tensors[f"{name}.weight_scale"] = torch.ones(rows, 2, dtype = torch.float32)
        tensors[f"{name}.weight_shape"] = torch.tensor([rows, 64])
    path = str(tmp_path / "model.safetensors")
    save_file(tensors, path)
    ct_config = _build_quantization_config(_w4a16())
    skipped, _ = adopt_int4_packed_linears(
        copy.deepcopy(model), ct_config, [path], torch.bfloat16, skip_modules = ["proj"]
    )
    assert skipped == []  # a caller-skipped module stays with the decompress converter
    swapped, leftover = adopt_int4_packed_linears(model, ct_config, [path], torch.bfloat16)
    assert swapped == ["proj"] and leftover == ["gate"]
    # The stacked expert is not in the checkpoint's layout any more, so a full save must not claim it is.
    assert model.__dict__.get("_unsloth_int4_stacked_experts") is True
    assert isinstance(model.proj, Int4PackedLinear)
    assert model.proj.weight_packed.dtype == torch.int32 and model.proj.weight_packed.shape == (
        16,
        8,
    )
    assert model.proj.weight_scale.dtype == torch.float32
    assert type(model.gate) is Router
    # The leftover converter pattern renames only the router's keys, and only the suffix.
    patterns = [f"(?<={re.escape(n)}\\.){s}$" for n in leftover for s in _PACKED_SUFFIXES]
    rx = re.compile("|".join(patterns))
    m = rx.search("gate.weight_packed")
    assert m and "gate.weight_packed".replace(m.group(0), "weight", 1) == "gate.weight"
    assert rx.search("proj.weight_packed") is None


@needs_gpu
@pytest.mark.skipif(
    not (HAS_CT and HAS_CONVERTERS), reason = "needs compressed-tensors and the transformers 5 loader"
)
@pytest.mark.usefixtures("restore_llama_patches")
def test_a_full_save_of_the_packed_route_reloads(tmp_path, monkeypatch):
    # The saved tensors stay packed, so the saved config must be the checkpoint's, not the runtime bnb one.
    import json
    from transformers import AutoModelForCausalLM
    from unsloth import FastLanguageModel
    from test_compressed_tensors_bnb import _tokenizer_free_load, _write_tiny_packed_llama

    monkeypatch.setenv("UNSLOTH_COMPRESSED_TENSORS_INT4", "packed")
    packed_dir, _ = _write_tiny_packed_llama(str(tmp_path))
    _tokenizer_free_load(packed_dir, str(tmp_path))
    kw = dict(max_seq_length = 64, dtype = torch.bfloat16)
    model, tokenizer = FastLanguageModel.from_pretrained(packed_dir, load_in_4bit = True, **kw)
    runtime = model.config.quantization_config
    out = str(tmp_path / "saved")
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    assert model.config.quantization_config is runtime
    saved = json.load(open(f"{out}/config.json"))["quantization_config"]
    assert saved["quant_method"] == "compressed-tensors"
    ids = torch.randint(0, 256, (1, 16), device = "cuda:0")
    with torch.no_grad():
        want = model(input_ids = ids).logits.float()
        again, _ = FastLanguageModel.from_pretrained(out, load_in_4bit = True, **kw)
        assert torch.equal(again(input_ids = ids).logits.float(), want)
        plain = AutoModelForCausalLM.from_pretrained(out, dtype = torch.bfloat16, device_map = {"": 0})
        got = plain(input_ids = ids).logits.float()
    assert (got - want).abs().max() < 1e-2 * want.abs().max()
    del plain, again
    merged = FastLanguageModel.get_peft_model(
        model, r = 4, target_modules = ["q_proj"]
    ).merge_and_unload()
    merged.save_pretrained(str(tmp_path / "merged"))
    tokenizer.save_pretrained(str(tmp_path / "merged"))
    # A permanent merge must not keep the packed weights stashed for an unmerge that can never come.
    assert not any("_unsloth_int4_packed_state" in m.__dict__ for m in merged.modules())
    ignore = json.load(open(f"{tmp_path}/merged/config.json"))["quantization_config"]["ignore"]
    assert (
        "model.layers.0.self_attn.q_proj" in ignore
        and "model.layers.0.self_attn.k_proj" not in ignore
    )
    with torch.no_grad():
        want = merged(input_ids = ids).logits.float()
        plain = AutoModelForCausalLM.from_pretrained(
            str(tmp_path / "merged"), dtype = torch.bfloat16, device_map = {"": 0}
        )
        got = plain(input_ids = ids).logits.float()
    assert (got - want).abs().max() < 1e-2 * want.abs().max()


@needs_gpu
@needs_ct
@pytest.mark.parametrize("grad", [True, False])
@pytest.mark.parametrize("lead", [(0,), (2, 0)])
def test_an_empty_batch_returns_an_empty_output(grad, lead):
    from unsloth.kernels.int4_packed import Int4QuantState, int4_matmul

    packed, _, gs = _packed_layer(200, 512, 4, 128, True, False, torch.bfloat16)
    qs = Int4QuantState(packed["weight_scale"], None, None, (200, 512), 4, gs, torch.bfloat16)
    x = torch.randn(*lead, 512, device = "cuda", dtype = torch.bfloat16)
    with torch.set_grad_enabled(grad):
        assert int4_matmul(x, packed["weight_packed"], qs).shape == (*lead, 200)


def test_a_mixed_packed_layout_refuses_a_full_save():
    # Packed layers beside converted experts: no single saved config could reload them.
    from torch import nn
    from unsloth.models.compressed_tensors_int4 import (
        Int4PackedLinear,
        refuse_mixed_packed_full_save,
    )

    class Model(nn.Module):
        def save_pretrained(self, *args, **kwargs):
            return "saved"

    model = Model()
    layer = Int4PackedLinear.__new__(Int4PackedLinear)
    nn.Module.__init__(layer)
    model.layer = layer
    refuse_mixed_packed_full_save(model)
    with pytest.raises(RuntimeError, match = "full save is refused"):
        model.save_pretrained("out")
    layer.__class__ = nn.Linear  # what a LoRA merge's densify leaves behind
    assert model.save_pretrained("out") == "saved"


@needs_gpu
@needs_ct
def test_training_single_row_forward_uses_the_backward_weights():
    # fast = False is the autograd forward: it must multiply the same rounded weights int4_matmul_t uses.
    from unsloth.kernels.int4_packed import Int4QuantState, int4_dequantize, int4_matmul

    packed, _, gs = _packed_layer(200, 512, 4, 128, True, True, torch.bfloat16)
    qs = Int4QuantState(
        packed["weight_scale"], None, packed.get("weight_g_idx"), (200, 512), 4, gs, torch.bfloat16
    )
    W = packed["weight_packed"]
    x = torch.randn(1, 512, device = "cuda", dtype = torch.bfloat16)
    want = x @ int4_dequantize(W, qs, torch.bfloat16).t()
    assert torch.equal(int4_matmul(x, W, qs, fast = False), want)
