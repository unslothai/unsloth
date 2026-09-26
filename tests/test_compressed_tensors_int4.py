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

import os
import sys

import pytest
import torch
from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py

import unsloth  # noqa: F401
from unsloth.models.compressed_tensors_bnb import _transformers_supports_weight_converters

try:
    import compressed_tensors  # noqa: F401

    HAS_CT = True
except Exception:
    HAS_CT = False

HAS_CONVERTERS = _transformers_supports_weight_converters()
# The tiny packed-Llama builder lives next to this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
needs_gpu = pytest.mark.skipif(not has_real_cuda(), reason = "needs a CUDA device")
needs_ct = pytest.mark.skipif(not HAS_CT, reason = "needs compressed-tensors")


def _packed_layer(out_f, in_f, bits, group_size, symmetric, actorder, scale_dtype, strategy = "group"):
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
def test_kernels_decode_bit_exact_and_multiply_like_dense(bits, group, sym, actorder, scale_dtype, strategy):
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
    # The fast_dequantize contract: a transposed packed view decodes to the transposed weight.
    assert torch.equal(int4_dequantize_weight(W.t(), qs), ref.t())
    for rows in (1, 3, 5, 64):
        x = torch.randn(rows, in_f, device = "cuda", dtype = torch.bfloat16)
        y = int4_matmul(x, W, qs)
        want = x.float() @ ref.float().t()
        assert y.shape == (rows, out_f) and y.dtype == torch.bfloat16
        assert ((y.float() - want).norm() / want.norm()) < 5e-3
        dy = torch.randn(rows, out_f, device = "cuda", dtype = torch.bfloat16)
        dx = int4_matmul_t(dy, W, qs)
        want = dy.float() @ ref.float()
        assert ((dx.float() - want).norm() / want.norm()) < 5e-3


@needs_gpu
@pytest.mark.skipif(not (HAS_CT and HAS_CONVERTERS), reason = "needs compressed-tensors and the transformers 5 loader")
@pytest.mark.parametrize("variant", ["symmetric", "asymmetric", "actorder", "int8"])
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
    # A cast moves the compute dtype, never the stored scales.
    lin = next(m for m in model_a.modules() if isinstance(m, Int4PackedLinear))
    scale = lin.weight_scale.clone()
    lin.to(torch.float16)
    assert torch.equal(lin.weight_scale, scale) and lin.quant_state.dtype == torch.float16


@needs_gpu
@pytest.mark.skipif(not (HAS_CT and HAS_CONVERTERS), reason = "needs compressed-tensors and the transformers 5 loader")
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
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model.train()
    ids = torch.randint(0, 256, (2, 16), device = "cuda:0")
    loss = model(input_ids = ids, labels = ids).loss
    loss.backward()
    grads = [p.grad for n, p in model.named_parameters() if "lora_B" in n]
    assert len(grads) == 14 and all(g is not None and torch.isfinite(g).all() for g in grads)
    assert all(g.abs().sum() > 0 for g in grads)
    # PEFT merge densifies the packed base, unmerge restores the exact packed tensors.
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
