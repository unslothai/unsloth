# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""compressed-tensors ``mxfp4-pack-quantized`` Linears stay packed and train through.

Offline tests pin the dequant against an independent float64 decode and against
compressed-tensors' own decompressor, the autograd function (exact forward and input
gradient, nothing but the packed bytes saved for backward), adopting compressed-tensors'
compressed modules, the loader's module matching, the fast LoRA paths and PEFT merge /
unmerge on a packed base. The GPU tests build a tiny Llama packed with compressed-tensors'
MXFP4 compressor and load it through ``FastLanguageModel.from_pretrained``: kept packed by
default, bitsandbytes 4-bit with ``UNSLOTH_MXFP4_KEEP_PACKED=0``, bit-identical to the bf16
decode on disk, and saved (full and merged) into checkpoints plain transformers reloads.
"""

import copy
import json
import os
import shutil

import pytest
import torch
from torch import nn

import unsloth  # noqa: F401
from unsloth.models.mxfp4_compressed_linear import (
    Mxfp4PackedLinear,
    _dequantize_torch,
    adopt_compressed_mxfp4_modules,
    dequantize_mxfp4_packed,
    make_mxfp4_packed_linear,
)

try:
    import compressed_tensors  # noqa: F401
    HAS_CT = True
except Exception:
    HAS_CT = False

from unsloth.models.compressed_tensors_bnb import _transformers_supports_weight_converters

# The bitsandbytes route needs the transformers 5 loader's `update_weight_conversions`
# (5.5+); on 5.4 an MXFP4 checkpoint loads through compressed-tensors' own quantizer.
HAS_CONVERTERS = _transformers_supports_weight_converters()

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _random_packed(
    out_features,
    in_features,
    seed = 0,
    low = 100,
    high = 150,
):
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (out_features, in_features // 2), generator = g, dtype = torch.uint8)
    scale = torch.randint(
        low, high, (out_features, in_features // 32), generator = g, dtype = torch.uint8
    )
    return packed, scale


def _reference_decode(packed, scale):
    """Element by element, in float64: low nibble first, sign in bit 3, times 2**(e - 127)."""
    out_features, half = packed.shape
    w = torch.empty(out_features, half * 2, dtype = torch.float64)
    for i in range(out_features):
        for j in range(half):
            byte = int(packed[i, j])
            for k, nib in enumerate((byte & 0x0F, byte >> 4)):
                col = 2 * j + k
                v = _E2M1[nib & 0x07] * (-1.0 if nib & 0x08 else 1.0)
                w[i, col] = v * 2.0 ** (int(scale[i, col // 32]) - 127)
    return w


@pytest.mark.parametrize("device", DEVICES)
def test_dequant_matches_an_independent_float64_decode(device):
    packed, scale = _random_packed(8, 64, low = 1, high = 254)
    packed[0, :16] = torch.arange(0, 256, 16, dtype = torch.uint8)
    packed[1, :16] = torch.arange(0, 16, dtype = torch.uint8)
    want = _reference_decode(packed, scale)
    for dtype in (torch.bfloat16, torch.float32):
        got = dequantize_mxfp4_packed(packed.to(device), scale.to(device), dtype).cpu()
        assert got.dtype == dtype and got.shape == (8, 64)
        assert torch.equal(got.double(), want.to(dtype).double()), dtype


@pytest.mark.skipif(not HAS_CT, reason = "needs compressed-tensors")
@pytest.mark.parametrize("device", DEVICES)
def test_dequant_matches_compressed_tensors_decompress(device):
    from compressed_tensors.compressors import BaseCompressor
    from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

    scheme = QuantizationScheme(
        targets = ["Linear"],
        weights = QuantizationArgs(
            num_bits = 4,
            type = "float",
            strategy = "group",
            group_size = 32,
            symmetric = True,
            scale_dtype = torch.uint8,
        ),
        format = "mxfp4-pack-quantized",
    )
    comp = BaseCompressor.get_value_from_registry("mxfp4-pack-quantized")
    for seed, (o, i) in enumerate([(128, 128), (96, 256), (3, 64)]):
        packed, scale = _random_packed(o, i, seed = seed, low = 1, high = 254)
        state = {
            "weight_packed": packed,
            "weight_scale": scale,
            "weight_shape": torch.tensor([o, i]),
        }
        want = comp.decompress(state, scheme)["weight"].to(torch.bfloat16)
        got = dequantize_mxfp4_packed(packed.to(device), scale.to(device), torch.bfloat16).cpu()
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), (o, i)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
def test_zoo_kernel_and_torch_path_agree():
    packed, scale = _random_packed(256, 512, low = 1, high = 254)
    packed, scale = packed.cuda(), scale.cuda()
    blocks = packed.view(256, 16, 16)
    for dtype in (torch.bfloat16, torch.float16):
        a = dequantize_mxfp4_packed(packed, scale, dtype)
        b = _dequantize_torch(blocks, scale, dtype)
        assert torch.equal(a.view(torch.int16), b.view(torch.int16)), dtype


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bias", [False, True])
def test_forward_and_input_gradient_are_exact_and_save_only_packed_bytes(device, bias):
    torch.manual_seed(0)
    out_f, in_f = 96, 64
    packed, scale = _random_packed(out_f, in_f)
    module = make_mxfp4_packed_linear(in_f, out_f, device = device)
    module.weight_packed.data.copy_(packed)
    module.weight_scale.data.copy_(scale)
    ref = nn.Linear(in_f, out_f, bias = bias, device = device, dtype = torch.bfloat16)
    ref.weight.data.copy_(module.dequantize_weight(torch.bfloat16))
    ref.weight.requires_grad_(False)
    if bias:
        module.bias = nn.Parameter(ref.bias.detach().clone())
    x = torch.randn(5, 7, in_f, device = device, dtype = torch.bfloat16)
    xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()

    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        ya = module(xa)
    yb = ref(xb)
    assert torch.equal(ya, yb)
    # Only the uint8 packed bytes and scales are kept for backward, never a float weight.
    assert saved and all(t.dtype == torch.uint8 for t in saved), [t.dtype for t in saved]
    g = torch.randn_like(ya)
    ya.backward(g)
    yb.backward(g)
    assert torch.equal(xa.grad, xb.grad)
    if bias:
        assert torch.allclose(module.bias.grad.float(), ref.bias.grad.float(), rtol = 1e-2, atol = 1e-2)
    assert module.weight_packed.grad is None and not module.weight_packed.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
def test_autocast_runs_the_matmul_in_the_autocast_dtype():
    packed, scale = _random_packed(64, 64)
    module = make_mxfp4_packed_linear(64, 64, device = "cuda")
    module.weight_packed.data.copy_(packed)
    module.weight_scale.data.copy_(scale)
    ref = nn.Linear(64, 64, bias = False, device = "cuda", dtype = torch.float32)
    ref.weight.data.copy_(module.dequantize_weight(torch.float32))
    x = torch.randn(4, 64, device = "cuda", requires_grad = True)
    x2 = x.detach().clone().requires_grad_()
    with torch.autocast("cuda", dtype = torch.bfloat16):
        ya, yb = module(x), ref(x2)
    assert ya.dtype == yb.dtype == torch.bfloat16
    assert torch.equal(ya, yb)
    ya.float().sum().backward()
    yb.float().sum().backward()
    assert x.grad.dtype == torch.float32
    assert torch.equal(x.grad, x2.grad)


def test_weight_property_is_an_exact_read_only_decode():
    packed, scale = _random_packed(32, 64)
    module = make_mxfp4_packed_linear(64, 32)
    module.weight_packed.data.copy_(packed)
    module.weight_scale.data.copy_(scale)
    w = module.weight
    assert w.dtype == torch.bfloat16 and w.shape == (32, 64)
    assert torch.equal(w.double(), _reference_decode(packed, scale).to(torch.bfloat16).double())
    assert "weight" not in dict(module.named_parameters())
    assert {n for n, _ in module.named_parameters()} == {"weight_packed", "weight_scale"}
    with pytest.raises(ValueError):
        make_mxfp4_packed_linear(48, 32)


def _ct_compressed_model(targets_ignore = (), groups = None):
    """Three Linears laid out by compressed-tensors itself: quantization config applied, then
    compressed in memory, with its decompress-on-first-forward hook registered. ``groups``
    replaces the one MXFP4 group targeting every Linear."""
    from compressed_tensors.compressors import ModelCompressor
    from compressed_tensors.quantization import QuantizationConfig, apply_quantization_config

    torch.manual_seed(0)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(64, 32, bias = False)
            self.b = nn.Linear(32, 64, bias = False)
            self.latent = nn.Linear(64, 64, bias = False)

        def forward(self, x):
            return self.latent(self.b(self.a(x)))

    model = Tiny().to(torch.bfloat16)
    quant = {
        "quant_method": "compressed-tensors",
        "format": "mxfp4-pack-quantized",
        "quantization_status": "compressed",
        "ignore": list(targets_ignore),
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "strategy": "group",
                    "group_size": 32,
                    "symmetric": True,
                    "scale_dtype": "torch.uint8",
                },
            }
        },
    }
    if groups is not None:
        quant["config_groups"] = groups
    qc = QuantizationConfig.model_validate(quant)
    apply_quantization_config(model, qc, run_compressed = False)
    for m in (model.a, model.b, model.latent):
        m.weight_scale.data.fill_(125 if m.weight_scale.dtype == torch.uint8 else 0.01)
    ModelCompressor(quantization_config = qc).compress_model(model)
    return model


_MXFP4_WEIGHTS = {
    "num_bits": 4,
    "type": "float",
    "strategy": "group",
    "group_size": 32,
    "symmetric": True,
    "scale_dtype": "torch.uint8",
}


@pytest.mark.skipif(not HAS_CT, reason = "needs compressed-tensors")
def test_adopting_compressed_tensors_modules_keeps_them_packed_and_matches_its_decompress():
    model = _ct_compressed_model()
    assert hasattr(model, "ct_decompress_hook")
    assert "weight_packed" in model.a._parameters and "weight" not in model.a._parameters
    reference = copy.deepcopy(model)
    x = torch.randn(3, 64, dtype = torch.bfloat16)
    with torch.no_grad():
        want = reference(x)  # compressed-tensors decompresses everything on this first forward
    assert "weight" in reference.a._parameters

    # `latent` is stored unpacked in the checkpoint: it goes back to a dense weight.
    n = adopt_compressed_mxfp4_modules(model, packed_names = {"a", "b"}, dtype = torch.bfloat16)
    assert n == 2
    assert type(model.a) is Mxfp4PackedLinear and type(model.b) is Mxfp4PackedLinear
    assert type(model.latent) is nn.Linear and "weight" in model.latent._parameters
    assert "weight_packed" not in model.latent._parameters
    assert not hasattr(model.latent, "quantization_scheme")
    # compressed-tensors' per-instance quantize-dequantize forward is gone, and so is its hook.
    assert all("forward" not in m.__dict__ for m in (model.a, model.b, model.latent))
    assert not hasattr(model, "ct_decompress_hook")
    model.latent.weight.data.copy_(reference.latent.weight.data)
    with torch.no_grad():
        got = model(x)
    assert torch.equal(got, want)
    assert "weight_packed" in model.a._parameters  # still packed after the forward


@pytest.mark.skipif(not HAS_CT, reason = "needs compressed-tensors")
def test_adoption_is_all_or_nothing(monkeypatch):
    model = _ct_compressed_model()
    model.b.__class__ = type("Other", (nn.Linear,), {})
    assert adopt_compressed_mxfp4_modules(model) == 0
    assert type(model.a) is nn.Linear and hasattr(model, "ct_decompress_hook")
    model = _ct_compressed_model()
    monkeypatch.setenv("UNSLOTH_MXFP4_KEEP_PACKED", "0")
    assert adopt_compressed_mxfp4_modules(model) == 0
    assert type(model.a) is nn.Linear


def _nested_model():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Linear(64, 32, bias = False)
            self.proj = nn.Linear(64, 64, bias = False)

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block(), Block()])

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    return Outer()


@pytest.mark.skipif(not HAS_CT, reason = "needs compressed-tensors")
def test_loader_matches_packed_checkpoint_names_and_swaps_in_empty_packed_modules():
    from unsloth.models.compressed_tensors_bnb import (
        _build_quantization_config,
        _keep_packed_linears,
        _match_packed_linears,
        plan_mxfp4_keep_packed,
    )

    with torch.device("meta"):
        model = _nested_model()
    keys = [
        "language_model.model.layers.0.w1.weight_packed",
        "language_model.model.layers.0.w1.weight_scale",
        "language_model.model.layers.1.w1.weight_packed",
        "language_model.model.layers.1.w1.weight_scale",
        "language_model.model.layers.0.proj.weight",
    ]
    assert set(_match_packed_linears(model, keys)) == {"model.layers.0.w1", "model.layers.1.w1"}
    # A packed module with no per-Linear home (a merged expert stack) keeps the old route.
    assert _match_packed_linears(model, keys + ["model.experts.gate_up_proj.weight_packed"]) is None
    assert _match_packed_linears(model, ["model.layers.0.proj.weight"]) is None
    plan = plan_mxfp4_keep_packed(model, keys)
    assert plan.blocks == [] and plan.linears == ["model.layers.0.w1", "model.layers.1.w1"]
    assert (
        plan_mxfp4_keep_packed(model, keys + ["model.experts.gate_up_proj.weight_packed"]) is None
    )
    assert plan_mxfp4_keep_packed(model, ["model.layers.0.proj.weight"]) is None

    ct_plan = {
        "quant_method": "compressed-tensors",
        "format": "mxfp4-pack-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "strategy": "group",
                    "group_size": 32,
                    "symmetric": True,
                    "scale_dtype": "torch.uint8",
                },
            }
        },
    }
    config = _build_quantization_config(ct_plan)
    assert _keep_packed_linears(model, plan.linears, config, torch.float16) == 2
    w1 = model.model.layers[1].w1
    assert type(w1) is Mxfp4PackedLinear and w1.compute_dtype == torch.float16
    assert w1.weight_packed.shape == (32, 32) and w1.weight_packed.dtype == torch.uint8
    assert w1.weight_scale.shape == (32, 2) and w1.weight_packed.is_meta
    assert type(model.model.layers[0].proj) is nn.Linear
    assert str(getattr(w1.quantization_status, "value", w1.quantization_status)) == "compressed"


def _filled(
    out_features,
    in_features,
    seed = 0,
    bias = False,
    device = "cpu",
    dtype = None,
):
    packed, scale = _random_packed(out_features, in_features, seed = seed)
    module = make_mxfp4_packed_linear(in_features, out_features, device = device, dtype = dtype)
    module.weight_packed.data.copy_(packed)
    module.weight_scale.data.copy_(scale)
    if bias:
        module.bias = nn.Parameter(torch.randn(out_features, device = device, dtype = torch.bfloat16))
    return module


def test_torch_fallback_without_the_zoo_kernel(monkeypatch):
    import unsloth.models.mxfp4_compressed_linear as mcl

    packed, scale = _random_packed(16, 64, low = 1, high = 254)
    with_zoo = dequantize_mxfp4_packed(packed, scale, torch.bfloat16)
    monkeypatch.setattr(mcl, "_zoo_mxfp4_dequantize", None)
    assert torch.equal(dequantize_mxfp4_packed(packed, scale, torch.bfloat16), with_zoo)


def test_dtype_casts_keep_the_packed_bytes_and_set_the_decode_dtype():
    module = _filled(32, 64, dtype = torch.float16)
    module.weight_scale.data.clamp_(110, 130)  # stay inside fp16's range
    before = module.weight_packed.clone(), module.weight_scale.clone()
    module.to(torch.bfloat16).half()
    assert module.weight_packed.dtype == torch.uint8 and torch.equal(
        module.weight_packed, before[0]
    )
    assert torch.equal(module.weight_scale, before[1])
    # `weight` (what PEFT and the fast paths read) decodes in the model's dtype, not always bf16.
    assert module.weight.dtype == torch.float16
    assert torch.equal(module.weight.float(), module.dequantize_weight(torch.bfloat16).float())


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
def test_device_move_carries_bytes_and_scales():
    module = _filled(32, 64).cuda()
    assert module.weight_packed.is_cuda and module.weight_scale.is_cuda
    assert module.weight.is_cuda


def test_fast_lora_paths_see_a_transient_decode_and_skip_the_fused_kernels():
    peft = pytest.importorskip("peft")
    from unsloth.kernels.utils import get_lora_parameters, get_lora_parameters_bias, has_mxfp4_base

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = _filled(32, 64)
            self.dense = nn.Linear(64, 64, bias = False)

    model = peft.get_peft_model(Block(), peft.LoraConfig(r = 4, target_modules = ["proj", "dense"]))
    proj = model.base_model.model.proj
    W, quant_state, A, B, scale = get_lora_parameters(proj)
    assert W.dtype == torch.bfloat16 and quant_state is None and A is not None
    assert torch.equal(W, proj.base_layer.dequantize_weight())
    assert get_lora_parameters_bias(proj)[1] is None
    assert has_mxfp4_base(proj) and has_mxfp4_base(proj.base_layer)
    assert not has_mxfp4_base(model.base_model.model.dense)


def _lora_pair(bias = False, scale_range = None):
    """PEFT LoRA on a packed Linear and on an nn.Linear holding its exact decode, with the same
    non-zero adapter weights."""
    peft = pytest.importorskip("peft")
    packed = _filled(32, 64, bias = bias)
    if scale_range is not None:
        packed.weight_scale.data.clamp_(*scale_range)
    dense = nn.Linear(64, 32, bias = bias, dtype = torch.bfloat16)
    dense.weight.data.copy_(packed.dequantize_weight())
    if bias:
        dense.bias.data.copy_(packed.bias.data)
    models = []
    for base in (packed, dense):
        holder = nn.Sequential(base)
        model = peft.get_peft_model(
            holder, peft.LoraConfig(r = 4, lora_alpha = 8, target_modules = ["0"])
        )
        g = torch.Generator().manual_seed(3)
        with torch.no_grad():
            for name, param in sorted(model.named_parameters()):
                if "lora_" in name:
                    param.copy_((torch.randn(param.shape, generator = g) * 0.1).to(param.dtype))
        models.append(model)
    return models


@pytest.mark.parametrize("bias", [False, True])
def test_peft_lora_on_a_packed_linear_matches_the_dense_decode(bias):
    packed_model, dense_model = _lora_pair(bias)
    x = torch.randn(3, 64, dtype = torch.bfloat16)
    xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
    ya, yb = packed_model(xa), dense_model(xb)
    assert torch.equal(ya, yb)
    ya.float().sum().backward()
    yb.float().sum().backward()
    assert torch.equal(xa.grad, xb.grad)
    grads = lambda m: {n: p.grad for n, p in m.named_parameters() if "lora_" in n}  # noqa: E731
    ga, gb = grads(packed_model), grads(dense_model)
    assert ga.keys() == gb.keys() and all(torch.equal(ga[n], gb[n]) for n in ga)


def test_peft_merge_densifies_exactly_and_unmerge_restores_the_packed_bytes():
    # Moderate scales and a fixed input: with weights up to 2**23 the bf16 merged-vs-unmerged
    # comparison below cancels catastrophically for some random inputs.
    packed_model, dense_model = _lora_pair(scale_range = (118, 134))
    x = torch.randn(3, 64, generator = torch.Generator().manual_seed(0)).to(torch.bfloat16)
    base = packed_model.base_model.model[0].base_layer
    packed_bytes, scale = base.weight_packed, base.weight_scale
    with torch.no_grad():
        with_lora = packed_model(x)
        packed_model.merge_adapter()
        dense_model.merge_adapter()
        merged_base = packed_model.base_model.model[0].base_layer
        # The delta lands in a real weight (a write into a fresh decode would be lost).
        assert type(merged_base) is nn.Linear and "weight_packed" not in merged_base._parameters
        assert torch.equal(merged_base.weight, dense_model.base_model.model[0].base_layer.weight)
        merged = packed_model(x)
        packed_model.unmerge_adapter()
        restored = packed_model.base_model.model[0].base_layer
        assert type(restored) is Mxfp4PackedLinear
        assert restored.weight_packed is packed_bytes and restored.weight_scale is scale
        assert "weight" not in restored._parameters
        assert torch.equal(packed_model(x), with_lora)
    assert not torch.equal(merged, packed_model.base_model.model[0].base_layer(x))
    torch.testing.assert_close(merged.float(), with_lora.float(), atol = 2e-2, rtol = 2e-2)
    unloaded = packed_model.merge_and_unload()
    assert type(unloaded[0]) is nn.Linear
    assert torch.equal(unloaded[0].weight, dense_model.base_model.model[0].base_layer.weight)


# ----------------------------------------------------------------------------- GPU loads


def _write_tiny_mxfp4_llama(root):
    """A 2-layer Llama with every Linear but lm_head packed MXFP4 by compressed-tensors'
    compressor. Returns (packed_dir, bf16_dir) where bf16_dir holds the exact decode."""
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM
    from compressed_tensors.compressors import BaseCompressor
    from compressed_tensors.quantization import QuantizationConfig

    torch.manual_seed(0)
    config = LlamaConfig(
        hidden_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        intermediate_size = 128,
        vocab_size = 32000,
        max_position_embeddings = 128,
        tie_word_embeddings = False,
    )
    model = LlamaForCausalLM(config).to(torch.bfloat16)
    quant = {
        "quant_method": "compressed-tensors",
        "format": "mxfp4-pack-quantized",
        "quantization_status": "compressed",
        "ignore": ["lm_head"],
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "strategy": "group",
                    "group_size": 32,
                    "symmetric": True,
                    "scale_dtype": "torch.uint8",
                    "dynamic": False,
                    "observer": "minmax",
                    "observer_kwargs": {},
                    "actorder": None,
                    "block_structure": None,
                },
                "input_activations": None,
                "output_activations": None,
            }
        },
    }
    scheme = list(QuantizationConfig.model_validate(quant).config_groups.values())[0]
    comp = BaseCompressor.get_value_from_registry("mxfp4-pack-quantized")
    linears = {n for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    packed, plain = {}, {}
    for k, w in model.state_dict().items():
        mod = k[: -len(".weight")] if k.endswith(".weight") else None
        if mod is None or mod not in linears or mod == "lm_head":
            packed[k] = plain[k] = w.contiguous()
            continue
        out_f, in_f = w.shape
        amax = w.float().reshape(out_f, in_f // 32, 32).abs().amax(-1).clamp_min(1e-8)
        exp = (torch.ceil(torch.log2(amax / 6.0)) + 127).clamp(1, 254).to(torch.uint8)
        state = {"weight": w.float(), "weight_scale": 2.0 ** (exp.float() - 127)}
        out = comp.compress(state, scheme)
        out["weight_scale"] = exp
        for pk, pv in out.items():
            if pk == "weight_shape":
                continue
            packed[mod + "." + pk] = pv.contiguous()
        plain[k] = (
            comp.decompress(
                {
                    "weight_packed": out["weight_packed"],
                    "weight_scale": exp,
                    "weight_shape": torch.tensor([out_f, in_f]),
                },
                scheme,
            )["weight"]
            .to(torch.bfloat16)
            .contiguous()
        )
    packed_dir, bf16_dir = os.path.join(root, "packed"), os.path.join(root, "bf16")
    os.makedirs(packed_dir)
    os.makedirs(bf16_dir)
    save_file(packed, os.path.join(packed_dir, "model.safetensors"), metadata = {"format": "pt"})
    save_file(plain, os.path.join(bf16_dir, "model.safetensors"), metadata = {"format": "pt"})
    cfg = config.to_dict()
    json.dump(cfg, open(os.path.join(bf16_dir, "config.json"), "w"))
    cfg["quantization_config"] = quant
    json.dump(cfg, open(os.path.join(packed_dir, "config.json"), "w"))
    return packed_dir, bf16_dir


def _add_tokenizer(path, root):
    from transformers import AutoTokenizer

    tok_dir = os.path.join(root, "tok")
    if not os.path.isdir(tok_dir):
        AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer").save_pretrained(
            tok_dir
        )
    for f in os.listdir(tok_dir):
        if not os.path.exists(os.path.join(path, f)):
            shutil.copy(os.path.join(tok_dir, f), os.path.join(path, f))


def _lora_losses(model, steps = 3):
    from unsloth import FastLanguageModel

    model = FastLanguageModel.get_peft_model(
        model,
        r = 4,
        lora_alpha = 8,
        random_state = 3407,
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
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 1e-2)
    torch.manual_seed(1)
    ids = torch.randint(0, 256, (2, 24), device = "cuda:0")
    losses = []
    for _ in range(steps):
        loss = model(input_ids = ids, labels = ids).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(loss.item())
    return losses


needs_gpu_loader_any = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device"),
    pytest.mark.skipif(
        not HAS_CT
        or not hasattr(__import__("transformers"), "__version__")
        or int(__import__("transformers").__version__.split(".")[0]) < 5,
        reason = "needs compressed-tensors and transformers 5",
    ),
]
needs_gpu_loader = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device"),
    pytest.mark.skipif(
        not (HAS_CT and HAS_CONVERTERS),
        reason = "needs compressed-tensors and the transformers 5 loader",
    ),
]


def _apply(marks):
    def wrap(fn):
        for m in marks:
            fn = m(fn)
        return fn

    return wrap


@_apply(needs_gpu_loader_any)
@pytest.mark.parametrize("load_in_4bit", [True, False])
def test_mxfp4_checkpoint_stays_packed_and_matches_its_bf16_decode(
    load_in_4bit, tmp_path, monkeypatch
):
    import unsloth.models.llama as llama_module
    from unsloth import FastLanguageModel

    packed_dir, bf16_dir = _write_tiny_mxfp4_llama(str(tmp_path))
    for d in (packed_dir, bf16_dir):
        _add_tokenizer(d, str(tmp_path))
    kw = dict(max_seq_length = 64, dtype = torch.bfloat16)
    model_a, _ = FastLanguageModel.from_pretrained(packed_dir, load_in_4bit = load_in_4bit, **kw)
    kept = [n for n, m in model_a.named_modules() if isinstance(m, Mxfp4PackedLinear)]
    assert len(kept) == 2 * 7, kept
    assert all(
        p.dtype == torch.uint8 for n, p in model_a.named_parameters() if "weight_packed" in n
    )
    model_b, _ = FastLanguageModel.from_pretrained(bf16_dir, load_in_4bit = False, **kw)
    ids = torch.randint(0, 256, (1, 16), device = "cuda:0")
    with torch.no_grad():
        assert torch.equal(model_a(input_ids = ids).logits, model_b(input_ids = ids).logits)
    losses_a = _lora_losses(model_a)
    # Packed layers take the PEFT forward instead of the fused LoRA kernels (those keep the 16-bit
    # weight until backward); the reference takes the same path, so the losses are bit-identical.
    from unsloth.kernels import apply_lora_o, apply_lora_qkv

    for layer in model_a.model.layers:
        assert "_unsloth_forward" not in layer.mlp.__dict__ and "forward" not in layer.mlp.__dict__
        assert getattr(layer.self_attn, "apply_qkv", None) is not apply_lora_qkv
        assert getattr(layer.self_attn, "apply_o", None) is not apply_lora_o
    monkeypatch.setattr(llama_module, "has_mxfp4_base", lambda *projs: True)
    assert losses_a == _lora_losses(model_b)


@_apply(needs_gpu_loader)
def test_keep_packed_off_restores_the_bitsandbytes_route(tmp_path, monkeypatch):
    import bitsandbytes as bnb
    from unsloth import FastLanguageModel

    monkeypatch.setenv("UNSLOTH_MXFP4_KEEP_PACKED", "0")
    packed_dir, _ = _write_tiny_mxfp4_llama(str(tmp_path))
    _add_tokenizer(packed_dir, str(tmp_path))
    model, _ = FastLanguageModel.from_pretrained(
        packed_dir, max_seq_length = 64, dtype = torch.bfloat16, load_in_4bit = True
    )
    assert not any(isinstance(m, Mxfp4PackedLinear) for m in model.modules())
    assert sum(isinstance(m, bnb.nn.Linear4bit) for m in model.modules()) == 2 * 7


def _plain_logits(path, ids):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(path, dtype = torch.bfloat16, device_map = {"": 0})
    with torch.no_grad():
        return model(input_ids = ids).logits


@_apply(needs_gpu_loader_any)
def test_full_and_merged_saves_reload_in_plain_transformers(tmp_path, monkeypatch):
    """A full save writes each packed Linear as its dense `weight` with no MXFP4 config, and a
    merged_16bit export decodes the packed checkpoint and folds the LoRA in: both reload in
    plain transformers and equal the in-memory merged model."""
    from safetensors import safe_open
    from unsloth import FastLanguageModel

    packed_dir, bf16_dir = _write_tiny_mxfp4_llama(str(tmp_path))
    _add_tokenizer(packed_dir, str(tmp_path))
    model, tokenizer = FastLanguageModel.from_pretrained(
        packed_dir, max_seq_length = 64, dtype = torch.bfloat16, load_in_4bit = False
    )
    ids = torch.randint(0, 256, (1, 16), device = "cuda:0")
    with torch.no_grad():
        base_logits = model(input_ids = ids).logits
    full = str(tmp_path / "full")
    model.save_pretrained(full)
    assert sum(isinstance(m, Mxfp4PackedLinear) for m in model.modules()) == 2 * 7  # put back
    with safe_open(os.path.join(full, "model.safetensors"), "pt") as f:
        keys = list(f.keys())
    assert not any("weight_packed" in k or "weight_scale" in k for k in keys)
    assert "quantization_config" not in json.load(open(os.path.join(full, "config.json")))
    # The exact decode: the same logits as plain transformers on the bf16-decoded checkpoint.
    assert torch.equal(_plain_logits(full, ids), _plain_logits(bf16_dir, ids))

    model = FastLanguageModel.get_peft_model(
        model,
        r = 4,
        lora_alpha = 8,
        random_state = 3407,
        target_modules = ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 0.05)
    merged_dir = str(tmp_path / "merged")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method = "merged_16bit")
    unloaded = model.merge_and_unload()
    # Merged layers are dense now; k_proj had no adapter and stays packed.
    assert [n for n, m in unloaded.named_modules() if isinstance(m, Mxfp4PackedLinear)] == [
        f"model.layers.{i}.self_attn.k_proj" for i in range(2)
    ]
    with torch.no_grad():
        unloaded_logits = unloaded(input_ids = ids).logits
    assert not torch.equal(unloaded_logits, base_logits)
    with safe_open(os.path.join(merged_dir, "model.safetensors"), "pt") as f:
        assert not any("weight_packed" in k or "weight_scale" in k for k in f.keys())
    unloaded_dir = str(tmp_path / "unloaded")
    unloaded.save_pretrained(unloaded_dir)
    merged_logits, unloaded_reload = (
        _plain_logits(merged_dir, ids),
        _plain_logits(unloaded_dir, ids),
    )
    torch.testing.assert_close(merged_logits, unloaded_reload, atol = 2e-2, rtol = 2e-2)
    torch.testing.assert_close(unloaded_reload, unloaded_logits, atol = 2e-2, rtol = 2e-2)
    assert not torch.equal(merged_logits, _plain_logits(full, ids))  # the LoRA is in


@_apply(needs_gpu_loader_any)
def test_full_finetuning_keeps_the_stock_compressed_tensors_route(tmp_path, monkeypatch):
    from unsloth.models.mxfp4_compressed_linear import install_compressed_tensors_keep_packed
    from transformers import AutoModelForCausalLM

    packed_dir, _ = _write_tiny_mxfp4_llama(str(tmp_path))
    assert install_compressed_tensors_keep_packed()
    monkeypatch.setenv("UNSLOTH_ENABLE_FULL_FINETUNING", "1")
    model = AutoModelForCausalLM.from_pretrained(packed_dir, dtype = torch.bfloat16)
    assert not any(isinstance(m, Mxfp4PackedLinear) for m in model.modules())
    monkeypatch.setenv("UNSLOTH_ENABLE_FULL_FINETUNING", "0")
    model = AutoModelForCausalLM.from_pretrained(packed_dir, dtype = torch.bfloat16)
    assert sum(isinstance(m, Mxfp4PackedLinear) for m in model.modules()) == 2 * 7


def _decompress_requests():
    """The ways this transformers asks compressed-tensors to decompress at load."""
    try:
        import inspect
        from transformers import CompressedTensorsConfig
    except Exception:
        return []
    params = inspect.signature(CompressedTensorsConfig).parameters
    return [{k: v} for k, v in (("run_compressed", False), ("dequantize", True)) if k in params]


@pytest.mark.skipif(not HAS_CT, reason = "needs compressed-tensors")
@pytest.mark.parametrize("request_kwargs", _decompress_requests(), ids = lambda kw: next(iter(kw)))
def test_an_explicit_decompress_request_keeps_the_stock_route(request_kwargs, tmp_path):
    """`run_compressed=False` (transformers 5.4) and `dequantize=True` (5.5+) ask compressed-tensors
    to decompress after the weights load: nothing may be adopted, or the adopted modules would
    lose their packed bytes to that decompress and fail on the first forward."""
    from transformers import AutoModelForCausalLM, CompressedTensorsConfig
    from unsloth.models.mxfp4_compressed_linear import install_compressed_tensors_keep_packed

    packed_dir, bf16_dir = _write_tiny_mxfp4_llama(str(tmp_path))
    assert install_compressed_tensors_keep_packed()
    # On the GPU when there is one: an earlier FastLanguageModel load patches Llama's forward.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        packed_dir,
        dtype = torch.bfloat16,
        device_map = {"": device},
        quantization_config = CompressedTensorsConfig(**request_kwargs),
    )
    assert not any(isinstance(m, Mxfp4PackedLinear) for m in model.modules())
    reference = AutoModelForCausalLM.from_pretrained(
        bf16_dir, dtype = torch.bfloat16, device_map = {"": device}
    )
    ids = torch.randint(0, 256, (1, 12), device = device)
    with torch.no_grad():
        assert torch.equal(model(input_ids = ids).logits, reference(input_ids = ids).logits)


@pytest.mark.parametrize("cast", [torch.float16, torch.float32])
def test_a_dtype_cast_reaches_the_merge_and_unmerge(cast):
    """`.to(dtype)` / `.half()` / `.float()` leave the uint8 bytes alone but must move the decode
    dtype with them, or a PEFT merge writes a weight in the old dtype and the merged forward fails."""
    packed_model, dense_model = _lora_pair(scale_range = (118, 134))  # inside fp16's range
    packed_model.to(cast)
    dense_model.to(cast)
    base = packed_model.base_model.model[0].base_layer
    assert base.compute_dtype == cast and base.weight.dtype == cast
    x = torch.randn(3, 64, dtype = cast)
    with torch.no_grad():
        packed_model.merge_adapter()
        dense_model.merge_adapter()
        assert packed_model.base_model.model[0].base_layer.weight.dtype == cast
        assert torch.equal(packed_model(x), dense_model(x))
        # Cast while merged, then unmerge: the packed module decodes in the model's new dtype.
        packed_model.to(torch.bfloat16)
        packed_model.unmerge_adapter()
        restored = packed_model.base_model.model[0].base_layer
        assert type(restored) is Mxfp4PackedLinear and restored.compute_dtype == torch.bfloat16
        packed_model.merge_adapter()
        assert packed_model(x.to(torch.bfloat16)).dtype == torch.bfloat16


@pytest.mark.skipif(not HAS_CT, reason = "needs compressed-tensors")
def test_the_sixteen_bit_route_decodes_in_the_load_dtype(tmp_path):
    """transformers 5.x does not hand the load dtype to the quantizer: the adopted modules take
    the model's dtype, and a PEFT merge on them runs."""
    peft = pytest.importorskip("peft")
    from transformers import AutoModelForCausalLM
    from unsloth.models.mxfp4_compressed_linear import install_compressed_tensors_keep_packed

    packed_dir, _ = _write_tiny_mxfp4_llama(str(tmp_path))
    assert install_compressed_tensors_keep_packed()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        packed_dir, dtype = torch.float16, device_map = {"": device}
    )
    packed = [m for m in model.modules() if isinstance(m, Mxfp4PackedLinear)]
    assert len(packed) == 2 * 7
    assert {m.compute_dtype for m in packed} == {torch.float16}
    assert {m.weight.dtype for m in packed} == {torch.float16}
    assert model.model.layers[0].input_layernorm.weight.dtype == torch.float16
    model = peft.get_peft_model(model, peft.LoraConfig(r = 2, target_modules = ["q_proj"]))
    merged = model.merge_and_unload()
    with torch.no_grad():
        assert merged(input_ids = torch.tensor([[1, 2, 3, 4]], device = device)).logits.isfinite().all()


@pytest.mark.skipif(not HAS_CT, reason = "needs compressed-tensors")
def test_adoption_declines_a_scheme_that_also_quantizes_activations():
    """The packed forward only dequantizes the weight; compressed-tensors' own forward also
    quantizes the input. Adopting such a module would silently drop that."""
    from unsloth.models.mxfp4_compressed_linear import is_mxfp4_scheme

    activations = {
        "num_bits": 8,
        "type": "int",
        "strategy": "token",
        "dynamic": True,
        "symmetric": True,
    }
    groups = {
        "group_0": {
            "targets": ["Linear"],
            "weights": _MXFP4_WEIGHTS,
            "input_activations": activations,
        }
    }
    model = _ct_compressed_model(groups = groups)
    assert not is_mxfp4_scheme(model.a.quantization_scheme, "mxfp4-pack-quantized")
    reference = copy.deepcopy(model)
    assert adopt_compressed_mxfp4_modules(model, dtype = torch.bfloat16) == 0
    assert type(model.a) is nn.Linear and hasattr(model, "ct_decompress_hook")
    x = torch.randn(3, 64, dtype = torch.bfloat16)
    with torch.no_grad():
        assert torch.equal(model(x), reference(x))


@pytest.mark.skipif(not HAS_CT, reason = "needs compressed-tensors")
def test_adoption_declines_when_another_compressed_format_needs_the_hook():
    """An FP8 module keeps its compressed weight under `weight`; compressed-tensors' model-wide
    hook still has to decompress it, so the MXFP4 modules are not adopted either."""
    fp8 = {
        "num_bits": 8,
        "type": "float",
        "strategy": "channel",
        "symmetric": True,
        "dynamic": False,
    }
    groups = {
        "group_0": {
            "targets": ["re:^a$", "re:^b$"],
            "weights": _MXFP4_WEIGHTS,
            "format": "mxfp4-pack-quantized",
        },
        "group_1": {"targets": ["re:^latent$"], "weights": fp8, "format": "float-quantized"},
    }
    model = _ct_compressed_model(groups = groups)
    assert model.latent.weight.dtype == torch.float8_e4m3fn
    reference = copy.deepcopy(model)
    assert adopt_compressed_mxfp4_modules(model, dtype = torch.bfloat16) == 0
    assert type(model.a) is nn.Linear and hasattr(model, "ct_decompress_hook")
    x = torch.randn(3, 64, dtype = torch.bfloat16)
    with torch.no_grad():
        assert torch.equal(model(x), reference(x))


@pytest.mark.parametrize("init", ["pissa", "olora"])
def test_initialisers_that_rewrite_the_base_weight_refuse_a_packed_base(init):
    """PiSSA / OLoRA (and CorDA, LoftQ, LoRA-GA) subtract their initial adapter from the base
    weight. On a packed base that write lands in a throwaway decode while the adapter keeps its
    value, which silently changes the model: refuse instead."""
    peft = pytest.importorskip("peft")
    holder = nn.Sequential(_filled(32, 64))
    with pytest.raises(NotImplementedError, match = "packed in MXFP4"):
        peft.get_peft_model(
            holder, peft.LoraConfig(r = 4, target_modules = ["0"], init_lora_weights = init)
        )
    # The default initialisation, and the same initialiser on a dense base, are untouched.
    peft.get_peft_model(nn.Sequential(_filled(32, 64)), peft.LoraConfig(r = 4, target_modules = ["0"]))
    dense = nn.Sequential(nn.Linear(64, 32, bias = False))
    peft.get_peft_model(dense, peft.LoraConfig(r = 4, target_modules = ["0"], init_lora_weights = init))


@_apply(needs_gpu_loader_any)
def test_the_sixteen_bit_route_declines_without_the_zoo_full_save_support(tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM
    from unsloth_zoo.temporary_patches import mxfp4 as zoo_mxfp4
    from unsloth.models.mxfp4_compressed_linear import install_compressed_tensors_keep_packed

    packed_dir, _ = _write_tiny_mxfp4_llama(str(tmp_path))
    assert install_compressed_tensors_keep_packed()
    monkeypatch.delattr(zoo_mxfp4, "_densified_module_names")
    model = AutoModelForCausalLM.from_pretrained(
        packed_dir, dtype = torch.bfloat16, device_map = {"": 0}
    )
    assert not any(isinstance(m, Mxfp4PackedLinear) for m in model.modules())


@_apply(needs_gpu_loader)
def test_a_partly_merged_bitsandbytes_route_model_saves_and_reloads(tmp_path):
    """4-bit route, LoRA on q_proj only, merge_and_unload, full save: the merged q_proj is a
    dense Linear now, the other packed Linears are written dense, and plain transformers reloads
    both as dense Linears (not bitsandbytes ones) with the in-memory model's outputs."""
    from transformers import AutoModelForCausalLM
    from unsloth import FastLanguageModel

    packed_dir, _ = _write_tiny_mxfp4_llama(str(tmp_path))
    _add_tokenizer(packed_dir, str(tmp_path))
    model, _ = FastLanguageModel.from_pretrained(
        packed_dir, max_seq_length = 64, dtype = torch.bfloat16, load_in_4bit = True
    )
    model = FastLanguageModel.get_peft_model(model, r = 4, lora_alpha = 8, target_modules = ["q_proj"])
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 0.05)
    unloaded = model.merge_and_unload()
    ids = torch.randint(0, 256, (1, 16), device = "cuda:0")
    with torch.no_grad():
        want = unloaded(input_ids = ids).logits
    out = str(tmp_path / "unloaded")
    unloaded.save_pretrained(out)
    reloaded = AutoModelForCausalLM.from_pretrained(out, dtype = torch.bfloat16, device_map = {"": 0})
    attn = reloaded.model.layers[0].self_attn
    assert type(attn.q_proj) is nn.Linear and type(attn.k_proj) is nn.Linear
    with torch.no_grad():
        got = reloaded(input_ids = ids).logits
    # Unsloth's fused kernels in memory vs plain transformers on reload.
    torch.testing.assert_close(got, want, atol = 2e-2, rtol = 2e-2)


def test_merge_and_unmerge_under_an_accelerate_hook():
    """A dispatched model wraps each module's forward and keeps the original as `_old_forward`;
    the class swaps of a merge and unmerge must carry it along, or the merged forward still runs
    the packed one on a module whose packed bytes were set aside."""
    pytest.importorskip("accelerate")
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    packed_model, dense_model = _lora_pair(scale_range = (118, 134))
    add_hook_to_module(
        packed_model.base_model.model[0].base_layer, AlignDevicesHook(io_same_device = True)
    )
    x = torch.randn(3, 64, generator = torch.Generator().manual_seed(0)).to(torch.bfloat16)
    with torch.no_grad():
        with_lora = packed_model(x)
        packed_model.merge_adapter()
        dense_model.merge_adapter()
        assert torch.equal(packed_model(x), dense_model(x))
        packed_model.unmerge_adapter()
        assert torch.equal(packed_model(x), with_lora)


@_apply(needs_gpu_loader_any)
def test_the_sixteen_bit_route_plans_only_mxfp4_checkpoints(tmp_path, monkeypatch):
    """Every compressed-tensors load passes through the hook; one that is not all MXFP4 (INT4
    Kimi-K2.7 has half a million keys) must not have its keys read or planned."""
    import importlib.util
    import unsloth.models.compressed_tensors_bnb as ctb
    from transformers import AutoModelForCausalLM
    from unsloth.models.mxfp4_compressed_linear import install_compressed_tensors_keep_packed

    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "test_compressed_tensors_bnb.py"
    )
    spec = importlib.util.spec_from_file_location("_k3s_ct_bnb_tests", path)
    ct_bnb_tests = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ct_bnb_tests)

    calls = []
    real = ctb._checkpoint_keys
    monkeypatch.setattr(ctb, "_checkpoint_keys", lambda files: calls.append(files) or real(files))
    assert install_compressed_tensors_keep_packed()
    int4_dir, _ = ct_bnb_tests._write_tiny_packed_llama(str(tmp_path / "int4"))
    AutoModelForCausalLM.from_pretrained(int4_dir, dtype = torch.bfloat16, device_map = {"": 0})
    assert calls == []
    mxfp4_dir, _ = _write_tiny_mxfp4_llama(str(tmp_path / "mxfp4"))
    model = AutoModelForCausalLM.from_pretrained(
        mxfp4_dir, dtype = torch.bfloat16, device_map = {"": 0}
    )
    assert (
        len(calls) == 1 and sum(isinstance(m, Mxfp4PackedLinear) for m in model.modules()) == 2 * 7
    )


@pytest.mark.parametrize("spelling", ["compressed-tensors", "compressed_tensors", "sparseml"])
def test_every_compressed_tensors_spelling_installs_the_keep_packed_hook(spelling, monkeypatch):
    """A 16-bit load (requantize_packed=False) never reaches the re-quantization branch, so the
    keep-packed hook must be installed for every spelling that branch accepts."""
    from types import SimpleNamespace
    from unsloth.models import loader_utils, mxfp4_compressed_linear

    calls = []
    monkeypatch.setattr(mxfp4_compressed_linear, "install_compressed_tensors_keep_packed", lambda: calls.append(1))
    config = SimpleNamespace(quantization_config = {"quant_method": spelling, "format": "mxfp4-pack-quantized"})
    loader_utils.check_and_disable_bitsandbytes_loading(
        config, load_in_4bit = False, verbose = False, requantize_packed = False,
    )
    assert calls == [1]
