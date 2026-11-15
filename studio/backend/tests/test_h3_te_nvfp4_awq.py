# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The reference workflow's nvfp4_awq conditioner layout: dequantization in the reference CUDA kernel's order, the
AWQ input scale, the int8 embedding, and the loader's state dict on a tiny synthetic file (CPU); the fused Triton
dequantization against the torch math on CUDA."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from core.inference import video_minimax_h3_te as te  # noqa: E402
from core.inference import video_minimax_h3_te_nvfp4 as nv  # noqa: E402
from core.inference.diffusion_nvfp4_ops import swizzle_sf  # noqa: E402

cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason = "the Triton kernel runs on CUDA only"
)


def _packed(
    rows,
    cols,
    seed = 0,
):
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(0, 256, (rows, cols // 2), dtype = torch.uint8, generator = g)
    plain = (torch.rand(rows, cols // 16, generator = g) * 400 + 1e-3).to(torch.float8_e4m3fn)
    return w, plain, torch.tensor(0.00351097, dtype = torch.float32)


def _kernel_math(w, plain, gs, dtype):
    """The reference CUDA kernel, written out: decode = float32(fp8) * global; value = e2m1 * decode; one rounding."""
    lut = torch.tensor(nv._E2M1, dtype = torch.float64)
    rows, half = w.shape
    codes = torch.stack((w >> 4, w & 15), dim = -1).reshape(rows, half * 2).long()
    decode = (plain.to(torch.float32) * gs).to(torch.float64)  # float32 product, as the kernel
    exact = lut[codes].reshape(rows, -1, 16) * decode.unsqueeze(
        -1
    )  # e2m1 x float32 is exact in float64
    return exact.to(torch.float32).reshape(rows, half * 2).to(dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dequantization_is_the_reference_kernel_math(dtype):
    w, plain, gs = _packed(256, 512)
    assert torch.equal(
        nv.dequantize_nvfp4_reference(w, plain, gs, dtype), _kernel_math(w, plain, gs, dtype)
    )
    assert torch.equal(nv.dequantize_nvfp4(w, plain, gs, dtype), _kernel_math(w, plain, gs, dtype))


def test_the_high_nibble_is_the_even_column():
    w = torch.tensor([[0x2D]], dtype = torch.uint8)  # hi 2 -> 1.0, lo 13 -> -3.0
    plain = torch.ones(1, 1, dtype = torch.float32).to(torch.float8_e4m3fn)
    out = nv.dequantize_nvfp4_reference(
        torch.cat([w] * 8, dim = 1), plain, torch.tensor(1.0), torch.float32
    )
    assert out[0, :2].tolist() == [1.0, -3.0]


def test_block_scales_unswizzle_from_the_128x4_layout():
    for rows, cols in ((1024, 5120), (300, 64)):
        _w, plain, _g = _packed(rows, cols)
        swizzled = swizzle_sf(plain, rows, cols).view(torch.float8_e4m3fn)
        got = nv.plain_block_scales(swizzled, rows, cols)
        assert torch.equal(got.view(torch.uint8), plain.view(torch.uint8))


def test_the_linear_scales_the_input_then_multiplies_by_the_dequantized_weight():
    w, plain, gs = _packed(64, 128, seed = 3)
    cls = nv.nvfp4_awq_linear_class()
    lin = cls(128, 64, pre_quant = True)
    pqs = (1 + torch.rand(128)).to(torch.bfloat16)
    lin.load_state_dict(
        {
            "weight": w,
            "weight_scale": plain,
            "weight_scale_2": gs.reshape(()),
            "pre_quant_scale": pqs,
        },
        assign = True,
    )
    x = torch.randn(3, 5, 128).to(torch.bfloat16)
    want = torch.nn.functional.linear(
        x * pqs, nv.dequantize_nvfp4_reference(w, plain, gs, torch.bfloat16)
    )
    assert torch.equal(lin(x), want)
    assert lin.weight.dtype == torch.uint8 and lin.weight_scale.dtype == torch.float8_e4m3fn
    plain_lin = cls(128, 64, pre_quant = False)
    plain_lin.load_state_dict(
        {"weight": w, "weight_scale": plain, "weight_scale_2": gs.reshape(())}, assign = True
    )
    want = torch.nn.functional.linear(
        x, nv.dequantize_nvfp4_reference(w, plain, gs, torch.bfloat16)
    )
    assert torch.equal(plain_lin(x), want)


def test_the_int8_embedding_is_dequantized_in_float32_then_rounded():
    q = torch.randint(-127, 128, (10, 8), dtype = torch.int8)
    s = torch.rand(10, 1) * 0.01
    got = nv.dequantize_int8_embedding(q, s, torch.bfloat16)
    assert torch.equal(got, (q.float() * s).to(torch.bfloat16))


def _meta(d):
    return torch.frombuffer(bytearray(json.dumps(d).encode()), dtype = torch.uint8).clone()


def _tiny_encoder():
    from torch import nn

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(128, 64, bias = False, device = "meta")
            self.o_proj = nn.Linear(64, 128, bias = False, device = "meta")

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attn()
            self.input_layernorm = nn.RMSNorm(128, device = "meta")

    class LM(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer()])
            self.embed_tokens = nn.Embedding(10, 128, device = "meta")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = LM()

    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Model()

    return Enc()


def _write_file(path, *, meta = None):
    tensors = {}
    for i, (name, rows, cols, pre) in enumerate(
        (
            ("model.layers.0.self_attn.q_proj", 64, 128, False),
            ("model.layers.0.self_attn.o_proj", 128, 64, True),
        )
    ):
        w, plain, gs = _packed(rows, cols, seed = i)
        tensors[name + ".weight"] = w
        tensors[name + ".weight_scale"] = (
            swizzle_sf(plain, rows, cols)
            .view(torch.float8_e4m3fn)
            .reshape((rows + 127) // 128 * 128, -1)
        )
        tensors[name + ".weight_scale_2"] = gs.reshape(())
        tensors[name + te._QUANT_SUFFIX] = _meta(
            meta or {"format": "nvfp4", "full_precision_matrix_mult": True}
        )
        if pre:
            tensors[name + ".pre_quant_scale"] = (1 + torch.rand(cols)).to(torch.bfloat16)
    tensors["model.embed_tokens.weight"] = torch.randint(-127, 128, (10, 128), dtype = torch.int8)
    tensors["model.embed_tokens.weight_scale"] = torch.rand(10, 1) * 0.01
    tensors["model.embed_tokens" + te._QUANT_SUFFIX] = _meta({"format": "int8_tensorwise"})
    tensors["model.layers.0.input_layernorm.weight"] = torch.ones(128, dtype = torch.float32)
    safetensors_torch.save_file(tensors, str(path))
    return tensors


def test_the_loader_swaps_projections_and_builds_the_reference_state(tmp_path):
    path = tmp_path / "te_nvfp4_awq.safetensors"
    tensors = _write_file(path)
    assert te._is_nvfp4_awq_path(str(path))
    enc = _tiny_encoder()
    state, quantized = te._h3_nvfp4_awq_state(enc, str(path), torch.bfloat16)
    assert quantized == {"model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.o_proj"}
    attn = enc.model.language_model.layers[0].self_attn
    cls = nv.nvfp4_awq_linear_class()
    assert isinstance(attn.q_proj, cls) and attn.q_proj.pre_quant_scale is None
    assert isinstance(attn.o_proj, cls) and attn.o_proj.pre_quant_scale is not None
    missing, unexpected = enc.load_state_dict(state, strict = False, assign = True)
    assert not unexpected and not missing
    emb = state["model.language_model.embed_tokens.weight"]
    assert emb.dtype == torch.bfloat16
    assert torch.equal(
        emb,
        nv.dequantize_int8_embedding(
            tensors["model.embed_tokens.weight"],
            tensors["model.embed_tokens.weight_scale"],
            torch.bfloat16,
        ),
    )
    assert state["model.language_model.layers.0.input_layernorm.weight"].dtype == torch.bfloat16
    x = torch.randn(2, 64).to(torch.bfloat16)
    pqs = tensors["model.layers.0.self_attn.o_proj.pre_quant_scale"]
    w, plain, gs = _packed(128, 64, seed = 1)
    assert torch.equal(
        attn.o_proj(x),
        torch.nn.functional.linear(
            x * pqs, nv.dequantize_nvfp4_reference(w, plain, gs, torch.bfloat16)
        ),
    )


def test_the_loader_refuses_another_quant_format(tmp_path):
    path = tmp_path / "bad.safetensors"
    _write_file(path, meta = {"format": "nvfp4", "full_precision_matrix_mult": False})
    with pytest.raises(ValueError, match = "full_precision_matrix_mult"):
        te._h3_nvfp4_awq_state(_tiny_encoder(), str(path), torch.bfloat16)


def test_an_int8_hosted_file_is_not_read_as_nvfp4(tmp_path):
    path = tmp_path / "int8.safetensors"
    safetensors_torch.save_file(
        {
            "model.layers.0.mlp.up_proj.weight": torch.zeros(4, 4, dtype = torch.int8),
            "model.layers.0.mlp.up_proj.weight_scale": torch.ones(4, 1),
        },  # fmt: skip
        str(path),
    )
    assert not te._is_nvfp4_awq_path(str(path))


@cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows, cols", [(256, 512), (1024, 5120), (130, 2048)])
def test_the_fused_dequantization_is_bit_identical_on_cuda(dtype, rows, cols):
    w, plain, gs = _packed(rows, cols, seed = rows)
    want = nv.dequantize_nvfp4_reference(w, plain, gs, dtype)
    got = nv.dequantize_nvfp4(w.cuda(), plain.cuda(), gs.cuda(), dtype)
    assert torch.equal(got.cpu(), want)
