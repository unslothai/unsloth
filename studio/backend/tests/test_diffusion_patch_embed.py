# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-VL patch embed as a matmul: the torch / cuDNN gate, exactness against Conv3d, install rules (CPU)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from core.inference import diffusion_patch_embed as pe


class _Target:
    def __init__(self, device: str) -> None:
        self.device = device


class _Component(torch.nn.Module):
    def __init__(self, *children: torch.nn.Module) -> None:
        super().__init__()
        self.children_ = torch.nn.ModuleList(children)


class _Pipe:
    def __init__(self, **components) -> None:
        self.components = components


def _embed(
    cls_name: str = "Qwen3VLVisionPatchEmbed",
    *,
    p: int = 4,
    t: int = 2,
    out: int = 8,
    bias: bool = True,
    stride = None,
    dtype = torch.float32,
) -> torch.nn.Module:
    cls = type(cls_name, (torch.nn.Module,), {})
    module = cls()
    kernel = (t, p, p)
    module.proj = torch.nn.Conv3d(3, out, kernel, stride = stride or kernel, bias = bias).to(dtype)
    return module


@pytest.mark.parametrize(
    "version, cudnn, expected",
    [
        ("2.9.1+cu128", 91002, True),  # the pip wheel: cuDNN 9.10.2
        ("2.9.0+cu126", 91002, True),
        ("2.9.1", 90800, True),
        ("2.9.1", 91499, True),
        ("2.9.1+cu128", 91500, False),  # cuDNN 9.15+: torch 2.9 uses cuDNN again
        ("2.9.1+cu128", 92801, False),
        ("2.9.1", 90700, False),
        ("2.10.0+cu128", 91002, False),  # torch 2.10 dropped the check
        ("2.8.0+cu128", 91002, False),  # torch 2.8 never had it
        ("2.14.1+cu130", 92400, False),
        ("2.9.1", None, False),
        ("garbage", 91002, False),
    ],
)
def test_gate_matches_torch_29_use_cudnn_window(version, cudnn, expected):
    assert pe.conv3d_skips_cudnn(version, cudnn) is expected


@pytest.mark.parametrize(
    "version, runtime, compiled, expected",
    [
        (
            "2.9.0+cu128",
            91501,
            91002,
            True,
        ),  # 2.9.0 reads the compile-time 9.10.2 whatever is loaded
        ("2.9.0+cu128", 92801, 90800, True),
        ("2.9.0+cu128", 91002, 90700, False),
        ("2.9.1+cu128", 91501, 91002, False),  # 2.9.1 reads the runtime cuDNN
        ("2.9.1+cu128", 91002, 92801, True),
        ("2.9.0rc1", 91501, 91002, True),
    ],
)
def test_torch_290_gates_on_the_compile_time_cudnn(version, runtime, compiled, expected):
    assert pe.conv3d_skips_cudnn(version, runtime, compiled) is expected


@pytest.mark.parametrize(
    "p, out, bias", [(14, 1280, False), (16, 1152, True)]
)  # Qwen2.5-VL, Qwen3-VL
def test_linear_equals_conv3d(p, out, bias):
    gen = torch.Generator().manual_seed(0)
    conv = torch.nn.Conv3d(3, out, (2, p, p), stride = (2, p, p), bias = bias)
    x = torch.randn(64, 3, 2, p, p, generator = gen)
    with torch.no_grad():
        ref = conv(x)
        got = pe.patch_conv_as_linear(x, conv.weight, conv.bias)
        assert got.shape == ref.shape
        assert (got - ref).abs().max().item() < 1e-4
        x64, w64 = x.double(), conv.weight.double()
        b64 = None if conv.bias is None else conv.bias.double()
        oracle = F.conv3d(x64, w64, b64, stride = (2, p, p))
        assert (pe.patch_conv_as_linear(x64, w64, b64) - oracle).abs().max().item() < 1e-10
        xb, wb = x.bfloat16(), conv.weight.bfloat16()
        bb = None if conv.bias is None else conv.bias.bfloat16()
        got_b = pe.patch_conv_as_linear(xb, wb, bb).double()
        # bf16 matmul accumulates in fp32: within bf16 rounding of the fp64 oracle.
        assert (got_b - oracle).abs().max().item() <= 0.02 * max(1.0, oracle.abs().max().item())


def test_patched_forward_takes_the_matmul_only_for_half_cuda_patch_rows(monkeypatch):
    module = _embed(dtype = torch.bfloat16)
    pipe = _Pipe(text_encoder = _Component(module))
    assert pe.install_patch_embed_linear(pipe, _Target("cuda"), affected = True) == 1
    conv = module.proj
    x = torch.randn(5, 3, 2, 4, 4).bfloat16()
    stock = torch.nn.Conv3d.forward(conv, x)
    assert torch.equal(conv(x), stock)
    calls = []
    real = pe.patch_conv_as_linear
    monkeypatch.setattr(pe, "_is_cuda", lambda _x: True)
    monkeypatch.setattr(pe, "patch_conv_as_linear", lambda *a: calls.append(1) or real(*a))
    got = conv(x)
    assert calls == [1]
    assert got.shape == stock.shape
    assert (got.float() - stock.float()).abs().max().item() <= 0.0625
    # fp32, or a tensor that is not one patch per row, keeps the stock conv.
    conv32 = _embed().proj
    pe.install_patch_embed_linear(
        _Pipe(te = _Component(_embed_holder(conv32))), _Target("cuda"), affected = True
    )
    conv32(torch.randn(2, 3, 2, 4, 4))
    conv(torch.randn(1, 3, 4, 8, 8).bfloat16())
    assert calls == [1]


def _embed_holder(conv: torch.nn.Module) -> torch.nn.Module:
    holder = type("Qwen2_5_VisionPatchEmbed", (torch.nn.Module,), {})()
    holder.proj = conv
    return holder


def test_install_is_gated_and_idempotent(monkeypatch):
    monkeypatch.delenv(pe.PATCH_EMBED_LINEAR_ENV, raising = False)
    module = _embed()
    pipe = _Pipe(text_encoder = _Component(module), vae = None)
    assert pe.install_patch_embed_linear(pipe, _Target("cuda"), affected = False) == 0
    assert "forward" not in module.proj.__dict__
    assert pe.install_patch_embed_linear(pipe, _Target("mps"), affected = True) == 0
    assert pe.install_patch_embed_linear(pipe, _Target("cpu"), affected = True) == 0
    monkeypatch.setenv(pe.PATCH_EMBED_LINEAR_ENV, "0")
    assert pe.install_patch_embed_linear(pipe, _Target("cuda"), affected = True) == 0
    monkeypatch.setenv(pe.PATCH_EMBED_LINEAR_ENV, "1")
    assert pe.install_patch_embed_linear(pipe, _Target("cuda"), affected = False) == 1
    assert pe.install_patch_embed_linear(pipe, _Target("cuda"), affected = False) == 0


def test_install_skips_everything_that_is_not_a_patchify_patch_embed(monkeypatch):
    monkeypatch.delenv(pe.PATCH_EMBED_LINEAR_ENV, raising = False)
    overlapping = _embed(stride = (1, 2, 2))
    pointwise = _embed(p = 1, t = 1)
    other_class = _embed("CausalConv3dBlock")
    padded = _embed()
    padded.proj.padding = (0, 1, 1)
    pipe = _Pipe(a = _Component(overlapping, pointwise, other_class, padded))
    assert pe.install_patch_embed_linear(pipe, _Target("cuda"), affected = True) == 0


def test_real_transformers_qwen_vl_patch_embeds_are_matched():
    transformers = pytest.importorskip("transformers")
    del transformers
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionPatchEmbed
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed

    q25 = Qwen2_5_VisionPatchEmbed(
        patch_size = 14, temporal_patch_size = 2, in_channels = 3, embed_dim = 32
    )
    q3 = Qwen3VLVisionPatchEmbed(Qwen3VLVisionConfig(hidden_size = 32, patch_size = 16))
    pipe = _Pipe(text_encoder = _Component(q25, q3))
    assert pe.install_patch_embed_linear(pipe, _Target("cuda"), affected = True) == 2
    for embed, p in ((q25, 14), (q3, 16)):
        x = torch.randn(7, 3 * 2 * p * p)
        with torch.no_grad():
            out = embed(x)
            ref = torch.nn.Conv3d.forward(embed.proj, x.view(-1, 3, 2, p, p)).view(-1, 32)
        assert torch.equal(out, ref)


def test_diffusion_load_installs_it_after_the_vae_hooks():
    src = Path(pe.__file__).with_name("diffusion.py").read_text(encoding = "utf-8")
    tree = ast.parse(src)
    names = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "install_patch_embed_linear" in names
    assert src.index("install_vae_sync(pipe, target") < src.index(
        "install_patch_embed_linear(pipe, target"
    )
