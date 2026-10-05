# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for ``diffusion_flux2_rope.py``: the fused FLUX.2 RoPE engages only for an fp16 FLUX.2 load (bf16 GPUs keep the
stock function), honours its kill switch, restores cleanly, and is bit-identical to stock (CUDA tests)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.inference import diffusion_flux2_rope as fr

torch = pytest.importorskip("torch")
fmod = pytest.importorskip("diffusers.models.transformers.transformer_flux2")
from diffusers.models.embeddings import apply_rotary_emb as stock_rope  # noqa: E402

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available()
    or bool(getattr(torch.version, "hip", None))
    or fr._kernel() is None,
    reason = "needs CUDA (not ROCm) and Triton",
)


class _Flux2Like:
    """A pipe whose transformer reports the FLUX.2 module, without building the model."""

    def __init__(self, module = fr._MODULE):
        self.transformer = type("Flux2Transformer2DModel", (), {"__module__": module})()


@pytest.fixture(autouse = True)
def _restore(monkeypatch):
    monkeypatch.delenv(fr.FLUX2_FUSED_ROPE_ENV, raising = False)
    fr.uninstall()
    yield
    fr.uninstall()
    assert fmod.apply_rotary_emb is stock_rope


@pytest.fixture
def fake_kernel(monkeypatch):
    """Gate tests run without a GPU: pretend Triton built the kernel on a CUDA (non-ROCm) torch."""
    monkeypatch.setattr(fr, "_kernel", lambda: (lambda x, cos, sin: None))
    monkeypatch.setattr(torch.version, "hip", None, raising = False)


def test_stock_symbol_is_the_one_flux2_calls():
    assert fmod.apply_rotary_emb is stock_rope
    src = Path(fmod.__file__).read_text(encoding = "utf-8")
    assert "apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)" in src


def test_fp16_flux2_load_installs(fake_kernel):
    assert fr.install_for_pipe(_Flux2Like(), torch.float16, "cuda")
    assert fr.is_installed()
    assert fmod.apply_rotary_emb is fr._fused_apply_rotary_emb


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_bf16_and_fp32_loads_keep_stock(fake_kernel, dtype):
    assert not fr.install_for_pipe(_Flux2Like(), getattr(torch, dtype), "cuda")
    assert not fr.is_installed()
    assert fmod.apply_rotary_emb is stock_rope


def test_fp16_load_after_bf16_load_and_back(fake_kernel):
    assert fr.install_for_pipe(_Flux2Like(), torch.float16, "cuda")
    assert not fr.install_for_pipe(_Flux2Like(), torch.bfloat16, "cuda")
    assert fmod.apply_rotary_emb is stock_rope


@pytest.mark.parametrize("device", ["cpu", "mps", "xpu"])
def test_non_cuda_devices_keep_stock(fake_kernel, device):
    assert not fr.install_for_pipe(_Flux2Like(), torch.float16, device)
    assert fmod.apply_rotary_emb is stock_rope


def test_rocm_keeps_stock(fake_kernel, monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "6.4", raising = False)
    assert not fr.install_for_pipe(_Flux2Like(), torch.float16, "cuda")


def test_other_families_keep_stock(fake_kernel):
    pipe = _Flux2Like(module = "diffusers.models.transformers.transformer_flux")
    assert not fr.install_for_pipe(pipe, torch.float16, "cuda")
    assert fmod.apply_rotary_emb is stock_rope


def test_other_family_load_removes_a_previous_install(fake_kernel):
    assert fr.install_for_pipe(_Flux2Like(), torch.float16, "cuda")
    assert not fr.install_for_pipe(_Flux2Like(module = "x.y"), torch.float16, "cuda")
    assert fmod.apply_rotary_emb is stock_rope


@pytest.mark.parametrize("value", ["0", "off", "false", "no"])
def test_kill_switch(fake_kernel, monkeypatch, value):
    monkeypatch.setenv(fr.FLUX2_FUSED_ROPE_ENV, value)
    assert not fr.install_for_pipe(_Flux2Like(), torch.float16, "cuda")
    assert fmod.apply_rotary_emb is stock_rope


def test_no_triton_keeps_stock(monkeypatch):
    monkeypatch.setattr(fr, "_kernel", lambda: None)
    assert not fr.install_for_pipe(_Flux2Like(), torch.float16, "cuda")


def test_install_is_idempotent_and_uninstall_restores(fake_kernel):
    assert fr.install(torch.float16, "cuda") and fr.install(torch.float16, "cuda")
    fr.uninstall()
    fr.uninstall()
    assert fmod.apply_rotary_emb is stock_rope


def test_cpu_tensors_fall_through_to_stock(fake_kernel):
    assert fr.install(torch.float16, "cuda")
    x = torch.randn(1, 6, 2, 8, dtype = torch.float32)
    pos = torch.randn(6, 4, dtype = torch.float64)
    freqs = (pos.cos().repeat_interleave(2, -1).float(), pos.sin().repeat_interleave(2, -1).float())
    assert torch.equal(
        fmod.apply_rotary_emb(x, freqs, sequence_dim = 1), stock_rope(x, freqs, sequence_dim = 1)
    )


def test_teardown_and_load_wiring():
    src = (Path(__file__).resolve().parents[1] / "core" / "inference" / "diffusion.py").read_text(
        encoding = "utf-8"
    )
    tree = ast.parse(src)
    teardown = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_uninstall_fused_dit_patches"
    )
    assert "uninstall_flux2_rope()" in ast.unparse(teardown)
    load = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "load_pipeline"
    )
    assert "install_flux2_rope(pipe, dtype, device, logger)" in ast.unparse(load)


def _case(shape, layout, freq_dtype):
    B, S, H, D = shape
    g = torch.Generator("cuda").manual_seed(0)
    if layout == "qkv_chunk":
        x = (
            torch.randn(B, S, 3 * H * D, device = "cuda", dtype = torch.float16, generator = g) * 4
        ).chunk(3, -1)[1]
        x = x.unflatten(-1, (H, D))
    elif layout == "permuted":
        x = (
            torch.randn(B, H, S, D, device = "cuda", dtype = torch.float16, generator = g) * 4
        ).transpose(1, 2)
    else:
        x = torch.randn(B, S, H, D, device = "cuda", dtype = torch.float16, generator = g) * 4
    pos = torch.randn(S, D // 2, device = "cuda", dtype = torch.float64, generator = g) * 50
    return x, (
        pos.cos().repeat_interleave(2, -1).to(freq_dtype),
        pos.sin().repeat_interleave(2, -1).to(freq_dtype),
    )


@needs_cuda
@pytest.mark.parametrize(
    "shape,layout,freq_dtype",
    [
        (
            (1, 4608, 24, 128),
            "contiguous",
            "float32",
        ),  # klein-4B 1024px: 512 text + 4096 image tokens
        ((2, 1000, 24, 128), "qkv_chunk", "float32"),
        ((1, 333, 5, 64), "qkv_chunk", "float16"),
        ((1, 777, 24, 128), "permuted", "float32"),
    ],
)
def test_bit_identical_to_stock(shape, layout, freq_dtype):
    x, freqs = _case(shape, layout, getattr(torch, freq_dtype))
    ref = stock_rope(x, freqs, sequence_dim = 1)
    assert fr.install(torch.float16, "cuda")
    out = fmod.apply_rotary_emb(x, freqs, sequence_dim = 1)
    assert out.dtype == ref.dtype and out.stride() == ref.stride()
    assert torch.equal(out, ref)


@needs_cuda
def test_grad_and_bf16_calls_use_stock():
    assert fr.install(torch.float16, "cuda")
    x, freqs = _case((1, 16, 2, 8), "contiguous", torch.float32)
    xg = x.clone().requires_grad_(True)
    fmod.apply_rotary_emb(xg, freqs, sequence_dim = 1).float().sum().backward()
    assert xg.grad is not None
    xb = x.bfloat16()
    assert torch.equal(
        fmod.apply_rotary_emb(xb, freqs, sequence_dim = 1), stock_rope(xb, freqs, sequence_dim = 1)
    )


@needs_cuda
def test_compiled_block_has_no_graph_break():
    """Studio compiles FLUX.2 blocks on fp16 explicit tiers: the patch must trace as stock, never split the block."""
    assert fr.install(torch.float16, "cuda")
    x, freqs = _case((1, 64, 4, 32), "contiguous", torch.float32)
    torch._dynamo.reset()
    compiled = torch.compile(
        lambda t: fmod.apply_rotary_emb(t, freqs, sequence_dim = 1), fullgraph = True
    )
    ref = torch.compile(lambda t: stock_rope(t, freqs, sequence_dim = 1), fullgraph = True)
    assert torch.equal(compiled(x), ref(x))


def test_call_after_uninstall_reaches_stock(fake_kernel):
    assert fr.install_for_pipe(_Flux2Like(), torch.float16, "cuda")
    patched = fmod.apply_rotary_emb
    fr.uninstall()
    x = torch.randn(1, 4, 2, 8)
    freqs = (torch.ones(4, 8), torch.zeros(4, 8))
    assert torch.equal(patched(x, freqs, sequence_dim = 1), x)
