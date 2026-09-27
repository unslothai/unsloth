# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for ``diffusion_int8_fused.py``. The kernel / model tests need CUDA, Triton and torchao's Int8Tensor."""

from __future__ import annotations

import pytest

from core.inference import diffusion_int8_fused as fused

torch = pytest.importorskip("torch")


def _cuda_int8_ready() -> bool:
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        return False
    try:
        from torchao.quantization import Int8DynamicActivationInt8WeightConfig  # noqa: F401
        from torchao.quantization.quantize_.workflows.int8.int8_tensor import Int8Tensor  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return fused._device_ok(torch.cuda.current_device())


needs_cuda = pytest.mark.skipif(not _cuda_int8_ready(), reason = "needs CUDA (not ROCm), Triton and torchao Int8Tensor")


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(fused.INT8_FUSED_ENV, raising = False)
    yield
    fused.uninstall()


@pytest.mark.parametrize(
    "version, ok",
    [("3.7.1", True), ("3.2.0", True), ("3.1.0", False), ("2.3.1", False), ("garbage", False), ("4.0", True)],
)
def test_triton_version_gate(version, ok):
    assert fused._triton_version_ok(version) is ok


@pytest.mark.parametrize("value, off", [("0", True), ("off", True), ("false", True), ("1", False), ("", False)])
def test_kill_switch(monkeypatch, value, off):
    monkeypatch.setenv(fused.INT8_FUSED_ENV, value)
    assert fused.int8_fused_disabled() is off


def test_install_noop_when_disabled(monkeypatch):
    monkeypatch.setenv(fused.INT8_FUSED_ENV, "0")
    mod = torch.nn.Sequential(torch.nn.Linear(8, 8))
    assert fused.install(mod) == 0
    assert "forward" not in mod[0].__dict__


def test_plain_tensor_is_not_eligible():
    assert fused._plain_int8_weight(torch.zeros(8, 8)) is False
    assert fused._plain_int8_weight(None) is False


def test_install_noop_on_cpu_module():
    from diffusers.models.attention import FeedForward

    ff = FeedForward(64, 64, activation_fn = "gelu-approximate")
    assert fused.install(ff) == 0
    assert "forward" not in ff.__dict__


def _rand_inputs(m, n, *, ws_dtype = torch.float32, bias = True, seed = 0):
    g = torch.Generator(device = "cpu").manual_seed(seed)
    c = torch.randint(-(2**20), 2**20, (m, n), generator = g, dtype = torch.int32).cuda()
    xs = (torch.rand(m, generator = g) * 1e-3 + 1e-5).to(torch.bfloat16).float().cuda()
    ws = (torch.rand(n, generator = g) * 1e-4 + 1e-6).to(ws_dtype).cuda()
    b = (torch.randn(n, generator = g) * 0.1).to(torch.bfloat16).cuda() if bias else None
    return c, xs, ws, b


@needs_cuda
@pytest.mark.parametrize(
    "m, n, ws_dtype, bias",
    [
        (4096, 12288, torch.float32, True),  # Qwen-Image / FLUX.1 MLP
        (4101, 3000, torch.float32, True),  # odd rows, N not a multiple of the chunk
        (512, 12288, torch.bfloat16, False),  # bf16 weight scale, no bias
        (17, 8, torch.float32, True),  # smallest eligible
    ],
)
def test_kernel_bit_exact_vs_eager_reference(m, n, ws_dtype, bias):
    c, xs, ws, b = _rand_inputs(m, n, ws_dtype = ws_dtype, bias = bias)
    q, s = fused._launch(c, xs, ws, b, None)
    q_ref, s_ref = fused.reference_dq_gelu_quant(c, xs, ws, b, None)
    assert torch.equal(s, s_ref)
    assert torch.equal(q, q_ref)


@needs_cuda
def test_kernel_small_activations_take_the_exact_amax_path():
    # Every pre-activation negative: max|gelu| comes from the negative lobe, not gelu(max y).
    c, xs, ws, b = _rand_inputs(256, 1024, bias = False)
    c = -c.abs() - 1
    q, s = fused._launch(c, xs, ws, b, None)
    q_ref, s_ref = fused.reference_dq_gelu_quant(c, xs, ws, b, None)
    assert torch.equal(s, s_ref) and torch.equal(q, q_ref)


@needs_cuda
@pytest.mark.parametrize("transposed", [False, True])
def test_kernel_prefix_segment(transposed):
    bsz, seq, heads, hd, n = 2, 300, 4, 64, 1024
    c, xs, ws, b = _rand_inputs(bsz * seq, n)
    if transposed:  # SDPA output layout [B, H, S, D] seen as [B, S, H, D]
        prefix = (torch.randn(bsz, heads, seq, hd, device = "cuda") * 3).to(torch.bfloat16).transpose(1, 2)
    else:
        prefix = (torch.randn(bsz, seq, heads, hd, device = "cuda") * 3).to(torch.bfloat16)
    q, s = fused._launch(c, xs, ws, b, prefix)
    q_ref, s_ref = fused.reference_dq_gelu_quant(c, xs, ws, b, prefix)
    assert q.shape == (bsz * seq, heads * hd + n)
    assert torch.equal(s, s_ref) and torch.equal(q, q_ref)


def _quantized_ff(dim = 256, inner = 1024, seed = 0):
    from diffusers.models.attention import FeedForward
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

    torch.manual_seed(seed)
    ff = FeedForward(dim, dim, mult = inner // dim, activation_fn = "gelu-approximate").cuda().to(torch.bfloat16).eval()
    for p in ff.parameters():
        p.data.normal_(0, 0.05)
    quantize_(ff, Int8DynamicActivationInt8WeightConfig())
    return ff


@needs_cuda
def test_feedforward_bit_identical_to_stock_eager():
    ff = _quantized_ff()
    x = torch.randn(2, 300, 256, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        ref = ff(x)
        assert fused.install(ff) == 1
        out = ff(x)
    assert torch.equal(out, ref)


@needs_cuda
def test_feedforward_small_m_keeps_stock_path():
    ff = _quantized_ff()
    x = torch.randn(1, 8, 256, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        ref = ff(x)
        fused.install(ff)
        out = ff(x)
    assert torch.equal(out, ref)


@needs_cuda
def test_feedforward_compiles_without_graph_break():
    ff = _quantized_ff()
    x = torch.randn(1, 512, 256, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        eager_fused = (fused.install(ff), ff(x))[1]
        torch._dynamo.reset()
        compiled = torch.compile(ff, fullgraph = True)
        out = compiled(x)
    # Only the pointwise epilogue after the second GEMM is recompiled, so the result stays within bf16 rounding.
    assert (out.float() - eager_fused.float()).abs().max().item() <= 2 ** -6 * eager_fused.float().abs().max().item()


@needs_cuda
def test_uninstall_restores_stock_forward():
    ff = _quantized_ff()
    fused.install(ff)
    assert "forward" in ff.__dict__
    fused.uninstall(ff)
    assert "forward" not in ff.__dict__
    assert not fused.is_installed(ff)


@needs_cuda
def test_flux_single_block_bit_identical_to_stock_eager():
    tf = pytest.importorskip("diffusers.models.transformers.transformer_flux")
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

    torch.manual_seed(0)
    blk = tf.FluxSingleTransformerBlock(dim = 256, num_attention_heads = 4, attention_head_dim = 64).cuda().to(torch.bfloat16).eval()
    for p in blk.parameters():
        p.data.normal_(0, 0.05)
    quantize_(blk, Int8DynamicActivationInt8WeightConfig(), filter_fn = lambda m, fqn: isinstance(m, torch.nn.Linear) and "norm" not in fqn)
    hid = torch.randn(1, 200, 256, device = "cuda", dtype = torch.bfloat16)
    enc = torch.randn(1, 40, 256, device = "cuda", dtype = torch.bfloat16)
    temb = torch.randn(1, 256, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        ref = blk(hid, enc, temb)
        assert fused.install(blk) == 1
        out = blk(hid, enc, temb)
    assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])
