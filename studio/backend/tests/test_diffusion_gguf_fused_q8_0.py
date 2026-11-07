# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fused Q8_0 dequant+cast for GGUFLinear: bit-identical to the stock dequant then cast, no graph breaks or
recompiles, other quant types untouched, reversible. Runs on CPU."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
gguf = pytest.importorskip("gguf")
gguf_utils = pytest.importorskip("diffusers.quantizers.gguf.utils")

from core.inference import diffusion_gguf_compile as gc  # noqa: E402

Q8_0 = gguf.GGMLQuantizationType.Q8_0
Q4_K = gguf.GGMLQuantizationType.Q4_K
needs_fused = pytest.mark.skipif(
    not gc._torch_supports_fused_q8_0(torch.__version__),
    reason = "fused Q8_0 needs torch >= 2.7",
)


@pytest.fixture(autouse = True)
def _clean():
    # Other Studio tests record inductor knobs (emulate_precision_casts among them) that the guarded compile
    # re-applies; start every test from torch's defaults so results do not depend on test order.
    from core.inference import diffusion_compile_config as cc

    cc._reset_for_tests()
    gc.uninstall_all()
    torch._dynamo.reset()
    with torch._inductor.config.patch(emulate_precision_casts = False):
        yield
    gc.uninstall_all()
    cc._reset_for_tests()


def _q8_0_bytes(
    rows,
    cols,
    seed = 0,
    adversarial = False,
):
    """Real Q8_0 blocks from gguf's own quantiser, or (adversarial) random bytes with the scale drawn over the whole
    fp16 range, so fp16 overflow / subnormal / tie cases of the fp16 product are exercised."""
    rng = np.random.default_rng(seed)
    if not adversarial:
        w = (rng.standard_normal((rows, cols)) * 0.02).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(gguf.quants.quantize(w, Q8_0)))
    raw = rng.integers(0, 256, size = (rows, cols // 32 * 34), dtype = np.uint8)
    return torch.from_numpy(raw)


def _param(raw, qtype = Q8_0):
    return gguf_utils.GGUFParameter(raw, quant_type = qtype)


def _stock_dense(
    raw,
    dtype,
    qtype = Q8_0,
):
    # diffusers' own eager path: GGUFLinear.forward_native's dequantize_gguf_tensor(weight).to(compute_dtype).
    return gguf_utils.dequantize_gguf_tensor(_param(raw, qtype)).to(dtype)


def _bits(t):
    return t.view(
        {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}[
            t.dtype
        ]
    )


@pytest.mark.parametrize(
    "emulate", [False, True], ids = ["inductor_default", "emulate_precision_casts"]
)
@pytest.mark.parametrize("adversarial", [False, True], ids = ["gguf_quantized", "random_bytes"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@needs_fused
def test_fused_q8_0_is_bit_identical_eager_and_compiled(adversarial, dtype, emulate):
    # Studio sets emulate_precision_casts on some renders, so the fused kernel must be exact with and without it.
    raw = _q8_0_bytes(96, 256, adversarial = adversarial)
    ref = _stock_dense(raw, dtype)
    eager = gc._q8_0_dequant_cast(raw, dtype)
    with torch._inductor.config.patch(emulate_precision_casts = emulate):
        compiled = gc._q8_0_dequant_cast(
            raw, dtype, torch.compile(gc._q8_0_blocks_dequant_cast, dynamic = True)
        )
    assert eager.shape == ref.shape == compiled.shape
    assert torch.equal(_bits(eager), _bits(ref))
    assert torch.equal(_bits(compiled), _bits(ref))


def _linear(
    raw,
    qtype,
    out_f,
    in_f,
    dtype = torch.bfloat16,
):
    lin = gguf_utils.GGUFLinear(in_f, out_f, bias = True, compute_dtype = dtype)
    lin.weight = _param(raw, qtype)
    lin.bias = torch.nn.Parameter(torch.randn(out_f, generator = torch.Generator().manual_seed(1)))
    return lin


@needs_fused
def test_naive_fusion_would_not_be_bit_identical():
    """Guards the int16 bitcast in _q8_0_dequant_cast: without it inductor skips the fp16 rounding."""

    def naive(raw, dtype):
        blocks = raw.reshape(-1, 34)
        d = blocks.view(torch.int16)[:, :1].view(torch.float16)
        return (d * blocks.view(torch.int8)[:, 2:]).reshape(raw.shape[0], -1).to(dtype)

    raw = _q8_0_bytes(96, 256, adversarial = True)
    got = torch.compile(naive, dynamic = True)(raw, torch.bfloat16)
    assert not torch.equal(_bits(got), _bits(_stock_dense(raw, torch.bfloat16)))


@needs_fused
def test_patched_q8_0_linear_skips_the_unfused_dequant_and_matches_stock(monkeypatch):
    raw = _q8_0_bytes(64, 128, seed = 3)
    lin = _linear(raw, Q8_0, 64, 128)
    x = torch.randn(5, 128, generator = torch.Generator().manual_seed(2)).to(torch.bfloat16)
    expected = lin.forward_native(x)  # stock, nothing installed

    assert gc.install_compiled_dequant() is True
    assert gc.is_fused_q8_0_installed() is True
    calls = []
    compiled_dequant = gguf_utils.dequantize_gguf_tensor
    monkeypatch.setattr(
        gguf_utils, "dequantize_gguf_tensor", lambda t: calls.append(1) or compiled_dequant(t)
    )
    got = lin(x)
    assert calls == []  # one fused kernel, not the compiled dequant plus an eager cast
    assert torch.equal(_bits(got), _bits(expected))


def test_other_quant_types_take_the_stock_forward(monkeypatch):
    rng = np.random.default_rng(4)
    block = gguf.GGML_QUANT_SIZES[Q4_K][1]
    raw = rng.integers(0, 256, size = (16, 256 // 256 * block), dtype = np.uint8)
    raw[:, :4] = 0x10  # small finite fp16 d / dmin
    raw = torch.from_numpy(raw)
    lin = _linear(raw, Q4_K, 16, 256)
    x = torch.randn(3, 256, generator = torch.Generator().manual_seed(5)).to(torch.bfloat16)
    stock_forward = gguf_utils.GGUFLinear.forward_native
    assert gc.install_compiled_dequant() is True
    expected = stock_forward(lin, x)  # the stock forward over the same compiled dequant
    calls = []
    compiled_dequant = gguf_utils.dequantize_gguf_tensor
    monkeypatch.setattr(
        gguf_utils, "dequantize_gguf_tensor", lambda t: calls.append(1) or compiled_dequant(t)
    )
    got = lin(x)
    assert calls == [1]
    assert torch.equal(_bits(got), _bits(expected))


@needs_fused
def test_fused_q8_0_has_no_graph_breaks_and_one_dynamic_graph():
    raw = _q8_0_bytes(32, 128)
    explain = torch._dynamo.explain(gc._q8_0_blocks_dequant_cast)(raw, torch.bfloat16)
    assert explain.graph_break_count == 0
    assert explain.graph_count == 1

    from torch._dynamo.testing import CompileCounter

    counter = CompileCounter()
    fn = torch.compile(gc._q8_0_blocks_dequant_cast, backend = counter, dynamic = True)
    for rows, cols in [(32, 128), (48, 256), (64, 4096), (17, 64)]:
        out = gc._q8_0_dequant_cast(_q8_0_bytes(rows, cols), torch.bfloat16, fn)
        assert out.shape == (rows, cols)
    assert counter.frame_count == 1  # new layer shapes reuse the graph: no recompiles


@needs_fused
def test_fused_q8_0_uninstall_and_kill_switch(monkeypatch):
    stock = gguf_utils.GGUFLinear.forward_native
    assert gc.install_compiled_dequant() is True
    assert gguf_utils.GGUFLinear.forward_native is not stock
    gc.uninstall_all()
    assert gguf_utils.GGUFLinear.forward_native is stock
    assert gc.is_fused_q8_0_installed() is False

    monkeypatch.setenv("UNSLOTH_DIFFUSION_GGUF_FUSED_Q8_0", "0")
    assert gc.install_compiled_dequant() is True
    assert gc.is_fused_q8_0_installed() is False
    assert gguf_utils.GGUFLinear.forward_native is stock


@pytest.mark.parametrize(
    "version, ok",
    [
        ("2.4.1+cu121", False),
        ("2.6.0+cu124", False),
        ("2.6.0+rocm6.2.4", False),
        ("2.7.1+cu126", True),
        ("2.11.0+rocm7.14.0", True),
        ("2.15.0.dev20261101+cu132", True),
        ("10.0.0", True),
        ("", False),
        ("unknown", False),
    ],
)
def test_fused_q8_0_torch_floor(version, ok):
    assert gc._torch_supports_fused_q8_0(version) is ok


def test_torch_below_the_floor_keeps_the_stock_forward(monkeypatch):
    stock = gguf_utils.GGUFLinear.forward_native
    monkeypatch.setattr(torch, "__version__", "2.6.0+cu124")
    assert gc.install_compiled_dequant() is True
    assert gc.is_fused_q8_0_installed() is False
    assert gguf_utils.GGUFLinear.forward_native is stock
