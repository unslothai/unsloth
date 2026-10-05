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


needs_cuda = pytest.mark.skipif(
    not _cuda_int8_ready(), reason = "needs CUDA (not ROCm), Triton and torchao Int8Tensor"
)


def _cuda_int8_toolchain() -> bool:
    """The probe's prerequisites without the probe: on such a host a refused probe is a failure, not a skip."""
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        return False
    try:
        from torchao.quantization.quantize_.workflows.int8.int8_tensor import Int8Tensor  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return fused._triton_version_ok() and fused._triton_jit_toolchain_ok()


@pytest.mark.skipif(
    not _cuda_int8_toolchain(), reason = "needs CUDA (not ROCm), Triton >= 3.2, torchao"
)
def test_device_probe_accepts_this_torchao():
    fused._device_ok.cache_clear()
    assert fused._device_ok(torch.cuda.current_device())


def _int8_config():
    """Studio's int8 config as an ``Int8Tensor`` (0.17 defaults to the legacy tensor), no process-wide setter."""
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig

    cfg = Int8DynamicActivationInt8WeightConfig(set_inductor_config = False)
    if hasattr(cfg, "version"):
        cfg.version = 2
    return cfg


def _require_int8tensor(module):
    """Skip when the torchao (<= 0.17) int8 config builds the legacy tensor: the fused path then keeps stock."""
    for m in module.modules():
        if isinstance(m, torch.nn.Linear) and type(m.weight).__name__ not in (
            "Parameter",
            "Tensor",
        ):
            if type(m.weight).__name__ != "Int8Tensor":
                pytest.skip(f"torchao int8 config builds {type(m.weight).__name__}, not Int8Tensor")
            return module
    return module


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(fused.INT8_FUSED_ENV, raising = False)
    yield


def test_int_mm_finds_safe_int_mm_after_torchao_kernel_was_removed(monkeypatch):
    """torchao main moved ``safe_int_mm`` out of ``torchao.kernel.intmm`` (and deleted ``torchao.kernel``):
    the fused forwards must find it in its new home, not fail every int8 render with ModuleNotFoundError."""
    import sys
    import types

    calls = []
    new_home = types.ModuleType("torchao.quantization.quantize_.workflows.int8.kernels")
    new_home.safe_int_mm = lambda a, b: calls.append((a, b)) or "out"
    monkeypatch.setitem(sys.modules, "torchao.kernel", None)
    monkeypatch.setitem(sys.modules, "torchao.kernel.intmm", None)
    monkeypatch.setitem(sys.modules, new_home.__name__, new_home)
    monkeypatch.setattr(fused, "_INTMM_MODULE", None, raising = False)
    weight = types.SimpleNamespace(qdata = torch.ones(3, 2, dtype = torch.int8))
    a = torch.ones(4, 2, dtype = torch.int8)
    assert fused._int_mm(a, weight) == "out"
    assert len(calls) == 1 and calls[0][0] is a and tuple(calls[0][1].shape) == (2, 3)


def test_int_mm_prefers_the_released_home(monkeypatch):
    """torchao <= 0.18 keeps ``torchao.kernel.intmm``: it wins over any other copy, so the capture-safe rebinding
    Studio installs there is the one that runs."""
    import sys
    import types

    old_home = types.ModuleType("torchao.kernel.intmm")
    old_home.safe_int_mm = lambda a, b: "old"
    new_home = types.ModuleType("torchao.quantization.quantize_.workflows.int8.kernels")
    new_home.safe_int_mm = lambda a, b: "new"
    monkeypatch.setitem(sys.modules, "torchao.kernel.intmm", old_home)
    monkeypatch.setitem(sys.modules, new_home.__name__, new_home)
    monkeypatch.setattr(fused, "_INTMM_MODULE", None, raising = False)
    weight = types.SimpleNamespace(qdata = torch.ones(3, 2, dtype = torch.int8))
    assert fused._int_mm(torch.ones(4, 2, dtype = torch.int8), weight) == "old"


@pytest.mark.parametrize(
    "version, ok",
    [
        ("3.7.1", True),
        ("3.2.0", True),
        ("3.1.0", False),
        ("2.3.1", False),
        ("garbage", False),
        ("4.0", True),
    ],
)
def test_triton_version_gate(version, ok):
    assert fused._triton_version_ok(version) is ok


@pytest.mark.parametrize(
    "value, off", [("0", True), ("off", True), ("false", True), ("1", False), ("", False)]
)
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
    FeedForward = pytest.importorskip("diffusers.models.attention").FeedForward

    ff = FeedForward(64, 64, activation_fn = "gelu-approximate")
    assert fused.install(ff) == 0
    assert "forward" not in ff.__dict__


def _rand_inputs(
    m,
    n,
    *,
    ws_dtype = torch.float32,
    bias = True,
    seed = 0,
    xs_dtype = torch.float32,
):
    g = torch.Generator(device = "cpu").manual_seed(seed)
    c = torch.randint(-(2**20), 2**20, (m, n), generator = g, dtype = torch.int32).cuda()
    xs = (torch.rand(m, generator = g) * 1e-3 + 1e-5).to(torch.bfloat16).to(xs_dtype).cuda()
    ws = (torch.rand(n, generator = g) * 1e-4 + 1e-6).to(ws_dtype).cuda()
    b = (torch.randn(n, generator = g) * 0.1).to(torch.bfloat16).cuda() if bias else None
    return c, xs, ws, b


_XS_DTYPES = pytest.mark.parametrize(
    "xs_dtype", [torch.float32, torch.bfloat16], ids = ["xs_fp32", "xs_bf16"]
)


@needs_cuda
@_XS_DTYPES
@pytest.mark.parametrize(
    "m, n, ws_dtype, bias",
    [
        (4096, 12288, torch.float32, True),  # Qwen-Image / FLUX.1 MLP
        (4101, 3000, torch.float32, True),  # odd rows, N not a multiple of the chunk
        (512, 12288, torch.bfloat16, False),  # bf16 weight scale, no bias
        (17, 8, torch.float32, True),  # smallest eligible
    ],
)
def test_kernel_bit_exact_vs_eager_reference(m, n, ws_dtype, bias, xs_dtype):
    c, xs, ws, b = _rand_inputs(m, n, ws_dtype = ws_dtype, bias = bias, xs_dtype = xs_dtype)
    q, s = fused._launch(c, xs, ws, b, None)
    q_ref, s_ref = fused.reference_dq_gelu_quant(c, xs, ws, b, None)
    assert s.dtype == s_ref.dtype == xs_dtype
    assert torch.equal(s, s_ref)
    assert torch.equal(q, q_ref)


@needs_cuda
@_XS_DTYPES
def test_kernel_small_activations_take_the_exact_amax_path(xs_dtype):
    # Every pre-activation negative: max|gelu| comes from the negative lobe, not gelu(max y).
    c, xs, ws, b = _rand_inputs(256, 1024, bias = False, xs_dtype = xs_dtype)
    c = -c.abs() - 1
    q, s = fused._launch(c, xs, ws, b, None)
    q_ref, s_ref = fused.reference_dq_gelu_quant(c, xs, ws, b, None)
    assert torch.equal(s, s_ref) and torch.equal(q, q_ref)


@needs_cuda
@_XS_DTYPES
@pytest.mark.parametrize("transposed", [False, True])
def test_kernel_prefix_segment(transposed, xs_dtype):
    bsz, seq, heads, hd, n = 2, 300, 4, 64, 1024
    c, xs, ws, b = _rand_inputs(bsz * seq, n, xs_dtype = xs_dtype)
    if transposed:  # SDPA output layout [B, H, S, D] seen as [B, S, H, D]
        prefix = (
            (torch.randn(bsz, heads, seq, hd, device = "cuda") * 3).to(torch.bfloat16).transpose(1, 2)
        )
    else:
        prefix = (torch.randn(bsz, seq, heads, hd, device = "cuda") * 3).to(torch.bfloat16)
    q, s = fused._launch(c, xs, ws, b, prefix)
    q_ref, s_ref = fused.reference_dq_gelu_quant(c, xs, ws, b, prefix)
    assert q.shape == (bsz * seq, heads * hd + n)
    assert torch.equal(s, s_ref) and torch.equal(q, q_ref)


def _quantized_ff(
    dim = 256,
    inner = 1024,
    seed = 0,
):
    FeedForward = pytest.importorskip("diffusers.models.attention").FeedForward
    from torchao.quantization import quantize_

    torch.manual_seed(seed)
    ff = (
        FeedForward(dim, dim, mult = inner // dim, activation_fn = "gelu-approximate")
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    for p in ff.parameters():
        p.data.normal_(0, 0.05)
    quantize_(ff, _int8_config())
    return _require_int8tensor(ff)


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
        eager = ff(x)
        torch._dynamo.reset()
        stock_compiled = torch.compile(ff, fullgraph = True)(x)
        fused.install(ff)
        torch._dynamo.reset()
        out = torch.compile(ff, fullgraph = True)(x)
    _assert_within_compile_floor(out, stock_compiled, eager)


@needs_cuda
def test_install_is_idempotent_and_holds_no_global_reference():
    import gc
    import weakref

    ff = _quantized_ff()
    assert fused.install(ff) == 1
    assert fused.install(ff) == 1
    ref = weakref.ref(ff)
    del ff
    gc.collect()
    assert ref() is None


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
    from torchao.quantization import quantize_

    torch.manual_seed(0)
    blk = (
        tf.FluxSingleTransformerBlock(dim = 256, num_attention_heads = 4, attention_head_dim = 64)
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    for p in blk.parameters():
        p.data.normal_(0, 0.05)
    quantize_(
        blk,
        _int8_config(),
        filter_fn = lambda m, fqn: isinstance(m, torch.nn.Linear) and "norm" not in fqn,
    )
    _require_int8tensor(blk)
    hid = torch.randn(1, 200, 256, device = "cuda", dtype = torch.bfloat16)
    enc = torch.randn(1, 40, 256, device = "cuda", dtype = torch.bfloat16)
    temb = torch.randn(1, 256, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        ref = blk(hid, enc, temb)
        assert fused.install(blk) == 1
        out = blk(hid, enc, temb)
    assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])


@needs_cuda
@_XS_DTYPES
@pytest.mark.parametrize(
    "m, n, gate_col, value_col", [(4352, 10240, 0, 10240), (300, 1024, 1024, 0), (17, 64, 0, 64)]
)
def test_swiglu_kernel_bit_exact_vs_eager_reference(m, n, gate_col, value_col, xs_dtype):
    c, xs, ws, b = _rand_inputs(m, 2 * n, bias = False, xs_dtype = xs_dtype)
    q, s = fused._launch_swiglu(c, xs, ws, None, gate_col, value_col, n)
    q_ref, s_ref = fused.reference_dq_swiglu_quant(c, xs, ws, None, gate_col, value_col, n)
    assert s.dtype == s_ref.dtype == xs_dtype
    assert torch.equal(s, s_ref) and torch.equal(q, q_ref)


def _bf16_tie_ints(m, n, seed):
    """int32 accumulators in [2^24, 2^30) within one fp32 ulp of a bf16 rounding midpoint, both signs."""
    g = torch.Generator(device = "cpu").manual_seed(seed)
    e = torch.randint(24, 30, (m, n), generator = g, dtype = torch.int64)
    one = torch.ones_like(e)
    mid = (
        torch.bitwise_left_shift(one, e)
        + torch.randint(0, 128, (m, n), generator = g, dtype = torch.int64)
        * torch.bitwise_left_shift(one, e - 7)
        + torch.bitwise_left_shift(one, e - 8)
    )
    half_ulp = torch.bitwise_left_shift(one, e - 24)
    off = torch.randint(-1, 2, (m, n), generator = g, dtype = torch.int64) * half_ulp
    off = off + torch.randint(-1, 2, (m, n), generator = g, dtype = torch.int64)
    sign = torch.randint(0, 2, (m, n), generator = g, dtype = torch.int64) * 2 - 1
    return ((mid + off) * sign).to(torch.int32).cuda()


def _double_rounding_hits(c):
    """How many int32 values round differently int32 -> bf16 directly (exact RNE in int64) than via fp32 (torch)."""
    v = c.cpu().to(torch.int64)
    a = v.abs()
    e = torch.floor(torch.log2(a.double().clamp(min = 1))).to(torch.int64)
    shift = (e - 7).clamp(min = 1)
    one = torch.ones_like(a)
    q = torch.bitwise_right_shift(a, shift)
    r = a - torch.bitwise_left_shift(q, shift)
    half = torch.bitwise_left_shift(one, shift - 1)
    q = q + ((r > half) | ((r == half) & (q % 2 == 1))).to(torch.int64)
    single = torch.bitwise_left_shift(q, shift) * v.sign()
    double = c.cpu().float().to(torch.bfloat16).double().to(torch.int64)
    return int((single != double)[a >= 2**24].sum())


@needs_cuda
@_XS_DTYPES
def test_kernels_round_large_accumulators_like_torchao(xs_dtype):
    m, n = 64, 2048
    c = _bf16_tie_ints(m, n, 3)
    assert _double_rounding_hits(c) > 100  # the inputs do reach the case
    _, xs, ws, b = _rand_inputs(m, n, xs_dtype = xs_dtype)
    xs = (xs * 2**-10).to(xs_dtype)  # |c * xs| stays O(1-100): GELU / SiLU in their curved range
    q, s = fused._launch(c, xs, ws, b, None)
    q_ref, s_ref = fused.reference_dq_gelu_quant(c, xs, ws, b, None)
    assert torch.equal(s, s_ref) and torch.equal(q, q_ref)
    q, s = fused._launch_swiglu(c, xs, ws, b, n // 2, 0, n // 2)
    q_ref, s_ref = fused.reference_dq_swiglu_quant(c, xs, ws, b, n // 2, 0, n // 2)
    assert torch.equal(s, s_ref) and torch.equal(q, q_ref)


@needs_cuda
def test_gelu_rounds_every_bf16_input_like_aten():
    assert fused._gelu_matches_aten(torch.device("cuda", torch.cuda.current_device()))


@needs_cuda
def test_epilogue_rounds_scale_product_and_bias_separately():
    assert fused._epilogue_matches_torchao(
        torch.device("cuda", torch.cuda.current_device()), 4101, 3000
    )


@needs_cuda
def test_reference_act_quant_is_torchao_own():
    from torchao.quantization.granularity import PerRow
    from torchao.quantization.quantize_.workflows.int8.int8_tensor import Int8Tensor

    g = torch.Generator(device = "cpu").manual_seed(5)
    h = torch.randn(300, 3072, generator = g) * (torch.rand(1, 3072, generator = g) * 6)
    h[:, :4] *= 80
    h[7] = 0
    h = h.to(torch.bfloat16).cuda()
    t = Int8Tensor.from_hp(h, PerRow())
    q, s = fused._reference_act_quant(h, t.scale.dtype)
    assert torch.equal(q, t.qdata) and torch.equal(s, t.scale.reshape(-1))


@needs_cuda
@_XS_DTYPES
def test_fake_ops_report_the_real_scale_dtype(xs_dtype):
    from torch._subclasses.fake_tensor import FakeTensorMode

    c, xs, ws, b = _rand_inputs(40, 256, xs_dtype = xs_dtype)
    op = fused._op()
    real = (op.gelu(c, xs, ws, b, None)[1], op.swiglu(c, xs, ws, b, 128, 0, 128)[1])
    with FakeTensorMode() as mode:
        fc, fxs, fws, fb = (mode.from_tensor(t) for t in (c, xs, ws, b))
        fake = (op.gelu(fc, fxs, fws, fb, None)[1], op.swiglu(fc, fxs, fws, fb, 128, 0, 128)[1])
    assert [t.dtype for t in fake] == [t.dtype for t in real] == [xs_dtype, xs_dtype]


@needs_cuda
def test_quantizing_leaves_fp32_matmul_precision_alone():
    # The default handler's recommended_inductor_config_setter() turns on TF32 process-wide; Studio opts out.
    before = torch.get_float32_matmul_precision()
    _quantized_ff()
    assert torch.get_float32_matmul_precision() == before


def _quantize(module):
    from torchao.quantization import quantize_

    for p in module.parameters():
        p.data.normal_(0, 0.05)
    quantize_(module, _int8_config())
    return _require_int8tensor(module)


@needs_cuda
@pytest.mark.parametrize("kind", ["diffusers_swiglu", "zimage", "flux2", "qwenimage21"])
def test_swiglu_mlps_bit_identical_to_stock_eager(kind, monkeypatch):
    monkeypatch.setattr(fused, "_SWIGLU_ALL_LAYOUTS", True)
    torch.manual_seed(0)
    if kind == "diffusers_swiglu":
        FeedForward = pytest.importorskip("diffusers.models.attention").FeedForward
        ff = FeedForward(256, inner_dim = 512, activation_fn = "swiglu", bias = False)
    elif kind == "zimage":
        zmod = pytest.importorskip("diffusers.models.transformers.transformer_z_image")
        ff = zmod.FeedForward(256, 512)
    elif kind == "flux2":
        f2 = pytest.importorskip("diffusers.models.transformers.transformer_flux2")
        ff = f2.Flux2FeedForward(256, inner_dim = 512)
    else:
        q21 = pytest.importorskip("diffusers.models.transformers.transformer_qwenimage21")
        ff = q21.QwenImage21SwiGLUFeedForward(256, 512)
    ff = _quantize(ff.cuda().to(torch.bfloat16).eval())
    x = torch.randn(2, 150, 256, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        ref = ff(x)
        torch._dynamo.reset()
        stock_compiled = torch.compile(ff, fullgraph = True)(x)
        assert fused.install(ff) == 1
        out = ff(x)
        torch._dynamo.reset()
        compiled = torch.compile(ff, fullgraph = True)(x)
    assert torch.equal(out, ref)
    _assert_within_compile_floor(compiled, stock_compiled, ref)


def _assert_within_compile_floor(compiled, stock_compiled, eager):
    # Inductor's own act-quant codegen is not eager-exact, so the bar is the stock compile's distance from eager.
    floor = (stock_compiled.float() - eager.float()).abs()
    ours = (compiled.float() - eager.float()).abs()
    assert ours.max().item() <= 1.5 * floor.max().item() + 1e-6
    assert ours.mean().item() <= 1.1 * floor.mean().item() + 1e-6


@needs_cuda
def test_cpu_placed_model_is_swapped_at_the_first_forward():
    ff = _quantized_ff().cpu()
    assert fused.install(ff) == 1
    assert not fused.is_installed(ff)
    ff = ff.cuda()
    x = torch.randn(1, 64, 256, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        ff(x)
    assert fused.is_installed(ff)
    assert (
        "_unsloth_first_call_hooks" in ff.__dict__ and not ff.__dict__["_unsloth_first_call_hooks"]
    )


@needs_cuda
def test_zimage_swiglu_deferred_swap_engages_under_inference_mode():
    zmod = pytest.importorskip("diffusers.models.transformers.transformer_z_image")
    torch.manual_seed(0)
    ff = _quantize(zmod.FeedForward(256, 512).to(torch.bfloat16).eval())
    assert fused.install(ff) == 1
    ff = ff.cuda()
    x = torch.randn(1, 64, 256, device = "cuda", dtype = torch.bfloat16)
    with torch.inference_mode():
        ref = type(ff).forward(ff, x)
        out = ff(x)
    assert fused.is_installed(ff)
    assert torch.equal(out, ref)


def test_offload_skips_install():
    assert fused.install(torch.nn.Linear(8, 8), offload_active = True) == 0


@needs_cuda
@pytest.mark.parametrize("kind", ["gelu", "swiglu"])
def test_convrot_linears_keep_the_stock_forward(kind, monkeypatch):
    # MiniMax-H3's ConvRotLinear rotates the input before the GEMM; the fused _int_mm would skip it.
    from core.inference.diffusion_convrot import _install_rotation

    if kind == "gelu":
        ff = _quantized_ff()
        lins = (ff.net[0].proj, ff.net[2])
    else:
        FeedForward = pytest.importorskip("diffusers.models.attention").FeedForward

        torch.manual_seed(0)
        ff = _quantize(
            FeedForward(256, inner_dim = 512, activation_fn = "swiglu", bias = False)
            .cuda()
            .to(torch.bfloat16)
            .eval()
        )
        lins = (ff.net[0].proj, ff.net[2])
    monkeypatch.setattr(
        fused, "_SWIGLU_ALL_LAYOUTS", True
    )  # the layout gate alone would already refuse diffusers SwiGLU
    for lin in lins:
        _install_rotation(lin, 16)
    assert fused.install(ff) == 0
    assert not fused.is_installed(ff)


def _swiglu_module(kind):
    if kind == "diffusers_swiglu":
        FeedForward = pytest.importorskip("diffusers.models.attention").FeedForward
        return FeedForward(256, inner_dim = 512, activation_fn = "swiglu", bias = False)
    if kind == "zimage":
        return pytest.importorskip("diffusers.models.transformers.transformer_z_image").FeedForward(
            256, 512
        )
    if kind == "flux2":
        return pytest.importorskip(
            "diffusers.models.transformers.transformer_flux2"
        ).Flux2FeedForward(256, inner_dim = 512)
    return pytest.importorskip(
        "diffusers.models.transformers.transformer_qwenimage21"
    ).QwenImage21SwiGLUFeedForward(256, 512)


@pytest.mark.parametrize("kind", ["diffusers_swiglu", "zimage", "flux2", "qwenimage21"])
def test_swiglu_quality_gate_allows_zimage_only(kind):
    # See _SWIGLU_ALL_LAYOUTS.
    ff = _swiglu_module(kind)
    assert fused._swiglu_candidate(ff) is (kind == "zimage")
    assert fused._swiglu_layout_allowed(ff) is (kind == "zimage")


@needs_cuda
@pytest.mark.parametrize("kind", ["zimage", "flux2", "qwenimage21"])
def test_swiglu_install_count_follows_the_quality_gate(kind):
    torch.manual_seed(0)
    ff = _quantize(_swiglu_module(kind).cuda().to(torch.bfloat16).eval())
    assert fused.install(ff) == (1 if kind == "zimage" else 0)
    assert fused.is_installed(ff) is (kind == "zimage")


def test_ineligible_resident_model_never_probes_the_kernels(monkeypatch):
    # A bf16 / fp16 DiT already on the GPU has nothing to fuse: the Triton validation launch must not run.
    probed = []
    monkeypatch.setattr(fused, "resident_cuda_device", lambda m: torch.device("cuda", 0))
    monkeypatch.setattr(fused, "_device_ok", lambda index: probed.append(index) or True)
    mod = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.GELU(), torch.nn.Linear(8, 8))
    assert fused.install(mod) == 0
    assert probed == []
    assert "forward" not in mod[0].__dict__


def test_uninstall_cancels_the_deferred_install(monkeypatch):
    mod = torch.nn.Sequential(torch.nn.Linear(8, 8))
    target = mod[0]
    monkeypatch.setattr(fused, "_ff_eligible", lambda m: m is target)
    finalized = []
    monkeypatch.setattr(fused, "_finalize", lambda t, logger = None: finalized.append(t) or 1)
    assert fused.install(mod) == 1
    assert "int8_fused" in mod.__dict__["_unsloth_first_call_hooks"]
    fused.uninstall(mod)
    mod(torch.randn(2, 8))
    assert finalized == []
    assert "int8_fused" not in mod.__dict__.get("_unsloth_first_call_hooks", {})
    assert not mod._forward_pre_hooks


def _tiny_flux():
    tf = pytest.importorskip("diffusers.models.transformers.transformer_flux")
    torch.manual_seed(0)
    return tf.FluxTransformer2DModel(
        patch_size = 1,
        in_channels = 4,
        num_layers = 1,
        num_single_layers = 3,
        attention_head_dim = 16,
        num_attention_heads = 2,
        joint_attention_dim = 32,
        pooled_projection_dim = 32,
        axes_dims_rope = (4, 6, 6),
    ).eval()


def _flux_inputs():
    g = torch.Generator().manual_seed(1)
    return dict(
        hidden_states = torch.randn(1, 16, 4, generator = g),
        encoder_hidden_states = torch.randn(1, 8, 32, generator = g),
        pooled_projections = torch.randn(1, 32, generator = g),
        timestep = torch.tensor([1.0]),
        img_ids = torch.zeros(16, 3),
        txt_ids = torch.zeros(8, 3),
        return_dict = False,
    )


def _fake_cuda_install(monkeypatch, model):
    """Real _finalize swap on CPU (fused forwards fall back off CUDA); returns the list the swapped forward appends to."""
    calls = []
    real = fused._flux_single_forward

    def spy(self, *args, **kwargs):
        calls.append(self)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(fused, "_flux_single_forward", spy)
    monkeypatch.setattr(fused, "resident_cuda_device", lambda m: torch.device("cuda", 0))
    monkeypatch.setattr(fused, "_device_ok", lambda index: True)
    monkeypatch.setattr(fused, "_op", lambda: None)
    monkeypatch.setattr(fused, "_ff_eligible", lambda m: False)
    monkeypatch.setattr(fused, "_prepare_swiglu", lambda m: False)
    monkeypatch.setattr(fused, "_has_eligible", lambda t: True)
    monkeypatch.setattr(
        fused, "_flux_single_eligible", lambda m: type(m).__name__ == "FluxSingleTransformerBlock"
    )
    assert fused.install(model) == len(model.single_transformer_blocks)
    return calls


def _fbcache(model):
    hooks = pytest.importorskip("diffusers.hooks")
    model.enable_cache(hooks.FirstBlockCacheConfig(threshold = 1e9))


def _two_steps(model):
    outs = []
    with torch.no_grad(), model.cache_context("cond"):
        for _ in range(2):
            outs.append(model(**_flux_inputs())[0])
    return outs


def test_fused_flux_single_keeps_fbcache_hooks_installed_before(monkeypatch):
    # Swap must go under the FBCache hooks, else the tail hook never records residuals and reuse reads None.
    import copy

    model = _tiny_flux()
    ref_model = copy.deepcopy(model)
    _fbcache(ref_model)
    ref = _two_steps(ref_model)

    _fbcache(model)
    wrappers = [b.__dict__.get("forward") for b in model.single_transformer_blocks]
    calls = _fake_cuda_install(monkeypatch, model)
    out = _two_steps(model)
    assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])
    assert [b.__dict__.get("forward") for b in model.single_transformer_blocks] == wrappers
    assert len(calls) == len(model.single_transformer_blocks)

    # Turning the cache off splices the hook's inner forward back: the fused forward must survive it.
    model.disable_cache()
    for b in model.single_transformer_blocks:
        assert b.forward.__func__ is fused._flux_single_forward
    fused.uninstall(model)
    for b in model.single_transformer_blocks:
        assert not fused.is_installed(b)
        assert b.forward.__func__ is type(b).forward


def test_fused_flux_single_then_fbcache_and_uninstall_keeps_hooks(monkeypatch):
    import copy

    model = _tiny_flux()
    ref_model = copy.deepcopy(model)
    _fbcache(ref_model)
    ref = _two_steps(ref_model)

    calls = _fake_cuda_install(monkeypatch, model)
    _fbcache(model)
    out = _two_steps(model)
    assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])
    assert len(calls) == len(model.single_transformer_blocks)
    wrappers = [b.__dict__.get("forward") for b in model.single_transformer_blocks]
    fused.uninstall(model)
    assert [b.__dict__.get("forward") for b in model.single_transformer_blocks] == wrappers
    calls.clear()
    out = _two_steps(model)
    # Both caches now hold the first run's residuals; a fresh reference would round step 0 differently on some CPUs.
    ref = _two_steps(ref_model)
    assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1]) and not calls


def test_fused_flux_single_rearms_a_compiled_cache_inner(monkeypatch):
    # Deferred swap must re-arm the hooks' compiled inner on the fused forward, not bypass it.
    import copy

    from core.inference import diffusion_cache

    model = _tiny_flux()
    ref_model = copy.deepcopy(model)
    _fbcache(ref_model)
    ref = _two_steps(ref_model)

    compiled = []

    def fake_compile(fn, **kwargs):
        def wrapper(*args, **kw):
            return fn(*args, **kw)

        wrapper.inner = fn
        compiled.append(fn)
        return wrapper

    monkeypatch.setattr(torch, "compile", fake_compile)
    _fbcache(model)
    for b in model.single_transformer_blocks:
        b._compiled_call_impl = b._call_impl  # stands in for compile_repeated_blocks
    armed = diffusion_cache._compile_hooked_block_inners(model)
    assert armed == len(model.single_transformer_blocks)
    calls = _fake_cuda_install(monkeypatch, model)
    for b in model.single_transformer_blocks:
        inner = b._diffusers_hook.hooks["fbc_block_hook"].fn_ref.original_forward
        assert inner.inner.__func__ is fused._flux_single_forward
    out = _two_steps(model)
    assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])
    assert len(calls) == len(model.single_transformer_blocks)

    fused.uninstall(model)
    for b in model.single_transformer_blocks:
        inner = b._diffusers_hook.hooks["fbc_block_hook"].fn_ref.original_forward
        assert inner.__func__ is type(b).forward


@needs_cuda
def test_int8_flux_under_fbcache_renders_through_the_fused_kernel():
    import copy

    tf = pytest.importorskip("diffusers.models.transformers.transformer_flux")
    hooks = pytest.importorskip("diffusers.hooks")
    from torchao.quantization import quantize_

    torch.manual_seed(0)
    model = (
        tf.FluxTransformer2DModel(
            patch_size = 1,
            in_channels = 16,
            num_layers = 1,
            num_single_layers = 2,
            attention_head_dim = 64,
            num_attention_heads = 4,
            joint_attention_dim = 64,
            pooled_projection_dim = 64,
            axes_dims_rope = (16, 24, 24),
        )
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )
    for p in model.parameters():
        p.data.normal_(0, 0.05)
    quantize_(
        model,
        _int8_config(),
        filter_fn = lambda m, fqn: isinstance(m, torch.nn.Linear)
        and "single_transformer_blocks" in fqn
        and "norm" not in fqn,
    )
    _require_int8tensor(model)
    ref_model = copy.deepcopy(model)
    g = torch.Generator(device = "cuda").manual_seed(1)
    kwargs = dict(
        hidden_states = torch.randn(1, 256, 16, device = "cuda", dtype = torch.bfloat16, generator = g),
        encoder_hidden_states = torch.randn(
            1, 64, 64, device = "cuda", dtype = torch.bfloat16, generator = g
        ),
        pooled_projections = torch.randn(1, 64, device = "cuda", dtype = torch.bfloat16, generator = g),
        timestep = torch.tensor([1.0], device = "cuda"),
        img_ids = torch.zeros(256, 3, device = "cuda"),
        txt_ids = torch.zeros(64, 3, device = "cuda"),
        return_dict = False,
    )

    def run(m):
        outs = []
        with torch.no_grad(), m.cache_context("cond"):
            for _ in range(2):
                outs.append(m(**kwargs)[0])
        return outs

    ref_model.enable_cache(hooks.FirstBlockCacheConfig(threshold = 1e9))
    ref = run(ref_model)
    model.enable_cache(hooks.FirstBlockCacheConfig(threshold = 1e9))
    assert fused.install(model) == 2
    out = run(model)
    assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])
    assert all(fused.is_installed(b) for b in model.single_transformer_blocks)
