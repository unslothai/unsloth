# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for ``diffusion_int8_gemm.py``: the per-arch gate and kill switch run anywhere; the kernel / Linear tests
need NVIDIA CUDA, Triton and torchao (they force the lever on, so an arch outside the shipped table still runs)."""

from __future__ import annotations

import pytest

from core.inference import diffusion_int8_gemm as g8

torch = pytest.importorskip("torch")


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(g8.INT8_GEMM_ENV, raising = False)
    monkeypatch.delenv(g8.INT8_ROTQUANT_ENV, raising = False)
    g8._DEVICE_CFG.clear()
    g8._ROTQ_DEVICE.clear()
    yield
    g8._DEVICE_CFG.clear()
    g8._ROTQ_DEVICE.clear()


@pytest.mark.parametrize(
    "cap, on",
    [
        ((7, 5), False),  # T4: Triton cannot lower the int8 dot
        ((8, 0), True),  # A100
        ((8, 6), False),  # unmeasured
        ((8, 9), True),  # L4
        ((9, 0), False),  # H100: unmeasured
        ((10, 0), False),  # B200: Triton int8 ~2.5x slower than cuBLAS
        ((12, 0), True),  # RTX PRO 6000
    ],
)
def test_arch_gate_auto(cap, on):
    assert (g8.arch_config(cap, "auto") is not None) is on


def test_kill_switch_turns_every_arch_off(monkeypatch):
    monkeypatch.setenv(g8.INT8_GEMM_ENV, "0")
    assert g8.int8_gemm_mode() == "off"
    for cap in ((8, 0), (8, 9), (12, 0)):
        assert g8.arch_config(cap) is None


def test_force_enables_unmeasured_sm80_plus_only(monkeypatch):
    monkeypatch.setenv(g8.INT8_GEMM_ENV, "1")
    assert g8.arch_config((9, 0)) == g8._FALLBACK_CONFIG
    assert g8.arch_config((10, 0)) == g8._FALLBACK_CONFIG
    assert g8.arch_config((7, 5)) is None


def test_install_is_a_noop_when_killed(monkeypatch):
    monkeypatch.setenv(g8.INT8_GEMM_ENV, "0")
    lin = torch.nn.Linear(64, 64)
    assert g8.install(lin) == 0 and not g8.is_installed(lin)


def test_offload_keeps_stock():
    assert g8.install(torch.nn.Linear(64, 64), offload_active = True) == 0


def test_rocm_never_probes(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "6.4", raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert g8.device_config(0) is None


def test_deferred_finalize_records_the_final_count():
    """A deferred install records the candidate count; a first forward that swaps nothing must clear it (status)."""
    holder = torch.nn.Sequential(torch.nn.Linear(64, 64))
    holder._unsloth_int8_gemm = 3
    assert g8._finalize(holder) == 0
    assert holder._unsloth_int8_gemm == 0


def test_dense_linear_is_not_eligible():
    assert g8._eligible(torch.nn.Linear(128, 128)) is None


def _cuda_ready() -> bool:
    if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
        return False
    if torch.cuda.get_device_capability() < (8, 0):
        return False
    try:
        import torchao  # noqa: F401
        import triton  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


needs_cuda = pytest.mark.skipif(
    not _cuda_ready(), reason = "needs NVIDIA sm80+ CUDA, Triton and torchao"
)


@pytest.fixture
def forced(monkeypatch):
    monkeypatch.setenv(g8.INT8_GEMM_ENV, "1")
    cfg = g8.device_config(torch.cuda.current_device())
    if cfg is None:
        pytest.skip("int8 GEMM probe refused this device")
    return cfg


def _int8_linear(k, n, bias, version):
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

    torch.manual_seed(0)
    lin = torch.nn.Linear(k, n, bias = bias).cuda().to(torch.bfloat16)
    cfg = Int8DynamicActivationInt8WeightConfig(set_inductor_config = False)
    if version is not None:
        if not hasattr(cfg, "version"):
            pytest.skip("torchao without config versions")
        cfg.version = version
    quantize_(lin, cfg)
    return lin


@needs_cuda
@pytest.mark.parametrize(
    "m, k, n, bias, xs32",
    [
        (4096, 4096, 4096, False, False),
        (1037, 520, 1400, True, False),
        (17, 256, 1024, True, True),
        (300, 12288, 256, False, False),
    ],
)
@pytest.mark.parametrize("ws32", [False, True])
def test_op_is_bit_exact_vs_torchao_epilogue(forced, m, k, n, bias, xs32, ws32):
    g = torch.Generator().manual_seed(m + k + n)
    a = torch.randint(-127, 128, (m, k), generator = g, dtype = torch.int8).cuda()
    w = torch.randint(-127, 128, (n, k), generator = g, dtype = torch.int8).cuda()
    xs = (torch.rand(m, generator = g) * 0.02 + 1e-4).to(torch.bfloat16)
    xs = (xs.float() if xs32 else xs).cuda()
    ws = (torch.rand(n, generator = g) * 0.002 + 1e-5).to(torch.bfloat16).cuda()
    ws = ws.float() if ws32 else ws
    b = (torch.randn(n, generator = g) * 0.1).to(torch.bfloat16).cuda() if bias else None
    out = g8._op()(a, w, xs, ws, b)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, g8.reference(a, w, xs, ws, b))


@needs_cuda
@pytest.mark.parametrize("version", [None, 2])
@pytest.mark.parametrize("bias", [False, True])
def test_linear_swap_is_bit_identical_eager_and_compiled(forced, version, bias):
    from torch._dynamo.utils import counters

    stock = _int8_linear(1024, 768, bias, version)
    fused = _int8_linear(1024, 768, bias, version)
    holder = torch.nn.Sequential(fused)
    assert g8.install(holder) == 1 and g8.is_installed(fused)
    x = torch.randn(2, 300, 1024, device = "cuda", dtype = torch.bfloat16)
    with torch.inference_mode():
        before = g8.call_count()
        assert torch.equal(fused(x), stock(x))
        assert g8.call_count() == before + 1
        counters.clear()
        torch._dynamo.reset()
        # Studio compiles with emulate_precision_casts (diffusion_speed); without it Inductor's stock epilogue keeps an
        # fp32 chain eager never had, and the fused kernel matches eager instead.
        with torch._inductor.config.patch(emulate_precision_casts = True):
            out = torch.compile(fused, fullgraph = True)(x)
            assert not counters["graph_break"]
            assert torch.equal(out, torch.compile(stock, fullgraph = True)(x))
    g8.uninstall(holder)
    assert not g8.is_installed(fused) and "forward" not in fused.__dict__


@needs_cuda
@pytest.mark.parametrize("bias", [False, True])
def test_prequant_style_fp32_weight_scale(forced, bias):
    """Studio's int8 prequant checkpoints rebuild an Int8Tensor with fp32 weight scales (bias added before rounding)."""
    from torch._dynamo.utils import counters

    stock = _int8_linear(1024, 768, bias, 2)
    fused = _int8_linear(1024, 768, bias, 2)
    for lin in (stock, fused):
        lin.weight.scale = lin.weight.scale.float()
    holder = torch.nn.Sequential(fused)
    assert g8.install(holder) == 1
    x = torch.randn(300, 1024, device = "cuda", dtype = torch.bfloat16) * 3
    with torch.inference_mode():
        assert torch.equal(fused(x), stock(x))
        counters.clear()
        torch._dynamo.reset()
        with torch._inductor.config.patch(emulate_precision_casts = True):
            out = torch.compile(fused, fullgraph = True)(x)
            assert not counters["graph_break"]
            assert torch.equal(out, torch.compile(stock, fullgraph = True)(x))


@needs_cuda
def test_small_m_and_misaligned_keep_stock(forced):
    lin = _int8_linear(1000, 768, False, None)  # K off the 64 grid
    assert g8._eligible(lin) is None
    ok = _int8_linear(1024, 768, False, None)
    holder = torch.nn.Sequential(ok)
    assert g8.install(holder) == 1
    before = g8.call_count()
    with torch.inference_mode():
        ok(
            torch.randn(16, 1024, device = "cuda", dtype = torch.bfloat16)
        )  # M = 16 < _int_mm's floor: stock
    assert g8.call_count() == before


@needs_cuda
def test_fused_mlp_down_projection_traces_without_breaks(forced, monkeypatch):
    """The fused MLP's down projection reaches the GEMM through ``linear_from_q`` inside the compiled block: no host
    sync, no graph break, and the same bits as the fused MLP's own stock epilogue (kill switch), eager and compiled."""
    from torch._dynamo.utils import counters

    from core.inference import diffusion_int8_fused as f8

    attention = pytest.importorskip("diffusers.models.attention")
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

    torch.manual_seed(0)
    ff = attention.FeedForward(1024, dim_out = 1024, mult = 4, activation_fn = "gelu-approximate")
    ff = ff.cuda().to(torch.bfloat16)
    quantize_(ff, Int8DynamicActivationInt8WeightConfig(version = 2, set_inductor_config = False))
    fused = torch.nn.Sequential(ff)
    if not f8.install(fused):
        pytest.skip("int8 fused MLP probe refused this device")
    g8.install(fused)  # as diffusion_speed does: sets the op handle linear_from_q calls
    x = torch.randn(2, 300, 1024, device = "cuda", dtype = torch.bfloat16)

    def run(mode):
        monkeypatch.setenv(g8.INT8_GEMM_ENV, mode)
        before = g8.call_count()
        eager = fused(x)
        counters.clear()
        torch._dynamo.reset()
        with torch._inductor.config.patch(emulate_precision_casts = True):
            compiled = torch.compile(fused, fullgraph = True)(x)
        assert not counters["graph_break"]
        return eager, compiled, g8.call_count() - before

    with torch.inference_mode():
        eager, compiled, calls = run("1")
        ref_eager, ref_compiled, ref_calls = run("0")
    assert calls == 2 and ref_calls == 0
    assert torch.equal(eager, ref_eager)
    assert torch.equal(compiled, ref_compiled)
    g8.uninstall(fused)
    f8.uninstall(fused)


def test_status_drops_int8_gemm_once_the_deferred_probe_swapped_nothing():
    from types import SimpleNamespace

    from core.inference.diffusion_speed import int8_gemm_live

    dit = SimpleNamespace(_unsloth_int8_gemm = 224)
    pipe = SimpleNamespace(transformer = dit)
    assert int8_gemm_live(pipe, ("compiled", "int8_gemm")) == ["compiled", "int8_gemm"]
    dit._unsloth_int8_gemm = 0
    assert int8_gemm_live(pipe, ("compiled", "int8_gemm")) == ["compiled"]
    assert int8_gemm_live(pipe, ("compiled",)) == ["compiled"]


def _convrot(lin, group = 256):
    from core.inference.diffusion_convrot import _install_rotation
    _install_rotation(lin, group)
    return lin


def test_convrot_linear_is_eligible_with_its_group(monkeypatch):
    monkeypatch.delenv(g8.INT8_GEMM_CONVROT_ENV, raising = False)
    scale = torch.ones(256, dtype = torch.bfloat16)
    monkeypatch.setattr(g8, "_v1_parts", lambda w: (torch.zeros(256, 512, dtype = torch.int8), scale))
    lin = _convrot(torch.nn.Linear(512, 256, bias = False))
    rec = g8._eligible(lin)
    assert rec is not None and rec[:2] == ("v1", 256) and rec[3] is lin.weight
    monkeypatch.setenv(g8.INT8_GEMM_CONVROT_ENV, "0")
    assert g8._eligible(lin) is None
    assert g8._eligible(torch.nn.Linear(512, 256, bias = False))[:2] == ("v1", None)


def test_other_linear_subclasses_and_bad_groups_stay_stock(monkeypatch):
    monkeypatch.delenv(g8.INT8_GEMM_CONVROT_ENV, raising = False)
    monkeypatch.setattr(
        g8,
        "_v1_parts",
        lambda w: (torch.zeros(256, 512, dtype = torch.int8), torch.ones(256, dtype = torch.bfloat16)),
    )

    class Other(torch.nn.Linear):
        pass

    assert g8._eligible(Other(512, 256, bias = False)) is None
    lin = _convrot(torch.nn.Linear(512, 256, bias = False))
    lin.convrot_groupsize = 1024  # does not divide in_features: the forward could not rotate it
    assert g8._eligible(lin) is None


@needs_cuda
@pytest.mark.parametrize("version", [None, 2])
def test_convrot_linear_swap_is_bit_identical_eager_and_compiled(forced, monkeypatch, version):
    """MiniMax-H3's rotated int8 Linear (and its fused QKV): rotation, act quant, fused GEMM == ConvRotLinear.forward."""
    from torch._dynamo.utils import counters

    monkeypatch.delenv(g8.INT8_GEMM_CONVROT_ENV, raising = False)
    stock = _convrot(_int8_linear(1024, 768, False, version))
    fused = _convrot(_int8_linear(1024, 768, False, version))
    holder = torch.nn.Sequential(fused)
    assert g8.install(holder) == 1 and g8.is_installed(fused)
    x = torch.randn(2, 300, 1024, device = "cuda", dtype = torch.bfloat16) * 3
    x[0, 7, :5] *= 200  # an outlier row, what the rotation exists for
    with torch.inference_mode():
        before = g8.call_count()
        assert torch.equal(fused(x), stock(x))
        assert g8.call_count() == before + 1
        counters.clear()
        torch._dynamo.reset()
        with torch._inductor.config.patch(emulate_precision_casts = True):
            out = torch.compile(fused, fullgraph = True)(x)
            assert not counters["graph_break"]
            if fused.__dict__[g8._REC][2]:
                # the opaque fused op is eager-exact; Inductor's compiled act quant is not on every torch
                assert torch.equal(out, stock(x))
            else:
                assert torch.equal(out, torch.compile(stock, fullgraph = True)(x))
    g8.uninstall(holder)
    assert not g8.is_installed(fused)


@needs_cuda
def test_convrot_kill_switch_keeps_rotated_linears_stock(forced, monkeypatch):
    monkeypatch.setenv(g8.INT8_GEMM_CONVROT_ENV, "0")
    rotated = _convrot(_int8_linear(1024, 768, False, None))
    plain = _int8_linear(1024, 768, False, None)
    holder = torch.nn.Sequential(rotated, plain)
    assert g8.install(holder) == 1
    assert g8.is_installed(plain) and not g8.is_installed(rotated)


@needs_cuda
@pytest.mark.parametrize("use_stream, version", [(False, None), (False, 2), (True, 2)])
def test_block_streamed_denoiser_installs_against_its_onload_device(
    forced, monkeypatch, use_stream, version
):
    """Group-offloaded int8 blocks (H3 on 40 GB): installed while on the host, the fused GEMM follows each onload."""
    hooks = pytest.importorskip("diffusers.hooks")
    monkeypatch.delenv(g8.INT8_GEMM_CONVROT_ENV, raising = False)
    monkeypatch.delenv(g8.INT8_GEMM_STREAMED_ENV, raising = False)

    class Holder(torch.nn.Module):
        def __init__(self, lin):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Sequential(lin)])

        def forward(self, x):
            return self.blocks[0](x)

    stock = _convrot(_int8_linear(1024, 768, False, version))
    fused = _convrot(_int8_linear(1024, 768, False, version))
    holder = (
        Holder(fused).requires_grad_(False).cpu()
    )  # Studio builds the streamed denoiser on the host
    hooks.apply_group_offloading(
        holder,
        onload_device = torch.device("cuda"),
        offload_device = torch.device("cpu"),
        offload_type = "block_level",
        num_blocks_per_group = 1,
        use_stream = use_stream,
    )
    assert fused.weight.device.type == "cpu"
    assert (
        g8.install(holder, offload_active = True) == 0
    )  # no onload device: an offloaded denoiser stays stock
    assert g8.install(holder, device = "cuda") == 1 and g8.is_installed(fused)
    x = torch.randn(300, 1024, device = "cuda", dtype = torch.bfloat16) * 3
    with torch.no_grad():  # group offload's swap_tensors onload cannot run on inference tensors
        before = g8.call_count()
        out = holder(x)
        assert g8.call_count() == before + 1
        assert torch.equal(out, stock(x))
        assert torch.equal(holder(x), out)  # second onload of the same block
    if not use_stream:  # the stream path keeps the last group resident until the next onload
        assert fused.weight.device.type == "cpu"


@needs_cuda
def test_streamed_kill_switch(forced, monkeypatch):
    monkeypatch.setenv(g8.INT8_GEMM_STREAMED_ENV, "0")
    holder = torch.nn.Sequential(_int8_linear(1024, 768, False, None))
    assert g8.install(holder, device = "cuda") == 0 and not g8.is_installed(holder[0])


@pytest.mark.parametrize(
    "cap, on",
    [
        ((8, 0), True),  # A100: measured end to end (streamed H3)
        ((8, 6), False),
        ((8, 9), False),  # L4: H3 Diffusers does not load there (host RAM floor), never measured
        ((9, 0), False),
        ((10, 0), False),
        ((12, 0), True),
    ],  # only the arches measured end to end (G4, A100) are on
)
def test_rotquant_arch_gate(cap, on):
    assert (g8.rotquant_config(cap, "auto") is not None) is on


def test_rotquant_kill_switch_and_force(monkeypatch):
    assert (
        g8.rotquant_config((12, 0), "off") is None
    )  # the fused GEMM's own kill switch also turns it off
    assert g8.rotquant_config((10, 0), "force") is not None
    assert g8.rotquant_config((7, 5), "force") is None
    assert g8.rotquant_config((8, 0), "off") is None
    monkeypatch.setenv(g8.INT8_ROTQUANT_ENV, "0")
    assert g8.rotquant_config((12, 0), "auto") is None
    assert g8.rotquant_config((8, 0), "auto") is None
    assert g8.rotquant_config((10, 0), "force") is None


@pytest.fixture
def forced_rotq(forced):
    cfg = g8.rotquant_device_config(torch.cuda.current_device())
    if cfg is None:
        pytest.skip("fused rotation probe refused this device")
    return cfg


def _act(m, k, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, k, generator = g) * (torch.rand(1, k, generator = g) * 4)
    x[:, :5] *= 80  # outlier channels, what the rotation is for
    x[min(1, m - 1)] = 0  # a zero row: the eps clamp
    return x.to(torch.bfloat16).cuda()


@needs_cuda
@pytest.mark.parametrize(
    "m, k", [(17, 256), (300, 256 * 3), (1037, 5376), (257, 7168), (129, 14336), (19, 256 * 128)]
)
@pytest.mark.parametrize("kind", ["v1", "v2"])
def test_rotquant_kernel_is_bit_exact_vs_rotation_then_torchao_quant(forced_rotq, m, k, kind):
    x = _act(m, k, m + k)
    q, s = g8._rotq_op()(x, 256, kind == "v2")
    rq, rs = g8.rotquant_reference(x, 256, kind)
    # torchao's own activation-scale dtype: bf16, except v2 on torchao >= 0.18 (fp32)
    want = torch.float32 if kind == "v2" and g8._v2_act_scale_fp32() else torch.bfloat16
    assert q.dtype == torch.int8 and s.dtype == want and s.shape == (m,)
    assert torch.equal(q, rq) and torch.equal(s, rs)


@needs_cuda
def test_rotquant_probe_accepts_this_torchao(forced):
    # with 0.17's bf16 roundings on 0.18, ~4% of codes differed and the probe refused every device
    assert g8.rotquant_device_config(torch.cuda.current_device()) is not None


@needs_cuda
def test_rotquant_fake_op_matches_the_real_scale_dtype(forced_rotq):
    from torch._subclasses.fake_tensor import FakeTensorMode
    x = _act(40, 768, 9)
    for v2 in (False, True):
        _, s = g8._rotq_op()(x, 256, v2)
        with FakeTensorMode() as mode:
            _, fs = g8._rotq_op()(mode.from_tensor(x), 256, v2)
        assert fs.dtype == s.dtype


@needs_cuda
def test_rotquant_unsupported_shapes_take_the_stock_math(forced_rotq):
    x = _act(40, 256 * 129, 3)  # more groups than one tile holds
    assert not g8.rotquant_supported(x, 256, forced_rotq)
    assert not g8.rotquant_supported(
        _act(40, 1024, 4), 64, forced_rotq
    )  # group outside the kernel's set
    assert not g8.rotquant_supported(_act(40, 1024, 5).half(), 256, forced_rotq)
    q, s = g8._rotq_op()(x, 256, True)
    rq, rs = g8.rotquant_reference(x, 256, "v2")
    assert torch.equal(q, rq) and torch.equal(s, rs)
    xt = _act(1024, 300, 6).t()  # non-contiguous input: made contiguous, same result
    q, s = g8._rotq_op()(xt, 256, True)
    rq, rs = g8.rotquant_reference(xt.contiguous(), 256, "v2")
    assert torch.equal(q, rq) and torch.equal(s, rs)


@needs_cuda
@pytest.mark.parametrize("version", [None, 2])
def test_convrot_linear_fuses_rotation_into_act_quant_eager_and_compiled(
    forced_rotq, monkeypatch, version
):
    """The rotated Linear quantizes its activation with the fused kernel: same output as ConvRotLinear.forward, eager
    and compiled, one graph, no graph break, no recompile on a second call."""
    from torch._dynamo.utils import counters

    monkeypatch.delenv(g8.INT8_GEMM_CONVROT_ENV, raising = False)
    stock = _convrot(_int8_linear(1024, 768, False, version))
    fused = _convrot(_int8_linear(1024, 768, False, version))
    holder = torch.nn.Sequential(fused)
    assert g8.install(holder) == 1 and fused.__dict__[g8._REC][2] is True
    x = torch.randn(2, 300, 1024, device = "cuda", dtype = torch.bfloat16) * 3
    x[0, 7, :5] *= 200
    x2 = (
        x * 0.5
    )  # made outside inference_mode like x, so a recompile here would be the op's own guards
    with torch.inference_mode():
        before = g8.rotquant_call_count()
        assert torch.equal(fused(x), stock(x))
        assert g8.rotquant_call_count() == before + 1
        counters.clear()
        torch._dynamo.reset()
        with torch._inductor.config.patch(emulate_precision_casts = True):
            compiled = torch.compile(fused, fullgraph = True)
            out = compiled(x)
            out2 = compiled(x2)
            assert not counters["graph_break"]
            assert counters["stats"]["unique_graphs"] == 1
            assert torch.equal(out, stock(x)) and torch.equal(out2, stock(x2))
    g8.uninstall(holder)


@needs_cuda
def test_rotquant_kill_switch_keeps_the_stock_rotation(forced, monkeypatch):
    monkeypatch.delenv(g8.INT8_GEMM_CONVROT_ENV, raising = False)
    monkeypatch.setenv(g8.INT8_ROTQUANT_ENV, "0")
    stock = _convrot(_int8_linear(1024, 768, False, 2))
    fused = _convrot(_int8_linear(1024, 768, False, 2))
    holder = torch.nn.Sequential(fused)
    assert g8.install(holder) == 1 and fused.__dict__[g8._REC][2] is False  # fused GEMM still on
    x = torch.randn(300, 1024, device = "cuda", dtype = torch.bfloat16)
    with torch.inference_mode():
        before, gemm = g8.rotquant_call_count(), g8.call_count()
        assert torch.equal(fused(x), stock(x))
        assert g8.rotquant_call_count() == before and g8.call_count() == gemm + 1
    g8.uninstall(holder)


@needs_cuda
def test_rotquant_probe_refusal_keeps_the_stock_rotation(forced, monkeypatch):
    monkeypatch.delenv(g8.INT8_GEMM_CONVROT_ENV, raising = False)
    monkeypatch.setattr(
        g8, "_rotq_probe", lambda index, cfg: False
    )  # e.g. an accumulation order that differs
    fused = _convrot(_int8_linear(1024, 768, False, 2))
    assert g8.install(torch.nn.Sequential(fused)) == 1 and fused.__dict__[g8._REC][2] is False
