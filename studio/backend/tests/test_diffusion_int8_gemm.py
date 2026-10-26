# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for ``diffusion_int8_gemm.py``: the per-arch gate and kill switch run anywhere; the kernel / Linear tests
need NVIDIA CUDA, Triton and torchao (they force the lever on, so an arch outside the shipped table still runs)."""

from __future__ import annotations

import types

import pytest

from core.inference import diffusion_convrot_quant as cq
from core.inference import diffusion_int8_gemm as g8

torch = pytest.importorskip("torch")


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(g8.INT8_GEMM_ENV, raising = False)
    monkeypatch.delenv(cq.INT8_ROTQUANT_ENV, raising = False)
    monkeypatch.delenv(g8.INT8_GEMM_TILES_ENV, raising = False)
    monkeypatch.delenv(cq.INT8_ROTQUANT_H16_ENV, raising = False)
    monkeypatch.delenv(cq.INT8_ROTQUANT_SWIGLU_ENV, raising = False)
    g8._DEVICE_CFG.clear()
    g8._DEVICE_TILES.clear()
    cq._ROTQ_DEVICE.clear()
    cq._ROTQ_DEVICE_K.clear()
    cq._H16_DEVICE.clear()
    cq._SWIGLU_DEVICE.clear()
    yield
    g8._DEVICE_CFG.clear()
    g8._DEVICE_TILES.clear()
    cq._ROTQ_DEVICE.clear()
    cq._ROTQ_DEVICE_K.clear()
    cq._H16_DEVICE.clear()
    cq._SWIGLU_DEVICE.clear()


@pytest.mark.parametrize(
    "cap, on",
    [
        ((7, 5), False),  # T4: Triton cannot lower the int8 dot
        ((8, 0), True),
        ((8, 6), False),
        ((8, 9), True),
        ((9, 0), False),
        ((10, 0), False),  # B200: Triton int8 ~2.5x slower than cuBLAS
        ((12, 0), True),
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


def test_probe_shapes_stay_on_the_16_byte_grid():
    ns = [n for _m, n, _k, _b, _x in g8._PROBE_SHAPES]
    ks = [k for _m, _n, k, _b, _x in g8._PROBE_SHAPES]
    assert all(n % 16 == 0 for n in ns) and all(k % 16 == 0 for k in ks)
    for bk in (64, 128):
        assert any(k % bk for k in ks)
    assert any(n % 128 for n in ns)


def test_misaligned_operands_take_the_stock_epilogue(monkeypatch):
    launched, stock = [], []
    monkeypatch.setattr(g8, "_launch", lambda a, w, *rest: launched.append(a.shape) or "fused")
    monkeypatch.setattr(g8, "reference", lambda a, w, *rest: stock.append(a.shape) or "stock")
    g8._DEVICE_CFG[None] = g8._FALLBACK_CONFIG  # CPU tensors report device index None
    xs, ws = torch.ones(32), torch.ones(64)

    def run(a, w):
        return g8._run(a, w, xs, ws, None)

    assert (
        run(torch.zeros(32, 1024, dtype = torch.int8), torch.zeros(64, 1024, dtype = torch.int8))
        == "fused"
    )
    assert (
        run(torch.zeros(32, 1000, dtype = torch.int8), torch.zeros(64, 1000, dtype = torch.int8))
        == "stock"
    )
    flat = torch.zeros(64 * 1024 + 8, dtype = torch.int8)
    assert (
        run(flat[8 : 8 + 32 * 1024].view(32, 1024), torch.zeros(64, 1024, dtype = torch.int8))
        == "stock"
    )
    assert run(torch.zeros(32, 1024, dtype = torch.int8), flat[8:].view(64, 1024)) == "stock"
    wide = torch.zeros(64, 1032, dtype = torch.int8)[:, :1024]  # row stride 1032: off 16
    assert run(torch.zeros(32, 1024, dtype = torch.int8), wide) == "stock"
    assert (
        run(torch.zeros(32, 1024, dtype = torch.int8), torch.zeros(72, 1024, dtype = torch.int8))
        == "stock"
    )
    assert len(launched) == 1 and len(stock) == 5


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
        (1037, 528, 1400, True, False),
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


def _tie_operands(k = 4096, rows = 32):
    """int8 a, w whose int32 products sit one or two units off a bf16 midpoint in [2^24, 2^26), both signs."""
    k1 = k - 128
    targets = []
    for e in (24, 25):
        for mult in (0, 3, 50, 100):
            mid = (1 << e) + mult * (1 << (e - 7)) + (1 << (e - 8))
            for d in (-1, 1, -2, 2):
                targets += [mid + d, -(mid + d)]
    w = torch.zeros(len(targets), k, dtype = torch.int8)
    for j, t in enumerate(targets):
        s1, s2 = divmod(abs(t), 127)
        full, rem = divmod(s1, 127)
        sign = 1 if t >= 0 else -1
        w[j, :full] = 127 * sign
        w[j, full] = rem * sign
        w[j, k1] = s2 * sign
    a = torch.ones(rows, k, dtype = torch.int8)
    a[:, :k1] = 127
    return a.cuda(), w.cuda(), torch.tensor(targets, dtype = torch.int32)


@needs_cuda
@pytest.mark.parametrize("cfg", sorted({g8._FALLBACK_CONFIG, *g8._ARCH_CONFIG.values()}))
@pytest.mark.parametrize("ws32", [False, True])
def test_epilogue_rounds_large_accumulators_twice_like_torch(cfg, ws32):
    a, w, targets = _tie_operands()
    assert torch.equal(torch._int_mm(a, w.t())[0].cpu(), targets)
    assert (
        int((_one_rounding_bf16(targets) != targets.float().to(torch.bfloat16).float()).sum()) >= 8
    )
    xs = torch.ones(a.shape[0], device = "cuda", dtype = torch.bfloat16)
    ws = torch.ones(w.shape[0], device = "cuda", dtype = torch.float32 if ws32 else torch.bfloat16)
    try:
        out = g8._launch(a, w, xs, ws, None, cfg)
    except Exception as exc:  # noqa: BLE001 - a tile this part's shared memory cannot hold
        pytest.skip(f"tile {cfg} does not launch here: {exc}")
    ref = g8.reference(a, w, xs, ws, None)
    assert torch.equal(ref[0].float().cpu(), targets.float().to(torch.bfloat16).float())
    assert torch.equal(out, ref)


@needs_cuda
def test_probe_covers_the_double_rounding():
    a, w = g8.tie_operands(torch.device("cuda"))
    c = torch._int_mm(a, w.t())[0].cpu()
    assert int((_one_rounding_bf16(c) != c.float().to(torch.bfloat16).float()).sum()) >= 8
    xs = torch.ones(a.shape[0], device = "cuda", dtype = torch.bfloat16)
    ws = torch.ones(w.shape[0], device = "cuda", dtype = torch.bfloat16)
    out = g8._launch(a, w, xs, ws, None, g8._FALLBACK_CONFIG)
    assert torch.equal(out, g8.reference(a, w, xs, ws, None))


def _one_rounding_bf16(c):
    """int32 -> bf16 with ONE round-to-nearest-even (exact, in int64), as fp32 values."""
    v = c.to(torch.int64)
    a = v.abs()
    e = torch.floor(torch.log2(a.double().clamp(min = 1))).to(torch.int64)
    shift = (e - 7).clamp(min = 1)
    q = torch.bitwise_right_shift(a, shift)
    r = a - torch.bitwise_left_shift(q, shift)
    half = torch.bitwise_left_shift(torch.ones_like(a), shift - 1)
    q = q + ((r > half) | ((r == half) & (q % 2 == 1))).to(torch.int64)
    out = (torch.bitwise_left_shift(q, shift) * v.sign()).double()
    return torch.where(a < 256, v.double(), out).float()


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
        # Studio compiles with emulate_precision_casts; without it Inductor keeps an fp32 chain eager lacks.
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
def test_no_compiled_variant_spills_to_local_memory(forced):
    g = torch.Generator().manual_seed(0)
    for m, k, n in ((4096, 3072, 3072), (300, 1040, 528), (300, 1040, 520), (300, 1000, 384)):
        a = torch.randint(-127, 128, (m, k), generator = g, dtype = torch.int8).cuda()
        w = torch.randint(-127, 128, (n, k), generator = g, dtype = torch.int8).cuda()
        xs = (torch.rand(m, generator = g) * 0.02).to(torch.bfloat16).cuda()
        ws = (torch.rand(n, generator = g) * 0.002).to(torch.bfloat16).cuda()
        assert torch.equal(g8._op()(a, w, xs, ws, None), g8.reference(a, w, xs, ws, None))
    torch.cuda.synchronize()
    caches = getattr(g8._kernels().i8mm_dq, "device_caches", None)
    if caches is None:
        pytest.skip("Triton without per-device kernel caches")
    variants = [ck for cache in caches.values() for ck in cache[0].values()]
    assert variants
    for ck in variants:
        ck._init_handles()
        assert ck.n_spills == 0, ck.n_spills


@needs_cuda
def test_small_m_and_misaligned_keep_stock(forced):
    lin = _int8_linear(1000, 768, False, None)
    assert g8._eligible(lin) is None
    ok = _int8_linear(1024, 768, False, None)
    holder = torch.nn.Sequential(ok)
    assert g8.install(holder) == 1
    before = g8.call_count()
    with torch.inference_mode():
        ok(
            torch.randn(16, 1024, device = "cuda", dtype = torch.bfloat16)
        )  # M = 16 is below _int_mm's floor
    assert g8.call_count() == before


@needs_cuda
def test_pinned_group_offloaded_denoiser_takes_the_gemm_and_survives_release(forced, monkeypatch):
    """Every group pinned: the GEMM engages after placement, bit-identical to torchao through release and restore."""
    pytest.importorskip("diffusers.hooks")
    from diffusers.hooks import apply_group_offloading
    import types as _types

    from core.inference import diffusion_memory as dm
    from core.inference import diffusion_speed as ds

    monkeypatch.delenv("UNSLOTH_DIFFUSION_PARTIAL_RESIDENT", raising = False)

    class Dit(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_in = torch.nn.Linear(256, 512).to(torch.bfloat16)
            self.blocks = torch.nn.ModuleList(
                torch.nn.Sequential(
                    torch.nn.Linear(512, 1024), torch.nn.GELU(), torch.nn.Linear(1024, 512)
                )
                for _ in range(4)
            )

        def forward(self, x):
            x = self.proj_in(x)
            for block in self.blocks:
                x = x + block(x)
            return x

    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

    torch.manual_seed(0)
    dit = Dit().cuda().to(torch.bfloat16)
    cfg = Int8DynamicActivationInt8WeightConfig(set_inductor_config = False)
    if not hasattr(cfg, "version"):
        pytest.skip("torchao without config versions")
    cfg.version = 2  # what streams under group offloading (v1's subclass rejects is_pinned)
    quantize_(dit.blocks, cfg)
    x = torch.randn(300, 256, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        ref = dit(x)
    dit.to("cpu")
    kwargs = dict(
        onload_device = torch.device("cuda"),
        offload_device = torch.device("cpu"),
        offload_type = "block_level",
        num_blocks_per_group = 1,
        use_stream = True,
        record_stream = True,
        non_blocking = True,
    )
    apply_group_offloading(dit, **dm._torchao_group_offload_kwargs(dit, kwargs, [0]))
    pipe = _types.SimpleNamespace(transformer = dit, components = {"transformer": dit})
    assert dm._keep_groups_resident(dit, 1024, "cuda") > 0
    assert dm.denoisers_pinned_resident(pipe)

    monkeypatch.setattr(ds, "_denoiser_dits", lambda p: [dit])
    applied = {"compiled": True, "int8_gemm": False}
    ds.engage_pinned_denoisers(pipe, applied)
    assert applied["int8_gemm"] and dit._unsloth_int8_gemm == 8
    # inference_mode rejects moving torchao weights
    with torch.no_grad():
        before = g8.call_count()
        assert torch.equal(dit(x), ref)
        assert g8.call_count() == before + 8
        restore = dm.release_resident_groups(pipe, 1024)
        assert restore is not None and not dm.denoisers_pinned_resident(pipe)
        for _ in range(2):
            assert torch.equal(dit(x), ref)
        restore()
        assert dm.denoisers_pinned_resident(pipe)
        assert torch.equal(dit(x), ref)
    g8.uninstall(dit)


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
    g8.install(fused)
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
    lin.convrot_groupsize = 1024
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
    x[0, 7, :5] *= 200
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
                # The fused op is eager-exact; Inductor's compiled act quant is not on every torch.
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
    holder = Holder(fused).requires_grad_(False).cpu()
    hooks.apply_group_offloading(
        holder,
        onload_device = torch.device("cuda"),
        offload_device = torch.device("cpu"),
        offload_type = "block_level",
        num_blocks_per_group = 1,
        use_stream = use_stream,
    )
    assert fused.weight.device.type == "cpu"
    assert g8.install(holder, offload_active = True) == 0
    assert g8.install(holder, device = "cuda") == 1 and g8.is_installed(fused)
    x = torch.randn(300, 1024, device = "cuda", dtype = torch.bfloat16) * 3
    with torch.no_grad():  # group offload's swap_tensors onload cannot run on inference tensors
        before = g8.call_count()
        out = holder(x)
        assert g8.call_count() == before + 1
        assert torch.equal(out, stock(x))
        assert torch.equal(holder(x), out)
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
        ((8, 0), True),
        ((8, 6), False),
        ((8, 9), True),
        ((9, 0), False),
        ((10, 0), False),
        ((12, 0), True),
    ],
)
def test_rotquant_arch_gate(cap, on):
    assert (cq.rotquant_config(cap, "auto") is not None) is on


def test_rotquant_kill_switch_and_force(monkeypatch):
    assert cq.rotquant_config((12, 0), "off") is None
    assert cq.rotquant_config((10, 0), "force") is not None
    assert cq.rotquant_config((7, 5), "force") is None
    assert cq.rotquant_config((8, 0), "off") is None
    monkeypatch.setenv(cq.INT8_ROTQUANT_ENV, "0")
    assert cq.rotquant_config((12, 0), "auto") is None
    assert cq.rotquant_config((8, 0), "auto") is None
    assert cq.rotquant_config((10, 0), "force") is None


@pytest.fixture
def forced_rotq(forced):
    cfg = cq.rotquant_device_config(torch.cuda.current_device())
    if cfg is None:
        pytest.skip("fused rotation probe refused this device")
    return cfg


@pytest.fixture
def dense_rotq(forced, monkeypatch):
    """The dense ``rotq_i8`` only (``UNSLOTH_DIFFUSION_INT8_ROTQUANT_H16=0``): codes bit-identical to the stock path."""
    monkeypatch.setenv(cq.INT8_ROTQUANT_H16_ENV, "0")
    cfg = cq.rotquant_device_config(torch.cuda.current_device())
    if cfg is None:
        pytest.skip("fused rotation probe refused this device")
    assert cq._H16_DEVICE[torch.cuda.current_device()] is False
    return cfg


@pytest.fixture
def h16_rotq(forced):
    cfg = cq.rotquant_device_config(torch.cuda.current_device())
    if cfg is None or not cq._H16_DEVICE.get(torch.cuda.current_device()):
        pytest.skip("rotation probes refused this device")
    return cfg


def _act(m, k, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, k, generator = g) * (torch.rand(1, k, generator = g) * 4)
    x[:, :5] *= 80
    x[min(1, m - 1)] = 0
    return x.to(torch.bfloat16).cuda()


@needs_cuda
@pytest.mark.parametrize(
    "m, k", [(17, 256), (300, 256 * 3), (1037, 5376), (257, 7168), (129, 14336), (19, 256 * 128)]
)
@pytest.mark.parametrize("kind", ["v1", "v2"])
def test_rotquant_kernel_is_bit_exact_vs_rotation_then_torchao_quant(dense_rotq, m, k, kind):
    x = _act(m, k, m + k)
    q, s = cq._rotq_op()(x, 256, kind == "v2")
    rq, rs = cq.rotquant_reference(x, 256, kind)
    # torchao act-scale dtype: bf16, except v2 on torchao >= 0.18 (fp32).
    want = torch.float32 if kind == "v2" and cq._v2_act_scale_fp32() else torch.bfloat16
    assert q.dtype == torch.int8 and s.dtype == want and s.shape == (m,)
    assert torch.equal(q, rq) and torch.equal(s, rs)


@needs_cuda
def test_rotquant_probe_accepts_this_torchao(forced):
    assert cq.rotquant_device_config(torch.cuda.current_device()) is not None


@needs_cuda
def test_rotquant_fake_op_matches_the_real_scale_dtype(dense_rotq):
    from torch._subclasses.fake_tensor import FakeTensorMode
    x = _act(40, 768, 9)
    for v2 in (False, True):
        _, s = cq._rotq_op()(x, 256, v2)
        with FakeTensorMode() as mode:
            _, fs = cq._rotq_op()(mode.from_tensor(x), 256, v2)
        assert fs.dtype == s.dtype


@needs_cuda
def test_rotquant_unsupported_shapes_take_the_stock_math(dense_rotq):
    x = _act(40, 256 * 129, 3)
    assert not cq.rotquant_supported(x, 256, dense_rotq)
    assert not cq.rotquant_supported(_act(40, 1024, 4), 64, dense_rotq)
    assert not cq.rotquant_supported(_act(40, 1024, 5).half(), 256, dense_rotq)
    q, s = cq._rotq_op()(x, 256, True)
    rq, rs = cq.rotquant_reference(x, 256, "v2")
    assert torch.equal(q, rq) and torch.equal(s, rs)
    xt = _act(1024, 300, 6).t()
    q, s = cq._rotq_op()(xt, 256, True)
    rq, rs = cq.rotquant_reference(xt.contiguous(), 256, "v2")
    assert torch.equal(q, rq) and torch.equal(s, rs)


@needs_cuda
@pytest.mark.parametrize("version", [None, 2])
def test_convrot_linear_fuses_rotation_into_act_quant_eager_and_compiled(
    dense_rotq, monkeypatch, version
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
    x2 = x * 0.5  # made outside inference_mode, so a recompile would come from the op's own guards
    with torch.inference_mode():
        before = cq.rotquant_call_count()
        assert torch.equal(fused(x), stock(x))
        assert cq.rotquant_call_count() == before + 1
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
    monkeypatch.setenv(cq.INT8_ROTQUANT_ENV, "0")
    stock = _convrot(_int8_linear(1024, 768, False, 2))
    fused = _convrot(_int8_linear(1024, 768, False, 2))
    holder = torch.nn.Sequential(fused)
    assert g8.install(holder) == 1 and fused.__dict__[g8._REC][2] is False
    x = torch.randn(300, 1024, device = "cuda", dtype = torch.bfloat16)
    with torch.inference_mode():
        before, gemm = cq.rotquant_call_count(), g8.call_count()
        assert torch.equal(fused(x), stock(x))
        assert cq.rotquant_call_count() == before and g8.call_count() == gemm + 1
    g8.uninstall(holder)


@needs_cuda
def test_rotquant_probe_refusal_keeps_the_stock_rotation(forced, monkeypatch):
    monkeypatch.delenv(g8.INT8_GEMM_CONVROT_ENV, raising = False)
    monkeypatch.setattr(cq, "_rotq_probe", lambda index, cfg: False)
    fused = _convrot(_int8_linear(1024, 768, False, 2))
    assert g8.install(torch.nn.Sequential(fused)) == 1 and fused.__dict__[g8._REC][2] is False


def test_tile_for_picks_the_shape_rule_then_the_arch_default(monkeypatch):
    default, wide = (128, 128, 64, 8, 4, 4), (128, 256, 64, 8, 8, 3)
    g8._DEVICE_CFG[0] = default
    g8._DEVICE_TILES[0] = ((16384, 65536, 4096, 8192, wide),)
    assert g8.tile_for(0, 4096, 21504, 5376) == wide
    assert g8.tile_for(0, 4096, 5376, 5376) == default
    assert g8.tile_for(0, 4096, 21504, 14336) == default
    assert g8.tile_for(0, g8._SHAPE_MIN_M - 1, 21504, 5376) == default
    assert g8.tile_for(1, 4096, 21504, 5376) is None


def test_tiles_kill_switch_and_unmeasured_arch(monkeypatch):
    monkeypatch.setitem(g8._SHAPE_TILES, (12, 0), ((1, 2, 3, 4, (64, 64, 64, 8, 4, 3)),))
    assert g8.shape_tiles((12, 0))
    assert g8.shape_tiles((8, 6)) == ()
    monkeypatch.setenv(g8.INT8_GEMM_TILES_ENV, "0")
    assert g8.shape_tiles((12, 0)) == ()


def test_shape_tile_launch_failure_falls_back_to_the_default(monkeypatch):
    default, wide = (128, 128, 64, 8, 4, 4), (128, 256, 64, 8, 8, 3)
    g8._DEVICE_CFG[None] = default
    g8._DEVICE_TILES[None] = ((1, 1 << 20, 1, 1 << 20, wide),)
    used = []

    def launch(a, w, xs, ws, bias, cfg):
        used.append(cfg)
        if cfg == wide:
            raise RuntimeError("out of shared memory")
        return "fused"

    monkeypatch.setattr(g8, "_launch", launch)
    monkeypatch.setattr(g8, "reference", lambda *a: "stock")
    a = torch.zeros(2048, 1024, dtype = torch.int8)
    w = torch.zeros(1024, 1024, dtype = torch.int8)
    xs = torch.ones(2048, dtype = torch.bfloat16)
    ws = torch.ones(1024, dtype = torch.bfloat16)
    assert g8._run(a, w, xs, ws, None) == "fused"
    assert used == [wide, default]
    assert g8._DEVICE_TILES[None] == () and g8._DEVICE_CFG[None] == default
    assert g8._run(a, w, xs, ws, None) == "fused" and used[-1] == default


@needs_cuda
def test_failed_shape_tile_probe_keeps_its_shapes_on_the_default(monkeypatch):
    cap = torch.cuda.get_device_capability()
    bad, good = (64, 64, 64, 8, 4, 3), (128, 64, 64, 8, 4, 3)
    monkeypatch.setenv(g8.INT8_GEMM_ENV, "1")
    monkeypatch.setitem(
        g8._SHAPE_TILES, cap, ((1, 4096, 1, 4096, bad), (4097, 1 << 20, 1, 1 << 20, good))
    )
    real = g8._probe
    monkeypatch.setattr(g8, "_probe", lambda index, cfg: cfg != bad and real(index, cfg))
    cfg = g8.device_config(torch.cuda.current_device())
    if cfg is None:
        pytest.skip("int8 GEMM probe refused this device")
    rules = g8._DEVICE_TILES[torch.cuda.current_device()]
    assert [r[4] for r in rules] == [good]
    assert g8.tile_for(torch.cuda.current_device(), 4096, 1024, 1024) == cfg


def _shipped_tiles():
    return sorted({rule[4] for rules in g8._SHAPE_TILES.values() for rule in rules})


@needs_cuda
@pytest.mark.parametrize("tile", _shipped_tiles() or [None])
@pytest.mark.parametrize("ws32", [False, True])
def test_every_shipped_shape_tile_is_bit_exact(forced, tile, ws32):
    """Every tile in the shape table, at a shape it is picked for and ragged neighbours: int32 accumulation makes the
    output independent of the tile, so each must equal the eager torchao epilogue bit for bit (and the tie operands)."""
    if tile is None:
        pytest.skip("no per-shape tiles shipped")
    g = torch.Generator().manual_seed(7)
    for m, n, k in ((1100, 1024, 1024), (2061, 2064, 1088), (17, 512, 256)):
        a = torch.randint(-127, 128, (m, k), generator = g, dtype = torch.int8).cuda()
        w = torch.randint(-127, 128, (n, k), generator = g, dtype = torch.int8).cuda()
        xs = (torch.rand(m, generator = g) * 0.02 + 1e-4).to(torch.bfloat16).cuda()
        ws = (torch.rand(n, generator = g) * 0.002 + 1e-5).to(torch.bfloat16).cuda()
        ws = ws.float() if ws32 else ws
        try:
            out = g8._launch(a, w, xs, ws, None, tile)
        except Exception as exc:  # noqa: BLE001 - a tile that does not fit this part is dropped by the probe
            if g8._probe(torch.cuda.current_device(), tile):
                raise
            pytest.skip(
                f"tile {tile} does not fit this part ({type(exc).__name__}); probe drops it"
            )
        assert torch.equal(out, g8.reference(a, w, xs, ws, None))
    a, w = g8.tie_operands(torch.device("cuda"))
    ones = torch.ones(a.shape[0], device = "cuda", dtype = torch.bfloat16)
    wones = torch.ones(w.shape[0], device = "cuda", dtype = torch.bfloat16)
    assert torch.equal(
        g8._launch(a, w, ones, wones, None, tile), g8.reference(a, w, ones, wones, None)
    )


@pytest.fixture
def rotq_only(monkeypatch):
    """The fused rotation forced on, the fused GEMM forced off (what sm100 and other unfused archs run)."""
    monkeypatch.setenv(cq.INT8_ROTQUANT_ENV, "1")
    monkeypatch.setenv(
        cq.INT8_ROTQUANT_H16_ENV, "0"
    )  # the bit-identity tests below run the dense kernel
    monkeypatch.setattr(g8, "device_config", lambda index: None)
    monkeypatch.setattr(g8, "arch_config", lambda cap, mode = None: None)
    cfg = cq.rotquant_device_config(torch.cuda.current_device())
    if cfg is None:
        pytest.skip("fused rotation probe refused this device")
    return cfg


def test_rotquant_mode_follows_its_own_switch_and_the_gemm_master(monkeypatch):
    assert cq.rotquant_mode() == "auto"
    monkeypatch.setenv(cq.INT8_ROTQUANT_ENV, "1")
    assert cq.rotquant_mode() == "force" and cq.rotquant_config((9, 0)) == cq._ROTQ_FALLBACK
    monkeypatch.setenv(g8.INT8_GEMM_ENV, "0")
    assert cq.rotquant_mode() == "off" and cq.rotquant_config((12, 0)) is None
    monkeypatch.setenv(g8.INT8_GEMM_ENV, "1")
    monkeypatch.delenv(cq.INT8_ROTQUANT_ENV)
    assert cq.rotquant_mode() == "force"


@needs_cuda
@pytest.mark.parametrize("version", [None, 2])
@pytest.mark.parametrize("bias", [False, True])
def test_rotated_linear_without_the_fused_gemm_is_bit_identical(
    rotq_only, monkeypatch, version, bias
):
    """Fused GEMM off, rotation kernel on: only the ConvRot Linear is swapped; rotq + cuBLAS _int_mm + torchao's
    epilogue equals ConvRotLinear.forward eager, compiled fullgraph keeps one graph, the plain Linear stays stock."""
    from torch._dynamo.utils import counters

    monkeypatch.delenv(g8.INT8_GEMM_CONVROT_ENV, raising = False)
    stock = _convrot(_int8_linear(1024, 768, bias, version))
    fused = _convrot(_int8_linear(1024, 768, bias, version))
    plain = _int8_linear(1024, 768, bias, version)
    holder = torch.nn.Sequential(fused, torch.nn.Linear(768, 1024).cuda().to(torch.bfloat16), plain)
    assert g8.install(holder) == 1
    assert g8.is_installed(fused) and not g8.is_installed(plain)
    assert fused.__dict__[g8._REC][2] is True and fused.__dict__[g8._REC][4] is False
    x = torch.randn(2, 300, 1024, device = "cuda", dtype = torch.bfloat16) * 3
    x[0, 7, :5] *= 200
    with torch.inference_mode():
        before, gemm = cq.rotquant_call_count(), g8.call_count()
        assert torch.equal(fused(x), stock(x))
        assert cq.rotquant_call_count() == before + 1 and g8.call_count() == gemm
        counters.clear()
        torch._dynamo.reset()
        with torch._inductor.config.patch(emulate_precision_casts = True):
            compiled = torch.compile(fused, fullgraph = True)
            out = compiled(x)
            assert not counters["graph_break"]
            assert counters["stats"]["unique_graphs"] == 1
            # Codes / scales come from the eager-exact op; the epilogue's bf16 roundings are emulated.
            assert torch.equal(out, stock(x))
    g8.uninstall(holder)


@needs_cuda
def test_rotated_linear_without_either_kernel_is_left_alone(monkeypatch):
    monkeypatch.setattr(g8, "device_config", lambda index: None)
    monkeypatch.setattr(cq, "rotquant_device_config", lambda index: None)
    fused = _convrot(_int8_linear(1024, 768, False, 2))
    assert g8.install(torch.nn.Sequential(fused)) == 0 and not g8.is_installed(fused)


@needs_cuda
def test_stock_gemm_matches_torchao_linear_bit_for_bit(rotq_only):
    for version in (None, 2):
        lin = _int8_linear(1024, 768, True, version)
        x = torch.randn(300, 1024, device = "cuda", dtype = torch.bfloat16) * 2
        parts = g8._v1_parts(lin.weight) or g8._v2_parts(lin.weight)
        kind = "v1" if g8._v1_parts(lin.weight) else "v2"
        xq, xs = g8._act_quant_v1(x) if kind == "v1" else g8._act_quant_v2(x, lin.weight)
        with torch.inference_mode():
            assert torch.equal(g8.stock_gemm(xq, parts[0], xs, parts[1], lin.bias), lin(x))


@needs_cuda
def test_convrot_act_quant_api(dense_rotq):
    x = _act(300, 1024, 11)
    q, s = cq.convrot_act_quant(x, 256, True)
    rq, rs = cq.rotquant_reference(x, 256, "v2")
    assert torch.equal(q, rq) and torch.equal(s, rs)
    assert cq.convrot_act_quant(x, 64, True) is None
    assert cq.convrot_act_quant(x.half(), 256, True) is None
    assert cq.convrot_act_quant(x.cpu(), 256, True) is None


def test_rotq_tile_for_picks_the_k_rule_only_where_a_row_fits(monkeypatch):
    default, narrow = (128, 32, 8, 3), (64, 32, 4, 3)
    cq._ROTQ_DEVICE[0] = default
    cq._ROTQ_DEVICE_K[0] = ((5000, 20000, narrow),)
    assert cq.rotq_tile_for(0, 5376) == narrow
    assert cq.rotq_tile_for(0, 4096) == default
    assert cq.rotq_tile_for(0, 256 * 70) == default  # 70 groups do not fit a 64-row tile
    assert cq.rotq_tile_for(1, 5376) is None
    monkeypatch.setitem(cq._ROTQ_K_TILES, (8, 0), ((1, 2, narrow),))
    assert cq.rotq_k_tiles((8, 0))
    monkeypatch.setenv(g8.INT8_GEMM_TILES_ENV, "0")
    assert cq.rotq_k_tiles((8, 0)) == ()


@needs_cuda
def test_every_shipped_rotq_k_tile_is_bit_exact(forced):
    tiles = sorted({rule[2] for rules in cq._ROTQ_K_TILES.values() for rule in rules})
    if not tiles:
        pytest.skip("no per-K rotq tiles shipped")
    for tile in tiles:
        assert cq._rotq_probe(torch.cuda.current_device(), tile), tile
        for m, k in ((1037, 5376), (300, 4096), (129, 14336), (77, 10240)):
            if cq._rotq_rows(k, 256, tile) < 1:
                continue
            x = _act(m, k, m * k)
            for kind in ("v1", "v2"):
                q, s = cq._rotq_launch(x, 256, kind, tile)
                rq, rs = cq.rotquant_reference(x, 256, kind)
                assert torch.equal(q, rq) and torch.equal(s, rs), (tile, m, k, kind)


# ---- rotq_h16 (H16 x H16 rotation), the shared act quant and the SwiGLU prologue ----


def _exact_codes(x, kind):
    """Codes of an exact (fp64) rotation, rounded to bf16, then the stock quant: the target both kernels approximate."""
    from core.inference.diffusion_convrot import build_convrot_hadamard

    h = build_convrot_hadamard(256, device = x.device, dtype = torch.float64)
    xr = (x.double().reshape(x.shape[0], -1, 256) @ h).reshape(x.shape).to(torch.bfloat16)
    if kind == "v1":
        return cq.act_quant_v1(xr)[0]
    from torchao.quantization.granularity import PerRow
    from torchao.quantization.quantize_.workflows.int8.int8_tensor import Int8Tensor

    return Int8Tensor.from_hp(xr, PerRow()).qdata


def test_codes_close_tolerance():
    rq = torch.randint(-120, 120, (64, 1024), dtype = torch.int8)
    rs = torch.rand(64) + 0.5
    assert cq.codes_close(rq.clone(), rs.clone(), rq, rs)
    q = rq.clone()
    q[3, 7] += 1
    q[9, 1] -= 2
    assert cq.codes_close(q, rs, rq, rs)
    q[5, 5] += 3  # three steps: a broken kernel
    assert not cq.codes_close(q, rs, rq, rs)
    q = rq.clone()
    q[:2] += 1  # 2048 codes off on rows whose scale matches
    assert not cq.codes_close(q, rs, rq, rs)
    s = rs.clone()
    s[4] = s[4] * (1 + 2.0**-8)  # one row's amax rounded the other way
    assert cq.codes_close(rq, s, rq, rs)
    s[4] = s[4] * 1.1
    assert not cq.codes_close(rq, s, rq, rs)
    assert not cq.codes_close(rq, rs.to(torch.bfloat16), rq, rs)


def test_h16_plan_keeps_every_chunk_full():
    for ng in (1, 3, 12, 16, 21, 40, 48, 56, 129, 160):
        cg, rows, nch, one = cq._h16_plan(ng)
        assert ng % cg == 0 and nch * cg == ng and cg * rows == cq._H16_TILE_ROWS
        assert one == (nch == 1)
    assert cq._h16_plan(16) == (16, 1, 1, True) and cq._h16_plan(4) == (4, 4, 1, True)
    assert cq._h16_plan(48) == (16, 1, 3, False)  # Qwen-Image-2.1's K 12288


@needs_cuda
@pytest.mark.parametrize(
    "m, k",
    [
        (17, 256),
        (300, 256 * 3),
        (1037, 5376),
        (257, 4096),
        (129, 12288),
        (19, 256 * 129),
        (4, 256 * 16),
    ],
)
@pytest.mark.parametrize("tf32x3", [False, True])
@pytest.mark.parametrize("kind", ["v1", "v2"])
def test_h16_kernel_matches_the_stock_codes_and_the_exact_rotation(h16_rotq, m, k, kind, tf32x3):
    """Within ``codes_close`` of the stock rotation + quant, and no further from the exact rotation than the stock
    (dense GEMM) path itself: the two only differ where an fp32 sum lands on the other side of a bf16 rounding."""
    x = _act(m, k, m * 7 + k)
    q, s = cq._h16_launch(x, kind, tf32x3 = tf32x3)
    rq, rs = cq.rotquant_reference(x, 256, kind)
    assert q.dtype == torch.int8 and s.dtype == rs.dtype and s.shape == (m,)
    assert cq.codes_close(q, s, rq, rs)
    ex = _exact_codes(x, kind)
    assert int((q != ex).sum()) <= int((rq != ex).sum()) + max(4, q.numel() // 200000)


@needs_cuda
def test_h16_is_the_default_and_its_kill_switch_keeps_the_dense_kernel(forced, monkeypatch):
    index = torch.cuda.current_device()
    if cq.rotquant_device_config(index) is None:
        pytest.skip("fused rotation probe refused this device")
    if not cq._H16_DEVICE[index]:
        pytest.skip("rotq_h16 probe refused this device")
    x = _act(300, 256 * 129, 5)  # 129 groups: no dense tile holds the row, rotq_h16 does
    before = cq.h16_call_count()
    q, s = cq.convrot_act_quant(x, 256, True)
    assert cq.h16_call_count() == before + 1
    assert cq.codes_close(q, s, *cq.rotquant_reference(x, 256, "v2"))
    cq._ROTQ_DEVICE.clear()
    cq._H16_DEVICE.clear()
    monkeypatch.setenv(cq.INT8_ROTQUANT_H16_ENV, "0")
    assert cq.rotquant_device_config(index) is not None and cq._H16_DEVICE[index] is False
    assert cq.convrot_act_quant(x, 256, True) is None  # dense tiles stop at 128 groups
    x = _act(300, 1024, 6)
    q, s = cq.convrot_act_quant(x, 256, True)
    rq, rs = cq.rotquant_reference(x, 256, "v2")
    assert torch.equal(q, rq) and torch.equal(s, rs) and cq.h16_call_count() == before + 1


@needs_cuda
def test_h16_mode_follows_the_arch(forced):
    index = torch.cuda.current_device()
    if cq.rotquant_device_config(index) is None or not cq._H16_DEVICE.get(index):
        pytest.skip("rotation probes refused this device")
    cap = tuple(torch.cuda.get_device_capability(index))
    want = "tf32x3" if cap in cq._H16_TF32X3_ARCHS else "ieee"
    assert cq._H16_DEVICE[index] == want


@needs_cuda
def test_h16_probe_refusal_keeps_the_dense_kernel(forced, monkeypatch):
    monkeypatch.setattr(cq, "_h16_probe", lambda index: False)
    index = torch.cuda.current_device()
    if cq.rotquant_device_config(index) is None:
        pytest.skip("fused rotation probe refused this device")
    assert cq._H16_DEVICE[index] is False and cq._SWIGLU_DEVICE[index] is False
    x = _act(300, 1024, 6)
    q, s = cq.convrot_act_quant(x, 256, True)
    rq, rs = cq.rotquant_reference(x, 256, "v2")
    assert torch.equal(q, rq) and torch.equal(s, rs)


def _swiglu_halves(m, k, seed):
    gate = (_act(m, k, seed).float() * 0.05).to(torch.bfloat16)
    gate[0, :6] = torch.tensor([-90.0, -20.0, -1e-3, 0.0, 1e-3, 90.0])
    return gate, _act(m, k, seed + 1)


@needs_cuda
@pytest.mark.parametrize("m, k", [(300, 4096), (129, 12288), (33, 5376)])
@pytest.mark.parametrize("kind", ["v1", "v2"])
def test_swiglu_prologue_matches_the_stock_product_rotation_and_quant(h16_rotq, m, k, kind):
    if not cq._SWIGLU_DEVICE.get(torch.cuda.current_device()):
        pytest.skip("SwiGLU prologue probe refused this device")
    gate, value = _swiglu_halves(m, k, m + k)
    q, s = cq._h16_launch(gate, kind, value)
    rq, rs = cq.rotquant_reference(gate, 256, kind, value)
    assert cq.codes_close(q, s, rq, rs)
    # a different tile (rows per program) gives the same codes bit for bit: the rotation is per group
    q2, s2 = cq._h16_launch(gate, kind, value, tile_rows = 32)
    assert torch.equal(q, q2) and torch.equal(s, s2)
    before = cq.swiglu_call_count()
    out = cq.convrot_swiglu_act_quant(gate, value, 256, kind == "v2")
    assert out is not None and cq.swiglu_call_count() == before + 1
    assert torch.equal(out[0], q) and torch.equal(out[1], s)
    assert cq.convrot_swiglu_act_quant(gate, value[:, :256], 256, True) is None
    assert cq.convrot_swiglu_act_quant(gate, value, 64, True) is None


@needs_cuda
def test_swiglu_kill_switch(forced, monkeypatch):
    monkeypatch.setenv(cq.INT8_ROTQUANT_SWIGLU_ENV, "0")
    index = torch.cuda.current_device()
    if cq.rotquant_device_config(index) is None:
        pytest.skip("fused rotation probe refused this device")
    assert cq._SWIGLU_DEVICE[index] is False
    gate, value = _swiglu_halves(64, 1024, 3)
    assert cq.convrot_swiglu_act_quant(gate, value, 256, True) is None


@needs_cuda
@pytest.mark.parametrize("version", [None, 2])
def test_shared_input_linears_quantize_once_and_match_the_separate_calls(
    h16_rotq, monkeypatch, version
):
    """Q / K / V on one input: one fused rotation + act quant, three GEMMs, outputs bit-identical to the three swapped
    Linears called one by one (each of those quantizes the same input to the same codes)."""
    monkeypatch.delenv(g8.INT8_GEMM_CONVROT_ENV, raising = False)
    lins = [
        _convrot(_int8_linear(1024, n, bias, version))
        for n, bias in ((1024, False), (512, True), (1024, False))
    ]
    holder = torch.nn.Sequential(*lins)
    assert g8.install(holder) == 3
    x = torch.randn(2, 300, 1024, device = "cuda", dtype = torch.bfloat16) * 3
    x[0, 7, :5] *= 200
    with torch.inference_mode():
        separate = [m(x) for m in lins]
        before = cq.rotquant_call_count()
        shared = g8.shared_rotq_linears(x, tuple(lins))
        assert cq.rotquant_call_count() == before + 1
    assert shared is not None and all(torch.equal(a, b) for a, b in zip(shared, separate))
    # a plain (unrotated) Linear, a replaced weight or a wrapped forward: not shareable
    plain = _int8_linear(1024, 256, False, version)
    g8.install(torch.nn.Sequential(plain))
    assert g8.shared_rotq_linears(x, (lins[0], plain)) is None
    lins[1]._diffusers_hook = object()  # a leaf-level offload hook on one of them
    assert g8.shared_rotq_linears(x, tuple(lins)) is None
    del lins[1]._diffusers_hook
    g8.uninstall(holder)


def _swiglu_ff(dim, hidden, version):
    """A module shaped and named like diffusers' QwenImage21SwiGLUFeedForward, int8 + ConvRot."""

    def forward(self, hidden_states):
        return self.out(
            self.activation_fn(self.gate_layer(hidden_states)) * self.proj(hidden_states)
        )

    cls = type("QwenImage21SwiGLUFeedForward", (torch.nn.Module,), {"forward": forward})
    ff = cls()
    ff.proj = _convrot(_int8_linear(dim, hidden, False, version))
    ff.out = _convrot(_int8_linear(hidden, dim, False, version))
    ff.gate_layer = _convrot(_int8_linear(dim, hidden, False, version))
    ff.activation_fn = torch.nn.SiLU()
    return ff


def _swapped_unfused(ff):
    """``ff`` with its Linears swapped but the MLP forward left stock: each Linear quantizes on its own."""
    assert g8.install(torch.nn.Sequential(ff)) == 3
    ff.__dict__.pop("forward", None)
    ff.__dict__.pop(g8._FF_MARK, None)
    return ff


@needs_cuda
@pytest.mark.parametrize("version", [None, 2])
def test_swiglu_mlp_fuses_the_product_into_the_down_quant_eager_and_compiled(
    h16_rotq, monkeypatch, version
):
    """The patched MLP: gate / value share one act quant, the down projection takes its codes from the halves.
    Output within the stock MLP's own int8 rounding (codes_close upstream), compiled fullgraph, one graph, no
    recompile on a second input, uninstall restores the class forward."""
    from torch._dynamo.utils import counters

    if not cq._SWIGLU_DEVICE.get(torch.cuda.current_device()):
        pytest.skip("SwiGLU prologue probe refused this device")
    monkeypatch.delenv(g8.INT8_GEMM_CONVROT_ENV, raising = False)
    stock = _swiglu_ff(1024, 3072, version)
    unfused = _swapped_unfused(_swiglu_ff(1024, 3072, version))
    ff = _swiglu_ff(1024, 3072, version)
    holder = torch.nn.Sequential(ff)
    assert g8.install(holder) == 3
    assert g8.is_installed(ff) and ff.__dict__.get(g8._FF_MARK)
    x = torch.randn(2, 300, 1024, device = "cuda", dtype = torch.bfloat16)
    x[0, 7, :5] *= 50
    x2 = x * 0.5
    with torch.inference_mode():
        before, sw = cq.rotquant_call_count(), cq.swiglu_call_count()
        out = ff(x)
        assert cq.rotquant_call_count() == before + 2 and cq.swiglu_call_count() == sw + 1
        # vs the same kernels unfused (four quants, the product through HBM) and vs the stock ConvRotLinear MLP
        for ref in (unfused(x), stock(x)):
            cos = torch.nn.functional.cosine_similarity(
                out.float().flatten(), ref.float().flatten(), dim = 0
            )
            assert cos > 0.9999
            assert (out.float() - ref.float()).abs().max() <= 0.02 * ref.float().abs().max()
        counters.clear()
        torch._dynamo.reset()
        with torch._inductor.config.patch(emulate_precision_casts = True):
            compiled = torch.compile(ff, fullgraph = True)
            c1 = compiled(x)
            c2 = compiled(x2)
            assert not counters["graph_break"]
            assert counters["stats"]["unique_graphs"] == 1
        assert torch.equal(c1, out)  # opaque kernels + emulated epilogue: eager-exact
        assert torch.equal(c2, ff(x2))
    g8.uninstall(holder)
    assert "forward" not in ff.__dict__ and not g8.is_installed(ff)


@needs_cuda
def test_swiglu_mlp_without_the_prologue_still_shares_the_gate_value_quant(h16_rotq, monkeypatch):
    monkeypatch.delenv(g8.INT8_GEMM_CONVROT_ENV, raising = False)
    cq._SWIGLU_DEVICE[torch.cuda.current_device()] = False
    unfused = _swapped_unfused(_swiglu_ff(1024, 3072, 2))
    ff = _swiglu_ff(1024, 3072, 2)
    assert g8.install(torch.nn.Sequential(ff)) == 3 and ff.__dict__.get(g8._FF_MARK)
    x = torch.randn(300, 1024, device = "cuda", dtype = torch.bfloat16)
    with torch.inference_mode():
        sw = cq.swiglu_call_count()
        before = cq.rotquant_call_count()
        out = ff(x)
        # shared gate / value quant + the down projection's own: two kernels, no SwiGLU prologue
        assert cq.rotquant_call_count() == before + 2 and cq.swiglu_call_count() == sw
        before = cq.rotquant_call_count()
        ref = unfused(x)
        assert cq.rotquant_call_count() == before + 3
        assert torch.equal(out, ref)
    g8.uninstall(torch.nn.Sequential(ff))


def _host_swiglu_ff(monkeypatch):
    ff = torch.nn.Module()
    ff.gate_layer, ff.proj, ff.out = (
        torch.nn.Linear(8, 8),
        torch.nn.Linear(8, 8),
        torch.nn.Linear(8, 8),
    )
    ff.activation_fn = torch.nn.SiLU()
    monkeypatch.setattr(g8, "_swiglu_ff_eligible", lambda m: m is ff)
    return ff


@pytest.mark.parametrize("streamed, patched", [(False, False), (True, True)])
def test_streamed_install_patches_the_host_swiglu_mlp(monkeypatch, streamed, patched):
    monkeypatch.delenv(g8.INT8_SWIGLU_STREAMED_ENV, raising = False)
    ff = _host_swiglu_ff(monkeypatch)
    assert g8._install_swiglu_ff(torch.nn.Sequential(ff), 0, streamed = streamed) == int(patched)
    assert bool(ff.__dict__.get(g8._FF_MARK)) is patched


def test_streamed_install_keeps_a_swiglu_mlp_on_another_card(monkeypatch):
    ff = _host_swiglu_ff(monkeypatch)
    monkeypatch.setattr(g8, "_weight_card", lambda m: 1)
    assert g8._install_swiglu_ff(torch.nn.Sequential(ff), 0, streamed = True) == 0
    assert g8._FF_MARK not in ff.__dict__


def test_swap_passes_streamed_only_for_the_onload_device_install(monkeypatch):
    seen = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        g8, "_swap_on", lambda t, lg, i, streamed = False: seen.append((i, streamed)) or 0
    )
    import core.inference.diffusion_int8_fused as fused

    monkeypatch.setattr(fused, "resident_cuda_device", lambda t: torch.device("cuda", 0))
    g8._swap(torch.nn.Linear(2, 2), device = "cuda:0")
    g8._swap(torch.nn.Linear(2, 2))
    assert seen == [(0, True), (0, False)]


@needs_cuda
def test_block_streamed_swiglu_mlp_runs_fused_after_the_onload(h16_rotq, monkeypatch):
    """Streamed host MLP runs fused, bit-identical to resident; the kill switch keeps it stock."""
    hooks = pytest.importorskip("diffusers.hooks")
    if not cq._SWIGLU_DEVICE.get(torch.cuda.current_device()):
        pytest.skip("SwiGLU prologue probe refused this device")
    monkeypatch.delenv(g8.INT8_GEMM_CONVROT_ENV, raising = False)
    monkeypatch.delenv(g8.INT8_GEMM_STREAMED_ENV, raising = False)

    class Holder(torch.nn.Module):
        def __init__(self, ff):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Sequential(ff)])

        def forward(self, x):
            return self.blocks[0](x)

    resident = _swiglu_ff(1024, 3072, 2)
    assert g8.install(torch.nn.Sequential(resident)) == 3 and resident.__dict__.get(g8._FF_MARK)
    x = torch.randn(300, 1024, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        ref = resident(x)
    for kill, fused in ((None, True), ("0", False)):
        if kill is None:
            monkeypatch.delenv("UNSLOTH_DIFFUSION_INT8_SWIGLU_STREAMED", raising = False)
        else:
            monkeypatch.setenv("UNSLOTH_DIFFUSION_INT8_SWIGLU_STREAMED", kill)
        ff = _swiglu_ff(1024, 3072, 2)
        holder = Holder(ff).requires_grad_(False).cpu()
        hooks.apply_group_offloading(
            holder,
            onload_device = torch.device("cuda"),
            offload_device = torch.device("cpu"),
            offload_type = "block_level",
            num_blocks_per_group = 1,
        )
        assert ff.out.weight.device.type == "cpu"
        assert g8.install(holder, device = "cuda") == 3
        assert bool(ff.__dict__.get(g8._FF_MARK)) is fused
        with torch.no_grad():
            sw = cq.swiglu_call_count()
            out = holder(x)
            assert cq.swiglu_call_count() == sw + int(fused)
        if fused:
            assert torch.equal(out, ref)
        assert ff.out.weight.device.type == "cpu"


def test_hooked_linear_is_not_shared():
    """A forward hook or pre-hook (module-level or global) on a swapped Linear must see its call, so the shared QKV /
    SwiGLU paths, which run the GEMM without calling the module, decline it and the per-module call runs instead."""
    lin = torch.nn.Linear(8, 8)
    lin.__dict__[g8._REC] = ("v2", 64, True, lin.weight, False)
    lin.__dict__[g8._MARK] = True
    lin.forward = types.MethodType(g8._linear_forward, lin)
    assert g8._swapped_rec(lin) is not None
    noop = lambda *a, **k: None  # noqa: E731
    for register in (
        lin.register_forward_hook,
        lin.register_forward_pre_hook,
        lambda f: lin.register_forward_hook(f, with_kwargs = True),
        torch.nn.modules.module.register_module_forward_hook,
        torch.nn.modules.module.register_module_forward_pre_hook,
    ):
        handle = register(noop)
        try:
            assert g8._swapped_rec(lin) is None
        finally:
            handle.remove()
        assert g8._swapped_rec(lin) is not None


def test_replaced_linear_forward_is_not_shared():
    """An instance ``forward`` assigned over the installed ``_linear_forward`` (adapter / instrumentation code) is
    what a module call runs, so the shared paths, which skip the call, decline it; restoring it re-enables them."""
    lin = torch.nn.Linear(8, 8)
    lin.__dict__[g8._REC] = ("v2", 64, True, lin.weight, False)
    lin.__dict__[g8._MARK] = True
    installed = types.MethodType(g8._linear_forward, lin)
    lin.forward = installed
    assert g8._swapped_rec(lin) is not None
    lin.forward = lambda t: t
    assert g8._swapped_rec(lin) is None
    lin.forward = types.MethodType(lambda self, t: t, lin)
    assert g8._swapped_rec(lin) is None
    del lin.forward
    assert g8._swapped_rec(lin) is None
    lin.forward = installed
    assert g8._swapped_rec(lin) is not None


def test_hooked_activation_declines_the_swiglu_fusion(monkeypatch):
    """The fused SwiGLU path never calls ``activation_fn``: a hook (module-level or global) or an instance
    ``forward`` on it has to see the call, so the MLP takes ``out(act(gate) * value)`` instead."""
    ff = torch.nn.Module()
    ff.gate_layer, ff.proj, ff.out = (
        torch.nn.Linear(8, 8),
        torch.nn.Linear(8, 8),
        torch.nn.Linear(8, 8),
    )
    ff.activation_fn = torch.nn.SiLU()
    fused = object()
    monkeypatch.setattr(g8, "shared_rotq_linears", lambda x, linears: None)
    monkeypatch.setattr(g8, "_swiglu_down", lambda down, gate, value: fused)
    x = torch.randn(3, 8)
    assert g8._swiglu_ff_forward(ff, x) is fused
    seen = []
    for register in (
        lambda f: ff.activation_fn.register_forward_hook(lambda m, a, o: seen.append(1)),
        lambda f: ff.activation_fn.register_forward_pre_hook(lambda m, a: seen.append(1)),
        lambda f: torch.nn.modules.module.register_module_forward_hook(
            lambda m, a, o: seen.append(1)
        ),
    ):
        handle = register(None)
        try:
            seen.clear()
            y = g8._swiglu_ff_forward(ff, x)
            assert y is not fused and seen
            ref = ff.out(torch.nn.functional.silu(ff.gate_layer(x)) * ff.proj(x))
            assert torch.equal(y, ref)
        finally:
            handle.remove()
        assert g8._swiglu_ff_forward(ff, x) is fused
    ff.activation_fn.forward = lambda t: t  # an instance override replacing the activation
    assert g8._swiglu_ff_forward(ff, x) is not fused


def test_replaced_activation_declines_the_swiglu_fusion(monkeypatch):
    """The fused prologue hard-codes SiLU: an ``activation_fn`` swapped after install for GELU, a custom module or a
    ``SiLU`` subclass (no hooks, no instance ``forward``) has to run itself, and restoring a plain SiLU fuses again."""

    class _SiLUSub(torch.nn.SiLU):
        def forward(self, t):
            return torch.nn.functional.silu(t) * 2

    class _Custom(torch.nn.Module):
        def forward(self, t):
            return t * 3

    ff = torch.nn.Module()
    ff.gate_layer, ff.proj, ff.out = (
        torch.nn.Linear(8, 8),
        torch.nn.Linear(8, 8),
        torch.nn.Linear(8, 8),
    )
    ff.activation_fn = torch.nn.SiLU()
    fused = object()
    monkeypatch.setattr(g8, "shared_rotq_linears", lambda x, linears: None)
    monkeypatch.setattr(g8, "_swiglu_down", lambda down, gate, value: fused)
    x = torch.randn(3, 8)
    assert g8._swiglu_ff_forward(ff, x) is fused
    for act in (torch.nn.GELU(), torch.nn.GELU(approximate = "tanh"), _SiLUSub(), _Custom()):
        ff.activation_fn = act
        y = g8._swiglu_ff_forward(ff, x)
        assert y is not fused
        assert torch.equal(y, ff.out(act(ff.gate_layer(x)) * ff.proj(x)))
    ff.activation_fn = torch.nn.SiLU()
    assert g8._swiglu_ff_forward(ff, x) is fused


def test_codes_close_rejects_non_finite_scales():
    """NaN compares false against every tolerance: a NaN / inf scale on one row (e.g. the probe's all-zero row) must
    still fail, on either side."""
    rq = torch.randint(-120, 120, (64, 1024), dtype = torch.int8)
    rs = torch.rand(64) + 0.5
    for bad in (float("nan"), float("inf"), float("-inf")):
        s = rs.clone()
        s[1] = bad
        assert not cq.codes_close(rq, s, rq, rs)
        assert not cq.codes_close(rq, rs, rq, s)


@needs_cuda
def test_h16_probe_covers_the_one_pass_branch_and_rejects_nan_scales(forced):
    """The per-device probe runs both ``rotq_h16`` branches (one pass, two pass) on v1 / v2, and a kernel that returns
    a NaN scale on one row is refused."""
    index = torch.cuda.current_device()
    real = cq._h16_launch
    seen = set()

    def recording(x, kind, *a, **k):
        seen.add((cq._h16_plan(x.shape[-1] // 256)[3], kind))
        return real(x, kind, *a, **k)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cq, "_h16_launch", recording)
        if not cq._h16_probe(index):
            pytest.skip("rotq_h16 probe refused this device")
    assert seen == {(True, "v1"), (True, "v2"), (False, "v1"), (False, "v2")}

    def nan_row(x, kind, *a, **k):
        q, s = real(x, kind, *a, **k)
        s = s.clone()
        s[1] = float("nan")  # the probe's all-zero row
        return q, s

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cq, "_h16_launch", nan_row)
        assert cq._h16_probe(index) is False
        assert cq._swiglu_probe(index) is False
