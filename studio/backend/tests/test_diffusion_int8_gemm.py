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
    g8._DEVICE_CFG.clear()
    yield
    g8._DEVICE_CFG.clear()


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


def test_probe_shapes_stay_on_the_16_byte_grid():
    """A probe K off 16 compiles the spilling kernel variant; its local memory stays reserved for the process."""
    ks = [k for _m, _n, k, _b, _x in g8._PROBE_SHAPES]
    assert all(k % 16 == 0 for k in ks)
    for bk in (64, 128):  # every shipped BLOCK_K still sees a ragged K
        assert any(k % bk for k in ks)


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
    assert len(launched) == 1 and len(stock) == 4


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
def test_no_compiled_variant_spills_to_local_memory(forced):
    """Local memory a kernel spills to is reserved by the driver for every resident thread of the device and kept for
    the process (0.7 GB on B200, 1.4 GB on A100 for the K-off-16 variant), outside PyTorch's allocator."""
    g = torch.Generator().manual_seed(0)
    for m, k, n in ((4096, 3072, 3072), (300, 1040, 520), (300, 1000, 384)):
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
def test_pinned_group_offloaded_denoiser_takes_the_gemm_and_survives_release(forced, monkeypatch):
    """The Qwen-Image-2.1 16 GB placement: diffusers group offloading on an int8 DiT with every group pinned resident.
    The speed layer refused the GEMM on the plan; after placement engage_pinned_denoisers installs it, bit-identical to
    the stock torchao Linear, and it stays correct when an oversized request streams the groups again and restores."""
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
    cfg.version = (
        2  # what streams under diffusers group offloading (v1's subclass rejects is_pinned)
    )
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
    # Studio's own torchao kwargs (frozen weights, the swap retry, the pin decision), as apply_memory_plan passes them
    apply_group_offloading(dit, **dm._torchao_group_offload_kwargs(dit, kwargs, [0]))
    pipe = _types.SimpleNamespace(transformer = dit, components = {"transformer": dit})
    assert dm._keep_groups_resident(dit, 1024, "cuda") > 0
    assert dm.denoisers_pinned_resident(pipe)

    monkeypatch.setattr(ds, "_denoiser_dits", lambda p: [dit])
    applied = {"compiled": True, "int8_gemm": False}
    ds.engage_pinned_denoisers(pipe, applied)
    assert applied["int8_gemm"] and dit._unsloth_int8_gemm == 8
    # no_grad, as Studio renders an offloaded torchao denoiser (inference_mode rejects moving torchao weights)
    with torch.no_grad():
        before = g8.call_count()
        assert torch.equal(dit(x), ref)
        assert g8.call_count() == before + 8
        restore = dm.release_resident_groups(pipe, 1024)
        assert restore is not None and not dm.denoisers_pinned_resident(pipe)
        for _ in range(2):  # first streamed forward traces the prefetch order
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
