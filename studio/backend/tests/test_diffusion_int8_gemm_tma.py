# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the TMA int8 GEMM; kernel tests need an sm90+ GPU with Triton host descriptors."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_int8_gemm as g8


def _reset():
    for d in (
        g8._DEVICE_CFG,
        g8._DEVICE_TILES,
        g8._DEVICE_TMA,
        g8._UNLISTED_PROBE,
        g8._UNLISTED_OK,
    ):
        d.clear()
    g8._UNLISTED_CARDS.clear()


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    for name in (
        g8.INT8_GEMM_ENV,
        g8.INT8_GEMM_TMA_ENV,
        g8.INT8_GEMM_TILES_ENV,
        g8.INT8_GEMM_UNLISTED_ENV,
    ):
        monkeypatch.delenv(name, raising = False)
    _reset()
    yield
    _reset()


class _Fake:
    def __init__(
        self,
        monkeypatch,
        cap,
        probe = True,
        tma_probe = True,
    ):
        self.probes = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: cap)
        monkeypatch.setattr(g8, "is_rocm_torch", lambda t: False)
        monkeypatch.setattr(g8, "_triton_version_ok", lambda version = None: True)
        monkeypatch.setattr(g8, "_kernels", lambda: object())
        monkeypatch.setattr(g8, "_op", lambda: "op")
        monkeypatch.setattr(g8, "_tensor_descriptor", lambda: object())

        def fake_probe(
            index,
            cfg,
            tma = False,
        ):
            self.probes.append((cfg, tma))
            if tma and isinstance(tma_probe, Exception):
                raise tma_probe
            return tma_probe if tma else probe

        monkeypatch.setattr(g8, "_probe", fake_probe)


@pytest.mark.parametrize("cap", [(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 1)])
def test_tma_only_on_measured_archs(monkeypatch, cap):
    fake = _Fake(monkeypatch, cap)
    g8.device_config(0)
    assert g8._DEVICE_TMA.get(0) is None
    assert all(not tma for _, tma in fake.probes)


def test_sm120_probes_and_arms_the_tma_tile(monkeypatch):
    fake = _Fake(monkeypatch, (12, 0))
    assert g8.device_config(0) == g8._ARCH_CONFIG[(12, 0)]
    assert g8._DEVICE_TMA[0] == g8._TMA_CONFIG[(12, 0)]
    assert (g8._TMA_CONFIG[(12, 0)], True) in fake.probes


def test_tma_tile_only_from_shape_min_m():
    g8._DEVICE_TMA[0] = g8._TMA_CONFIG[(12, 0)]
    assert g8.tma_tile_for(0, g8._SHAPE_MIN_M - 1) is None
    assert g8.tma_tile_for(0, g8._SHAPE_MIN_M) == g8._TMA_CONFIG[(12, 0)]
    assert g8.tma_tile_for(1, 19315) is None


@pytest.mark.parametrize("value", ["0", "off", "false"])
def test_kill_switch_keeps_pointer_loads_and_never_probes(monkeypatch, value):
    monkeypatch.setenv(g8.INT8_GEMM_TMA_ENV, value)
    fake = _Fake(monkeypatch, (12, 0))
    assert g8.device_config(0) == g8._ARCH_CONFIG[(12, 0)]
    assert g8._DEVICE_TMA.get(0) is None
    assert all(not tma for _, tma in fake.probes)


@pytest.mark.parametrize("result", [False, RuntimeError("no TMA")])
def test_failed_tma_probe_keeps_the_pointer_kernel(monkeypatch, result):
    _Fake(monkeypatch, (12, 0), tma_probe = result)
    assert g8.device_config(0) == g8._ARCH_CONFIG[(12, 0)]
    assert g8._DEVICE_TMA.get(0) is None
    assert g8._DEVICE_TILES[0]


def test_no_descriptor_api_keeps_the_pointer_kernel(monkeypatch):
    fake = _Fake(monkeypatch, (12, 0))
    monkeypatch.setattr(g8, "_tensor_descriptor", lambda: None)
    assert g8.device_config(0) == g8._ARCH_CONFIG[(12, 0)]
    assert g8._DEVICE_TMA.get(0) is None
    assert all(not tma for _, tma in fake.probes)


def test_failed_pointer_probe_leaves_everything_off(monkeypatch):
    _Fake(monkeypatch, (12, 0), probe = False)
    assert g8.device_config(0) is None
    assert g8._DEVICE_TMA.get(0) is None


def test_reprobe_drops_a_stale_tma_tile(monkeypatch):
    _Fake(monkeypatch, (12, 0))
    g8.device_config(0)
    assert g8._DEVICE_TMA[0]
    g8._DEVICE_CFG.clear()
    monkeypatch.setenv(g8.INT8_GEMM_TMA_ENV, "0")
    g8.device_config(0)
    assert g8._DEVICE_TMA.get(0) is None


class _T:
    def __init__(self, shape):
        self.shape = shape
        self.device = types_device

    def stride(self, dim = None):
        s = (self.shape[1], 1)
        return s if dim is None else s[dim]

    def data_ptr(self):
        return 0

    def contiguous(self):
        return self


class _Dev:
    index = 0

    def __eq__(self, other):
        return isinstance(other, _Dev)


types_device = _Dev()


def _armed(monkeypatch, tma_raises = False):
    calls = []
    g8._DEVICE_CFG[0] = g8._ARCH_CONFIG[(12, 0)]
    g8._DEVICE_TILES[0] = ()
    g8._DEVICE_TMA[0] = g8._TMA_CONFIG[(12, 0)]

    def tma(a, w, xs, ws, bias, cfg):
        calls.append(("tma", cfg))
        if tma_raises:
            raise RuntimeError("descriptor refused")
        return "tma-out"

    def ptr(a, w, xs, ws, bias, cfg):
        calls.append(("ptr", cfg))
        return "ptr-out"

    monkeypatch.setattr(g8, "_launch_tma", tma)
    monkeypatch.setattr(g8, "_launch", ptr)
    monkeypatch.setattr(g8, "reference", lambda *a: "stock-out")
    return calls


def test_run_takes_the_tma_kernel_and_counts_it(monkeypatch):
    calls = _armed(monkeypatch)
    before = g8.tma_call_count()
    assert g8._run(_T((2048, 4096)), _T((4096, 4096)), None, None, None) == "tma-out"
    assert calls == [("tma", g8._TMA_CONFIG[(12, 0)])]
    assert g8.tma_call_count() == before + 1


def test_tma_launch_failure_falls_back_to_pointer_loads_for_good(monkeypatch):
    calls = _armed(monkeypatch, tma_raises = True)
    before = g8.tma_call_count()
    assert g8._run(_T((2048, 4096)), _T((4096, 4096)), None, None, None) == "ptr-out"
    assert g8._DEVICE_TMA[0] is None
    assert g8._run(_T((2048, 4096)), _T((4096, 4096)), None, None, None) == "ptr-out"
    assert [c[0] for c in calls] == ["tma", "ptr", "ptr"]
    assert g8.tma_call_count() == before


def test_small_m_takes_the_pointer_kernel(monkeypatch):
    calls = _armed(monkeypatch)
    assert g8._run(_T((300, 4096)), _T((4096, 4096)), None, None, None) == "ptr-out"
    assert calls == [("ptr", g8._ARCH_CONFIG[(12, 0)])]


def test_misaligned_operands_skip_both_kernels(monkeypatch):
    calls = _armed(monkeypatch)
    assert g8._run(_T((2048, 4100)), _T((4096, 4100)), None, None, None) == "stock-out"
    assert calls == []


def _tma_ready() -> bool:
    try:
        if not torch.cuda.is_available() or g8.is_rocm_torch(torch):
            return False
        if torch.cuda.get_device_capability(0) < (9, 0):
            return False
        return g8._kernels() is not None and g8._tensor_descriptor() is not None
    except Exception:  # noqa: BLE001
        return False


needs_tma = pytest.mark.skipif(
    not _tma_ready(), reason = "needs an sm90+ NVIDIA GPU and Triton host descriptors"
)


@needs_tma
@pytest.mark.parametrize(
    "m, k, n, bias, xs32",
    [
        (4096, 4096, 4096, False, False),
        (1037, 528, 1400, True, False),
        (17, 256, 1024, True, True),
        (300, 12288, 256, False, False),
        (2053, 5376, 5376, False, True),
    ],
)
@pytest.mark.parametrize("ws32", [False, True])
@pytest.mark.parametrize("cfg", sorted(set(g8._TMA_CONFIG.values())))
def test_tma_kernel_is_bit_exact_vs_torchao_epilogue(m, k, n, bias, xs32, ws32, cfg):
    g = torch.Generator().manual_seed(m + k + n)
    a = torch.randint(-127, 128, (m, k), generator = g, dtype = torch.int8).cuda()
    w = torch.randint(-127, 128, (n, k), generator = g, dtype = torch.int8).cuda()
    xs = (torch.rand(m, generator = g) * 0.02 + 1e-4).to(torch.bfloat16)
    xs = (xs.float() if xs32 else xs).cuda()
    ws = (torch.rand(n, generator = g) * 0.002 + 1e-5).to(torch.bfloat16).cuda()
    ws = ws.float() if ws32 else ws
    b = (torch.randn(n, generator = g) * 0.1).to(torch.bfloat16).cuda() if bias else None
    assert torch.equal(g8._launch_tma(a, w, xs, ws, b, cfg), g8.reference(a, w, xs, ws, b))


@needs_tma
@pytest.mark.parametrize("cfg", sorted(set(g8._TMA_CONFIG.values())))
def test_tma_probe_passes_including_the_rounding_ties(cfg):
    assert g8._probe(0, cfg, tma = True)
    a, w = g8.tie_operands(torch.device("cuda", 0))
    xs = torch.ones(a.shape[0], device = "cuda", dtype = torch.bfloat16)
    ws = torch.ones(w.shape[0], device = "cuda", dtype = torch.bfloat16)
    assert torch.equal(g8._launch_tma(a, w, xs, ws, None, cfg), g8.reference(a, w, xs, ws, None))


@needs_tma
def test_compiled_swapped_linear_runs_the_tma_kernel_without_breaks():
    from torch._dynamo.utils import counters
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

    if g8.device_config(0) is None or not g8._DEVICE_TMA.get(0):
        pytest.skip("TMA tile not armed on this device")

    def lin():
        torch.manual_seed(0)
        m = torch.nn.Linear(1024, 768, bias = True).cuda().to(torch.bfloat16)
        quantize_(m, Int8DynamicActivationInt8WeightConfig(version = 2, set_inductor_config = False))
        return m

    stock, fused = lin(), lin()
    holder = torch.nn.Sequential(fused)
    assert g8.install(holder) == 1
    x = torch.randn(2, 600, 1024, device = "cuda", dtype = torch.bfloat16)  # M >= _SHAPE_MIN_M
    with torch.inference_mode():
        before = g8.tma_call_count()
        assert torch.equal(fused(x), stock(x))
        assert g8.tma_call_count() == before + 1
        counters.clear()
        torch._dynamo.reset()
        with torch._inductor.config.patch(emulate_precision_casts = True):
            fn = torch.compile(fused, fullgraph = True)
            out = fn(x)
            again = fn(x)
            assert torch.equal(out, again)
            assert torch.equal(out, torch.compile(stock, fullgraph = True)(x))
        assert not counters["graph_break"]
        assert counters["stats"]["unique_graphs"] == 2  # this Linear + the stock one, no recompile
        assert g8.tma_call_count() >= before + 3
    g8.uninstall(holder)
