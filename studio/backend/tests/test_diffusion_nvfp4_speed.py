# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Kernel-side tests for the NVFP4 flashinfer path: the device guard and the ordering barrier.

The hermetic half stubs ``torch`` and ``flashinfer``, because neither which device a launch sees nor
the call ORDER inside the GEMM op is observable from outside an opaque custom op on one GPU.
"""

from __future__ import annotations

import contextlib
import sys
import types

import pytest

from core.inference import diffusion_nvfp4_dispatch as dispatch
from core.inference import diffusion_nvfp4_linear as nl
from core.inference import diffusion_nvfp4_ops as ops




class _FakeDevice:
    def __init__(
        self,
        index,
        kind = "cuda",
    ):
        self.type = kind
        self.index = index

    def __eq__(self, other):
        return isinstance(other, _FakeDevice) and (self.type, self.index) == (
            other.type,
            other.index,
        )

    def __hash__(self):
        return hash((self.type, self.index))

    def __repr__(self):  # pragma: no cover - debug aid
        return f"cuda:{self.index}"


class _FakeTensor:
    """A tensor-shaped object whose every method returns a tensor on the SAME device."""

    def __init__(
        self,
        shape = (1,),
        device = None,
        dtype = None,
    ):
        self.shape = tuple(shape)
        self.device = device if device is not None else _FakeDevice(0)
        self.dtype = dtype

    def _same(self, shape = None):
        return _FakeTensor(self.shape if shape is None else shape, self.device, self.dtype)

    def float(self):
        return self._same()

    def abs(self):
        return self._same()

    def amax(self):
        return self._same((1,))

    def clamp(self, **_kw):
        return self._same((1,))

    def all(self):
        return self._same((1,))

    def item(self):
        return True

    def reshape(self, *shape):
        flat = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        return self._same(tuple(flat) if not isinstance(flat, int) else (flat,))

    def contiguous(self):
        return self

    def is_contiguous(self):
        return True

    def to(
        self,
        device = None,
        **_kw,
    ):
        out = self._same()
        if isinstance(device, _FakeDevice):
            out.device = device
        return out

    def zero_(self):
        _RECORDER.launches.append(("zero_", _RECORDER.current))
        return self

    def add_(self, _other):
        return self

    @property
    def T(self):
        return self._same(tuple(reversed(self.shape)))

    def __mul__(self, _other):
        return self._same()

    __rmul__ = __mul__

    def __truediv__(self, _other):
        return self._same()

    def __rtruediv__(self, _other):
        return self._same()


class _Recorder:
    """The process-wide "current device", plus the log every assertion below reads."""

    def __init__(self):
        self.current = 0
        self.launches: list[tuple] = []
        self.stack: list[int] = []

    def reset(self):
        self.current = 0
        self.launches.clear()
        self.stack.clear()


_RECORDER = _Recorder()


@contextlib.contextmanager
def _fake_device_guard(device):
    index = device.index if isinstance(device, _FakeDevice) else int(device)
    _RECORDER.stack.append(_RECORDER.current)
    _RECORDER.current = index
    try:
        yield
    finally:
        _RECORDER.current = _RECORDER.stack.pop()


def _record(name):
    def _call(*args, **kwargs):
        _RECORDER.launches.append((name, _RECORDER.current))
        if name == "nvfp4_quantize":
            x = args[0]
            return (_FakeTensor(x.shape, x.device), _FakeTensor(x.shape, x.device))
        out = kwargs.get("out")
        return out if out is not None else _FakeTensor((1,), _FakeDevice(_RECORDER.current))

    return _call


def _fake_alloc(name):
    def _call(
        *shape,
        device = None,
        dtype = None,
        **_kw,
    ):
        flat = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        _RECORDER.launches.append((name, _RECORDER.current))
        return _FakeTensor(tuple(flat), device, dtype)

    return _call


def _fake_torch():
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"
    torch.float32 = "float32"
    torch.uint8 = "uint8"
    torch.zeros = _fake_alloc("zeros")
    torch.empty = _fake_alloc("empty")
    torch.randn = _fake_alloc("randn")
    torch.isfinite = lambda t: t

    def _device(kind, index = None):
        if isinstance(kind, _FakeDevice):
            return kind
        if isinstance(kind, int):
            return _FakeDevice(kind, "cuda")
        return _FakeDevice(0 if index is None else index, str(kind))

    torch.device = _device
    torch.inference_mode = contextlib.nullcontext
    cuda = types.SimpleNamespace(
        is_available = lambda: True,
        device = _fake_device_guard,
        current_device = lambda: _RECORDER.current,
        device_count = lambda: 2,
        get_device_capability = lambda dev = None: (10, 0),
        get_device_name = lambda dev = None: "stub B200",
        synchronize = lambda dev = None: _RECORDER.launches.append(("synchronize", _RECORDER.current)),
        is_current_stream_capturing = lambda: False,
    )
    torch.cuda = cuda
    return torch


def _fake_flashinfer():
    fi = types.ModuleType("flashinfer")
    fi.__version__ = "0.6.6"
    fi.nvfp4_quantize = _record("nvfp4_quantize")
    fi.mm_fp4 = _record("mm_fp4")
    fi.autotune = lambda _on: contextlib.nullcontext()
    return fi


@pytest.fixture
def stub_kernels(monkeypatch):
    """Install the stubbed torch and flashinfer for the duration of one test."""
    _RECORDER.reset()
    # A verdict left behind by another file would take a path the stub does not model.
    dispatch.reset()
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "flashinfer", _fake_flashinfer())
    monkeypatch.setattr(ops, "register_ops", lambda: None)
    monkeypatch.setattr(nl, "register_ops", lambda: None)
    ops.reset_preflight_cache()
    yield _RECORDER
    _RECORDER.reset()
    ops.reset_preflight_cache()
    dispatch.reset()


def _launch_devices(recorder, *names) -> list[int]:
    return [device for name, device in recorder.launches if name in names]




def test_the_stub_records_the_device_a_launch_actually_sees(stub_kernels):
    """The detector's own control: without a guard the launch reads the CURRENT device."""
    import flashinfer

    flashinfer.mm_fp4()
    assert _launch_devices(stub_kernels, "mm_fp4") == [0]


def test_the_quantize_body_enters_the_tensors_device(stub_kernels):
    x = _FakeTensor((128, 256), _FakeDevice(1))
    ops._quantize_impl(x, _FakeTensor((1,), _FakeDevice(1)))
    assert _launch_devices(stub_kernels, "nvfp4_quantize") == [1]
    assert stub_kernels.current == 0


def test_the_mm_body_enters_the_tensors_device_for_the_barrier_and_the_gemm(stub_kernels):
    device = _FakeDevice(1)
    ops._mm_impl(
        _FakeTensor((4096, 1536), device),
        _FakeTensor((12288, 1536), device),
        _FakeTensor((4096, 192), device),
        _FakeTensor((12288, 192), device),
        _FakeTensor((1,), device),
        12288,
        "cutlass",
    )
    # The barrier is only a barrier if it fires on the card the GEMM will read from.
    assert set(_launch_devices(stub_kernels, "zeros", "empty", "mm_fp4")) == {1}
    assert stub_kernels.current == 0


def test_the_preflight_enters_the_probed_device(stub_kernels):
    record = ops.nvfp4_preflight(_FakeDevice(1))
    assert record["ok"] is True, record["reason"]
    assert set(_launch_devices(stub_kernels, "nvfp4_quantize", "mm_fp4", "synchronize")) == {1}
    assert stub_kernels.current == 0


def test_the_prewarm_enters_each_layers_device(stub_kernels):
    class NVFP4FlashInferLinear:
        """Named for what ``is_nvfp4_flashinfer_linear`` keys on: the class NAME, not an import."""

        def __init__(self):
            self.in_features = 1536
            self.out_features = 12288
            self.a_gsf = _FakeTensor((1,), _FakeDevice(1))
            self.wq = _FakeTensor((12288, 768), _FakeDevice(1))
            self._tuned = False

        def __call__(self, _x):
            _RECORDER.launches.append(("layer", _RECORDER.current))
            return _FakeTensor((1,), self.wq.device)

    layer = NVFP4FlashInferLinear()
    tree = types.SimpleNamespace(named_modules = lambda: [("blocks.0.ff", layer)])

    nl.reset_tuned_shapes()
    try:
        assert nl.nvfp4_prewarm(tree, (512,)) == 1
    finally:
        nl.reset_tuned_shapes()
    assert set(_launch_devices(stub_kernels, "layer", "zeros", "synchronize")) == {1}
    assert stub_kernels.current == 0


def test_no_guard_is_left_open_when_a_launch_raises(stub_kernels, monkeypatch):
    """A leaked guard is worse than no guard: it moves the whole process onto another card."""
    import flashinfer

    def _boom(*_a, **_kw):
        raise RuntimeError("kernel refused")

    monkeypatch.setattr(flashinfer, "nvfp4_quantize", _boom)
    with pytest.raises(RuntimeError):
        ops._quantize_impl(_FakeTensor((8, 16), _FakeDevice(1)), _FakeTensor((1,), _FakeDevice(1)))
    assert stub_kernels.current == 0
    assert stub_kernels.stack == []




def test_a_layer_on_card_one_runs_correctly_while_the_current_device_is_card_zero():
    """Unguarded, the cutlass FP4 GEMM launches against whatever context is current."""
    torch = pytest.importorskip("torch")
    if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two visible CUDA devices")
    if tuple(torch.cuda.get_device_capability(1)) not in ops.NVFP4_FLASHINFER_CAPS:
        pytest.skip("device 1 has no flashinfer NVFP4 kernels")
    pytest.importorskip("flashinfer")
    pytest.importorskip("torchao")

    import torch.nn as nn
    from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
    from torchao.quantization import quantize_

    torch.manual_seed(0)
    with torch.cuda.device(1):
        linear = nn.Linear(3072, 3072, bias = True).to("cuda:1", torch.bfloat16)
        quantize_(linear, NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel = False))
        x = torch.randn(512, 3072, device = "cuda:1", dtype = torch.bfloat16) * 0.05
        converted = nl.nvfp4_linear_from_torchao(linear, ops.global_scale(x))
        with torch.inference_mode():
            want = converted(x)
        torch.cuda.synchronize(1)

    assert torch.cuda.current_device() == 0
    with torch.inference_mode():
        got = converted(x)
    torch.cuda.synchronize(1)
    assert bool(torch.isfinite(got).all())
    assert torch.equal(got, want)




def _mm_once(
    device_index = 1,
    m = 4096,
    n = 12288,
    k = 3072,
):
    device = _FakeDevice(device_index)
    return ops._mm_impl(
        _FakeTensor((m, k // 2), device),
        _FakeTensor((n, k // 2), device),
        _FakeTensor((m, k // 16), device),
        _FakeTensor((n, k // 16), device),
        _FakeTensor((1,), device),
        n,
        "cutlass",
    )


@pytest.fixture(autouse = True)
def _clean_barriers():
    ops.reset_barriers()
    yield
    ops.reset_barriers()


def test_the_barrier_is_allocated_once_per_device(stub_kernels):
    for _ in range(5):
        _mm_once(1)
    for _ in range(5):
        _mm_once(0)
    barrier_allocs = [1 for name, _ in stub_kernels.launches if name == "empty"]
    assert len(barrier_allocs) == 12, stub_kernels.launches
    assert sorted(ops._BARRIERS) == [0, 1]
    assert ops._BARRIERS[0] is not ops._BARRIERS[1]


def test_the_barrier_fill_precedes_every_gemm(stub_kernels):
    """A kernel MUST exist between the activation quantiser and the GEMM (PDL without
    griddepcontrol)."""
    for _ in range(3):
        _mm_once(1)
    order = [name for name, _ in stub_kernels.launches if name in ("zero_", "mm_fp4")]
    assert order == ["zero_", "mm_fp4"] * 3


def test_the_barrier_is_not_cached_when_the_stream_is_capturing(stub_kernels, monkeypatch):
    """An allocation made inside a capture belongs to the graph's private pool and dies with it."""
    monkeypatch.setattr(ops, "_is_capturing", lambda: True)
    _mm_once(1)
    assert ops._BARRIERS == {}
    # It still fires: an uncached buffer is a cost, a missing barrier is a wrong answer.
    order = [name for name, _ in stub_kernels.launches if name in ("zero_", "mm_fp4")]
    assert order == ["zero_", "mm_fp4"]

    monkeypatch.setattr(ops, "_is_capturing", lambda: False)
    _mm_once(1)
    warm = ops._BARRIERS[1]
    monkeypatch.setattr(ops, "_is_capturing", lambda: True)
    _mm_once(1)
    assert ops._BARRIERS[1] is warm


def test_reset_barriers_drops_them(stub_kernels):
    _mm_once(1)
    assert list(ops._BARRIERS) == [1]
    ops.reset_barriers()
    assert ops._BARRIERS == {}
    _mm_once(1)
    assert list(ops._BARRIERS) == [1]


def test_the_zero_buffer_env_restores_the_full_memset(stub_kernels, monkeypatch):
    monkeypatch.setenv(ops.NVFP4_ZERO_BUFFER_ENV, "1")
    _mm_once(1)
    names = [name for name, _ in stub_kernels.launches]
    # The M x N zeros IS the barrier in this mode: no separate fill, no buffer.
    assert "zeros" in names and "zero_" not in names
    assert ops._BARRIERS == {}


def test_the_barrier_is_not_an_op_argument():
    """A mutable tensor input would go through ``auto_functionalized`` and be CLONED per call."""
    schema = "(Tensor xq, Tensor wq, Tensor x_sf, Tensor w_sf, Tensor alpha, int n, str backend)"
    import inspect

    assert schema.count("Tensor") == 5
    params = list(inspect.signature(ops._mm_impl).parameters)
    assert params == ["xq", "wq", "x_sf", "w_sf", "alpha", "n", "backend"]




def _nvfp4_cuda_or_skip():
    torch = pytest.importorskip("torch")
    if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if tuple(torch.cuda.get_device_capability(0)) not in ops.NVFP4_FLASHINFER_CAPS:
        pytest.skip("this device has no flashinfer NVFP4 kernels")
    pytest.importorskip("flashinfer")
    pytest.importorskip("torchao")
    return torch


def _real_operands(
    torch,
    m,
    k,
    n,
    seed = 0,
):
    import flashinfer

    torch.manual_seed(seed)
    with torch.cuda.device(0):
        x = torch.randn(m, k, device = "cuda", dtype = torch.bfloat16) * 0.05
        w = torch.randn(n, k, device = "cuda", dtype = torch.bfloat16) * 0.02
        a_gsf, w_gsf = ops.global_scale(x), ops.global_scale(w)
        xq, x_sf = flashinfer.nvfp4_quantize(x, a_gsf, do_shuffle = False)
        wq, w_sf = flashinfer.nvfp4_quantize(w, w_gsf, do_shuffle = False)
        alpha = (1.0 / (a_gsf * w_gsf)).float()
    return xq, wq, x_sf, w_sf, alpha


def test_the_persistent_barrier_is_bit_identical_to_the_per_call_one():
    """50 iterations, because the fault the barrier prevents is intermittent by nature."""
    torch = _nvfp4_cuda_or_skip()
    ops.reset_barriers()
    xq, wq, x_sf, w_sf, alpha = _real_operands(torch, 4096, 3072, 12288)

    with torch.inference_mode():
        reference = None
        for _ in range(50):
            out = ops._mm_impl(xq, wq, x_sf, w_sf, alpha, 12288, ops.DEFAULT_MM_BACKEND)
            assert bool(torch.isfinite(out).all())
            if reference is None:
                reference = out.clone()
            else:
                assert torch.equal(out, reference)
        pointer = ops._BARRIERS[0].data_ptr()
        for _ in range(10):
            ops._mm_impl(xq, wq, x_sf, w_sf, alpha, 12288, ops.DEFAULT_MM_BACKEND)
        assert ops._BARRIERS[0].data_ptr() == pointer
    ops.reset_barriers()


def test_the_barrier_pointer_is_stable_across_a_capture_and_replay():
    """The barrier is warmed OUTSIDE the capture and must be the one the replay fires."""
    torch = _nvfp4_cuda_or_skip()
    ops.reset_barriers()
    xq, wq, x_sf, w_sf, alpha = _real_operands(torch, 512, 3072, 3072, seed = 5)

    with torch.cuda.device(0), torch.inference_mode():
        eager = ops._mm_impl(xq, wq, x_sf, w_sf, alpha, 3072, ops.DEFAULT_MM_BACKEND).clone()
        before = ops._BARRIERS[0].data_ptr()
        torch.cuda.synchronize(0)

        pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                ops._mm_impl(xq, wq, x_sf, w_sf, alpha, 3072, ops.DEFAULT_MM_BACKEND)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph, pool = pool):
            captured = ops._mm_impl(xq, wq, x_sf, w_sf, alpha, 3072, ops.DEFAULT_MM_BACKEND)
        assert ops._BARRIERS[0].data_ptr() == before
        for _ in range(5):
            graph.replay()
            torch.cuda.synchronize(0)
            assert torch.equal(captured, eager)
        assert ops._BARRIERS[0].data_ptr() == before
    del graph
    ops.reset_barriers()



BIAS_SHAPES = (
    (16384, 12288),
    (4096, 3072),
    (1024, 12288),
    (333, 4096),
    (1, 512),
    (9304, 3072),
    (7, 1024),
)


def _bias_pair(
    torch,
    m,
    n,
    dtype = None,
    device = "cuda",
):
    dtype = torch.bfloat16 if dtype is None else dtype
    torch.manual_seed(m * 31 + n)
    out = torch.randn(m, n, device = device, dtype = dtype)
    bias = torch.randn(n, device = device, dtype = dtype)
    return out, bias


def test_the_fast_bias_falls_back_without_triton(monkeypatch):
    torch = pytest.importorskip("torch")
    from core.inference import diffusion_nvfp4_bias as fb

    monkeypatch.setattr(fb, "_HAVE_TRITON", False)
    out = torch.zeros(4, 8)
    bias = torch.arange(8, dtype = torch.float32)
    assert fb.fused_bias_add_(out, bias) is out
    assert torch.equal(out[0], bias)


@pytest.mark.parametrize("value", ["0", "off", "FALSE", " no "])
def test_the_env_switch_takes_the_kernel_out_of_the_path(monkeypatch, value):
    pytest.importorskip("torch")
    from core.inference import diffusion_nvfp4_bias as fb

    monkeypatch.setenv(fb.NVFP4_FAST_BIAS_ENV, value)
    assert fb.fast_bias_enabled() is False


@pytest.mark.parametrize("value", ["", "auto", "1", "AUTO"])
def test_auto_and_one_both_leave_it_on(monkeypatch, value):
    pytest.importorskip("torch")
    from core.inference import diffusion_nvfp4_bias as fb

    monkeypatch.setenv(fb.NVFP4_FAST_BIAS_ENV, value)
    assert fb.fast_bias_enabled() is True


def test_the_kernel_declines_a_shape_or_dtype_it_does_not_cover():
    torch = pytest.importorskip("torch")
    from core.inference import diffusion_nvfp4_bias as fb

    if not fb._HAVE_TRITON:
        pytest.skip("needs triton")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    out, bias = _bias_pair(torch, 8, 16)
    assert fb._eligible(out, bias)
    assert not fb._eligible(*_bias_pair(torch, 8, 16, dtype = torch.float32))
    assert not fb._eligible(out.T.contiguous().T, bias)
    assert not fb._eligible(out.cpu(), bias.cpu())
    assert not fb._eligible(out, bias[:8])
    assert not fb._eligible(out, bias.reshape(1, 16))


def test_the_kernel_declines_while_tracing(monkeypatch):
    """Under tracing it falls back to ``add_``, so inductor can keep fusing the bias."""
    torch = pytest.importorskip("torch")
    from core.inference import diffusion_nvfp4_bias as fb

    calls = []
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(fb, "_eligible", lambda *a: calls.append("eligible") or True)
    out = torch.zeros(4, 8)
    fb.fused_bias_add_(out, torch.ones(8))
    assert calls == []  # short-circuited before eligibility was even asked
    assert float(out[0, 0]) == 1.0


@pytest.mark.parametrize("m,n", BIAS_SHAPES)
def test_the_fused_bias_is_bit_identical_to_add_(m, n):
    torch = pytest.importorskip("torch")
    from core.inference import diffusion_nvfp4_bias as fb

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if not fb._HAVE_TRITON:
        pytest.skip("needs triton")

    out, bias = _bias_pair(torch, m, n)
    want = out.clone().add_(bias)
    got = fb.fused_bias_add_(out, bias)
    assert got is out
    # Bit-identical, not close: the accuracy gates compare renders against stored references.
    assert torch.equal(got, want), float((got.float() - want.float()).abs().max())


def test_an_empty_output_is_left_alone():
    torch = pytest.importorskip("torch")
    from core.inference import diffusion_nvfp4_bias as fb

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    out = torch.zeros(0, 32, device = "cuda", dtype = torch.bfloat16)
    assert fb.fused_bias_add_(out, torch.ones(32, device = "cuda", dtype = torch.bfloat16)) is out


def test_m3_the_fused_bias_against_add_at_the_bench_shapes(capsys):
    """M3, reported rather than asserted: a timing threshold in a test file is a flake."""
    torch = pytest.importorskip("torch")
    from core.inference import diffusion_nvfp4_bias as fb

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if not fb._HAVE_TRITON:
        pytest.skip("needs triton")
    import time

    lines = []
    for m, n in ((4096, 12288), (4096, 3072), (4096, 4096), (16384, 12288)):
        out, bias = _bias_pair(torch, m, n)
        row = {}
        for name, fn in (
            ("add_", lambda: out.add_(bias)),
            ("fused", lambda: fb.fused_bias_add_(out, bias)),
        ):
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            best = float("inf")
            for _ in range(5):
                start = time.perf_counter()
                for _ in range(20):
                    fn()
                torch.cuda.synchronize()
                best = min(best, (time.perf_counter() - start) / 20 * 1e3)
            row[name] = best
        lines.append(
            f"  bias {m}x{n}: add_ {row['add_']:.4f} ms, fused {row['fused']:.4f} ms, "
            f"{row['add_'] / row['fused']:.2f}x"
        )
    with capsys.disabled():
        print("\n" + "\n".join(lines))
