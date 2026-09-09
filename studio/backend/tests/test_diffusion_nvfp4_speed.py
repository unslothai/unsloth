# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Kernel-side tests for the NVFP4 flashinfer path: the device guard and the ordering barrier.

The hermetic half runs against a stubbed ``torch`` and a stubbed ``flashinfer``, because the two
properties under test are not observable from outside an opaque custom op on a single-GPU host:

* **which device each launch sees.** ``torch.cuda.device`` is a push/pop of a process-wide value,
  so a stub that records the value at every stubbed FlashInfer call answers "did this launch reach
  the tensor's card" exactly, with the tensors on device 1 and the current device 0, on a machine
  with one GPU and no FlashInfer at all. The real thing costs a card when it is wrong.
* **the call ORDER inside the GEMM op.** The ordering barrier only works if a kernel fires between
  the activation quantiser and the GEMM; a stub call log is what makes "before" a testable word.

The CUDA-gated half is the same two claims against real hardware, and skips itself when the host
cannot host them (one visible card, no FlashInfer, not a Blackwell).
"""

from __future__ import annotations

import contextlib
import sys
import types

import pytest

from core.inference import diffusion_nvfp4_linear as nl
from core.inference import diffusion_nvfp4_ops as ops


# ── the stub: just enough torch and flashinfer to record devices and call order ────────────────


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
    """A tensor-shaped object that answers every call the NVFP4 module bodies make of one.

    Every arithmetic and reshaping method returns a tensor on the SAME device, which is the only
    property any assertion here reads.
    """

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
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "flashinfer", _fake_flashinfer())
    monkeypatch.setattr(ops, "register_ops", lambda: None)
    monkeypatch.setattr(nl, "register_ops", lambda: None)
    ops.reset_preflight_cache()
    yield _RECORDER
    _RECORDER.reset()
    ops.reset_preflight_cache()


def _launch_devices(recorder, *names) -> list[int]:
    return [device for name, device in recorder.launches if name in names]


# ── T-GUARD-1: every launch sees the tensor's own device, never the current one ────────────────


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


# ── T-CUDA-8: the same claim on real cards ────────────────────────────────────────────────────


def test_a_layer_on_card_one_runs_correctly_while_the_current_device_is_card_zero():
    """The bug this whole guard exists for, reproduced as an assertion.

    Unguarded, FlashInfer's cutlass FP4 GEMM takes the stream from the tensor and launches it
    against whatever context is current; on this host that left three cards in "GPU requires
    reset". Guarded, the answer is bit-identical to running with the device already current.
    """
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

    # The hypothesis: current device 0, tensors on card 1. Only the layer's own guard saves this.
    assert torch.cuda.current_device() == 0
    with torch.inference_mode():
        got = converted(x)
    torch.cuda.synchronize(1)
    assert bool(torch.isfinite(got).all())
    assert torch.equal(got, want)
