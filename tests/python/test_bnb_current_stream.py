# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""CPU/mock behavioral coverage for live streams in direct BNB native calls."""

from __future__ import annotations

import importlib.util
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import triton.language as tl
from packaging.version import Version


REPO_ROOT = Path(__file__).resolve().parents[2]
UTILS_PATH = REPO_ROOT / "unsloth" / "kernels" / "utils.py"

DEVICE_INDEX = 5
CACHED_STREAM = 0xCA11ED
LIVE_STREAMS = (0xA11CE, 0xB0B)


class _Tensor:
    def __init__(
        self,
        shape,
        dtype,
        device = None,
        storage = None,
    ):
        self.shape = (shape,) if isinstance(shape, int) else tuple(shape)
        self.dtype = dtype
        self.device = device or SimpleNamespace(type = "cuda", index = DEVICE_INDEX)
        self.storage = storage if storage is not None else object()

    def numel(self):
        total = 1
        for dimension in self.shape:
            total *= dimension
        return total

    def __iadd__(self, _other):
        return self

    def __getitem__(self, item):
        assert isinstance(item, slice)
        start = 0 if item.start is None else item.start
        stop = self.numel() if item.stop is None else item.stop
        return _Tensor((max(0, stop - start),), self.dtype, self.device, self.storage)

    def resize_(self, size):
        self.shape = (size,) if isinstance(size, int) else tuple(size)
        return self

    def t(self):
        return self

    def view(self, shape):
        return _Tensor(shape, self.dtype, self.device, self.storage)


def _install_module(monkeypatch, name: str, module: ModuleType):
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_utils(
    monkeypatch,
    backend: str,
    stream_values = LIVE_STREAMS,
    cached_stream = CACHED_STREAM,
):
    device_type = "xpu" if backend == "xpu" else backend
    device = SimpleNamespace(type = device_type, index = DEVICE_INDEX)
    controls = SimpleNamespace(
        device = device,
        native_calls = [],
        snapshot_calls = [],
        stream_lookups = [],
        import_phase = True,
        matmul_result = object(),
        fp8_result = object(),
    )
    streams = iter(stream_values)
    expected_route = "xpu" if backend == "xpu" else "cuda"

    def snapshot_getter(route):
        def get_current_raw_stream(index):
            if controls.import_phase:
                controls.snapshot_calls.append((route, index))
                return cached_stream if route == expected_route else CACHED_STREAM + 1
            if route != expected_route:
                raise AssertionError(f"unexpected live stream route: {route}")
            controls.stream_lookups.append(index)
            return next(streams)

        return get_current_raw_stream

    monkeypatch.setattr(
        torch._C,
        "_cuda_getCurrentRawStream",
        snapshot_getter("cuda"),
        raising = False,
    )
    monkeypatch.setattr(
        torch._C,
        "_xpu_getCurrentRawStream",
        snapshot_getter("xpu"),
        raising = False,
    )

    def device_factory(_index):
        return SimpleNamespace(idx = DEVICE_INDEX)

    monkeypatch.setattr(torch.cuda, "device", device_factory)
    monkeypatch.setattr(torch.xpu, "device", device_factory)
    if backend == "xpu" and not hasattr(tl.extra, "intel"):
        monkeypatch.setattr(
            tl.extra,
            "intel",
            SimpleNamespace(libdevice = SimpleNamespace(tanh = lambda value: value)),
            raising = False,
        )

    package_name = f"_bnb_stream_test_{backend}"
    package = ModuleType(package_name)
    package.__path__ = []
    kernels_package = ModuleType(f"{package_name}.kernels")
    kernels_package.__path__ = []
    _install_module(monkeypatch, package_name, package)
    _install_module(monkeypatch, f"{package_name}.kernels", kernels_package)

    device_module = ModuleType(f"{package_name}.device_type")
    device_module.is_hip = lambda: backend == "hip"
    device_module.get_device_type = lambda: device_type
    device_module.DEVICE_TYPE = device_type
    device_module.DEVICE_TYPE_TORCH = device_type
    device_module.DEVICE_COUNT = 0 if backend == "fallback" else 1
    device_module.ALLOW_PREQUANTIZED_MODELS = True
    _install_module(monkeypatch, device_module.__name__, device_module)

    availability_module = ModuleType(f"{package_name}.bnb_availability")
    availability_module.native_kernels_ready = lambda *_args, **_kwargs: True
    _install_module(monkeypatch, availability_module.__name__, availability_module)

    fp8_module = ModuleType(f"{package_name}.kernels.fp8")
    fp8_module.weight_dequant = lambda *_args, **_kwargs: controls.fp8_result
    fp8_module.fp8_linear = lambda *_args, **_kwargs: controls.fp8_result
    _install_module(monkeypatch, fp8_module.__name__, fp8_module)

    zoo_package = ModuleType("unsloth_zoo")
    zoo_package.__path__ = []
    zoo_utils = ModuleType("unsloth_zoo.utils")
    zoo_utils.Version = Version
    _install_module(monkeypatch, "unsloth_zoo", zoo_package)
    _install_module(monkeypatch, "unsloth_zoo.utils", zoo_utils)

    torchao_package = ModuleType("torchao")
    torchao_package.__path__ = []
    torchao_package.__spec__ = ModuleSpec("torchao", loader = None)
    torchao_quantization = ModuleType("torchao.quantization")
    torchao_quantization.__spec__ = ModuleSpec("torchao.quantization", loader = None)
    torchao_quantization.Float8Tensor = type("Float8Tensor", (), {})
    _install_module(monkeypatch, "torchao", torchao_package)
    _install_module(monkeypatch, "torchao.quantization", torchao_quantization)

    def record_native_call(name):
        def native_call(*args):
            controls.native_calls.append((name, args))

        return native_call

    functional_module = ModuleType("bitsandbytes.functional")
    functional_module.get_ptr = lambda value: value
    functional_module.lib = SimpleNamespace()
    for symbol in (
        "cdequantize_blockwise_fp32",
        "cdequantize_blockwise_fp16_nf4",
        "cdequantize_blockwise_bf16_nf4",
        "cgemv_4bit_inference_fp16",
        "cgemv_4bit_inference_bf16",
        "cgemm_4bit_inference_naive_fp16",
        "cgemm_4bit_inference_naive_bf16",
    ):
        setattr(functional_module.lib, symbol, record_native_call(symbol))
    functional_module.__spec__ = ModuleSpec("bitsandbytes.functional", loader = None)
    bnb_package = ModuleType("bitsandbytes")
    bnb_package.__path__ = []
    bnb_package.__version__ = "0.45.0"
    bnb_package.functional = functional_module
    bnb_package.__spec__ = ModuleSpec("bitsandbytes", loader = None)
    _install_module(monkeypatch, "bitsandbytes", bnb_package)
    _install_module(monkeypatch, "bitsandbytes.functional", functional_module)

    module_name = f"{package_name}.kernels.utils"
    spec = importlib.util.spec_from_file_location(module_name, UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    _install_module(monkeypatch, module_name, module)
    spec.loader.exec_module(module)

    def torch_empty(
        shape,
        dtype = None,
        device = None,
        **_kwargs,
    ):
        return _Tensor(shape, dtype, device)

    controls.import_phase = False
    module.torch_empty = torch_empty
    module.torch_matmul = lambda *_args, **_kwargs: controls.matmul_result
    controls.float16 = module.torch_float16
    controls.bfloat16 = module.torch_bfloat16
    controls.fp8_dtype = module.torch.float8_e4m3fn
    return module, controls


def _quant_state(controls, dtype):
    absmax = _Tensor((2,), controls.float16, controls.device)
    absmax2 = _Tensor((1,), controls.float16, controls.device)
    return SimpleNamespace(
        absmax = absmax,
        shape = (4, 6),
        dtype = dtype,
        blocksize = 64,
        code = object(),
        offset = 0.25,
        state2 = SimpleNamespace(absmax = absmax2, code = object(), blocksize = 256),
    )


def _invoke(
    name,
    module,
    controls,
    quant_state,
    use_global_buffer = False,
):
    weight = _Tensor((4, 3), object(), controls.device)
    if name == "fast_dequantize":
        return module.fast_dequantize(weight, quant_state, use_global_buffer = use_global_buffer)
    activation = _Tensor((1, 1, 6), quant_state.dtype, controls.device)
    return module.fast_gemv(activation, weight, quant_state)


def _stream_values(calls):
    return [0 if args[-1].value is None else args[-1].value for _name, args in calls]


@pytest.mark.parametrize("backend", ("cuda", "hip", "xpu"))
@pytest.mark.parametrize("name", ("fast_dequantize", "fast_gemv"))
def test_native_calls_use_the_live_stream_at_each_invocation(monkeypatch, backend, name):
    module, controls = _load_utils(monkeypatch, backend)
    _invoke(name, module, controls, _quant_state(controls, controls.float16))

    assert controls.stream_lookups == [DEVICE_INDEX, DEVICE_INDEX]
    assert _stream_values(controls.native_calls) == list(LIVE_STREAMS)
    assert CACHED_STREAM not in _stream_values(controls.native_calls)


@pytest.mark.parametrize(
    ("backend", "expected_route"),
    (("cuda", "cuda"), ("hip", "cuda"), ("xpu", "xpu")),
)
def test_backend_current_stream_getter_route(monkeypatch, backend, expected_route):
    _module, controls = _load_utils(monkeypatch, backend, stream_values = ())

    assert controls.snapshot_calls
    assert {route for route, _index in controls.snapshot_calls} == {expected_route}
    assert {index for _route, index in controls.snapshot_calls} == {DEVICE_INDEX}


@pytest.mark.parametrize("backend", ("cuda", "hip", "xpu"))
@pytest.mark.parametrize("owner_stream", (LIVE_STREAMS[0], 0))
def test_owner_stream_reuses_per_device_scratch(monkeypatch, backend, owner_stream):
    module, controls = _load_utils(
        monkeypatch,
        backend,
        stream_values = (owner_stream,) * 6,
        cached_stream = owner_stream,
    )
    state = _quant_state(controls, controls.float16)

    first = _invoke("fast_dequantize", module, controls, state, use_global_buffer = True)
    second = _invoke("fast_dequantize", module, controls, state, use_global_buffer = True)

    nested_calls = [controls.native_calls[index] for index in (0, 2)]
    assert first.storage is second.storage
    assert nested_calls[0][1][3].storage is nested_calls[1][1][3].storage
    assert module.WEIGHT_BUFFERS[DEVICE_INDEX].storage is first.storage
    assert module.ABSMAX_BUFFERS[DEVICE_INDEX].storage is nested_calls[0][1][3].storage
    assert _stream_values(controls.native_calls) == [owner_stream] * 4


@pytest.mark.parametrize("backend", ("cuda", "hip", "xpu"))
def test_nonowner_streams_use_call_local_scratch(monkeypatch, backend):
    owner_stream = LIVE_STREAMS[0]
    alternate_a, alternate_b = LIVE_STREAMS[1], CACHED_STREAM
    module, controls = _load_utils(
        monkeypatch,
        backend,
        stream_values = (alternate_a,) * 6 + (alternate_b,) * 3,
        cached_stream = owner_stream,
    )
    state = _quant_state(controls, controls.float16)

    outputs = [
        _invoke("fast_dequantize", module, controls, state, use_global_buffer = True)
        for _ in range(3)
    ]
    nested_scratch = [controls.native_calls[index][1][3] for index in (0, 2, 4)]
    final_outputs = [controls.native_calls[index][1][3] for index in (1, 3, 5)]

    assert len({output.storage for output in outputs}) == 3
    assert len({scratch.storage for scratch in nested_scratch}) == 3
    assert len({output.storage for output in final_outputs}) == 3
    assert module.WEIGHT_BUFFERS[DEVICE_INDEX] is None
    assert module.ABSMAX_BUFFERS[DEVICE_INDEX] is None
    assert _stream_values(controls.native_calls) == [alternate_a] * 4 + [alternate_b] * 2


@pytest.mark.parametrize("backend", ("cuda", "hip", "xpu"))
def test_cached_snapshot_only_gates_scratch_eligibility(monkeypatch, backend):
    owner_stream = LIVE_STREAMS[0]
    nested_stream, final_stream = LIVE_STREAMS[1], CACHED_STREAM
    module, controls = _load_utils(
        monkeypatch,
        backend,
        stream_values = (owner_stream, nested_stream, final_stream),
        cached_stream = owner_stream,
    )

    _invoke(
        "fast_dequantize",
        module,
        controls,
        _quant_state(controls, controls.float16),
        use_global_buffer = True,
    )

    assert module.WEIGHT_BUFFERS[DEVICE_INDEX] is not None
    assert module.ABSMAX_BUFFERS[DEVICE_INDEX] is not None
    assert _stream_values(controls.native_calls) == [nested_stream, final_stream]


@pytest.mark.parametrize("backend", ("cuda", "hip", "xpu"))
def test_early_returns_do_not_query_a_stream(monkeypatch, backend):
    module, controls = _load_utils(monkeypatch, backend, stream_values = ())
    weight = _Tensor((4, 3), object(), controls.device)
    activation = _Tensor((1, 1, 6), controls.float16, controls.device)

    assert module.fast_dequantize(weight, None) is weight
    assert module.fast_gemv(activation, weight, None) is controls.matmul_result
    weight.dtype = controls.fp8_dtype
    assert module.fast_dequantize(weight, object()) is controls.fp8_result
    assert controls.stream_lookups == []
    assert controls.native_calls == []
