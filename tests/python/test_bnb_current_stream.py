# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""CPU/mock coverage for live PyTorch streams in direct BNB native calls."""

from __future__ import annotations

import ast
import contextlib
import copy
import ctypes
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
UTILS_PATH = REPO_ROOT / "unsloth" / "kernels" / "utils.py"
UTILS_SOURCE = UTILS_PATH.read_text(encoding = "utf-8")
UTILS_TREE = ast.parse(UTILS_SOURCE)

DEVICE_INDEX = 5
CACHED_STREAM = 0xCA11ED
LIVE_STREAMS = (0xA11CE, 0xB0B)


class _Float8Tensor:
    pass


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


def _production_functions(name: str) -> dict[str, ast.FunctionDef]:
    functions = sorted(
        (
            node
            for node in ast.walk(UTILS_TREE)
            if isinstance(node, ast.FunctionDef) and node.name == name
        ),
        key = lambda node: node.lineno,
    )
    assert len(functions) == 3, f"expected XPU, CUDA/HIP, and fallback {name} variants"
    return dict(zip(("xpu", "cuda", "fallback"), functions))


def _load_function(name: str, backend: str, namespace: dict):
    variant = "xpu" if backend == "xpu" else "fallback" if backend == "fallback" else "cuda"
    function = copy.deepcopy(_production_functions(name)[variant])
    function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body = [function], type_ignores = []))
    exec(compile(module, str(UTILS_PATH), "exec"), namespace)
    return namespace[name]


def _mock_namespace(
    backend: str,
    stream_values = LIVE_STREAMS,
    cached_stream = CACHED_STREAM,
):
    device_type = "xpu" if backend == "xpu" else "cpu" if backend == "fallback" else backend
    device = SimpleNamespace(type = device_type, index = DEVICE_INDEX)
    fp8_dtype = object()
    float16 = object()
    bfloat16 = object()
    native_calls = []
    stream_lookups = []
    device_contexts = []
    matmul_result = object()
    fp8_result = object()
    streams = iter(stream_values)

    def record(name):
        def native_call(*args):
            native_calls.append((name, args))

        return native_call

    def get_tensor_stream(tensor):
        stream_lookups.append(tensor.device.index)
        return ctypes.c_void_p(next(streams))

    def torch_empty(
        shape,
        dtype = None,
        device = None,
        **_kwargs,
    ):
        return _Tensor(shape, dtype, device)

    @contextlib.contextmanager
    def torch_gpu_device(selected_device):
        assert selected_device.type == device.type
        device_contexts.append(selected_device.index)
        yield

    def torch_matmul(*_args, **_kwargs):
        return matmul_result

    def weight_dequant(*_args, **_kwargs):
        return fp8_result

    namespace = {
        "DEVICE_TYPE": device_type,
        "Float8Tensor": _Float8Tensor,
        "torch": SimpleNamespace(float8_e4m3fn = fp8_dtype, float32 = object()),
        "torch_float16": float16,
        "torch_bfloat16": bfloat16,
        "torch_float32": object(),
        "torch_empty": torch_empty,
        "torch_gpu_device": torch_gpu_device,
        "torch_matmul": torch_matmul,
        "weight_dequant": weight_dequant,
        "get_ptr": lambda value: value,
        "ctypes_c_int": ctypes.c_int,
        "ctypes_c_int32": ctypes.c_int32,
        "_get_tensor_stream": get_tensor_stream,
        "cdequantize_blockwise_fp32": record("nested_absmax"),
        "cdequantize_blockwise_fp16_nf4": record("nf4_fp16"),
        "cdequantize_blockwise_bf16_nf4": record("nf4_bf16"),
        "cgemm_4bit_inference_naive_fp16": record("gemv_fp16"),
        "cgemm_4bit_inference_naive_bf16": record("gemv_bf16"),
        # Snapshots gate scratch ownership but must not select native execution streams.
        "CUDA_STREAMS": tuple(ctypes.c_void_p(cached_stream) for _ in range(DEVICE_INDEX + 2)),
        "XPU_STREAMS": tuple(ctypes.c_void_p(cached_stream) for _ in range(DEVICE_INDEX + 2)),
        "WEIGHT_BUFFERS": [None] * (DEVICE_INDEX + 2),
        "ABSMAX_BUFFERS": [None] * (DEVICE_INDEX + 2),
    }
    controls = SimpleNamespace(
        device = device,
        fp8_dtype = fp8_dtype,
        float16 = float16,
        bfloat16 = bfloat16,
        native_calls = native_calls,
        stream_lookups = stream_lookups,
        device_contexts = device_contexts,
        matmul_result = matmul_result,
        fp8_result = fp8_result,
    )
    return namespace, controls


def _quant_state(controls, representation: str, dtype):
    absmax = _Tensor((2,), controls.float16, controls.device)
    absmax2 = _Tensor((1,), controls.float16, controls.device)
    code2 = object()
    stats = object()
    shape = (4, 6)
    blocksize = 64
    blocksize2 = 256
    offset = 0.25
    if representation == "object":
        state2 = SimpleNamespace(absmax = absmax2, code = code2, blocksize = blocksize2)
        return SimpleNamespace(
            absmax = absmax,
            shape = shape,
            dtype = dtype,
            blocksize = blocksize,
            code = stats,
            offset = offset,
            state2 = state2,
        )
    state2 = [absmax2, code2, blocksize2, None, None, None, None]
    return [absmax, shape, dtype, blocksize, [offset, state2], "nf4", stats]


def _invoke(
    name: str,
    function,
    controls,
    quant_state,
    use_global_buffer = False,
    device = None,
):
    device = controls.device if device is None else device
    weight = _Tensor((4, 3), object(), device)
    if name == "fast_dequantize":
        return function(weight, quant_state, use_global_buffer = use_global_buffer)
    activation = _Tensor(
        (1, 1, 6),
        quant_state.dtype if hasattr(quant_state, "dtype") else quant_state[2],
        controls.device,
    )
    return function(activation, weight, quant_state)


def _stream_values(calls):
    return [args[-1].value for _name, args in calls]


def test_get_tensor_stream_invokes_the_bound_getter_at_call_time():
    helper = copy.deepcopy(
        next(
            node
            for node in UTILS_TREE.body
            if isinstance(node, ast.FunctionDef) and node.name == "_get_tensor_stream"
        )
    )
    values = iter(LIVE_STREAMS)
    device_indices = []

    def current_raw_stream(device_index):
        device_indices.append(device_index)
        return next(values)

    namespace = {
        "torch_Tensor": object,
        "c_void_p": ctypes.c_void_p,
        "_gpu_getCurrentRawStream": current_raw_stream,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body = [helper], type_ignores = [])),
            str(UTILS_PATH),
            "exec",
        ),
        namespace,
    )
    tensor = _Tensor((1,), object(), SimpleNamespace(type = "cuda", index = DEVICE_INDEX))

    assert namespace["_get_tensor_stream"](tensor).value == LIVE_STREAMS[0]
    assert namespace["_get_tensor_stream"](tensor).value == LIVE_STREAMS[1]
    assert device_indices == [DEVICE_INDEX, DEVICE_INDEX]


@pytest.mark.parametrize("backend", ("cuda", "hip", "xpu"))
@pytest.mark.parametrize("name", ("fast_dequantize", "fast_gemv"))
def test_each_native_call_observes_its_own_live_stream(backend, name):
    namespace, controls = _mock_namespace(backend)
    function = _load_function(name, backend, namespace)
    state = _quant_state(controls, "object", controls.float16)

    _invoke(name, function, controls, state)

    assert controls.stream_lookups == [DEVICE_INDEX, DEVICE_INDEX]
    assert _stream_values(controls.native_calls) == list(LIVE_STREAMS)
    assert CACHED_STREAM not in _stream_values(controls.native_calls)


@pytest.mark.parametrize("backend", ("cuda", "hip", "xpu"))
@pytest.mark.parametrize("name", ("fast_dequantize", "fast_gemv"))
@pytest.mark.parametrize("representation", ("object", "list"))
@pytest.mark.parametrize("dtype_name", ("float16", "bfloat16"))
def test_quant_state_formats_and_dtype_specific_native_symbols(
    backend, name, representation, dtype_name
):
    namespace, controls = _mock_namespace(backend)
    function = _load_function(name, backend, namespace)
    dtype = getattr(controls, dtype_name)

    _invoke(name, function, controls, _quant_state(controls, representation, dtype))

    suffix = "fp16" if dtype_name == "float16" else "bf16"
    final_name = f"nf4_{suffix}" if name == "fast_dequantize" else f"gemv_{suffix}"
    assert [call_name for call_name, _args in controls.native_calls] == [
        "nested_absmax",
        final_name,
    ]
    assert _stream_values(controls.native_calls) == list(LIVE_STREAMS)


@pytest.mark.parametrize("backend", ("cuda", "hip", "xpu"))
def test_global_scratch_reuses_same_device_and_stream(backend):
    owner_stream = LIVE_STREAMS[0]
    namespace, controls = _mock_namespace(
        backend,
        (owner_stream,) * 6,
        cached_stream = owner_stream,
    )
    function = _load_function("fast_dequantize", backend, namespace)
    state = _quant_state(controls, "object", controls.float16)

    first = _invoke("fast_dequantize", function, controls, state, use_global_buffer = True)
    second = _invoke("fast_dequantize", function, controls, state, use_global_buffer = True)

    assert first.storage is second.storage
    nested_calls = [call for call in controls.native_calls if call[0] == "nested_absmax"]
    assert nested_calls[0][1][3].storage is nested_calls[1][1][3].storage
    assert namespace["WEIGHT_BUFFERS"][DEVICE_INDEX].storage is first.storage
    assert namespace["ABSMAX_BUFFERS"][DEVICE_INDEX].storage is nested_calls[0][1][3].storage
    assert _stream_values(controls.native_calls) == [owner_stream] * 4


@pytest.mark.parametrize("backend", ("cuda", "hip", "xpu"))
def test_cached_snapshot_only_gates_scratch_not_native_stream_selection(backend):
    owner_stream = LIVE_STREAMS[0]
    nested_stream = LIVE_STREAMS[1]
    final_stream = CACHED_STREAM
    namespace, controls = _mock_namespace(
        backend,
        (owner_stream, nested_stream, final_stream),
        cached_stream = owner_stream,
    )
    function = _load_function("fast_dequantize", backend, namespace)
    state = _quant_state(controls, "object", controls.float16)

    _invoke("fast_dequantize", function, controls, state, use_global_buffer = True)

    assert namespace["WEIGHT_BUFFERS"][DEVICE_INDEX] is not None
    assert namespace["ABSMAX_BUFFERS"][DEVICE_INDEX] is not None
    assert _stream_values(controls.native_calls) == [nested_stream, final_stream]


@pytest.mark.parametrize("backend", ("cuda", "hip", "xpu"))
def test_global_scratch_is_bypassed_for_nonowning_streams(backend):
    stream_a, stream_b = LIVE_STREAMS
    namespace, controls = _mock_namespace(
        backend,
        (stream_a, stream_a, stream_a, stream_b, stream_b, stream_b),
        cached_stream = CACHED_STREAM,
    )
    function = _load_function("fast_dequantize", backend, namespace)
    state = _quant_state(controls, "object", controls.float16)

    first = _invoke("fast_dequantize", function, controls, state, use_global_buffer = True)
    second = _invoke("fast_dequantize", function, controls, state, use_global_buffer = True)

    assert first.storage is not second.storage
    nested_calls = [call for call in controls.native_calls if call[0] == "nested_absmax"]
    assert nested_calls[0][1][3].storage is not nested_calls[1][1][3].storage
    final_calls = [call for call in controls.native_calls if call[0].startswith("nf4_")]
    assert final_calls[0][1][3].storage is not final_calls[1][1][3].storage
    assert namespace["WEIGHT_BUFFERS"][DEVICE_INDEX] is None
    assert namespace["ABSMAX_BUFFERS"][DEVICE_INDEX] is None
    assert _stream_values(controls.native_calls) == [stream_a, stream_a, stream_b, stream_b]


@pytest.mark.parametrize("backend", ("cuda", "hip", "xpu"))
def test_global_scratch_isolated_by_device_even_for_same_stream(backend):
    stream = LIVE_STREAMS[0]
    namespace, controls = _mock_namespace(backend, (stream,) * 6, cached_stream = stream)
    function = _load_function("fast_dequantize", backend, namespace)
    state = _quant_state(controls, "object", controls.float16)
    other_device = SimpleNamespace(type = controls.device.type, index = DEVICE_INDEX + 1)

    first = _invoke("fast_dequantize", function, controls, state, use_global_buffer = True)
    second = _invoke(
        "fast_dequantize",
        function,
        controls,
        state,
        use_global_buffer = True,
        device = other_device,
    )

    assert first.storage is not second.storage
    nested_calls = [call for call in controls.native_calls if call[0] == "nested_absmax"]
    assert nested_calls[0][1][3].storage is not nested_calls[1][1][3].storage
    assert (
        namespace["WEIGHT_BUFFERS"][DEVICE_INDEX].storage
        is not namespace["WEIGHT_BUFFERS"][DEVICE_INDEX + 1].storage
    )
    assert (
        namespace["ABSMAX_BUFFERS"][DEVICE_INDEX].storage
        is not namespace["ABSMAX_BUFFERS"][DEVICE_INDEX + 1].storage
    )
    assert controls.stream_lookups == [DEVICE_INDEX] * 3 + [DEVICE_INDEX + 1] * 3
    assert controls.device_contexts == [DEVICE_INDEX, DEVICE_INDEX + 1]


@pytest.mark.parametrize("backend", ("cuda", "hip", "xpu"))
def test_default_stream_zero_is_a_reusable_scratch_key(backend):
    namespace, controls = _mock_namespace(backend, (0,) * 6, cached_stream = 0)
    function = _load_function("fast_dequantize", backend, namespace)
    state = _quant_state(controls, "object", controls.bfloat16)

    first = _invoke("fast_dequantize", function, controls, state, use_global_buffer = True)
    second = _invoke("fast_dequantize", function, controls, state, use_global_buffer = True)

    assert first.storage is second.storage
    assert namespace["WEIGHT_BUFFERS"][DEVICE_INDEX].storage is first.storage
    nested_calls = [call for call in controls.native_calls if call[0] == "nested_absmax"]
    assert namespace["ABSMAX_BUFFERS"][DEVICE_INDEX].storage is nested_calls[0][1][3].storage


@pytest.mark.parametrize("backend", ("cuda", "hip", "xpu"))
def test_gemv_preserves_backend_specific_dimension_abi(backend):
    namespace, controls = _mock_namespace(backend)
    function = _load_function("fast_gemv", backend, namespace)

    _invoke("fast_gemv", function, controls, _quant_state(controls, "object", controls.float16))

    _call_name, args = controls.native_calls[-1]
    expected_mn = (1, 4) if backend == "xpu" else (4, 1)
    assert (args[0].value, args[1].value) == expected_mn
    assert len(args) == 13


@pytest.mark.parametrize("backend", ("cuda", "hip", "xpu"))
def test_unquantized_and_fp8_early_returns_do_not_look_up_a_stream(backend):
    namespace, controls = _mock_namespace(backend, ())
    dequantize = _load_function("fast_dequantize", backend, namespace)
    gemv = _load_function("fast_gemv", backend, namespace)
    weight = _Tensor((4, 3), object(), controls.device)
    activation = _Tensor((1, 1, 6), controls.float16, controls.device)

    assert dequantize(weight, None) is weight
    assert gemv(activation, weight, None) is controls.matmul_result

    weight.dtype = controls.fp8_dtype
    assert dequantize(weight, object()) is controls.fp8_result
    assert controls.stream_lookups == []
    assert controls.native_calls == []


@pytest.mark.parametrize("name", ("fast_dequantize", "fast_gemv"))
def test_legacy_no_stream_fallback_keeps_native_abi_without_stream_argument(name):
    namespace, controls = _mock_namespace("fallback", ())
    function = _load_function(name, "fallback", namespace)

    _invoke(name, function, controls, _quant_state(controls, "list", controls.bfloat16))

    assert controls.stream_lookups == []
    assert [len(args) for _call_name, args in controls.native_calls] == (
        [6, 6] if name == "fast_dequantize" else [6, 12]
    )


@pytest.mark.parametrize(
    ("backend", "expected_getter"),
    (("cuda", "cuda"), ("hip", "cuda"), ("xpu", "xpu")),
)
def test_backend_binds_the_expected_current_stream_getter(backend, expected_getter):
    binding = next(
        node
        for node in UTILS_TREE.body
        if isinstance(node, ast.If)
        and any(
            isinstance(target, ast.Name) and target.id == "_gpu_getCurrentRawStream"
            for assignment in ast.walk(node)
            if isinstance(assignment, ast.Assign)
            for target in assignment.targets
        )
    )
    cuda_getter = object()
    xpu_getter = object()
    namespace = {
        "DEVICE_TYPE": backend,
        "torch": SimpleNamespace(
            _C = SimpleNamespace(
                _cuda_getCurrentRawStream = cuda_getter,
                _xpu_getCurrentRawStream = xpu_getter,
            )
        ),
    }

    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body = [copy.deepcopy(binding)], type_ignores = [])),
            str(UTILS_PATH),
            "exec",
        ),
        namespace,
    )

    expected = cuda_getter if expected_getter == "cuda" else xpu_getter
    assert namespace["_gpu_getCurrentRawStream"] is expected


def test_xpu_and_cuda_hip_native_symbol_bindings_remain_distinct():
    assert "bnb_functional.lib.cgemv_4bit_inference_fp16" in UTILS_SOURCE
    assert "bnb_functional.lib.cgemv_4bit_inference_bf16" in UTILS_SOURCE
    assert "bnb_functional.lib.cgemm_4bit_inference_naive_fp16" in UTILS_SOURCE
    assert "bnb_functional.lib.cgemm_4bit_inference_naive_bf16" in UTILS_SOURCE


def test_accelerator_paths_ignore_cached_snapshots_and_add_no_coordination():
    assert "CUDA_STREAMS = tuple(CUDA_STREAMS)" in UTILS_SOURCE
    assert "XPU_STREAMS = tuple(XPU_STREAMS)" in UTILS_SOURCE
    assert UTILS_SOURCE.count("WEIGHT_BUFFERS = [None] *") == 2
    assert UTILS_SOURCE.count("ABSMAX_BUFFERS = [None] *") == 2
    assert ".get(stream_key)" not in UTILS_SOURCE
    forbidden_calls = {
        "Event",
        "record_event",
        "record_stream",
        "synchronize",
        "wait_event",
        "wait_stream",
        "Lock",
        "RLock",
        "sleep",
    }
    for name in ("fast_dequantize", "fast_gemv"):
        for backend in ("xpu", "cuda"):
            function = _production_functions(name)[backend]
            names = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}
            attributes = {
                node.attr for node in ast.walk(function) if isinstance(node, ast.Attribute)
            }
            calls = {
                node.func.id
                for node in ast.walk(function)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            }
            native_calls = [
                node
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id
                in {
                    "cdequantize_blockwise_fp32",
                    "cdequantize_blockwise_fp16_nf4",
                    "cdequantize_blockwise_bf16_nf4",
                    "fx",
                }
            ]
            assert len(native_calls) == 2
            for native_call in native_calls:
                stream_argument = native_call.args[-1]
                assert isinstance(stream_argument, ast.Call)
                assert isinstance(stream_argument.func, ast.Name)
                assert stream_argument.func.id == "_get_tensor_stream"
                assert len(stream_argument.args) == 1
                assert isinstance(stream_argument.args[0], ast.Name)
                assert stream_argument.args[0].id == "W"
                native_names = {
                    node.id for node in ast.walk(native_call) if isinstance(node, ast.Name)
                }
                assert native_names.isdisjoint({"CUDA_STREAMS", "XPU_STREAMS"})
            if name == "fast_dequantize":
                expected_snapshot = "XPU_STREAMS" if backend == "xpu" else "CUDA_STREAMS"
                other_snapshot = "CUDA_STREAMS" if backend == "xpu" else "XPU_STREAMS"
                assert expected_snapshot in names
                assert other_snapshot not in names
            else:
                assert names.isdisjoint({"CUDA_STREAMS", "XPU_STREAMS"})
            assert attributes.isdisjoint(forbidden_calls)
            assert calls.isdisjoint(forbidden_calls)
