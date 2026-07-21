from __future__ import annotations

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
FP8_SOURCE = REPO_ROOT / "unsloth" / "kernels" / "fp8.py"


class _FakeDeviceModule:
    def __init__(self, device_count: int) -> None:
        self._device_count = device_count
        self.device_calls = []

    def device_count(self) -> int:
        return self._device_count

    def device(self, device):
        self.device_calls.append(device)
        return ("device-context", device)


class _FakeTorch:
    Tensor = object

    def __init__(
        self,
        cuda_device_count: int,
        xpu_device_count: int = 0,
    ) -> None:
        self.cuda = _FakeDeviceModule(cuda_device_count)
        self.xpu = _FakeDeviceModule(xpu_device_count)


class _LaunchVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.guarded_launches: set[str] = set()
        self.unguarded_launches: set[str] = set()
        self._inside_fp8_device_context = 0

    def visit_With(self, node: ast.With) -> None:
        enters_context = any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Name)
            and item.context_expr.func.id == "_fp8_triton_device_context"
            for item in node.items
        )
        if enters_context:
            self._inside_fp8_device_context += 1
        for statement in node.body:
            self.visit(statement)
        if enters_context:
            self._inside_fp8_device_context -= 1

    def visit_Call(self, node: ast.Call) -> None:
        launch_name = self._triton_launch_name(node)
        if launch_name is not None:
            if self._inside_fp8_device_context:
                self.guarded_launches.add(launch_name)
            else:
                self.unguarded_launches.add(launch_name)
        self.generic_visit(node)

    @staticmethod
    def _triton_launch_name(node: ast.Call) -> str | None:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "triton_quantize_fp8_block"
        ):
            return node.func.id
        if not isinstance(node.func, ast.Subscript):
            return None
        if not isinstance(node.func.value, ast.Name):
            return None
        return node.func.value.id


def _load_device_context_helper(fake_torch: _FakeTorch):
    source = FP8_SOURCE.read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "_fp8_triton_device_context"
        ):
            namespace = {"torch": fake_torch, "nullcontext": nullcontext}
            exec(ast.get_source_segment(source, node), namespace)
            return namespace["_fp8_triton_device_context"]
    raise AssertionError("_fp8_triton_device_context was not found")


def test_fp8_device_context_selects_cuda_tensor_device_on_multi_gpu() -> None:
    fake_torch = _FakeTorch(cuda_device_count = 2)
    helper = _load_device_context_helper(fake_torch)
    tensor = SimpleNamespace(device = SimpleNamespace(type = "cuda"))

    context = helper(tensor)

    assert context == ("device-context", tensor.device)
    assert fake_torch.cuda.device_calls == [tensor.device]


def test_fp8_device_context_is_noop_for_single_cuda_device() -> None:
    fake_torch = _FakeTorch(cuda_device_count = 1)
    helper = _load_device_context_helper(fake_torch)
    tensor = SimpleNamespace(device = SimpleNamespace(type = "cuda"))

    context = helper(tensor)

    assert isinstance(context, nullcontext)
    assert fake_torch.cuda.device_calls == []


def test_fp8_device_context_selects_xpu_tensor_device_on_multi_gpu() -> None:
    fake_torch = _FakeTorch(cuda_device_count = 0, xpu_device_count = 2)
    helper = _load_device_context_helper(fake_torch)
    tensor = SimpleNamespace(device = SimpleNamespace(type = "xpu"))

    context = helper(tensor)

    assert context == ("device-context", tensor.device)
    assert fake_torch.xpu.device_calls == [tensor.device]


def test_fp8_device_context_is_noop_for_single_xpu_device() -> None:
    fake_torch = _FakeTorch(cuda_device_count = 0, xpu_device_count = 1)
    helper = _load_device_context_helper(fake_torch)
    tensor = SimpleNamespace(device = SimpleNamespace(type = "xpu"))

    context = helper(tensor)

    assert isinstance(context, nullcontext)
    assert fake_torch.xpu.device_calls == []


def test_fp8_device_context_is_noop_for_non_cuda_tensor() -> None:
    fake_torch = _FakeTorch(cuda_device_count = 8)
    helper = _load_device_context_helper(fake_torch)
    tensor = SimpleNamespace(device = SimpleNamespace(type = "cpu"))

    context = helper(tensor)

    assert isinstance(context, nullcontext)
    assert fake_torch.cuda.device_calls == []


def test_fp8_triton_launches_enter_tensor_device_context() -> None:
    tree = ast.parse(FP8_SOURCE.read_text())
    function_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    assert "_fp8_triton_device_context" in function_names

    visitor = _LaunchVisitor()
    visitor.visit(tree)

    expected_launches = {
        "weight_dequant_kernel",
        "act_quant_kernel",
        "_w8a8_block_fp8_matmul",
        "triton_quantize_fp8_block",
    }
    assert expected_launches <= visitor.guarded_launches
    assert not (expected_launches & visitor.unguarded_launches)


def _require_two_cuda_devices():
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires at least two CUDA devices")
    return torch


def test_weight_dequant_block_runs_on_tensor_device_when_current_device_differs() -> (
    None
):
    torch = _require_two_cuda_devices()
    from unsloth.kernels.fp8 import weight_dequant_block

    previous_device = torch.cuda.current_device()
    try:
        torch.cuda.set_device(0)
        x = torch.arange(256 * 256, device = "cuda:1", dtype = torch.float32).reshape(
            256, 256
        )
        scales = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], device = "cuda:1", dtype = torch.float32
        )

        actual = weight_dequant_block(x, scales, block_size = 128, dtype = torch.float32)

        expanded_scales = scales.repeat_interleave(128, dim = 0).repeat_interleave(
            128, dim = 1
        )
        expected = x * expanded_scales

        assert actual.device == x.device
        assert torch.cuda.current_device() == 0
        torch.testing.assert_close(actual, expected)
    finally:
        torch.cuda.set_device(previous_device)


def test_act_quant_runs_on_tensor_device_when_current_device_differs() -> None:
    torch = _require_two_cuda_devices()
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("requires torch.float8_e4m3fn")
    if torch.cuda.get_device_capability(1)[0] < 9:
        pytest.skip("requires FP8-capable CUDA hardware")

    from unsloth.kernels.fp8 import act_quant

    previous_device = torch.cuda.current_device()
    try:
        torch.cuda.set_device(0)
        x = torch.arange(256, device = "cuda:1", dtype = torch.float32).reshape(2, 128)

        y, scales = act_quant(x, block_size = 128)

        assert y.device == x.device
        assert scales.device == x.device
        assert torch.cuda.current_device() == 0
    finally:
        torch.cuda.set_device(previous_device)


def test_w8a8_block_fp8_matmul_triton_runs_on_tensor_device_when_current_device_differs() -> (
    None
):
    torch = _require_two_cuda_devices()
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("requires torch.float8_e4m3fn")
    if torch.cuda.get_device_capability(1)[0] < 9:
        pytest.skip("requires FP8-capable CUDA hardware")

    from unsloth.kernels.fp8 import w8a8_block_fp8_matmul_triton

    previous_device = torch.cuda.current_device()
    try:
        torch.cuda.set_device(0)
        A = torch.ones((128, 128), device = "cuda:1", dtype = torch.float32).to(
            torch.float8_e4m3fn
        )
        B = torch.ones((128, 128), device = "cuda:1", dtype = torch.float32).to(
            torch.float8_e4m3fn
        )
        As = torch.ones((128, 1), device = "cuda:1", dtype = torch.float32)
        Bs = torch.ones((1, 1), device = "cuda:1", dtype = torch.float32)

        actual = w8a8_block_fp8_matmul_triton(
            A,
            B,
            As,
            Bs,
            block_size = [128, 128],
            output_dtype = torch.float32,
        )

        expected = torch.full((128, 128), 128.0, device = "cuda:1", dtype = torch.float32)
        assert actual.device == A.device
        assert torch.cuda.current_device() == 0
        torch.testing.assert_close(actual, expected)
    finally:
        torch.cuda.set_device(previous_device)
