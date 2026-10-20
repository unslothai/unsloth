# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TMA support is process-wide, so on H100 (cuda:0) + A100 (cuda:1) it must hold on every visible
card, not just the current one; launches and Triton scratch must follow the operands' device."""

import ast
import contextlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
import torch._dynamo  # noqa: E402,F401  # import before custom_op is stubbed out below
from torch._subclasses.fake_tensor import FakeTensorMode  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GG_DIR = ROOT / "unsloth" / "kernels" / "moe" / "grouped_gemm"
PKG = "_unsloth_gg_spoof"


def _cuda_visible_capabilities():
    """The real unsloth.device_type helper, without importing the GPU-only package."""
    source = (ROOT / "unsloth" / "device_type.py").read_text(encoding = "utf-8")
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "cuda_visible_capabilities"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module([node], []), "device_type.py", "exec"), namespace)
    return namespace["cuda_visible_capabilities"]


class _Kernel:
    def __init__(self, launches, guard_stack):
        self.launches, self.guard_stack, self.best_config = launches, guard_stack, None

    def __getitem__(self, grid):
        def launch(**kwargs):
            self.launches.append(self.guard_stack[-1] if self.guard_stack else None)
            return types.SimpleNamespace(name = "k", n_regs = 0, n_spills = 0, metadata = {})

        return launch


def _load(
    monkeypatch,
    caps,
    triton_has_tma = True,
):
    """interface.py under a throwaway package, with stubbed unsloth imports and spoofed cards."""
    launches, guard_stack = [], []

    @contextlib.contextmanager
    def torch_gpu_device(device):
        guard_stack.append(device)
        try:
            yield
        finally:
            guard_stack.pop()

    state = {"current": 0}
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(caps))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device = None: caps[state["current"] if device is None else device],
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device = None: types.SimpleNamespace(multi_processor_count = 4),
    )
    # No custom-op registration from the spoofed module.
    monkeypatch.setattr(torch.library, "custom_op", None, raising = False)

    import triton.language as tl

    for name in ("make_tensor_descriptor", "_experimental_make_tensor_descriptor"):
        if triton_has_tma:
            monkeypatch.setattr(tl, name, object(), raising = False)
        else:
            monkeypatch.delattr(tl, name, raising = False)

    stubs = {
        "unsloth": types.SimpleNamespace(DEVICE_TYPE = "cuda"),
        "unsloth.device_type": types.SimpleNamespace(
            cuda_visible_capabilities = _cuda_visible_capabilities()
        ),
        "unsloth.kernels.utils": types.SimpleNamespace(torch_gpu_device = torch_gpu_device),
        PKG: types.ModuleType(PKG),
        f"{PKG}.kernels": types.ModuleType(f"{PKG}.kernels"),
        f"{PKG}.kernels.backward": types.SimpleNamespace(
            **{
                n: _Kernel(launches, guard_stack)
                for n in (
                    "_autotuned_grouped_gemm_dW_kernel",
                    "_autotuned_grouped_gemm_dX_kernel",
                    "_grouped_gemm_dW_kernel",
                    "_grouped_gemm_dX_kernel",
                )
            }
        ),
        f"{PKG}.kernels.forward": types.SimpleNamespace(
            _autotuned_grouped_gemm_forward_kernel = _Kernel(launches, guard_stack),
            _grouped_gemm_forward_kernel = _Kernel(launches, guard_stack),
        ),
        f"{PKG}.kernels.tuning": types.SimpleNamespace(
            KernelConfigBackward_dW = object,
            KernelConfigBackward_dX = object,
            KernelConfigForward = object,
        ),
    }
    stubs[PKG].__path__ = [str(GG_DIR)]
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(f"{PKG}.interface", GG_DIR / "interface.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module._is_tracing = lambda *tensors: False  # fake tensors stand in for real cuda ones
    return module, launches


H100, A100, B200, L4 = (9, 0), (8, 0), (10, 0), (8, 9)


@pytest.mark.parametrize(
    "caps, expected",
    [
        pytest.param([H100, A100], False, id = "h100+a100"),
        pytest.param([A100, H100], False, id = "a100+h100"),
        pytest.param([H100, L4, H100], False, id = "h100+l4+h100"),
        pytest.param([H100, H100], True, id = "h100x2"),
        pytest.param([H100, B200], True, id = "h100+b200"),
        pytest.param([H100], True, id = "h100"),
        pytest.param([A100], False, id = "a100"),
        pytest.param([A100, A100], False, id = "a100x2"),
    ],
)
def test_tma_needs_every_visible_card(monkeypatch, caps, expected):
    module, _ = _load(monkeypatch, caps)
    assert module.supports_tma() is expected


def test_tma_still_needs_the_triton_api(monkeypatch):
    module, _ = _load(monkeypatch, [H100, H100], triton_has_tma = False)
    assert module.supports_tma() is False


def test_an_unreadable_card_disables_tma(monkeypatch):
    module, _ = _load(monkeypatch, [H100, H100])

    def get_device_capability(device = None):
        if device == 1:
            raise RuntimeError("unreadable")
        return H100

    monkeypatch.setattr(torch.cuda, "get_device_capability", get_device_capability)
    assert module._check_tma_support() is False


def test_launches_are_guarded_on_the_operands_device(monkeypatch):
    module, launches = _load(monkeypatch, [H100, A100])
    allocations = []
    monkeypatch.setattr(module.triton, "set_allocator", allocations.append, raising = False)
    monkeypatch.setattr(module, "_HAS_SET_ALLOCATOR", True)
    E, T, N, K = 2, 8, 16, 16
    with FakeTensorMode():
        dev = torch.device("cuda", 1)
        X = torch.empty(T, K, device = dev, dtype = torch.bfloat16)
        W = torch.empty(E, N, K, device = dev, dtype = torch.bfloat16)
        dY = torch.empty(T, N, device = dev, dtype = torch.bfloat16)
        m_sizes = torch.empty(E, device = dev, dtype = torch.int32)
        common = dict(m_sizes = m_sizes, gather_indices = None, topk = 1, autotune = True)
        module._grouped_gemm_forward_impl(X = X, W = W, **common)
        module._grouped_gemm_dX_impl(dY = dY, W = W, **common)
        module._grouped_gemm_dW_impl(X = X, dY = dY, **common)
        scratch = [alloc(128, 128, 0).device for alloc in allocations]
    assert launches == [dev, dev, dev]
    assert scratch == [dev, dev, dev]
