# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the NVFP4 flashinfer ops module (``diffusion_nvfp4_ops.py``)."""

from __future__ import annotations

import ast
import pathlib

import pytest

from core.inference import diffusion_nvfp4_ops as ops

_INFERENCE_DIR = pathlib.Path(ops.__file__).resolve().parent


@pytest.fixture(autouse = True)
def _clean_backend_state(monkeypatch):
    monkeypatch.delenv(ops.NVFP4_BACKEND_ENV, raising = False)
    ops.reset_preflight_cache()
    yield
    ops.reset_preflight_cache()


def _stub(
    monkeypatch,
    *,
    available = True,
    capability = (10, 0),
    preflight_ok = True,
):
    monkeypatch.setattr(
        ops,
        "_flashinfer_available",
        lambda: (available, "0.6.6" if available else "ImportError: no flashinfer"),
    )
    monkeypatch.setattr(ops, "_device_capability", lambda device = None: capability)
    monkeypatch.setattr(
        ops,
        "nvfp4_preflight",
        lambda device = None, **kw: {
            "ok": preflight_ok,
            "capability": capability,
            "name": "stub",
            "reason": "ok" if preflight_ok else "JIT build failed",
        },
    )




def test_auto_selects_flashinfer_when_import_capability_and_preflight_all_pass(monkeypatch):
    _stub(monkeypatch)
    assert ops.select_nvfp4_backend(0) == "flashinfer"
    assert "preflight ok" in ops.nvfp4_backend_reason(0)


@pytest.mark.parametrize("capability", [(10, 0), (10, 3), (12, 0)])
def test_every_capability_in_the_flashinfer_set_selects_it(monkeypatch, capability):
    _stub(monkeypatch, capability = capability)
    assert ops.select_nvfp4_backend(0) == "flashinfer"


@pytest.mark.parametrize("capability", [(8, 9), (9, 0), (12, 1)])
def test_a_capability_outside_the_set_falls_to_torchao_and_names_it(monkeypatch, capability):
    _stub(monkeypatch, capability = capability)
    assert ops.select_nvfp4_backend(0) == "torchao"
    assert "sm_%d%d" % capability in ops.nvfp4_backend_reason(0)


def test_no_cuda_capability_is_torchao(monkeypatch):
    _stub(monkeypatch, capability = None)
    assert ops.select_nvfp4_backend() == "torchao"
    assert "capability" in ops.nvfp4_backend_reason()


def test_import_failure_is_torchao_which_is_how_windows_resolves(monkeypatch):
    _stub(monkeypatch, available = False)
    assert ops.select_nvfp4_backend(0) == "torchao"
    assert "flashinfer unavailable" in ops.nvfp4_backend_reason(0)


def test_a_failed_preflight_is_torchao_even_on_a_supported_capability(monkeypatch):
    _stub(monkeypatch, preflight_ok = False)
    assert ops.select_nvfp4_backend(0) == "torchao"
    assert "preflight failed: JIT build failed" in ops.nvfp4_backend_reason(0)


def test_explicit_torchao_never_probes(monkeypatch):
    def _boom(*a, **kw):  # pragma: no cover - reached only on a regression
        raise AssertionError("the probe ran under UNSLOTH_NVFP4_BACKEND=torchao")

    monkeypatch.setattr(ops, "_flashinfer_available", _boom)
    monkeypatch.setattr(ops, "nvfp4_preflight", _boom)
    monkeypatch.setenv(ops.NVFP4_BACKEND_ENV, "torchao")
    assert ops.select_nvfp4_backend(0) == "torchao"
    assert ops.nvfp4_backend_reason(0).endswith("=torchao")


def test_explicit_flashinfer_that_fails_a_check_warns_and_uses_torchao(monkeypatch):
    _stub(monkeypatch, preflight_ok = False)
    monkeypatch.setenv(ops.NVFP4_BACKEND_ENV, "flashinfer")
    with pytest.warns(RuntimeWarning, match = "preflight failed"):
        assert ops.select_nvfp4_backend(0) == "torchao"


def test_explicit_flashinfer_that_passes_warns_about_nothing(monkeypatch):
    _stub(monkeypatch)
    monkeypatch.setenv(ops.NVFP4_BACKEND_ENV, "flashinfer")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ops.select_nvfp4_backend(0) == "flashinfer"


@pytest.mark.parametrize("value", ["", "  ", "cutlass", "TorchAO nope"])
def test_an_unrecognised_env_value_reads_as_auto(monkeypatch, value):
    _stub(monkeypatch)
    monkeypatch.setenv(ops.NVFP4_BACKEND_ENV, value)
    assert ops.nvfp4_backend_env() == "auto"
    assert ops.select_nvfp4_backend(0) == "flashinfer"


@pytest.mark.parametrize("value", ["FlashInfer", " flashinfer "])
def test_the_env_value_is_case_and_space_insensitive(monkeypatch, value):
    _stub(monkeypatch, preflight_ok = False)
    monkeypatch.setenv(ops.NVFP4_BACKEND_ENV, value)
    with pytest.warns(RuntimeWarning):
        assert ops.select_nvfp4_backend(0) == "torchao"




def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _is_guard(node: ast.AST) -> bool:
    """``torch.cuda.device(...)`` or the module's own ``_device_guard(...)`` alias."""
    if not isinstance(node, ast.Call):
        return False
    name = _dotted(node.func)
    return name.endswith("torch.cuda.device") or name.endswith("_device_guard")


# FlashInfer's PRIVATE dispatch entry points: imported by name, so a "flashinfer." prefix check
# cannot see them, and each one launches or allocates on the current device.
_PRIVATE_LAUNCHES = frozenset(
    {
        "choose_one",
        "cutlass_fp4_gemm_runner",
        "get_cutlass_fp4_gemm_module",
        "_get_cache_buf",
        "fp4_quantize_sm100",
        "get_fp4_quantization_module",
    }
)


def _is_launch(node: ast.Call) -> str:
    # A Triton launch is a Call on a SUBSCRIPT (``kernel[grid](...)``) rather than on a name, and
    # needs the guard just as much: Triton takes its device from the CURRENT context.
    if isinstance(node.func, ast.Subscript):
        name = _dotted(node.func.value)
        return name if name.endswith("_kernel") else ""
    name = _dotted(node.func)
    if name.startswith("flashinfer.") or name.startswith("_fi."):
        return name
    if "torch.ops.unsloth_nvfp4" in name:
        return name
    if name.rsplit(".", 1)[-1] in _PRIVATE_LAUNCHES:
        return name
    return ""


class _LaunchVisitor(ast.NodeVisitor):
    """Records every launch whose lexical ancestry contains no device guard."""

    def __init__(self) -> None:
        self.depth = 0
        self.unguarded: list[tuple[int, str]] = []

    def _visit_with(self, node) -> None:
        guarded = 0
        for item in node.items:
            self.visit(item.context_expr)
            if _is_guard(item.context_expr):
                guarded = 1
        self.depth += guarded
        for stmt in node.body:
            self.visit(stmt)
        self.depth -= guarded

    visit_With = _visit_with
    visit_AsyncWith = _visit_with

    def visit_Call(self, node: ast.Call) -> None:
        launch = _is_launch(node)
        if launch and self.depth == 0:
            self.unguarded.append((node.lineno, launch))
        self.generic_visit(node)


def _nvfp4_sources() -> list[pathlib.Path]:
    return sorted(_INFERENCE_DIR.glob("diffusion_nvfp4_*.py"))


def test_the_guard_visitor_catches_an_unguarded_launch():
    tree = ast.parse(
        "import flashinfer\n"
        "def f(x):\n"
        "    with torch.cuda.device(x.device):\n"
        "        flashinfer.mm_fp4(x)\n"
        "    flashinfer.nvfp4_quantize(x)\n"
        "    torch.ops.unsloth_nvfp4.mm(x)\n"
        "    _bias_add_kernel[grid](x)\n"
        "    with torch.cuda.device(x.device):\n"
        "        _bias_add_kernel[grid](x)\n"
        "    _get_cache_buf('ws', 1, x.device)\n"
    )
    visitor = _LaunchVisitor()
    visitor.visit(tree)
    assert [name for _, name in visitor.unguarded] == [
        "flashinfer.nvfp4_quantize",
        "torch.ops.unsloth_nvfp4.mm",
        "_bias_add_kernel",
        "_get_cache_buf",
    ]


def test_every_flashinfer_launch_in_the_nvfp4_modules_sits_inside_a_device_guard():
    sources = _nvfp4_sources()
    assert sources, "no diffusion_nvfp4_*.py modules found to guard"
    offences: list[str] = []
    for path in sources:
        visitor = _LaunchVisitor()
        visitor.visit(ast.parse(path.read_text(encoding = "utf-8")))
        offences += [
            f"{path.name}:{line}: {name} outside torch.cuda.device"
            for line, name in visitor.unguarded
        ]
    assert not offences, "\n".join(offences)


_STREAM_BANNED = ("set_stream", "set_device", "setDevice")


def _banned_stream_calls(source: str) -> list[tuple[int, str]]:
    """Lines that switch the current device or stream behind the guard's back: ``set_stream``
    silently sets the current DEVICE as well, and ``set_device`` moves what the guard restores."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr in _STREAM_BANNED:
            found.append((node.lineno, _dotted(node) or node.attr))
        elif isinstance(node, ast.Name) and node.id in _STREAM_BANNED:
            found.append((node.lineno, node.id))
    return found


def test_the_banned_call_detector_sees_an_aliased_set_stream():
    assert _banned_stream_calls("import torch\ncuda = torch.cuda\ncuda.set_stream(s)\n")
    assert _banned_stream_calls("from torch.cuda import set_device\nset_device(1)\n")
    assert not _banned_stream_calls("def reset_stream_cache():\n    return None\n")


def test_the_modules_never_set_the_current_stream_or_device():
    offences: list[str] = []
    for path in _nvfp4_sources():
        offences += [
            f"{path.name}:{line}: {name} switches the current device behind the guard"
            for line, name in _banned_stream_calls(path.read_text(encoding = "utf-8"))
        ]
    assert not offences, "\n".join(offences)




@pytest.mark.parametrize(
    "m,k,n",
    [
        (1, 3072, 18432),
        (1, 256, 15360),
        (77, 3072, 3072),
        (129, 256, 15360),
        (512, 3072, 12288),
        (4096, 3072, 3072),
        (16384, 3072, 18432),
    ],
)
def test_the_fake_impls_reproduce_the_flashinfer_allocation(m, k, n):
    torch = pytest.importorskip("torch")

    x = torch.empty((m, k), dtype = torch.bfloat16, device = "meta")
    gsf = torch.empty((1,), dtype = torch.float32, device = "meta")
    xq, x_sf = ops._quantize_fake(x, gsf)
    cols = k // 16
    total = ops._swizzled_sf_numel(m, cols, 128)

    assert tuple(xq.shape) == (m, k // 2) and xq.dtype == torch.uint8
    assert tuple(x_sf.shape) == (total // cols, cols) and x_sf.dtype == torch.uint8
    assert x_sf.numel() == total
    assert x_sf.shape[0] >= m

    w = torch.empty((n, k // 2), dtype = torch.uint8, device = "meta")
    w_sf = torch.empty(ops.sf_matrix_shape(n, cols), dtype = torch.uint8, device = "meta")
    alpha = torch.empty((1,), dtype = torch.float32, device = "meta")
    out = ops._mm_fake(xq, w, x_sf, w_sf, alpha, n, "cutlass")
    assert tuple(out.shape) == (m, n) and out.dtype == torch.bfloat16


def test_swizzled_numel_pads_rows_to_128_and_columns_to_4():
    assert ops._swizzled_sf_numel(1, 16, 128) == 128 * 16
    assert ops._swizzled_sf_numel(129, 16, 128) == 256 * 16
    assert ops._swizzled_sf_numel(128, 17, 128) == 128 * 20
    assert ops._swizzled_sf_numel(512, 192, 128) == 512 * 192


def test_sf_matrix_shape_is_the_flat_buffer_reshaped():
    assert ops.sf_matrix_shape(3072, 192) == (3072, 192)
    assert ops.sf_matrix_shape(1, 16) == (128, 16)
    assert ops.sf_matrix_shape(15360, 16) == (15360, 16)


def test_register_ops_is_idempotent():
    pytest.importorskip("torch")
    import torch

    ops.register_ops()
    ops.register_ops()
    assert hasattr(torch.ops.unsloth_nvfp4, "quantize")
    assert hasattr(torch.ops.unsloth_nvfp4, "mm")




def test_real_preflight_reports_ok_on_a_blackwell_card_with_flashinfer():
    torch = pytest.importorskip("torch")
    if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    capability = tuple(torch.cuda.get_device_capability(0))
    if capability not in ops.NVFP4_FLASHINFER_CAPS:
        pytest.skip("sm_%d%d has no flashinfer NVFP4 kernels" % capability)
    pytest.importorskip("flashinfer")

    ops.reset_preflight_cache()
    record = ops.nvfp4_preflight(0)
    assert record["capability"] == capability
    assert record["name"]
    assert record["ok"] is True, record["reason"]
    assert record["reason"] == "ok"
    ops.nvfp4_preflight(0)
    assert ops.nvfp4_preflight(0) == record
    assert ops.select_nvfp4_backend(0) == "flashinfer"
