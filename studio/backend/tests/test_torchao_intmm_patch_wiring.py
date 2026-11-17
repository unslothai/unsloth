# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Invariants for Studio's copy of the torchao ``safe_int_mm`` fix, none of which raise on their own."""

from __future__ import annotations

import ast
import types
from pathlib import Path

import pytest

from core.inference.diffusion_torchao_patches import (
    _TORCHAO_INTMM_MODULES,
    _TorchaoIntmmLoader,
    _TorchaoIntmmPatchFinder,
    _patch_torchao_intmm_module,
    install_torchao_int_mm_patch,
)

_BACKEND = Path(__file__).resolve().parent.parent
_CORE = _BACKEND / "core"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_IMPORT_FIXES = _REPO_ROOT / "unsloth" / "import_fixes.py"
_PATCH_MODULE = _CORE / "inference" / "diffusion_torchao_patches.py"
_INSTALL = install_torchao_int_mm_patch.__name__

# Every module that can be the first thing in its process to reach torchao.
_ENTRYPOINTS = [
    _CORE / "inference" / "diffusion.py",
    _CORE / "inference" / "video.py",
    _CORE / "inference" / "diffusion_transformer_quant.py",
    _CORE / "training" / "diffusion_train_common.py",
]

_SHARED_FUNCTIONS = ("_is_fake_tensor", "_make_safe_int_mm", "_patch_torchao_intmm_module")


def _install_call_linenos(node) -> list[int]:
    return [
        c.lineno
        for c in ast.walk(node)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == _INSTALL
    ]


def test_all_entrypoints_install_the_patch():
    for path in _ENTRYPOINTS:
        assert _install_call_linenos(ast.parse(path.read_text(encoding = "utf-8"))), (
            f"{path.relative_to(_BACKEND)} never calls {_INSTALL}() -- torchao's int8 GEMM "
            "keeps syncing the device on every linear and cannot be CUDA-graph captured."
        )


def _strip_docstrings(node):
    for child in ast.walk(node):
        body = getattr(child, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            del body[0]
    return node


def _function_dump(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.dump(_strip_docstrings(node))
    raise AssertionError(f"{path} has no top-level def {name}")


@pytest.mark.parametrize("name", _SHARED_FUNCTIONS)
def test_studio_copy_matches_unsloth_import_fixes(name):
    if not _IMPORT_FIXES.is_file():
        pytest.skip("unsloth/import_fixes.py is not in this checkout")
    assert _function_dump(_PATCH_MODULE, name) == _function_dump(_IMPORT_FIXES, name), (
        f"DRIFT: {name}() differs between core/inference/diffusion_torchao_patches.py and "
        "unsloth/import_fixes.py. The two copies must stay interchangeable, because whichever "
        "one runs first in a process is the one that patches torchao."
    )


def _fake_intmm_module(safe_int_mm) -> types.ModuleType:
    module = types.ModuleType("torchao_intmm_stand_in")
    module.safe_int_mm = safe_int_mm
    module.out_dtype = lambda *args, **kwargs: None
    module.dynamo_is_compiling = lambda: False
    return module


def test_patch_leaves_an_already_patched_module_alone():
    """Why the two copies do not fight: both mark their replacement ``__unsloth_patched__``."""

    def already_patched(input, mat2):
        return None

    already_patched.__unsloth_patched__ = True
    module = _fake_intmm_module(already_patched)

    assert _patch_torchao_intmm_module(module) is False
    assert module.safe_int_mm is already_patched


def test_patch_refuses_an_unrecognised_body():
    def rewritten_upstream(input, mat2):
        return input @ mat2

    module = _fake_intmm_module(rewritten_upstream)

    assert _patch_torchao_intmm_module(module) is False
    assert module.safe_int_mm is rewritten_upstream


def test_finder_answers_for_both_torchao_homes(monkeypatch):
    """torchao main moved ``safe_int_mm`` into the int8 workflow module (pytorch/ao#4718), probe intact."""
    import importlib.machinery
    import importlib.util

    assert _TORCHAO_INTMM_MODULES == (
        "torchao.kernel.intmm",
        "torchao.quantization.quantize_.workflows.int8.kernels",
    )
    if _IMPORT_FIXES.is_file():
        text = _IMPORT_FIXES.read_text(encoding = "utf-8")
        for name in _TORCHAO_INTMM_MODULES:
            assert f'"{name}"' in text, f"unsloth/import_fixes.py does not list {name}"

    class _Loader:
        def exec_module(self, module):
            pass

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda fullname, *a, **k: importlib.machinery.ModuleSpec(fullname, _Loader()),
    )
    finder = _TorchaoIntmmPatchFinder()
    for name in _TORCHAO_INTMM_MODULES:
        spec = finder.find_spec(name)
        assert spec is not None and isinstance(spec.loader, _TorchaoIntmmLoader), name
    assert finder.find_spec("torchao.kernel.other") is None


def test_real_torchao_int_mm_is_patched_and_bit_identical():
    pytest.importorskip("torchao")
    torch = pytest.importorskip("torch")
    import importlib

    intmm = None
    for name in _TORCHAO_INTMM_MODULES:
        try:
            candidate = importlib.import_module(name)
        except ImportError:
            continue
        if callable(getattr(candidate, "safe_int_mm", None)):
            intmm = candidate
            break
    if intmm is None:
        pytest.skip("torchao does not define safe_int_mm under any known module name")

    install_torchao_int_mm_patch()
    patched = intmm.safe_int_mm
    assert getattr(patched, "__unsloth_patched__", False) is True
    # int_scaled_matmul resolves the name through module globals, so the rebind must reach it
    assert intmm.int_scaled_matmul.__globals__["safe_int_mm"] is patched
    original = patched.__unsloth_original__

    generator = torch.Generator().manual_seed(0)

    def randint8(*shape):
        return torch.randint(-127, 127, shape, dtype = torch.int8, generator = generator)

    cases = [
        (randint8(64, 64), randint8(64, 64)),
        (randint8(40, 24), randint8(24, 72)),
        # mat2 non-contiguous: torchao silently returns a wrong answer without the .contiguous()
        (randint8(64, 64), randint8(64, 64).t().contiguous().t()),
        # j = 20 is not a nonzero multiple of 8, so cuBLAS is skipped for the fallback path
        (randint8(40, 20), randint8(20, 64)),
    ]
    for a, b in cases:
        assert torch.equal(
            patched(a, b), original(a, b)
        ), f"patched safe_int_mm diverged from torchao's on {tuple(a.shape)} x {tuple(b.shape)}"


def _capturing_safe_int_mm():
    seen = []

    def original(input, mat2):
        return None

    module = _fake_intmm_module(original)
    module.out_dtype = lambda op, dtype, a, b: seen.append(b) or a
    from core.inference.diffusion_torchao_patches import _make_safe_int_mm

    return _make_safe_int_mm(module, original), seen


def test_transposed_weight_reaches_the_gemm_without_a_row_major_copy():
    torch = pytest.importorskip("torch")
    patched, seen = _capturing_safe_int_mm()
    a = torch.empty((1184, 7168), dtype = torch.int8, device = "meta")
    weight = torch.empty((5376, 7168), dtype = torch.int8, device = "meta")
    mat2 = weight.t()
    patched(a, mat2)
    assert seen[-1] is mat2
    assert seen[-1].stride() == (1, 7168)


def test_row_major_and_strided_mat2_keep_their_handling():
    torch = pytest.importorskip("torch")
    patched, seen = _capturing_safe_int_mm()
    a = torch.empty((64, 64), dtype = torch.int8, device = "meta")
    row_major = torch.empty((64, 32), dtype = torch.int8, device = "meta")
    patched(a, row_major)
    assert seen[-1] is row_major
    strided = torch.empty((64, 64), dtype = torch.int8, device = "meta")[:, ::2]
    assert not strided.is_contiguous() and not strided.t().is_contiguous()
    patched(a, strided)
    assert seen[-1] is not strided and seen[-1].is_contiguous()
