# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Invariants for Studio's copy of the torchao ``safe_int_mm`` fix.

``core/inference/diffusion_torchao_patches.py`` replaces torchao's int8 GEMM, whose
``"FakeTensor" in input.__repr__()`` probe calls ``.item()`` on a real CUDA tensor: a device
sync per eager int8 linear, and ``cudaErrorStreamCaptureUnsupported`` under
``torch.cuda.graph``. Three things can rot and none of them raise on their own:

1. a module stops calling ``install_torchao_int_mm_patch()`` and its process silently syncs
   again (the int8 prequant path never calls ``quantize_``, so nothing else would install it);
2. the Studio copy and the canonical one in ``unsloth/import_fixes.py`` drift apart, which is
   how two copies of one patch stop being interchangeable;
3. the structural gate stops refusing bodies it does not recognise, and the copy starts
   impersonating a torchao it was never verified against.

Mostly CPU-only ``ast`` work; the last test needs a real torchao and runs the GEMM on CPU.
"""

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

_BACKEND = Path(__file__).resolve().parent.parent  # studio/backend
_CORE = _BACKEND / "core"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_IMPORT_FIXES = _REPO_ROOT / "unsloth" / "import_fixes.py"
_PATCH_MODULE = _CORE / "inference" / "diffusion_torchao_patches.py"
_INSTALL = install_torchao_int_mm_patch.__name__  # a rename breaks the import loudly

# Every module that can be the first thing in its process to reach torchao: the image backend,
# the video backend, the transformer-quant module (also imported alone by the spawned smoke-probe
# child) and the diffusion trainers' shared entry.
_ENTRYPOINTS = [
    _CORE / "inference" / "diffusion.py",
    _CORE / "inference" / "video.py",
    _CORE / "inference" / "diffusion_transformer_quant.py",
    _CORE / "training" / "diffusion_train_common.py",
]

# The three functions that are one implementation living in two files.
_SHARED_FUNCTIONS = ("_is_fake_tensor", "_make_safe_int_mm", "_patch_torchao_intmm_module")


def _install_call_linenos(node) -> list[int]:
    """Line numbers of every ``install_torchao_int_mm_patch()`` call under ``node``."""
    return [
        c.lineno
        for c in ast.walk(node)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == _INSTALL
    ]


def test_all_entrypoints_install_the_patch():
    """Presence check: torchao is patched by whichever of these a process imports first."""
    for path in _ENTRYPOINTS:
        assert _install_call_linenos(ast.parse(path.read_text(encoding = "utf-8"))), (
            f"{path.relative_to(_BACKEND)} never calls {_INSTALL}() -- torchao's int8 GEMM "
            "keeps syncing the device on every linear and cannot be CUDA-graph captured."
        )


def _strip_docstrings(node):
    """Drop every docstring under ``node`` so the comparison is about code only."""
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
    """The two homes must stay one implementation.

    Compared as ast dumps with docstrings removed, so comments and wording are free to differ
    per home while the code cannot. Skipped when the checkout has no ``unsloth/`` (Studio ships
    on its own), which is the only reason this file may be absent.
    """
    if not _IMPORT_FIXES.is_file():
        pytest.skip("unsloth/import_fixes.py is not in this checkout")
    assert _function_dump(_PATCH_MODULE, name) == _function_dump(_IMPORT_FIXES, name), (
        f"DRIFT: {name}() differs between core/inference/diffusion_torchao_patches.py and "
        "unsloth/import_fixes.py. The two copies must stay interchangeable, because whichever "
        "one runs first in a process is the one that patches torchao."
    )


def _fake_intmm_module(safe_int_mm) -> types.ModuleType:
    """A stand-in for ``torchao.kernel.intmm`` carrying the names the gate resolves."""
    module = types.ModuleType("torchao_intmm_stand_in")
    module.safe_int_mm = safe_int_mm
    module.out_dtype = lambda *args, **kwargs: None
    module.dynamo_is_compiling = lambda: False
    return module


def test_patch_leaves_an_already_patched_module_alone():
    """Idempotence, and the reason the two copies do not fight: the Studio module and
    ``unsloth/import_fixes.py`` both mark their replacement ``__unsloth_patched__``, so the
    second one to run recognises the first one's work rather than wrapping it."""

    def already_patched(input, mat2):
        return None

    already_patched.__unsloth_patched__ = True
    module = _fake_intmm_module(already_patched)

    assert _patch_torchao_intmm_module(module) is False
    assert module.safe_int_mm is already_patched


def test_patch_refuses_an_unrecognised_body():
    """The replacement is a full copy of torchao 0.17.0's body, so it may only stand in for a
    body that still matches: no cuBLAS dimension guards in the source, no patch."""

    def rewritten_upstream(input, mat2):
        # None of the markers the gate looks for
        return input @ mat2

    module = _fake_intmm_module(rewritten_upstream)

    assert _patch_torchao_intmm_module(module) is False
    assert module.safe_int_mm is rewritten_upstream


def test_finder_answers_for_both_torchao_homes(monkeypatch):
    """torchao main moved ``safe_int_mm`` into the int8 workflow module (pytorch/ao#4718) with the
    probe intact, so the finder must wrap both names. The table is also pinned against the
    canonical copy so the two homes cannot disagree on WHICH modules they cover."""
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
        importlib.util, "find_spec",
        lambda fullname, *a, **k: importlib.machinery.ModuleSpec(fullname, _Loader()),
    )
    finder = _TorchaoIntmmPatchFinder()
    for name in _TORCHAO_INTMM_MODULES:
        spec = finder.find_spec(name)
        assert spec is not None and isinstance(spec.loader, _TorchaoIntmmLoader), name
    assert finder.find_spec("torchao.kernel.other") is None

def test_real_torchao_int_mm_is_patched_and_bit_identical():
    """With torchao installed: the patch lands on every binding that matters, and the copy
    returns exactly what the original returned. The shapes cover both cuBLAS guards (a good j
    and k, a j that is not a nonzero multiple of 8) and the silent-wrong-answer contiguity fix
    on mat2."""
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
