# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""[Windows, Linux, WSL, macOS] x [NVIDIA, AMD/ROCm, CPU-only] for the #7897 fix.

The fix is pure path arithmetic and imports no GPU library, so the GPU axis is an
invariance check: the stem and its destination must be byte-identical in every
cell. No per-vendor expectations are invented, because none exist.

UNSLOTH_SIM_GPU (nvidia|rocm|cpu) picks the cell and is applied at import, since the
spoofs mutate torch globals and cannot be undone in-process. One process per cell.
The OS axis is monkeypatch-scoped and needs no isolation.
"""

from __future__ import annotations

import ast
import ntpath
import os
import posixpath
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SAVE_PY = _REPO_ROOT / "unsloth" / "save.py"

_GPU_CELL = os.environ.get("UNSLOTH_SIM_GPU", "cpu").lower()


# GPU cell: applied before anything torch-touching -------------------------
def _apply_gpu_cell(cell: str) -> dict:
    """Returns a description of what the process now claims to be."""
    if cell == "cpu":
        return {"cell": "cpu", "cuda": False, "hip": None}

    sys.path.insert(0, str(_REPO_ROOT / "tests"))
    if cell == "nvidia":
        import _zoo_aggressive_cuda_spoof as spoof
        spoof.apply()
    elif cell == "rocm":
        import _zoo_rocm_spoof as spoof

        # gfx1100 == RX 7900 XTX, the card in issue #7897.
        spoof.apply("gfx1100")
    else:
        raise AssertionError(f"unknown UNSLOTH_SIM_GPU={cell!r}")

    import torch

    return {
        "cell": cell,
        "cuda": torch.cuda.is_available(),
        "hip": getattr(torch.version, "hip", None),
    }


try:
    _GPU_STATE = _apply_gpu_cell(_GPU_CELL)
except Exception as exc:  # noqa: BLE001 -- torch absent is a legitimate cell
    _GPU_STATE = {"cell": _GPU_CELL, "error": str(exc)}


# The helper under test, lifted without importing unsloth ------------------
def _load_helper():
    src = _SAVE_PY.read_text(encoding = "utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == "_model_basename":
            ns: dict = {"os": os}
            exec(compile(ast.Module([node], []), str(_SAVE_PY), "exec"), ns)
            return ns["_model_basename"]
    raise AssertionError("unsloth/save.py defines no _model_basename")


# OS flavour -> (path module, a base-model path that OS actually produces)
_OS_CELLS = {
    "windows": (ntpath, r"D:\Models\Merged Models\MyModel"),
    "linux": (posixpath, "/home/u/models/MyModel"),
    # WSL reaches a Windows drive through drvfs;
    # it is an ordinary POSIX path.
    "wsl": (posixpath, "/mnt/d/Models/MyModel"),
    "macos": (posixpath, "/Users/u/models/MyModel"),
}

# Cells that are not real products.
# passing here is NOT a claim that Unsloth supports CUDA or ROCm on macOS.
_UNREAL_CELLS = {("macos", "nvidia"), ("macos", "rocm")}


@pytest.mark.parametrize("os_name", sorted(_OS_CELLS))
def test_stem_is_identical_in_every_cell(os_name):
    flavour, base = _OS_CELLS[os_name]
    stem = _load_helper()(base)
    assert stem == "MyModel", f"cell {os_name}/{_GPU_CELL}: {base!r} -> {stem!r}"


@pytest.mark.parametrize("os_name", sorted(_OS_CELLS))
def test_destination_is_identical_in_every_cell(os_name):
    """Same stem, same join result, regardless of GPU vendor."""
    flavour, base = _OS_CELLS[os_name]
    gguf_dir = (
        r"C:\Users\u\.unsloth\exports\run_gguf"
        if flavour is ntpath
        else "/home/u/.unsloth/exports/run_gguf"
    )
    stem = _load_helper()(base)
    out = flavour.join(gguf_dir, f"{stem}.Q5_K_M.gguf")
    assert flavour.dirname(out) == gguf_dir, f"cell {os_name}/{_GPU_CELL}: {out!r}"
    assert flavour.basename(out) == "MyModel.Q5_K_M.gguf"


def test_the_gpu_cell_really_is_what_it_claims():
    """Guard the harness itself: a silently-inert spoof would fake 12 green cells."""
    if "error" in _GPU_STATE:
        pytest.skip(f"torch unavailable for cell {_GPU_CELL}: {_GPU_STATE['error']}")
    if _GPU_CELL == "nvidia":
        assert _GPU_STATE["cuda"] is True
        assert not _GPU_STATE["hip"]
    elif _GPU_CELL == "rocm":
        assert _GPU_STATE["cuda"] is True
        assert _GPU_STATE["hip"], "ROCm cell has no torch.version.hip"
    elif _GPU_CELL == "cpu":
        assert _GPU_STATE["cuda"] is False


def test_the_fix_imports_no_gpu_library():
    """_model_basename must stay pure: no torch, no accelerator probing."""
    src = _SAVE_PY.read_text(encoding = "utf-8")
    fn = next(
        n
        for n in ast.parse(src).body
        if isinstance(n, ast.FunctionDef) and n.name == "_model_basename"
    )
    body = ast.get_source_segment(src, fn)
    for forbidden in ("torch", "cuda", "hip", "device", "unsloth_zoo"):
        assert forbidden not in body, f"_model_basename references {forbidden!r}"
    assert not any(
        isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(fn)
    ), "_model_basename must not import anything"


def test_unreal_cells_are_declared_not_claimed():
    """macOS x NVIDIA / macOS x ROCm do not exist; this documents that."""
    for os_name, gpu in _UNREAL_CELLS:
        assert os_name in _OS_CELLS and gpu in {"nvidia", "rocm"}
