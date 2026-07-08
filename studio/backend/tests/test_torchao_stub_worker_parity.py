# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Invariant: every subprocess entrypoint that imports transformers / sentence-transformers must first
install the torchao Windows-ROCm stub.

``core/_torchao_stub.py:install_torchao_windows_rocm_stub`` stubs torchao so ``transformers.modeling_utils``
can expose ``PreTrainedModel`` without importing an absent c10d/RCCL backend on Windows ROCm (it is a no-op
on every other runtime). If transformers is imported before the stub is in place, a legacy Windows-ROCm venv
that still carries a real torchao crashes on import (issue #6833) instead of merely warning.

Three workers already guard this (training, export, rag); the inference worker -- the most-used path --
regressed by omitting it. These tests lock parity so it cannot silently drift again:

* ``test_all_entrypoints_call_stub`` -- each of the four entrypoints actually *calls* the stub.
* ``test_inference_worker_stubs_before_transformers`` -- in the inference worker the stub call runs BEFORE
  the transformers-importing statements (both the direct ``import transformers`` and the transitive import
  via ``from core.inference.inference import InferenceBackend``, whose module imports transformers at load).

CPU-only, no torch/transformers/GPU/weights needed: parses source with ``ast``.
"""

from __future__ import annotations

import ast
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent  # studio/backend
_CORE = _BACKEND_DIR / "core"

_ENTRYPOINTS = [
    _CORE / "training" / "worker.py",
    _CORE / "export" / "worker.py",
    _CORE / "rag" / "embeddings.py",
    _CORE / "inference" / "worker.py",
]

_STUB_FUNC = "install_torchao_windows_rocm_stub"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding = "utf-8"), filename = str(path))


def _stub_call_linenos(tree: ast.AST) -> list[int]:
    """Line numbers of every ``install_torchao_windows_rocm_stub()`` call in the tree."""
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == _STUB_FUNC
    ]


def test_all_entrypoints_call_stub():
    """Each subprocess/embedder entrypoint that imports transformers must call the stub."""
    for path in _ENTRYPOINTS:
        assert path.exists(), f"missing entrypoint: {path}"
        calls = _stub_call_linenos(_parse(path))
        assert calls, (
            f"{path.relative_to(_BACKEND_DIR)} never calls {_STUB_FUNC}() -- transformers would import "
            "unguarded and crash on a legacy Windows-ROCm venv (issue #6833)."
        )


def test_inference_worker_stubs_before_transformers():
    """In the inference worker the stub call precedes every statement that imports transformers,
    directly (``import transformers``) or transitively (``from core.inference.inference import ...``)."""
    path = _CORE / "inference" / "worker.py"
    tree = _parse(path)

    stub_linenos = _stub_call_linenos(tree)
    assert stub_linenos, f"{path.name}: stub never called"
    first_stub = min(stub_linenos)

    transformers_import_linenos = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "transformers" or alias.name.startswith("transformers.") for alias in node.names):
                transformers_import_linenos.append(node.lineno)
        elif isinstance(node, ast.ImportFrom):
            # Transitive: core.inference.inference imports transformers at module load.
            if node.module == "core.inference.inference":
                transformers_import_linenos.append(node.lineno)

    assert transformers_import_linenos, (
        f"{path.name}: expected an `import transformers` / `from core.inference.inference import ...` "
        "statement; the test's anchor is stale -- update it to the new import site."
    )
    earliest_import = min(transformers_import_linenos)
    assert first_stub < earliest_import, (
        f"{path.name}: {_STUB_FUNC}() at line {first_stub} must run BEFORE the transformers import at "
        f"line {earliest_import}; otherwise torchao imports unguarded on Windows ROCm (issue #6833)."
    )
