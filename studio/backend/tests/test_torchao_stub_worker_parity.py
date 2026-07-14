# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Invariant: the inference subprocess must install the torchao Windows-ROCm stub before it imports
transformers.

``core/_torchao_stub.py:install_torchao_windows_rocm_stub`` stubs torchao so transformers can import
without an absent RCCL backend on Windows ROCm (no-op on every other runtime). If transformers imports
first, a legacy Windows-ROCm venv that still carries a real torchao crashes on import (issue #6833).
Three entrypoints already guard this (the training and export workers, and the main-process rag
embedder); the inference worker -- the most-used path -- never had the call.

CPU-only: parses source with ``ast``, no torch/transformers/GPU/weights needed.
"""

from __future__ import annotations

import ast
from pathlib import Path

from core._torchao_stub import install_torchao_windows_rocm_stub

_BACKEND = Path(__file__).resolve().parent.parent  # studio/backend
_CORE = _BACKEND / "core"
_STUB = install_torchao_windows_rocm_stub.__name__  # a rename breaks the import loudly

_ENTRYPOINTS = [
    _CORE / "training" / "worker.py",
    _CORE / "export" / "worker.py",
    _CORE / "rag" / "embeddings.py",
    _CORE / "inference" / "worker.py",
]


def _stub_call_linenos(node) -> list[int]:
    """Line numbers of every ``install_torchao_windows_rocm_stub()`` call under ``node``."""
    return [
        c.lineno
        for c in ast.walk(node)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == _STUB
    ]


def _func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def test_all_entrypoints_call_stub():
    """Every entrypoint that imports transformers must call the stub at all -- this is the exact
    gap that shipped (the inference worker never gained the call). This is a presence check (the call
    exists in the file); ordering is asserted only for the inference worker below, the path this fix
    hardened. The other three import transformers at structurally different sites."""
    for path in _ENTRYPOINTS:
        assert _stub_call_linenos(ast.parse(path.read_text(encoding = "utf-8"))), (
            f"{path.relative_to(_BACKEND)} never calls {_STUB}() -- transformers would import "
            "unguarded and crash on a legacy Windows-ROCm venv (issue #6833)."
        )


_INFERENCE_MOD = "core.inference.inference"


def _imports_transformers(node) -> bool:
    """A statement that imports transformers directly (``import transformers[.x]`` /
    ``from transformers[.x] import ...``) or transitively at load: any absolute or relative import
    form resolving to ``core.inference.inference`` (whose module imports transformers), so a style
    refactor of the section-2 import can't slip past the anchor."""
    if isinstance(node, ast.Import):
        return any(
            a.name.split(".")[0] == "transformers"
            or a.name == _INFERENCE_MOD
            or a.name.startswith(_INFERENCE_MOD + ".")
            for a in node.names
        )
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if node.level == 0:
            return (
                module.split(".")[0] == "transformers"
                or module == _INFERENCE_MOD
                or module.startswith(_INFERENCE_MOD + ".")
                or (module == "core.inference" and any(a.name == "inference" for a in node.names))
            )
        # Relative forms inside core/inference/worker.py: ``from .inference import X`` and
        # ``from . import inference`` both resolve to core.inference.inference.
        return module == "inference" or (
            not module and any(a.name == "inference" for a in node.names)
        )
    return False


def test_inference_worker_stubs_before_transformers():
    """In ``run_inference_process`` the stub must precede every path that reaches transformers: the
    section-2 imports (direct ``import transformers`` and the transitive ``core.inference.inference``
    import), and -- the reason it sits at the top of the function -- the ``_resolve_base_model`` call,
    which pulls transformers via ``utils.models`` for a local LoRA adapter with no recorded base.
    Scoped to the function (mirrors ``test_ssm_runtime``) so a stub call elsewhere in the module can't
    mask a drop from the function that actually runs the import. The ``_activate_transformers_version``
    call inside the MLX branch is not an anchor: MLX is never Windows ROCm, so it needs no stub."""
    tree = ast.parse((_CORE / "inference" / "worker.py").read_text(encoding = "utf-8"))
    fn = _func(tree, "run_inference_process")
    assert (
        fn is not None
    ), "run_inference_process not found in inference/worker.py -- renamed? update this test."

    stub = _stub_call_linenos(fn)
    assert stub, f"run_inference_process must call {_STUB}()"

    dangers = []
    for node in ast.walk(fn):
        if _imports_transformers(node):
            dangers.append(node.lineno)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_resolve_base_model"
        ):
            dangers.append(node.lineno)
    assert dangers, (
        "no transformers-reaching site found in run_inference_process -- the anchors are stale, update "
        "them to the new import/resolution sites."
    )

    assert min(stub) < min(dangers), (
        f"{_STUB}() at line {min(stub)} must run before the first transformers-reaching site at line "
        f"{min(dangers)}; otherwise torchao imports unguarded on Windows ROCm (issue #6833)."
    )
