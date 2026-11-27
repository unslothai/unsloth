# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""unsloth/kernels/utils.py must not query CUDA streams at import: the first stream query opens a CUDA context on the
current card, which in a worker that picks its card after `import unsloth` (torch.multiprocessing.spawn +
set_device) is GPU 0. Static: module-level code only (function bodies run later, on the caller's card)."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_STREAM_QUERIES = {
    "_cuda_getCurrentRawStream",
    "current_stream",
    "default_stream",
    "_cuda_getDefaultStream",
}


def _module_level_calls(tree):
    stack = list(tree.body)
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(node, ast.Call):
            yield node
        stack.extend(ast.iter_child_nodes(node))


def test_no_cuda_stream_query_at_import():
    tree = ast.parse((ROOT / "unsloth" / "kernels" / "utils.py").read_text(encoding = "utf-8"))
    hits = []
    for call in _module_level_calls(tree):
        name = getattr(call.func, "attr", None) or getattr(call.func, "id", None)
        if name in _STREAM_QUERIES:
            hits.append(f"line {call.lineno}: {name}")
    assert hits == [], hits
