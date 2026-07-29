# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stack-dependent startup work must not run before the socket binds.

Two pieces of work import the ML stack. The MLX availability probe imports the
MLX runtime, and the RAG embedder warm pulls sentence-transformers/transformers/
torch. Uvicorn binds only once the lifespan yields, so anything that does either
on the lifespan thread -- or on a thread started early enough to race the
coordinated warm for the GIL and the import locks -- puts that import back in
front of the login screen.

Both now run on the post-warm thread, which joins the coordinated warm first.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

_MAIN = _BACKEND / "main.py"


def _lifespan_body() -> ast.AsyncFunctionDef:
    tree = ast.parse(_MAIN.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan":
            return node
    raise AssertionError("lifespan not found in main.py")


def _post_warm_body() -> ast.FunctionDef:
    tree = ast.parse(_MAIN.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_post_warm_background_work":
            return node
    raise AssertionError("_post_warm_background_work not found in main.py")


def _names_called(node: ast.AST) -> set[str]:
    """Every bare name and attribute tail invoked or referenced under `node`."""
    found: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            found.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            found.add(sub.attr)
    return found


def test_lifespan_does_not_probe_mlx_before_yielding():
    referenced = _names_called(_lifespan_body())
    assert "start_mlx_autorepair_if_needed" not in referenced, (
        "the MLX autorepair probe is back on the lifespan thread; "
        "mlx_stack_available() imports the MLX runtime and the socket binds "
        "only after this coroutine yields"
    )


def test_lifespan_does_not_start_the_rag_warm_before_yielding():
    referenced = _names_called(_lifespan_body())
    assert "_warm_rag_embedder" not in referenced, (
        "the RAG embedder warm is started from the lifespan again; it imports "
        "sentence-transformers/transformers/torch and races the coordinated warm"
    )


def test_post_warm_thread_joins_the_warm_before_doing_stack_work():
    body = _post_warm_body()
    statements = [n for n in body.body if not isinstance(n, ast.Expr) or True]

    # The join has to come first, otherwise these imports race the warm rather
    # than building on it.
    join_line = None
    work_lines = []
    for sub in ast.walk(body):
        if isinstance(sub, ast.Name) and sub.id == "join_background_warm":
            join_line = sub.lineno if join_line is None else min(join_line, sub.lineno)
        if isinstance(sub, ast.Name) and sub.id in {
            "start_mlx_autorepair_if_needed",
            "_warm_rag_embedder",
        }:
            work_lines.append(sub.lineno)

    assert join_line is not None, "_post_warm_background_work must join the warm"
    assert work_lines, "_post_warm_background_work must do the deferred work"
    assert join_line < min(work_lines), (
        "join_background_warm() must precede the stack-dependent work, "
        f"got join at {join_line} and work at {sorted(work_lines)}"
    )
    assert statements, "body must not be empty"


def test_post_warm_thread_is_started_by_the_lifespan():
    referenced = _names_called(_lifespan_body())
    assert "_post_warm_background_work" in referenced, (
        "nothing starts the post-warm thread, so the MLX autorepair and the RAG "
        "warm would never run"
    )
