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
    tree = ast.parse(_MAIN.read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan":
            return node
    raise AssertionError("lifespan not found in main.py")


def _post_warm_body() -> ast.FunctionDef:
    tree = ast.parse(_MAIN.read_text(encoding = "utf-8"))
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

    # The join has to come first, or these imports race the warm instead of
    # building on it.
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


def test_post_warm_work_honours_the_coordinated_warm_kill_switch():
    """The switch must gate the torch-dependent work, and only that.

    It used to be the first statement, which also skipped MLX autorepair -- and
    that one is not torch, has always had its own UNSLOTH_DISABLE_MLX_AUTOREPAIR
    opt-out, and ran in the lifespan unconditionally before the deferral, so
    skipping it left a broken-MLX Mac chat-only for good. Assert the guard exists
    and returns, and that the thing it protects is the RAG warm; the ordering
    against autorepair is pinned in test_warm_window_review_fixes.py.
    """
    body = _post_warm_body()
    guards = [
        node
        for node in body.body
        if isinstance(node, ast.If) and "DISABLE_ENV_VAR" in ast.unparse(node.test)
    ]
    assert guards, "the kill-switch guard is gone; the RAG warm would import torch anyway"
    guard = guards[0]
    condition = ast.unparse(guard.test)
    assert "os.environ" in condition
    assert any(isinstance(node, ast.Return) for node in guard.body)

    # Everything after the guard is what the switch actually disables.
    after = body.body[body.body.index(guard) + 1 :]
    protected = {
        sub.func.id
        for stmt in after
        for sub in ast.walk(stmt)
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
    }
    assert "_warm_rag_embedder" in protected, (
        "the RAG warm is no longer behind the kill switch, so a --no-torch host "
        "still pulls sentence-transformers and torch at startup"
    )


def test_post_warm_thread_is_started_by_the_lifespan():
    """The lifespan must still put the thread up -- now through a helper.

    It used to construct the Thread inline. That thread was untracked, so a
    shutdown landing while it was parked in the warm join could not stop it and a
    second lifespan stacked another one, which is why the start moved behind
    _start_post_warm_thread(). Follow the indirection rather than pinning the
    inline form.
    """
    referenced = _names_called(_lifespan_body())
    assert "_start_post_warm_thread" in referenced or (
        "_post_warm_background_work" in referenced
    ), (
        "nothing starts the post-warm thread, so the MLX autorepair and the RAG "
        "warm would never run"
    )

    # ...and the helper must target the real work, or a do-nothing thread passes.
    tree = ast.parse(_MAIN.read_text(encoding = "utf-8"))
    starter = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_start_post_warm_thread"
    )
    assert "_post_warm_background_work" in _names_called(starter)


def test_post_warm_actually_runs_both_pieces_of_work(monkeypatch):
    """Deferring must not become dropping.

    The lexical tests above prove the lifespan no longer does this work. They
    would still pass if the deferred work never ran at all, which would silently
    end MLX self-healing and leave the RAG embedder cold on every launch.
    """
    import main as main_mod
    import utils.mlx_repair as mlx_mod

    order: list[str] = []
    monkeypatch.setattr(main_mod, "join_background_warm", lambda *a, **k: order.append("join"))
    monkeypatch.setattr(
        mlx_mod, "start_mlx_autorepair_if_needed", lambda: order.append("mlx") or False
    )
    monkeypatch.setattr(main_mod, "_warm_rag_embedder", lambda: order.append("rag"))

    main_mod._post_warm_background_work()

    assert order == [
        "join",
        "mlx",
        "rag",
    ], f"post-warm work did not run in the intended order, got {order}"


def test_a_failing_mlx_probe_does_not_strand_the_rag_warm(monkeypatch):
    """One deferred stage failing must not take the other down with it."""
    import main as main_mod
    import utils.mlx_repair as mlx_mod

    ran: list[str] = []
    monkeypatch.setattr(main_mod, "join_background_warm", lambda *a, **k: None)

    def _boom():
        raise RuntimeError("mlx probe exploded")

    monkeypatch.setattr(mlx_mod, "start_mlx_autorepair_if_needed", _boom)
    monkeypatch.setattr(main_mod, "_warm_rag_embedder", lambda: ran.append("rag"))

    main_mod._post_warm_background_work()

    assert ran == ["rag"], "a failing MLX probe must not skip the RAG warm"
