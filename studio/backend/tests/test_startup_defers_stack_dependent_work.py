# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Optional GPU consumers must stay cold through normal backend startup.

MLX repair remains deferred until after the coordinated warm, and linked-folder lifecycle
management remains active. The RAG embedder is different: startup must never warm it. A linked
folder sync with real queued ingestion may activate embeddings through the ordinary operation.
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


def test_no_startup_worker_warms_the_rag_embedder():
    referenced = _names_called(_lifespan_body()) | _names_called(_post_warm_body())
    assert "_warm_rag_embedder" not in referenced
    assert (
        "warm" not in referenced
    ), "startup calls an eager warm method; embeddings must load only from a real RAG operation"


def test_post_warm_thread_joins_before_platform_repair_and_folder_scheduling():
    body = _post_warm_body()
    join_line = None
    work_lines = []
    for sub in ast.walk(body):
        if isinstance(sub, ast.Name) and sub.id == "join_background_warm":
            join_line = sub.lineno if join_line is None else min(join_line, sub.lineno)
        if isinstance(sub, ast.Name) and sub.id in {
            "start_mlx_autorepair_if_needed",
            "_start_linked_folder_auto_sync",
        }:
            work_lines.append(sub.lineno)

    assert join_line is not None, "_post_warm_background_work must join the warm"
    assert work_lines, "post-warm lifecycle work was dropped"
    assert join_line < min(work_lines)


def test_post_warm_work_has_no_rag_warm_kill_switch_branch():
    """With no eager torch/RAG work, the post-warm worker needs no torch kill-switch gate."""
    body = _post_warm_body()
    guards = [
        node
        for node in body.body
        if isinstance(node, ast.If) and "DISABLE_ENV_VAR" in ast.unparse(node.test)
    ]
    assert guards == []
    assert "_warm_rag_embedder" not in _names_called(body)


def test_post_warm_thread_is_started_by_the_lifespan():
    """The lifespan must still put the thread up, now through a helper. The old inline Thread
    was untracked, so a shutdown landing while it was parked in the warm join could not stop
    it and a second lifespan stacked another. Follow the indirection, not the inline form."""
    referenced = _names_called(_lifespan_body())
    assert "_start_post_warm_thread" in referenced or (
        "_post_warm_background_work" in referenced
    ), (
        "nothing starts the post-warm thread, so MLX autorepair and linked-folder "
        "lifecycle management would never run"
    )

    # ...and the helper must target the real work, or a do-nothing thread passes.
    tree = ast.parse(_MAIN.read_text(encoding = "utf-8"))
    starter = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_start_post_warm_thread"
    )
    assert "_post_warm_background_work" in _names_called(starter)


def test_post_warm_runs_repair_and_linked_folder_lifecycle(monkeypatch):
    import main as main_mod
    import utils.mlx_repair as mlx_mod

    order: list[str] = []
    monkeypatch.setattr(main_mod, "join_background_warm", lambda *a, **k: order.append("join"))
    monkeypatch.setattr(
        mlx_mod, "start_mlx_autorepair_if_needed", lambda: order.append("mlx") or False
    )
    monkeypatch.setattr(
        main_mod,
        "_start_linked_folder_auto_sync",
        lambda generation: order.append("folder-sync"),
    )

    main_mod._post_warm_background_work()

    assert order == ["join", "mlx", "folder-sync"]


def test_a_failing_mlx_probe_does_not_strand_linked_folder_startup(monkeypatch):
    import main as main_mod
    import utils.mlx_repair as mlx_mod

    ran: list[str] = []
    monkeypatch.setattr(main_mod, "join_background_warm", lambda *a, **k: None)

    def _boom():
        raise RuntimeError("mlx probe exploded")

    monkeypatch.setattr(mlx_mod, "start_mlx_autorepair_if_needed", _boom)
    monkeypatch.setattr(
        main_mod, "_start_linked_folder_auto_sync", lambda generation: ran.append("folder-sync")
    )

    main_mod._post_warm_background_work()

    assert ran == ["folder-sync"]
