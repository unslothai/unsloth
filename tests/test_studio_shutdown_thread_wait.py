# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Source-level regression tests for terminal shutdown ordering (no backend import)."""

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_RUN_PY = _ROOT / "studio" / "backend" / "run.py"
_STUDIO_CLI_PY = _ROOT / "unsloth_cli" / "commands" / "studio.py"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding = "utf-8"))


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def _calls_name(tree: ast.AST, name: str) -> int:
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    )


def _calls_shutdown_wait_getattr(tree: ast.AST) -> int:
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Call):
            continue
        if not (isinstance(func.func, ast.Name) and func.func.id == "getattr"):
            continue
        if len(func.args) < 2:
            continue
        target, attr = func.args[:2]
        if (
            isinstance(target, ast.Name)
            and target.id == "run_mod"
            and isinstance(attr, ast.Constant)
            and attr.value == "_wait_for_server_shutdown"
        ):
            count += 1
    return count


def test_run_server_records_uvicorn_thread_for_terminal_shutdown_wait():
    tree = _parse(_RUN_PY)
    run_server = _function(tree, "run_server")

    assigns_thread_global = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "_server_thread"
            for target in node.targets
        )
        for node in ast.walk(run_server)
    )

    assert (
        assigns_thread_global
    ), "run_server must retain the uvicorn thread so terminal shutdown can join it"


def test_wait_for_server_shutdown_joins_uvicorn_thread():
    tree = _parse(_RUN_PY)
    wait_func = _function(tree, "_wait_for_server_shutdown")

    joins_thread = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        for node in ast.walk(wait_func)
    )

    assert (
        joins_thread
    ), "_wait_for_server_shutdown must join the uvicorn thread before process exit"


def test_wait_for_server_shutdown_join_is_bounded():
    tree = _parse(_RUN_PY)
    wait_func = _function(tree, "_wait_for_server_shutdown")

    bounded_join = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and any(kw.arg == "timeout" for kw in node.keywords)
        for node in ast.walk(wait_func)
    )

    assert bounded_join, "the join must pass a timeout so a stalled uvicorn shutdown cannot hang the terminal"


def test_direct_backend_entrypoint_waits_before_returning_to_shell():
    tree = _parse(_RUN_PY)

    assert (
        _calls_name(tree, "_wait_for_server_shutdown") >= 1
    ), "run.py must wait after the main shutdown event loop before returning to the shell"


def test_signal_handler_restores_default_handlers_for_force_quit():
    tree = _parse(_RUN_PY)
    handler = _function(tree, "_signal_handler")

    restores_default = any(
        isinstance(node, ast.Attribute) and node.attr == "SIG_DFL"
        for node in ast.walk(handler)
    )

    assert (
        restores_default
    ), "the signal handler must restore SIG_DFL so a second Ctrl+C can force-quit"


def test_cli_entrypoints_wait_before_returning_to_shell():
    tree = _parse(_STUDIO_CLI_PY)

    assert (
        _calls_shutdown_wait_getattr(tree) >= 3
    ), "Unsloth CLI terminal paths must wait for the backend thread after requesting shutdown"
