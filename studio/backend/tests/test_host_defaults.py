# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Checks run.py defaults the host to loopback 127.0.0.1, not 0.0.0.0, read by AST without the venv."""

import ast
from pathlib import Path

_RUN_PY = Path(__file__).resolve().parent.parent / "run.py"


def _parse_function_param_defaults(source: str, func_name: str) -> dict:
    """Handles only ast.Constant defaults, i.e. literal strings, ints and bools."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            result = {}
            all_args = node.args.args
            defaults = node.args.defaults
            offset = len(all_args) - len(defaults)
            for i, default in enumerate(defaults):
                arg_name = all_args[offset + i].arg
                if isinstance(default, ast.Constant):
                    result[arg_name] = default.value
            return result
    return {}


def _parse_argparse_add_argument_default(source: str, option_name: str):
    """Walks the whole module so the add_argument call may sit in __main__ or a helper; literals only."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        if not (isinstance(first_arg, ast.Constant) and first_arg.value == option_name):
            continue
        for kw in node.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                return kw.value.value
    return None


def test_run_server_default_host_is_loopback():
    """run_server must default host to loopback: 0.0.0.0 exposes the service on every interface."""
    source = _RUN_PY.read_text(encoding = "utf-8")
    defaults = _parse_function_param_defaults(source, "run_server")
    assert "host" in defaults, "run_server() must have a 'host' parameter with a default"
    host_default = defaults["host"]
    assert host_default == "127.0.0.1", (
        f"run_server() host default must be '127.0.0.1' (loopback) "
        f"but got '{host_default}'. Binding to '{host_default}' by default "
        f"exposes the service beyond localhost."
    )


def test_argparse_default_host_is_loopback():
    """The argparse --host default must match the function default, since direct python run.py uses it."""
    source = _RUN_PY.read_text(encoding = "utf-8")
    host_default = _parse_argparse_add_argument_default(source, "--host")
    assert host_default is not None, "Could not find add_argument('--host', ...) in run.py"
    assert (
        host_default == "127.0.0.1"
    ), f"run.py argparse --host default must be '127.0.0.1', got '{host_default}'"
