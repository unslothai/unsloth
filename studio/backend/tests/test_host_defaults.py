# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests that Unsloth Studio defaults to 127.0.0.1 (loopback) not 0.0.0.0.

Uses AST parsing to inspect source-level defaults without requiring the
full studio venv (run.py has heavy dependencies like structlog/uvicorn).
"""

import ast
from pathlib import Path

_RUN_PY = Path(__file__).resolve().parent.parent / "run.py"


def _parse_function_param_defaults(source: str, func_name: str) -> dict:
    """Return {param_name: default_value} for a named function in *source*.

    Only handles ast.Constant defaults (strings, ints, bools).
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == func_name
        ):
            result = {}
            all_args = node.args.args
            defaults = node.args.defaults
            # Defaults are right-aligned against the args list
            offset = len(all_args) - len(defaults)
            for i, default in enumerate(defaults):
                arg_name = all_args[offset + i].arg
                if isinstance(default, ast.Constant):
                    result[arg_name] = default.value
            return result
    return {}


def _parse_argparse_add_argument_default(source: str, option_name: str):
    """Return the 'default' kwarg value for add_argument(option_name, ...) in *source*.

    Walks the entire module so the call can live in __main__ or in a helper
    function — only handles ast.Constant defaults.
    """
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
    """run_server() parameter default for 'host' must be 127.0.0.1, not 0.0.0.0.

    Binding to 0.0.0.0 by default exposes the service on all network
    interfaces, contradicting the documented "privacy first / 100% local"
    guarantee.  Loopback (127.0.0.1) is the least-permissive default;
    users who need network access can pass -H 0.0.0.0 explicitly.
    """
    source = _RUN_PY.read_text()
    defaults = _parse_function_param_defaults(source, "run_server")
    assert (
        "host" in defaults
    ), "run_server() must have a 'host' parameter with a default"
    host_default = defaults["host"]
    assert host_default == "127.0.0.1", (
        f"run_server() host default must be '127.0.0.1' (loopback) "
        f"but got '{host_default}'. Binding to '{host_default}' by default "
        f"exposes the service beyond localhost."
    )


def test_argparse_default_host_is_loopback():
    """argparse --host add_argument default must be 127.0.0.1.

    When run.py is invoked directly (python run.py), the argparse default
    should match the function default so direct execution is equally safe.
    """
    source = _RUN_PY.read_text()
    host_default = _parse_argparse_add_argument_default(source, "--host")
    assert (
        host_default is not None
    ), "Could not find add_argument('--host', ...) in run.py"
    assert (
        host_default == "127.0.0.1"
    ), f"run.py argparse --host default must be '127.0.0.1', got '{host_default}'"
