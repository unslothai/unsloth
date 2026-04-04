"""Tests that the 'unsloth studio' CLI defaults to 127.0.0.1.

TDD: these tests should FAIL before the CLI host default is changed,
and PASS afterwards.

Uses AST parsing to inspect source-level defaults without requiring the
full unsloth_cli dependencies (pydantic, etc. are not available in unit
test environments without the full venv).

Run with:
    python -m pytest tests/studio/test_cli_studio_defaults.py -v
"""

import ast
from pathlib import Path

import pytest

_STUDIO_CMD_PY = Path(__file__).resolve().parents[2] / "unsloth_cli" / "commands" / "studio.py"


def _find_typer_option_default(source: str, func_name: str, long_option: str):
    """Return the default value of a typer.Option() parameter in *func_name*.

    Matches by the long option name (e.g. '--host') among the positional args
    of the typer.Option() call and returns the first arg (the default value).
    Only handles ast.Constant defaults (strings, ints, bools).
    """
    tree = ast.parse(source)
    for func_node in ast.walk(tree):
        if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if func_node.name != func_name:
            continue
        args = func_node.args.args
        defaults = func_node.args.defaults
        offset = len(args) - len(defaults)
        for i, default in enumerate(defaults):
            if not isinstance(default, ast.Call):
                continue
            # Identify this Option by the presence of the long option name string
            has_long_option = any(
                isinstance(a, ast.Constant) and a.value == long_option
                for a in default.args
            )
            if not has_long_option:
                continue
            # First positional arg is the default value
            if default.args and isinstance(default.args[0], ast.Constant):
                return default.args[0].value
    return None


def test_studio_cli_default_host_is_loopback():
    """'unsloth studio' with no -H flag must default to 127.0.0.1, not 0.0.0.0.

    The typer.Option default is the value forwarded as --host when the user
    invokes 'unsloth studio' without an explicit -H flag.  Defaulting to
    0.0.0.0 exposes the service on all interfaces without the user opting in.
    """
    source = _STUDIO_CMD_PY.read_text()
    host_default = _find_typer_option_default(source, "studio_default", "--host")
    assert host_default is not None, (
        "Could not find typer.Option('...', '--host', ...) in studio_default(); "
        "check that the parameter exists with a long-form '--host' option."
    )
    assert host_default == "127.0.0.1", (
        f"studio_default() host typer.Option default must be '127.0.0.1', "
        f"got '{host_default}'. This is the value forwarded to run.py when "
        f"the user invokes 'unsloth studio' without -H."
    )


def test_studio_cli_option_accepts_explicit_override():
    """The --host option must still accept arbitrary values like 0.0.0.0.

    Verify that the option is not locked to 127.0.0.1 — the default should
    be loopback but users must be able to opt in to all-interfaces binding.
    This is confirmed by the option having a configurable typer.Option (not
    a hard-coded constant), identified by the '--host'/'-H' option names.
    """
    source = _STUDIO_CMD_PY.read_text()
    # Confirm the option exists (is a typer.Option call, not a plain default)
    tree = ast.parse(source)
    found_option_call = False
    for func_node in ast.walk(tree):
        if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if func_node.name != "studio_default":
            continue
        for default in func_node.args.defaults:
            if not isinstance(default, ast.Call):
                continue
            has_host_option = any(
                isinstance(a, ast.Constant) and a.value in ("--host", "-H")
                for a in default.args
            )
            if has_host_option:
                found_option_call = True
                break
    assert found_option_call, (
        "studio_default() 'host' parameter must be a typer.Option() call with "
        "'--host'/'-H' flags so users can override it; plain default found instead."
    )
