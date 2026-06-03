"""Tests that the 'unsloth studio' CLI defaults to 127.0.0.1.

Uses AST parsing to inspect source-level defaults without requiring the
full unsloth_cli dependencies (typer/pydantic) at test-collection time.
"""

import ast
from pathlib import Path

_STUDIO_CMD_PY = (
    Path(__file__).resolve().parents[2] / "unsloth_cli" / "commands" / "studio.py"
)


def _find_typer_option_default(source: str, func_name: str, long_option: str):
    """Return the default value of a typer.Option(...) parameter in *func_name*.

    Matches by the long option name (e.g. '--host') among the positional args
    of the typer.Option() call and returns the first positional arg (the
    default value). Only handles ast.Constant defaults.
    """
    tree = ast.parse(source)
    for func_node in ast.walk(tree):
        if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if func_node.name != func_name:
            continue
        # Walk both regular args and kwonly args, each paired with its default.
        all_args = func_node.args.args + func_node.args.kwonlyargs
        all_defaults = func_node.args.defaults + [
            d for d in func_node.args.kw_defaults if d is not None
        ]
        # ast pads defaults right-aligned against args (ignoring kwonly). We
        # iterate calls directly, which is simpler and robust.
        for default in all_defaults:
            if not isinstance(default, ast.Call):
                continue
            call_func = default.func
            is_typer_option = (
                isinstance(call_func, ast.Attribute)
                and call_func.attr == "Option"
                and isinstance(call_func.value, ast.Name)
                and call_func.value.id == "typer"
            )
            if not is_typer_option:
                continue
            # First positional is the default value; remaining positionals are
            # option flags like "--host", "-H".
            if not default.args:
                continue
            flags = [
                a.value
                for a in default.args[1:]
                if isinstance(a, ast.Constant) and isinstance(a.value, str)
            ]
            if long_option not in flags:
                continue
            first = default.args[0]
            if isinstance(first, ast.Constant):
                return first.value
    return None


def test_studio_default_host_is_loopback():
    """`unsloth studio` (studio_default) --host typer Option default must be 127.0.0.1."""
    source = _STUDIO_CMD_PY.read_text()
    host_default = _find_typer_option_default(source, "studio_default", "--host")
    assert (
        host_default is not None
    ), "Could not find --host typer.Option default in studio_default()"
    assert host_default == "127.0.0.1", (
        f"studio_default() --host default must be '127.0.0.1' (loopback) "
        f"but got '{host_default}'."
    )


def test_studio_run_host_is_loopback():
    """`unsloth studio run` --host typer Option default must be 127.0.0.1."""
    source = _STUDIO_CMD_PY.read_text()
    host_default = _find_typer_option_default(source, "run", "--host")
    assert (
        host_default is not None
    ), "Could not find --host typer.Option default in run()"
    assert host_default == "127.0.0.1", (
        f"`unsloth studio run` --host default must be '127.0.0.1' (loopback) "
        f"but got '{host_default}'."
    )
