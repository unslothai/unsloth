"""Tests that ``unsloth run`` is registered as a top-level alias for
``unsloth studio run``.

AST-based to avoid importing ``unsloth_cli`` (which pulls in the heavy
training stack) at test-collection time.
"""

from __future__ import annotations

import ast
from pathlib import Path

_CLI_INIT = Path(__file__).resolve().parents[2] / "unsloth_cli" / "__init__.py"


def _module_calls(source: str):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node


def test_top_level_run_alias_registered():
    """`app.command("run", ...)` must be invoked with studio_run as its target."""
    source = _CLI_INIT.read_text()

    # Find ``app.command("run", ...)`` call -- the decorator-call form.
    found_decorator_call = False
    for call in _module_calls(source):
        # Match ``app.command(...)`` syntactically.
        if not (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "command"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "app"
        ):
            continue
        # Decorator-call form has a string literal "run" as the first
        # positional or as keyword ``name="run"``.
        first_pos = call.args[0] if call.args else None
        keyword_name = next(
            (kw.value for kw in call.keywords if kw.arg == "name"), None
        )
        is_run = (isinstance(first_pos, ast.Constant) and first_pos.value == "run") or (
            isinstance(keyword_name, ast.Constant) and keyword_name.value == "run"
        )
        if is_run:
            found_decorator_call = True
            break
    assert (
        found_decorator_call
    ), 'Expected `app.command("run", ...)` registration in unsloth_cli/__init__.py'


def test_studio_run_imported_for_alias():
    """The alias must wire up to the studio.run function, not redefine it."""
    source = _CLI_INIT.read_text()
    tree = ast.parse(source)
    has_import = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != "unsloth_cli.commands.studio":
            continue
        for alias in node.names:
            if alias.name == "run":
                has_import = True
                break
    assert has_import, "Expected `from unsloth_cli.commands.studio import run` in unsloth_cli/__init__.py"
