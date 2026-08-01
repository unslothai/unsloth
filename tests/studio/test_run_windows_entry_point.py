"""AST test that `unsloth run` looks for the console script Windows actually ships.

The venv's entry point is `unsloth.exe` on Windows, so resolving the bare name made
`studio_bin.is_file()` false on every Windows install and aborted with "Unsloth venv
missing 'unsloth' entry point". Pinned by AST because reaching the assignment at
runtime means standing up a studio venv.
"""

from __future__ import annotations

import ast
from pathlib import Path

_STUDIO = Path(__file__).resolve().parents[2] / "unsloth_cli" / "commands" / "studio.py"


def _run_function() -> ast.FunctionDef:
    tree = ast.parse(_STUDIO.read_text(encoding = "utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            return node
    raise AssertionError("no top-level `run` command in unsloth_cli/commands/studio.py")


def _studio_bin_value() -> ast.expr:
    for node in ast.walk(_run_function()):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id == "studio_bin"
                and node.value is not None
            ):
                if not (isinstance(node.value, ast.Constant) and node.value.value is None):
                    return node.value
    raise AssertionError("`run` never assigns a studio_bin path")


def test_the_entry_point_name_is_chosen_per_platform():
    value = _studio_bin_value()
    assert isinstance(value, ast.BinOp) and isinstance(value.op, ast.Div), (
        "expected studio_bin to be built as `studio_python.parent / <name>`, got "
        f"{ast.dump(value)}"
    )
    names = {
        node.value
        for node in ast.walk(value.right)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert {
        "unsloth",
        "unsloth.exe",
    } <= names, f"studio_bin must pick between 'unsloth' and 'unsloth.exe'; got {sorted(names)}"


def test_the_windows_branch_is_the_exe():
    """A swapped conditional would still hold the two names but break both platforms."""
    branch = next(
        node for node in ast.walk(_studio_bin_value().right) if isinstance(node, ast.IfExp)
    )
    assert isinstance(branch.body, ast.Constant) and branch.body.value == "unsloth.exe"
    assert isinstance(branch.orelse, ast.Constant) and branch.orelse.value == "unsloth"
    assert "Windows" in {
        node.value
        for node in ast.walk(branch.test)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }, "the .exe branch must be gated on platform.system() == 'Windows'"
