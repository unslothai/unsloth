# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The declared typer floor has to admit the annotations unsloth_cli actually uses.

unsloth_cli annotates typer options with ``typing.Literal`` (the speculative decoding
kind, the gpu memory mode, the reasoning switch). typer only grew Literal support in
0.19.0; below that every command dies before it parses a single argument with
``RuntimeError: Type not yet supported: typing.Literal[...]``. A floor below 0.19.0
is therefore not merely lax: pip honours it, keeps a preinstalled 0.12 to 0.18, and
the CLI is dead on arrival.

The floor is declared twice, in pyproject.toml and in the no-torch requirements file
that the studio installer applies with --no-deps. Both are checked, because the second
one is what an updating Studio user actually gets.

Runs under the GPU-free tests/conftest.py.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
NO_TORCH_RUNTIME = REPO_ROOT / "studio" / "backend" / "requirements" / "no-torch-runtime.txt"
CLI_ROOT = REPO_ROOT / "unsloth_cli"

# typer 0.19.0, "Support typing.Literal to define a set of predefined choices"
# (fastapi/typer#429, docs/release-notes.md). Raise this only alongside evidence
# from typer's own release notes; do not lower it while the CLI annotates options
# with Literal, which the first test below proves it does.
LITERAL_SUPPORTED_FROM = (0, 19, 0)


def _version_tuple(text):
    return tuple(int(part) for part in text.split("."))


def _declared_floor(text):
    """The lower bound of the first bare ``typer>=X`` requirement in *text*."""
    match = re.search(r"^\s*[\"']?typer\s*>=\s*([0-9]+(?:\.[0-9]+)*)", text, re.MULTILINE)
    assert match is not None, "no `typer>=` requirement found to check"
    return _version_tuple(match.group(1))


def _literal_annotated_typer_options():
    """Every ``name: ...Literal[...] = <default>`` parameter in a typer command module.

    Structural, via ast: a grep would also match Literal in pydantic models and in
    plain helper functions, which typer never sees.
    """
    found = []
    for path in sorted(CLI_ROOT.rglob("*.py")):
        if "tests" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding = "utf-8"))
        except SyntaxError:  # pragma: no cover - a parse error is another test's problem
            continue
        decorated = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any("command" in ast.dump(dec) for dec in node.decorator_list)
        ]
        for func in decorated:
            args = func.args
            for arg in list(args.args) + list(args.kwonlyargs) + list(args.posonlyargs):
                if arg.annotation is None:
                    continue
                if "Literal" in ast.dump(arg.annotation):
                    found.append(f"{path.relative_to(REPO_ROOT)}::{func.name}::{arg.arg}")
    return found


def test_the_cli_really_does_annotate_typer_options_with_literal():
    """The premise of the floor. Without this the next test would pass vacuously."""
    annotated = _literal_annotated_typer_options()
    assert annotated, (
        "no typer command annotates an option with Literal any more; if that is "
        "deliberate, the floor below can be revisited, but do not delete this test "
        "and leave the floor unexplained"
    )


@pytest.mark.parametrize(
    "path",
    [PYPROJECT, NO_TORCH_RUNTIME],
    ids = ["pyproject", "no-torch-runtime"],
)
def test_the_declared_typer_floor_supports_literal(path):
    floor = _declared_floor(path.read_text(encoding = "utf-8"))
    assert floor >= LITERAL_SUPPORTED_FROM, (
        f"{path.relative_to(REPO_ROOT)} declares typer>={'.'.join(map(str, floor))}, which lets "
        f"pip keep a preinstalled typer that cannot build "
        f"{_literal_annotated_typer_options()[:3]}"
    )


def test_both_declarations_agree():
    """The requirements file is installed --no-deps, so it cannot inherit the pyproject bound."""
    assert _declared_floor(PYPROJECT.read_text(encoding = "utf-8")) == _declared_floor(
        NO_TORCH_RUNTIME.read_text(encoding = "utf-8")
    )
