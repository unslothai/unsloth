# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The app is "Unsloth" or "Unsloth Studio" in backend messages, never a bare "Studio"."""

import ast
import re
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
BARE_STUDIO = re.compile(
    r"(?<!Unsloth )(?<!Unsloth-)(?<!LM )(?<!Recipe )(?<!Visual )(?<!Android )"
    r"(?<![A-Za-z0-9_./\\-])Studio(?![A-Za-z0-9_])"
)
ROUTE = re.compile(r"\.(get|post|put|patch|delete|api_route|websocket)$")
# Not the product name: a photography term, a fixed chat-template probe and an
# internal row name.
ALLOWED = (
    '"Studio photograph',
    "\"Studio '",
    "Studio replay probe",
    "after the Studio row committed",
)


def _offenders(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    skip = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if not (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            continue
        # Route docstrings are OpenAPI descriptions; other docstrings are code notes.
        is_route = any(
            ROUTE.search(ast.unparse(d.func if isinstance(d, ast.Call) else d))
            for d in getattr(node, "decorator_list", [])
        ) and not isinstance(node, ast.ClassDef)
        if not is_route:
            skip.add(id(body[0].value))
    lines = path.read_text(encoding = "utf-8").split("\n")
    found = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in skip
            and BARE_STUDIO.search(node.value)
        ):
            text = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            for line in text.split("\n"):
                if BARE_STUDIO.search(line) and not line.lstrip().startswith("//"):
                    if not any(a in line for a in ALLOWED):
                        found.append(f"{path.relative_to(BACKEND)}:{line.strip()}")
    return found


def test_backend_strings_never_call_the_app_a_bare_studio():
    offenders = []
    for path in sorted(BACKEND.rglob("*.py")):
        rel = path.relative_to(BACKEND).parts
        if "tests" in rel or "vendor" in rel or rel[0].startswith("."):
            continue
        offenders.extend(_offenders(path))
    assert offenders == []
