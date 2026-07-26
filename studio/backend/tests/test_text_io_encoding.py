# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Text I/O must name its encoding, or Windows silently uses the ANSI codepage.

``open()``, ``Path.read_text()`` and ``subprocess(text = True)`` fall back to
``locale.getencoding()`` when no ``encoding`` is passed. On Windows that is
cp1252 (or cp932, cp1251, ... by system locale), not UTF-8, so a chat template,
model config or path containing ``ä ö ü → 世`` mojibakes or raises
``UnicodeDecodeError`` mid-load. Studio's files are UTF-8, so say so.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


BACKEND_ROOT = Path(__file__).resolve().parent.parent

# Not Studio's own runtime source. Shipped plugins under plugins/*/src are, so
# only their build artifacts are skipped.
_SKIPPED_DIRS = ("node_modules", "build", "tests", "__pycache__", "sandbox_site")

# Path.open() takes a mode first and only these keywords. fitz.open(stream=...)
# and av.open(..., metadata_errors=...) are other libraries' open(), so matching
# the signature is what separates them.
_FILE_MODE_CHARS = set("rwxabt+")
_PATH_OPEN_KWARGS = {"mode", "buffering", "encoding", "errors", "newline"}

_SUBPROCESS_CALLS = {"run", "Popen", "check_output", "check_call", "call"}


def _studio_sources() -> list[Path]:
    return [
        path
        for path in sorted(BACKEND_ROOT.rglob("*.py"))
        if not any(part in _SKIPPED_DIRS for part in path.relative_to(BACKEND_ROOT).parts)
    ]


def _has_keyword(node: ast.Call, name: str) -> bool:
    return any(keyword.arg == name for keyword in node.keywords)


def _mode_is_binary(node: ast.Call) -> bool:
    mode: str | None = None
    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
        value = node.args[1].value
        mode = value if isinstance(value, str) else None
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            value = keyword.value.value
            if isinstance(value, str):
                mode = value
    return bool(mode and "b" in mode)


def _path_open_mode(node: ast.Call) -> str | None:
    if node.args and isinstance(node.args[0], ast.Constant):
        value = node.args[0].value
        if isinstance(value, str):
            return value
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Constant):
            value = keyword.value.value
            if isinstance(value, str):
                return value
    return None


def _is_path_open(node: ast.Call) -> bool:
    """True only for calls matching ``Path.open``'s signature."""
    if len(node.args) > 1:
        return False
    if any(k.arg not in _PATH_OPEN_KWARGS for k in node.keywords):
        return False
    mode = _path_open_mode(node)
    if mode is not None:
        return bool(mode) and set(mode) <= _FILE_MODE_CHARS
    return not node.args


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_subprocess_call(node: ast.Call) -> bool:
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr not in _SUBPROCESS_CALLS:
        return False
    value = func.value
    return isinstance(value, ast.Name) and value.id == "subprocess"


def _text_mode_subprocess(node: ast.Call) -> bool:
    for keyword in node.keywords:
        if keyword.arg not in ("text", "universal_newlines"):
            continue
        if isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
            return True
    return False


def _offenders(path: Path) -> list[str]:
    source = path.read_text(encoding = "utf-8")
    tree = ast.parse(source, filename = str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)

        if _is_subprocess_call(node):
            if _text_mode_subprocess(node) and not _has_keyword(node, "encoding"):
                found.append(f"{path.name}:{node.lineno}: subprocess(text = True) without encoding")
            continue

        if name == "open" and isinstance(node.func, ast.Name):
            if _mode_is_binary(node) or _has_keyword(node, "encoding"):
                continue
            found.append(f"{path.name}:{node.lineno}: open() without encoding")
            continue

        if name == "open" and isinstance(node.func, ast.Attribute):
            if not _is_path_open(node) or _has_keyword(node, "encoding"):
                continue
            if _path_open_mode(node) and "b" in _path_open_mode(node):
                continue
            found.append(f"{path.name}:{node.lineno}: Path.open() without encoding")
            continue

        if name in ("read_text", "write_text") and isinstance(node.func, ast.Attribute):
            if _has_keyword(node, "encoding"):
                continue
            # importlib.metadata Distribution.read_text() takes no encoding kwarg.
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "dist":
                continue
            found.append(f"{path.name}:{node.lineno}: {name}() without encoding")
    return found


@pytest.mark.parametrize("path", _studio_sources(), ids = lambda p: str(p.name))
def test_text_io_names_its_encoding(path: Path) -> None:
    offenders = _offenders(path)
    assert not offenders, (
        "Text I/O without an explicit encoding falls back to the Windows ANSI "
        "codepage and corrupts non-ASCII (ä ö ü → 世). Pass encoding = \"utf-8\":\n  "
        + "\n  ".join(offenders)
    )

