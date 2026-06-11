# Unsloth - 2x faster, 60% less VRAM LLM training and finetuning
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

"""Regression tests for unslothai/unsloth#2660.

On Windows the default text encoding is the locale code page (e.g. cp1252),
not UTF-8. ``subprocess.Popen`` / ``subprocess.run`` opened in text mode
(``text=True`` / ``universal_newlines=True``) without an explicit
``encoding`` therefore decode child-process output with cp1252. When
llama.cpp / Ollama emit a byte that is undefined in cp1252 (e.g. ``0x9d``,
which appears inside the UTF-8 encoding of common punctuation and box-drawing
glyphs), the read raises ``UnicodeDecodeError`` and aborts the GGUF export.

Two checks:

* ``test_save_subprocess_text_calls_declare_utf8_encoding`` -- a source-level
  drift detector. It parses ``unsloth/save.py`` (no import, so it runs under
  the GPU/torch-free harness) and fails if any text-mode subprocess call is
  missing ``encoding="utf-8"``. This is the regression guard: it is red
  before the fix and green after.
* ``test_utf8_replace_decodes_non_cp1252_subprocess_output`` -- a behavioural
  check that documents the bug and the fix deterministically on any platform:
  raw child output that is invalid under cp1252 raises, while the
  ``encoding="utf-8", errors="replace"`` kwargs used by the fix read it
  cleanly.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SAVE_PY = Path(__file__).resolve().parents[2] / "unsloth" / "save.py"


def _is_subprocess_call(node: ast.Call) -> bool:
    """True for ``subprocess.Popen(...)`` / ``subprocess.run(...)``."""
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr in {"Popen", "run"}
        and isinstance(func.value, ast.Name)
        and func.value.id == "subprocess"
    )


def _kw(node: ast.Call, name: str):
    for kw in node.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _is_true(value) -> bool:
    return isinstance(value, ast.Constant) and value.value is True


def _is_text_mode(node: ast.Call) -> bool:
    """Text mode = ``text=True`` or ``universal_newlines=True``."""
    return _is_true(_kw(node, "text")) or _is_true(_kw(node, "universal_newlines"))


def _collect_text_mode_subprocess_calls() -> list[ast.Call]:
    tree = ast.parse(SAVE_PY.read_text(encoding = "utf-8"), filename = str(SAVE_PY))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _is_subprocess_call(node)
        and _is_text_mode(node)
    ]


def test_text_mode_subprocess_calls_exist():
    """Guard the guard: if save.py stops using text-mode subprocess calls the
    drift test below would vacuously pass, so make sure we are actually
    inspecting something."""
    calls = _collect_text_mode_subprocess_calls()
    assert len(calls) >= 6, (
        f"Expected several text-mode subprocess calls in {SAVE_PY.name}, "
        f"found {len(calls)} -- has the file been restructured?"
    )


def test_save_subprocess_text_calls_declare_utf8_encoding():
    """Every text-mode subprocess call in save.py must pin encoding='utf-8'.

    Without it, reading llama.cpp/Ollama output crashes on Windows (cp1252).
    Fails before the #2660 fix, passes after.
    """
    offenders = []
    for node in _collect_text_mode_subprocess_calls():
        enc = _kw(node, "encoding")
        ok = isinstance(enc, ast.Constant) and enc.value == "utf-8"
        if not ok:
            offenders.append(node.lineno)

    assert not offenders, (
        "Text-mode subprocess call(s) in unsloth/save.py missing "
        'encoding="utf-8" (UnicodeDecodeError on Windows, #2660) at line(s): '
        + ", ".join(map(str, sorted(offenders)))
    )


def test_utf8_replace_decodes_non_cp1252_subprocess_output():
    """Document the failure and the fix with a real subprocess.

    The child emits U+201D (right double quote), whose UTF-8 encoding
    ``E2 80 9D`` contains byte 0x9D -- undefined in cp1252. Decoding the raw
    bytes as cp1252 raises (the bug); the fix's kwargs read it cleanly.
    """
    # All-ASCII argv; the child builds the non-ASCII char itself so this is
    # deterministic regardless of the parent's locale.
    child = (
        "import sys; "
        "sys.stdout.buffer.write(('tensor ' + chr(0x201D) + ' x\\n').encode('utf-8'))"
    )

    raw = subprocess.run(
        [sys.executable, "-c", child], capture_output = True
    ).stdout
    assert b"\x9d" in raw  # precondition: output carries the cp1252-undefined byte

    # Failing behaviour before the fix: cp1252 (the Windows default) cannot
    # decode this output.
    with pytest.raises(UnicodeDecodeError):
        raw.decode("cp1252")

    # Correct behaviour after the fix: the exact kwargs save.py now uses.
    result = subprocess.run(
        [sys.executable, "-c", child],
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "replace",
    )
    assert result.stdout.startswith("tensor ")
    assert "”" in result.stdout
