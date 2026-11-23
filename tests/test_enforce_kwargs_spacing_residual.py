# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""--residual: the two passes ruff-unsloth leaves to Python, skipped where they cannot apply.

The prefilters only decide whether a file reaches the AST; when one does, the result
must be exactly what the full post-pass (spacing and blank lines off) would write.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPTS = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import enforce_kwargs_spacing as K  # noqa: E402

# The full pass looks into a suite only when its parent suite holds two or more
# statements, so each `pass` case sits under a module with a second statement.
_TWO = "import os\n"

# Each changes under the full post-pass, so the prefilter must let it through.
_CHANGES = {
    "same_line_concatenation": 'x = "a" "b"\n',
    "plain_into_f_string": 'x = f"{y}" "tail {z}"\n',
    "pass_after_a_statement": _TWO + "def f():\n    y = 1\n    pass\n",
    "pass_before_a_statement": _TWO + "def f():\n    pass\n    return 1\n",
    "pass_after_a_compound_sibling": _TWO + "def f(x):\n    if x:\n        y()\n    pass\n",
    "pass_after_a_closing_bracket": _TWO + "def f():\n    g(\n        1,\n    )\n    pass\n",
    "pass_after_a_docstring_closing_at_column_0": _TWO + 'def f():\n    """doc\n"""\n    pass\n',
    "pass_after_a_comment_row": _TWO + "def f():\n    y = 1\n    # note\n    pass\n",
    "pass_after_a_backslash_continuation": _TWO + "def f():\n    y = 1 + \\\n2\n    pass\n",
}

# None of these changes, and none needs the AST to know it.
_UNTOUCHED = {
    "join_on_a_space": 'x = " ".join(parts)\n',
    "bytes_stay_side_by_side": 'x = b"a" b"b"\n',
    "f_strings_stay_side_by_side": 'x = f"{a}" f"{b}"\n',
    "mixed_quotes": "x = \"a\" 'b'\n",
    "lone_pass_in_a_class": "class A:\n    pass\n",
    "lone_pass_in_a_handler": "try:\n    import x\nexcept ImportError:\n    pass\n",
    "no_strings_no_pass": "def f(a = 1):\n    return a\n",
}


def _full(text: str) -> tuple[str, bool]:
    return K._post(text, spacing = False, blanks = False)


@pytest.mark.parametrize("name", sorted(_CHANGES))
def test_a_file_the_full_pass_changes_gets_the_same_change(name):
    text = _CHANGES[name]
    expected = _full(text)
    assert expected[1], f"{name} no longer changes under the full pass; the case is stale"
    assert K.residual(text) == expected


@pytest.mark.parametrize("name", sorted(_UNTOUCHED))
def test_a_file_nothing_applies_to_never_reaches_the_ast(name, monkeypatch):
    text = _UNTOUCHED[name]
    assert _full(text) == (text, False)

    def refuse(*_args, **_kwargs):
        raise AssertionError(f"{name} reached the AST pass")

    monkeypatch.setattr(K, "_post", refuse)
    assert K.residual(text) == (text, False)


def test_residual_leaves_kwarg_spacing_and_import_blank_lines_alone():
    # ruff-unsloth owns both; doing them again here would fight it.
    text = 'def f():\n    import os\n\n    return g(a=1, s="x" "y")\n'
    updated, changed = K.residual(text)
    assert changed
    assert updated == 'def f():\n    import os\n\n    return g(a=1, s="xy")\n'


def test_the_residual_flag_runs_only_the_residual_passes(tmp_path):
    target = tmp_path / "sample.py"
    target.write_text('x = f(a=1, b="p" "q")\n', encoding = "utf-8")
    script = Path(_SCRIPTS) / "enforce_kwargs_spacing.py"
    subprocess.run(
        [sys.executable, str(script), "--residual", str(target)], check = True, capture_output = True
    )
    assert target.read_text(encoding = "utf-8") == 'x = f(a=1, b="pq")\n'


def test_this_repos_scripts_agree_with_the_full_pass():
    # Real files, not snippets: every one either skips the AST or matches the full pass.
    for path in sorted(Path(_SCRIPTS).glob("*.py")):
        text = path.read_text(encoding = "utf-8")
        assert K.residual(text) == _full(text), path.name
