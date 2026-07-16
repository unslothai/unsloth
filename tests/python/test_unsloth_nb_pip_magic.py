# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""Regression tests for docker/unsloth_nb_pip_magic.py.

The input transformer rewrites explicit `!<python> -m pip|uv ...` shell lines
to `!pip|uv ...` so they resolve to the PATH shim. IPython input transformers
see the RAW cell text (brace expansion like `{sys.executable}` happens later,
in the system() execution path), so the braced and absolute-interpreter forms
notebooks use to target the running kernel must be rewritten too (item
3567875025); only matching literal `python`/`py` let module-pip bypass the
shim entirely.
"""

import importlib.util
import pathlib

_MOD_PATH = pathlib.Path(__file__).resolve().parents[2] / "docker" / "unsloth_nb_pip_magic.py"
_spec = importlib.util.spec_from_file_location("unsloth_nb_pip_magic", _MOD_PATH)
magic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(magic)


def _rewrite(line):
    return magic._rewrite_python_dash_m([line])[0]


def test_literal_python_rewritten():
    assert _rewrite("!python -m pip install peft\n") == "!pip install peft\n"


def test_literal_python_version_rewritten():
    assert _rewrite("!python3.12 -m pip install peft") == "!pip install peft"


def test_sys_executable_braces_rewritten():
    assert _rewrite("!{sys.executable} -m pip install peft\n") == "!pip install peft\n"


def test_sys_executable_braces_quoted_rewritten():
    assert _rewrite('!"{sys.executable}" -m pip install peft') == "!pip install peft"


def test_sys_executable_braces_spaced_rewritten():
    assert _rewrite("!{ sys.executable } -m pip install peft") == "!pip install peft"


def test_absolute_interpreter_path_rewritten():
    assert _rewrite("!/opt/unsloth-venv/bin/python -m pip install peft\n") == "!pip install peft\n"


def test_absolute_interpreter_versioned_path_rewritten():
    assert _rewrite("!/usr/bin/python3.11 -m uv pip install peft") == "!uv pip install peft"


def test_quoted_interpreter_path_rewritten():
    assert _rewrite('!"/opt/unsloth venv/bin/python" -m pip install peft') == "!pip install peft"


def test_indent_preserved():
    assert _rewrite("    !{sys.executable} -m pip install peft") == "    !pip install peft"


def test_python_script_not_rewritten():
    line = "!python train.py --epochs 3"
    assert _rewrite(line) == line


def test_module_other_than_pip_not_rewritten():
    line = "!python -m venv .venv"
    assert _rewrite(line) == line


def test_non_shell_line_not_rewritten():
    line = "x = '{sys.executable} -m pip install peft'"
    assert _rewrite(line) == line
