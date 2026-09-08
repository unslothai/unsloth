# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""Regression tests for docker/unsloth_nb_pip_magic.py.

The input transformer rewrites `!<python> -m pip|uv ...` to `!pip|uv ...` so it
resolves to the PATH shim. Transformers see the RAW cell text (brace expansion
happens later, in the system() path), so the braced and absolute-interpreter forms
notebooks use to target the running kernel must be rewritten too.
"""

import importlib.util
import pathlib
import pytest

_MOD_PATH = pathlib.Path(__file__).resolve().parents[2] / "docker" / "unsloth_nb_pip_magic.py"
_spec = importlib.util.spec_from_file_location("unsloth_nb_pip_magic", _MOD_PATH)
magic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(magic)


def _rewrite(line):
    return magic._rewrite_python_dash_m([line])[0]


@pytest.mark.parametrize(
    "_p0, expected",
    [
        pytest.param("!python -m pip install peft\n", "!pip install peft\n", id = "literal_python_rewritten"),
        pytest.param("!python3.12 -m pip install peft", "!pip install peft", id = "literal_python_version_rewritten"),
        pytest.param("!{sys.executable} -m pip install peft\n", "!pip install peft\n", id = "sys_executable_braces_rewritten"),
        pytest.param('!"{sys.executable}" -m pip install peft', "!pip install peft", id = "sys_executable_braces_quoted_rewritten"),
        pytest.param("!{ sys.executable } -m pip install peft", "!pip install peft", id = "sys_executable_braces_spaced_rewritten"),
        pytest.param("!/opt/unsloth-venv/bin/python -m pip install peft\n", "!pip install peft\n", id = "absolute_interpreter_path_rewritten"),
        pytest.param("!/usr/bin/python3.11 -m uv pip install peft", "!uv pip install peft", id = "absolute_interpreter_versioned_path_rewritten"),
        pytest.param('!"/opt/unsloth venv/bin/python" -m pip install peft', "!pip install peft", id = "quoted_interpreter_path_rewritten"),
        pytest.param("    !{sys.executable} -m pip install peft", "    !pip install peft", id = "indent_preserved"),
    ],
)
def test_module_cases(_p0, expected):
    assert _rewrite(_p0) == expected




def test_python_script_not_rewritten():
    line = "!python train.py --epochs 3"
    assert _rewrite(line) == line


def test_module_other_than_pip_not_rewritten():
    line = "!python -m venv .venv"
    assert _rewrite(line) == line


def test_non_shell_line_not_rewritten():
    line = "x = '{sys.executable} -m pip install peft'"
    assert _rewrite(line) == line
