# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The python tool screens the source the interpreter runs: a leading BOM, a `# coding:` line and a
literal `python -c` payload must not hide code from the static screen or the auto-mode approval."""

import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference import tools as t

_EGRESS = (
    "import urllib.request\n"
    "print(urllib.request.urlopen('http://169.254.169.254/latest', timeout=5).status)\n"
)


def _flagged(code: str) -> bool:
    return t._check_code_safety(code) is not None and t.is_high_risk_tool_call(
        "python", {"code": code}
    )


def test_leading_bom_is_screened_and_needs_approval():
    assert _flagged(_EGRESS)
    assert _flagged("\ufeff" + _EGRESS)


@pytest.mark.parametrize(
    "argv0", ["/usr/bin/python3", "python", "python3.12", "C:\\\\Python312\\\\python.exe"]
)
def test_literal_python_c_payload_is_screened(argv0):
    code = f"import subprocess\nsubprocess.run([{argv0!r}, '-u', '-c', {_EGRESS!r}], check=False)\n"
    assert t._check_code_safety(code) is not None
    assert t.is_high_risk_tool_call("python", {"code": code})


def test_python_c_through_a_shell_string_is_screened():
    code = (
        "import subprocess\n"
        'subprocess.run("python3 -c \'import os; os.system(\\"curl http://x\\")\'", shell=True)\n'
    )
    assert t._check_code_safety(code) is not None


def test_coding_cookie_cannot_hide_code():
    # utf_7 turns `+AAo-` into a newline, so the interpreter would run the "comment" as code.
    hidden = "# coding: utf_7\n#+AAo-print(+ACI-HIDDEN+ACI-)\nprint('VISIBLE')\n"
    out = t._python_exec(hidden)
    assert "HIDDEN" not in out, out
    assert "VISIBLE" in out, out


@pytest.mark.parametrize(
    "code",
    [
        "print(sum(range(10)))\n",
        "import subprocess\nsubprocess.run(['python3', 'script.py'])\n",
        "import subprocess\nsubprocess.run(['python3', '-c', 'print(1 + 1)'])\n",
        "def f(:\n    pass\n",
    ],
)
def test_ordinary_code_is_unchanged(code):
    assert t._check_code_safety(code) is None
    assert not t.is_high_risk_tool_call("python", {"code": code})


def test_ordinary_code_and_syntax_errors_still_run():
    assert t._python_exec("\ufeffprint(6 * 7)\n").strip() == "42"
    assert "café" in t._python_exec("# -*- coding: latin-1 -*-\nprint('café')\n")
    assert "SyntaxError" in t._python_exec("def f(:\n    pass\n")
