# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`scripts/lint_exec_literals.py` fails CI on a new non-literal exec/eval/compile.

The rule is coarse on purpose, so what these pin is the shape of the coarseness: which
calls it fires on, which it leaves alone, and that the baseline cannot be used to
smuggle a new call site past it.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "lint_exec_literals.py"
BASELINE = SCRIPT.parent / "exec_literals_baseline.json"


def _module():
    spec = importlib.util.spec_from_file_location("lint_exec_literals_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _findings(
    tmp_path,
    source,
    name = "sample.py",
):
    sample = tmp_path / name
    sample.write_text(source, encoding = "utf-8")
    return _module().scan_file(sample, name)


REPORTED = {
    "an f-string with a placeholder": 'def f(n):\n    exec(f"import {n}")\n',
    "concatenation with a name": 'def f(n):\n    eval("torch." + n)\n',
    "a bare name": "def f(source):\n    exec(source)\n",
    "a call": "def f():\n    exec(download())\n",
    "a value built by replace": 'def f(n):\n    exec("import X".replace("X", n))\n',
    "compile of a name": 'def f(s):\n    compile(s, "<x>", "exec")\n',
    "an f-string through a keyword sink": 'def f(n):\n    eval(f"{n}", {})\n',
}

QUIET = {
    "a written-out string": 'exec("import torch")\n',
    "an f-string with no placeholder": 'exec(f"import torch")\n',
    "two written-out strings added": 'exec("import " + "torch")\n',
    "bytes": 'exec(b"import torch")\n',
    "re.compile, which is not the builtin": "import re\ndef f(p):\n    re.compile(p)\n",
    "model.eval, which is not the builtin": "def f(m):\n    m.eval()\n",
    "a sink with no arguments": "exec()\n",
    "a name that merely shares the spelling": "def f(exec):\n    return exec\n",
}


@pytest.mark.parametrize("description", sorted(REPORTED))
def test_a_value_that_is_not_written_out_is_reported(tmp_path, description):
    assert _findings(tmp_path, REPORTED[description]), description


@pytest.mark.parametrize("description", sorted(QUIET))
def test_a_written_out_value_is_left_alone(tmp_path, description):
    assert not _findings(tmp_path, QUIET[description]), description


def test_a_notebook_cell_is_read_as_python(tmp_path):
    """The notebooks in this repository are the maintained source, not the .py files."""
    sample = tmp_path / "demo.ipynb"
    sample.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "metadata": {}, "source": ["# not code\n"]},
                    {
                        "cell_type": "code",
                        "metadata": {},
                        "execution_count": None,
                        "outputs": [],
                        "source": [
                            "%pip install unsloth\n",
                            "name = input()\n",
                            'exec(f"import {name}")\n',
                        ],
                    },
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        )
    )
    found = _module().scan_file(sample, "demo.ipynb")
    assert len(found) == 1, found


def test_a_magic_line_does_not_shift_the_reported_line_number(tmp_path):
    """A `%magic` is blanked rather than dropped, so the number still points at the cell."""
    sample = tmp_path / "demo.ipynb"
    sample.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "metadata": {},
                        "execution_count": None,
                        "outputs": [],
                        "source": [
                            "%pip install unsloth\n",
                            "!echo hi\n",
                            "import os\n",
                            'exec(f"import {os.name}")\n',
                        ],
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        )
    )
    found = _module().scan_file(sample, "demo.ipynb")
    assert [f["line"] for f in found] == [4], found


def test_the_baseline_matches_the_tree_it_was_recorded_against():
    """A stale entry silently re-permits whatever lands on that digest next."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)], capture_output = True, text = True, cwd = SCRIPT.parents[1]
    )
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"


def test_the_gate_fails_on_a_call_the_baseline_does_not_have(tmp_path):
    """The point of the gate: a new site is a build failure, not a note."""
    sample = tmp_path / "new_offender.py"
    sample.write_text('def f(n):\n    exec(f"import {n}")\n')
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--paths", str(sample)],
        capture_output = True,
        text = True,
        cwd = SCRIPT.parents[1],
    )
    assert proc.returncode == 1, f"{proc.stdout}\n{proc.stderr}"
    assert "not in the baseline" in proc.stdout, proc.stdout


def test_the_digest_follows_what_is_passed_rather_than_where_it_sits(tmp_path):
    """Moving a call must not churn the baseline; changing its argument must."""
    module = _module()
    moved = _findings(tmp_path, 'import os\n\n\ndef f(n):\n    exec(f"import {n}")\n', "a.py")
    same = _findings(tmp_path, 'def f(n):\n    exec(f"import {n}")\n', "a.py")
    changed = _findings(tmp_path, 'def f(n):\n    exec(f"import {n}!")\n', "a.py")
    assert moved[0]["digest"] == same[0]["digest"], "a line move changed the digest"
    assert moved[0]["digest"] != changed[0]["digest"], "a changed argument kept the digest"
    assert module is not None


def test_the_self_test_is_wired_into_ci():
    """Guards the guard: an unrun gate is not a gate."""
    workflow = (SCRIPT.parents[1] / ".github" / "workflows" / "lint-ci.yml").read_text(
        encoding = "utf-8"
    )
    assert "lint_exec_literals.py --self-test" in workflow
    assert "lint_exec_literals.py\n" in workflow


def test_the_baseline_targets_still_exist():
    """A target that resolves to nothing means the gate covers less than it claims."""
    document = json.loads(BASELINE.read_text(encoding = "utf-8"))
    root = SCRIPT.parents[1]
    missing = [t for t in document["targets"] if not (root / t).exists()]
    assert not missing, missing


def test_update_keeps_the_justifications_that_are_already_there(tmp_path, monkeypatch):
    """Rebuilding the list from scratch is how a reviewed entry becomes unreviewed."""
    module = _module()
    sample = tmp_path / "held.py"
    sample.write_text('def f(n):\n    exec(f"import {n}")\n')
    baseline = tmp_path / "baseline.json"
    monkeypatch.setattr(module, "BASELINE_PATH", baseline)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    baseline.write_text(json.dumps({"targets": ["held.py"], "entries": []}))

    monkeypatch.setattr(sys, "argv", ["x", "--update"])
    assert module.main() == 0
    written = json.loads(baseline.read_text())
    assert written["entries"], "nothing was recorded"
    written["entries"][0]["reason"] = "reviewed: the name is validated above"
    baseline.write_text(json.dumps(written))

    monkeypatch.setattr(sys, "argv", ["x", "--update"])
    assert module.main() == 0
    assert json.loads(baseline.read_text())["entries"][0]["reason"] == (
        "reviewed: the name is validated above"
    )


def test_an_entry_with_no_justification_fails_the_gate(tmp_path, monkeypatch):
    """A recorded call that nobody explained is not a reviewed call."""
    module = _module()
    sample = tmp_path / "held.py"
    sample.write_text('def f(n):\n    exec(f"import {n}")\n')
    baseline = tmp_path / "baseline.json"
    monkeypatch.setattr(module, "BASELINE_PATH", baseline)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    baseline.write_text(json.dumps({"targets": ["held.py"], "entries": []}))
    monkeypatch.setattr(sys, "argv", ["x", "--update"])
    assert module.main() == 0

    monkeypatch.setattr(sys, "argv", ["x"])
    assert module.main() == 1, "an unreviewed entry passed the gate"


def test_every_baseline_entry_carries_a_reason():
    """The committed baseline, not a synthetic one."""
    document = json.loads(BASELINE.read_text(encoding = "utf-8"))
    bare = [
        e["file"] for e in document["entries"] if not e.get("reason") or e["reason"] == "REVIEW ME"
    ]
    assert not bare, bare
