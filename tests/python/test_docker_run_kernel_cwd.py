# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`--out` must not move the notebook's execution directory.

nbconvert derives the kernel's working directory from the INPUT notebook's path
(``Exporter.from_filename`` sets ``resources["metadata"]["path"]`` to its dirname,
which nbclient passes to the kernel as cwd), so staging the input beside ``--out``
made every relative ``open()``, local import and save inside the notebook resolve
against the OUTPUT tree instead of the tree the notebook lives in.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = REPO_ROOT / "docker" / "unsloth_run.py"

NOTEBOOK = {
    "cells": [{"cell_type": "code", "source": ["print(1)\n"], "metadata": {}, "outputs": []}],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5,
}


@pytest.fixture()
def runner(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_NB_TF_MARKER", str(tmp_path / "marker"))
    spec = importlib.util.spec_from_file_location("unsloth_run_cwd_under_test", RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(runner, monkeypatch, argv):
    seen = {}

    def stub(cmd, env = None):
        # the input notebook is the positional argument before --output
        seen["input"] = cmd[cmd.index("--output") - 1]
        out_dir = cmd[cmd.index("--output-dir") + 1]
        name = cmd[cmd.index("--output") + 1]
        with open(os.path.join(out_dir, name), "w", encoding = "utf-8") as f:
            json.dump(NOTEBOOK, f)
        return 0

    monkeypatch.setattr(runner.subprocess, "call", stub)
    monkeypatch.setattr(runner.sys, "argv", ["unsloth-run", *argv])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 0
    return seen["input"]


def test_a_local_notebook_executes_from_its_own_directory(runner, monkeypatch, tmp_path):
    src_dir = tmp_path / "project"
    src_dir.mkdir()
    src = src_dir / "in.ipynb"
    src.write_text(json.dumps(NOTEBOOK), encoding = "utf-8")
    (src_dir / "data.json").write_text("{}", encoding = "utf-8")
    out = tmp_path / "results" / "out.ipynb"

    given = _run(runner, monkeypatch, [str(src), "--out", str(out)])

    assert Path(os.path.dirname(os.path.abspath(given))) == src_dir, (
        f"nbconvert was handed {given}; its dirname is the kernel cwd, so the "
        "notebook's relative paths would resolve against the --out tree"
    )
    assert out.is_file(), "the result still has to be published at --out"
    assert not list(out.parent.glob(".unsloth-run-in-*")), "no input staged beside --out"


def test_the_source_directory_is_not_written_to(runner, monkeypatch, tmp_path):
    src_dir = tmp_path / "project"
    src_dir.mkdir()
    src = src_dir / "in.ipynb"
    src.write_text(json.dumps(NOTEBOOK), encoding = "utf-8")
    out = tmp_path / "results" / "out.ipynb"

    _run(runner, monkeypatch, [str(src), "--out", str(out)])

    assert sorted(p.name for p in src_dir.iterdir()) == [
        "in.ipynb"
    ], "executing in place must not leave staging files in the user's tree"
    assert json.loads(src.read_text(encoding = "utf-8")) == NOTEBOOK, "input left unmodified"


def test_a_url_notebook_still_runs_beside_out(runner, monkeypatch, tmp_path):
    """A URL has no source tree; --out stays the run's working directory."""
    monkeypatch.setattr(runner, "_load", lambda *_args, **_kwargs: json.loads(json.dumps(NOTEBOOK)))
    out = tmp_path / "results" / "out.ipynb"

    given = _run(runner, monkeypatch, ["https://example.invalid/x.ipynb", "--out", str(out)])

    assert Path(os.path.dirname(os.path.abspath(given))) == out.parent
    assert out.is_file()
    assert not list(out.parent.glob(".unsloth-run-in-*")), "the staged input is cleaned up"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
