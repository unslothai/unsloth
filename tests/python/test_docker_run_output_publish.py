# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""`unsloth-run --out` must publish an output the host user can read.

tempfile.mkstemp() creates the staging file 0600, nbconvert writes its result by
TRUNCATING the same inode, and os.replace is a rename, so that root-owned 0600 is
carried onto the destination -- and re-running over an existing output replaces that
file's mode and owner too.
"""

from __future__ import annotations

import errno
import importlib.util
import json
import os
import stat
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
    assert RUNNER_PATH.is_file(), f"missing runner: {RUNNER_PATH}"
    spec = importlib.util.spec_from_file_location("unsloth_run_under_test", RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _nbconvert_stub(cmd, env = None):
    out_dir = cmd[cmd.index("--output-dir") + 1]
    name = cmd[cmd.index("--output") + 1]
    with open(os.path.join(out_dir, name), "w", encoding = "utf-8") as f:
        json.dump(NOTEBOOK, f)
    return 0


def _run(runner, monkeypatch, src, out):
    monkeypatch.setattr(runner.subprocess, "call", _nbconvert_stub)
    monkeypatch.setattr(runner.sys, "argv", ["unsloth-run", str(src), "--out", str(out)])
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 0


def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture()
def notebook(tmp_path):
    src = tmp_path / "in.ipynb"
    src.write_text(json.dumps(NOTEBOOK), encoding = "utf-8")
    return src


def test_a_new_output_is_not_published_as_0600(runner, monkeypatch, tmp_path, notebook):
    out = tmp_path / "out.ipynb"
    _run(runner, monkeypatch, notebook, out)
    assert out.is_file()
    umask = os.umask(0)
    os.umask(umask)
    assert _mode(out) == 0o666 & ~umask, (
        f"published mode {oct(_mode(out))}; the mkstemp 0600 survives os.replace, "
        "so a root container publishes an output the host user cannot read"
    )
    assert json.loads(out.read_text(encoding = "utf-8"))["cells"], "the result must survive"


def test_an_existing_outputs_mode_is_preserved(runner, monkeypatch, tmp_path, notebook):
    out = tmp_path / "out.ipynb"
    out.write_text("{}", encoding = "utf-8")
    os.chmod(out, 0o664)
    _run(runner, monkeypatch, notebook, out)
    assert (
        _mode(out) == 0o664
    ), f"re-running replaced the existing output's mode with {oct(_mode(out))}"


@pytest.mark.skipif(os.geteuid() != 0, reason = "chown to another uid needs root")
def test_an_existing_outputs_ownership_is_preserved(runner, monkeypatch, tmp_path, notebook):
    out = tmp_path / "out.ipynb"
    out.write_text("{}", encoding = "utf-8")
    os.chown(out, 1000, 1000)
    _run(runner, monkeypatch, notebook, out)
    st = os.stat(out)
    assert (st.st_uid, st.st_gid) == (1000, 1000), "the host user lost their own output file"


def test_no_staging_files_are_left_behind(runner, monkeypatch, tmp_path, notebook):
    out = tmp_path / "out.ipynb"
    _run(runner, monkeypatch, notebook, out)
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".unsloth-run-")]
    assert not leftovers, leftovers


def test_every_created_level_is_chowned_to_the_nearest_existing_ancestor(
    runner, monkeypatch, tmp_path
):
    # mkdir(2) uses the CALLER's uid, so a new `sub/dir` is root-owned and
    # _stage_metadata then gives the OUTPUT that owner too. Recorded, not observed:
    # chowning to another uid needs root.
    anchor = os.stat(tmp_path)
    chowned = []
    monkeypatch.setattr(runner.os, "chown", lambda p, u, g: chowned.append((str(p), u, g)))

    runner._makedirs_as_host(str(tmp_path / "sub" / "dir"))

    assert [p for p, _, _ in chowned] == [
        str(tmp_path / "sub"),
        str(tmp_path / "sub" / "dir"),
    ], "both created levels must be fixed, outermost first"
    assert {(u, g) for _, u, g in chowned} == {(anchor.st_uid, anchor.st_gid)}


def test_the_ancestor_is_the_nearest_one_that_exists(runner, monkeypatch, tmp_path):
    # the deepest EXISTING directory, not the mount root: a user can own
    # /workspace/host/projectA without owning everything above it
    base = tmp_path / "exists"
    base.mkdir()
    chowned = []
    monkeypatch.setattr(runner.os, "chown", lambda p, u, g: chowned.append(str(p)))

    runner._makedirs_as_host(str(base / "a" / "b"))

    assert chowned == [str(base / "a"), str(base / "a" / "b")]


@pytest.mark.skipif(os.geteuid() != 0, reason = "chown to another uid needs root")
def test_the_output_in_a_created_directory_is_not_root_owned(
    runner, monkeypatch, tmp_path, notebook
):
    host = tmp_path / "host"
    host.mkdir()
    os.chown(host, 1000, 1000)
    out = host / "results" / "run" / "out.ipynb"

    _run(runner, monkeypatch, notebook, out)

    for path in (host / "results", host / "results" / "run", out):
        st = os.stat(path)
        assert (st.st_uid, st.st_gid) == (1000, 1000), path


def test_an_existing_output_directory_is_left_alone(runner, tmp_path):
    existing = tmp_path / "already"
    existing.mkdir()
    before = os.stat(existing)
    runner._makedirs_as_host(str(existing))
    after = os.stat(existing)
    assert (after.st_uid, after.st_gid, after.st_mode) == (
        before.st_uid,
        before.st_gid,
        before.st_mode,
    )


# A bind-mounted OUTPUT FILE makes the destination a mount point, and rename(2) onto
# one returns EBUSY even though the file is writable, so the cleanup then deleted a
# finished run's result. Simulated by raising the kernel's errno.


def _replace_raising(errno_value):
    def _replace(src, dst):
        raise OSError(errno_value, os.strerror(errno_value))

    return _replace


def test_a_busy_destination_still_gets_the_executed_notebook(
    runner, monkeypatch, tmp_path, notebook
):
    out = tmp_path / "out.ipynb"
    out.write_text("{}", encoding = "utf-8")
    os.chmod(out, 0o664)
    monkeypatch.setattr(runner.os, "replace", _replace_raising(errno.EBUSY))

    _run(runner, monkeypatch, notebook, out)

    assert json.loads(out.read_text(encoding = "utf-8"))["cells"], (
        "the rename cannot work on a single-file bind mount, but the file itself "
        "is writable, so the finished notebook must still reach the user"
    )
    # writing through the existing inode is what a bind mount needs
    assert _mode(out) == 0o664
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".unsloth-run-")]
    assert not leftovers, leftovers


def test_an_unpublishable_result_is_kept_and_its_location_printed(
    runner, monkeypatch, tmp_path, notebook, capsys
):
    out = tmp_path / "out.ipynb"
    monkeypatch.setattr(runner.os, "replace", _replace_raising(errno.EBUSY))

    def _no_open(
        path,
        mode = "r",
        *args,
        **kwargs,
    ):
        if str(path) == str(out) and "w" in mode:
            raise OSError(errno.EACCES, os.strerror(errno.EACCES))
        return _real_open(path, mode, *args, **kwargs)

    _real_open = open
    monkeypatch.setattr(runner.subprocess, "call", _nbconvert_stub)
    monkeypatch.setattr(runner.sys, "argv", ["unsloth-run", str(notebook), "--out", str(out)])
    monkeypatch.setitem(runner.__dict__, "open", _no_open)

    with pytest.raises(OSError):
        runner.main()

    staged = [p for p in tmp_path.iterdir() if p.name.startswith(".unsloth-run-out-")]
    assert len(staged) == 1, "an executed notebook that cannot be published must be kept"
    assert json.loads(staged[0].read_text(encoding = "utf-8"))["cells"]
    assert str(staged[0]) in capsys.readouterr().err
