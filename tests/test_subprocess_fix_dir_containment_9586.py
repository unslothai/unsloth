# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression tests for issue #9586 channel 2: the subprocess-fix directory in /tmp.

``unsloth/import_fixes.py:_subprocess_fix_directory()`` builds
``<gettempdir()>/unsloth_subprocess_import_fix-<uid>``, writes a ``sitecustomize.py`` into
it and prepends the directory to ``PYTHONPATH``. That is correct for a real install -- the
point is that children inherit it -- but the path is scoped per **user**, not per **run**.

Three properties follow, and each has a test here:

* the directory a test session creates must not be the machine-shared one,
* a child interpreter must resolve the same per-run root, because a child that imports
  unsloth re-runs the write, and
* the redirect has to be applied at conftest **module** scope, because the write happens
  while that module is still being imported.

The write itself is gated on a torch/torchao version skew (torchao >= 0.18 against a torch
predating ``aten._grouped_mm``, which arrived in 2.8), so on a matched pair nothing is
written at all. These tests therefore assert *where* a temp path resolves rather than
requiring the skew to be present.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile
from pathlib import Path


_PREFIX = "unsloth-tests-"
_CONFTEST = Path(__file__).resolve().parent / "conftest.py"
_FIX_DIRNAME = "unsloth_subprocess_import_fix"


def test_the_session_tempdir_is_not_the_shared_one(shared_tempdir_before_redirect):
    resolved = Path(tempfile.gettempdir()).resolve()
    assert resolved.name.startswith(_PREFIX), resolved
    assert resolved != Path(shared_tempdir_before_redirect).resolve()


def test_the_fix_directory_would_land_off_the_shared_tempdir(shared_tempdir_before_redirect):
    """The path expression from `_subprocess_fix_directory`, built the same way.

    Asserted by construction rather than by calling the function, which is gated on a
    version skew that a matched torch/torchao pair does not have.
    """
    directory = (
        Path(tempfile.gettempdir())
        / f"{_FIX_DIRNAME}-{os.getuid() if hasattr(os, 'getuid') else 'user'}"
    )

    session_tmp = Path(tempfile.gettempdir()).resolve()
    assert session_tmp in directory.resolve().parents
    assert directory.resolve().parent != Path(shared_tempdir_before_redirect).resolve()


def test_no_fix_directory_is_added_to_the_shared_tempdir(
    shared_tempdir_before_redirect, pre_existing_fix_dirs
):
    """The observable the issue reports, checked where it would actually land.

    Compared against a snapshot taken before the redirect, not against an empty root. A
    machine affected by #9586 already carries a stray from before this fix, and failing on
    that would fail this suite on exactly the machines it is meant to protect.
    """
    shared = Path(shared_tempdir_before_redirect)
    if not shared.is_dir():
        return
    now = frozenset(glob.glob(os.path.join(str(shared), f"{_FIX_DIRNAME}*")))
    added = sorted(now - set(pre_existing_fix_dirs))
    assert not added, f"this session added subprocess-fix directories: {added}"


def test_the_temp_environment_points_at_the_session_tempdir():
    """`tempfile.tempdir` is a module attribute and does not survive exec.

    Children resolve their own temp root from the environment, so the environment has to
    be moved too or the containment stops at this process.
    """
    session_tmp = Path(tempfile.gettempdir()).resolve()
    # Anchored on the per-run prefix, not only on equality: with no redirect at all
    # gettempdir() and TEMP are BOTH the shared root, and comparing them alone passes.
    assert session_tmp.name.startswith(_PREFIX), session_tmp
    for name in ("TEMP", "TMP"):
        value = os.environ.get(name)
        assert value, f"{name} is not set"
        assert Path(value).resolve() == session_tmp, name


def test_tmpdir_is_left_posix_shaped_or_unset():
    """On Windows TMPDIR must be cleared, not set, and TEMP left authoritative.

    gettempdir() reads TMPDIR before TEMP on every platform, so a TMPDIR inherited from
    the surrounding session would win over the redirect and let a child resolve some
    other root. Clearing it also keeps a Windows path out of the POSIX spelling that
    install.sh and tests/sh read.
    """
    value = os.environ.get("TMPDIR")
    if os.name == "nt":
        assert value is None, f"TMPDIR must not be handed a Windows path: {value!r}"
        return
    assert value, "TMPDIR is not set"
    assert Path(value).resolve() == Path(tempfile.gettempdir()).resolve()


def test_a_child_interpreter_resolves_the_session_tempdir():
    """The property `tempfile.tempdir` alone does NOT have.

    Spawned the way `_run()` in tests/test_broken_tf_does_not_break_import.py spawns one:
    the parent's environment, passed through. Such a child imports unsloth, which re-runs
    the write via `_gpu_init.py`, so a child left on the shared root recreates the very
    directory this channel is about.
    """
    child = subprocess.run(
        [sys.executable, "-c", "import tempfile; print(tempfile.gettempdir())"],
        capture_output = True,
        text = True,
        env = dict(os.environ),
        timeout = 120,
    )

    assert child.returncode == 0, child.stderr
    resolved = Path(child.stdout.strip()).resolve()
    # The prefix check is what makes this non-vacuous. With no redirect anywhere, parent
    # and child both resolve the shared root and the equality below holds for the wrong
    # reason -- measured against origin/main, where this test passed until it was added.
    assert resolved.name.startswith(_PREFIX), child.stdout
    assert resolved == Path(tempfile.gettempdir()).resolve(), child.stdout


def test_the_session_tempdir_is_removed_at_normal_exit():
    """Runs the real containment block in a child and checks its root is gone after.

    pytest roots its tmp_path tree at `gettempdir()` and prunes only the `pytest-*`
    siblings beside it, so a fresh root per run means that pruning never sees an earlier
    run. Without the atexit cleanup every run leaks its whole fixture tree.

    The block is read from conftest rather than restated, so it cannot drift: if it moves,
    the extraction fails and this test says so.
    """
    source = _CONFTEST.read_text(encoding = "utf-8")
    start = source.index("# --- subprocess-fix directory containment")
    end = source.index("\n", source.index("_atexit.register(", start))
    block = source[start:end]

    child = subprocess.run(
        [sys.executable, "-c", block + "\nprint(_tempfile.tempdir)\n"],
        capture_output = True,
        text = True,
        timeout = 120,
    )

    assert child.returncode == 0, child.stderr
    root = Path(child.stdout.strip())
    assert root.name.startswith(_PREFIX), child.stdout
    assert not root.exists(), f"the per-run root outlived the process that made it: {root}"


def test_the_redirect_is_applied_at_module_scope():
    """Pins the placement, which is the whole mechanism.

    A fixture cannot be substituted: `_apply_upstream_import_fixes_for_tests()` runs while
    conftest is still being imported, so the write has already happened by the time any
    fixture could run. Read from disk rather than via `inspect`, because importing the
    module to inspect it would re-run the redirect and allocate a second directory.
    """
    source = _CONFTEST.read_text(encoding = "utf-8")

    # Anchored to line start: a bare substring matches the mention of the trigger inside
    # the comment above the redirect itself, which is earlier in the file, and the
    # assertion would then compare the redirect against its own documentation.
    redirect = source.index("\n_tempfile.tempdir = ")
    trigger = source.index("\n_apply_upstream_import_fixes_for_tests()")
    assert redirect < trigger, "the redirect must precede the import that triggers the write"
    assert redirect < source.index("\ndef "), "must precede every function in the module"
