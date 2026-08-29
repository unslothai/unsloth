# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression tests for issue #9586 channel 2: the subprocess-fix directory in /tmp.

``unsloth/import_fixes.py:_subprocess_fix_directory()`` builds
``<gettempdir()>/unsloth_subprocess_import_fix-<uid>``, writes a ``sitecustomize.py`` into
it and prepends the directory to ``PYTHONPATH``. That is correct for a real install -- the
point is that children inherit it -- but the path is scoped per **user**, not per **run**.

Two properties follow, and each has a test here:

* the directory a test session creates must not be the machine-shared one, and
* the redirect has to be applied at conftest **module** scope, because the write happens
  while that module is still being imported.

The write itself is gated on a torch/torchao version skew (torchao >= 0.18 against a torch
predating ``aten._grouped_mm``, which arrived in 2.8), so on a matched pair nothing is
written at all. These tests therefore assert *where* a temp path resolves rather than
requiring the skew to be present.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


_PREFIX = "unsloth-tests-"
_CONFTEST = Path(__file__).resolve().parent / "conftest.py"
_FIX_DIRNAME = "unsloth_subprocess_import_fix"


def test_the_session_tempdir_is_not_the_shared_one():
    resolved = Path(tempfile.gettempdir()).resolve()
    assert resolved.name.startswith(_PREFIX), resolved

    shared = Path(
        os.environ.get("TMPDIR") or os.environ.get("TEMP") or os.environ.get("TMP") or "/tmp"
    ).resolve()
    assert resolved != shared


def test_the_fix_directory_would_land_off_the_shared_tempdir():
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

    shared = Path(
        os.environ.get("TMPDIR") or os.environ.get("TEMP") or os.environ.get("TMP") or "/tmp"
    ).resolve()
    assert directory.resolve().parent != shared


def test_no_fix_directory_is_left_in_the_shared_tempdir():
    """The observable the issue reports, checked where it would actually land."""
    shared = Path(
        os.environ.get("TMPDIR") or os.environ.get("TEMP") or os.environ.get("TMP") or "/tmp"
    )
    if not shared.is_dir():
        return
    strays = sorted(shared.glob(f"{_FIX_DIRNAME}*"))
    assert not strays, f"subprocess-fix directories left in the shared tempdir: {strays}"


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
