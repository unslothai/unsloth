# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Regression test for the /recommended-folders 500 caused by an
unreadable model directory (e.g. a stock root-owned ``ollama`` install at
``/usr/share/ollama/.ollama/models``).

Root cause: ``routes.models.get_recommended_folders`` probed candidate
paths with a bare ``Path(p).is_dir()``. On Python <= 3.11 that returned
``False`` for an unreadable path; on Python >= 3.12 ``is_dir()`` propagates
``PermissionError`` (EACCES), so the endpoint 500-ed through the whole
middleware stack instead of just skipping the directory.

The check now goes through ``utils.fs_access.is_accessible_dir``, which is
stdlib-only and importable without the heavy backend dependencies — so
this regression is covered without standing up the FastAPI app.

Run:
    python -m pytest studio/backend/tests/test_recommended_folders_permission.py -v
"""

import os
import sys
from pathlib import Path

import pytest

# Mirror conftest: backend root on sys.path so `from utils...` resolves
# even when this file is run directly.
_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from utils.fs_access import is_accessible_dir  # noqa: E402

# Permission bits are bypassed for the superuser, so the chmod-000 setup
# below does not actually deny access when running as root.
_skip_as_root = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason = "root bypasses filesystem permission bits",
)


def test_readable_dir_is_accessible(tmp_path):
    assert is_accessible_dir(tmp_path) is True


def test_missing_path_is_not_accessible(tmp_path):
    assert is_accessible_dir(tmp_path / "does-not-exist") is False


def test_file_is_not_a_dir(tmp_path):
    f = tmp_path / "weights.gguf"
    f.write_bytes(b"x")
    assert is_accessible_dir(f) is False


@_skip_as_root
def test_unreadable_dir_returns_false_not_raises(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    os.chmod(locked, 0o000)
    try:
        # Must not raise PermissionError (the production bug).
        assert is_accessible_dir(locked) is False
    finally:
        os.chmod(locked, 0o755)


@_skip_as_root
def test_path_under_unreadable_parent_returns_false_not_raises(tmp_path):
    """The exact production scenario: stat()-ing a child of a mode-700
    system directory, e.g. ``/usr/share/ollama/.ollama/models``."""
    parent = tmp_path / "ollama"
    parent.mkdir()
    os.chmod(parent, 0o000)
    try:
        assert is_accessible_dir(parent / ".ollama" / "models") is False
    finally:
        os.chmod(parent, 0o755)


@_skip_as_root
@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason = "is_dir() only propagates PermissionError on Python >= 3.12",
)
def test_demonstrates_the_underlying_stdlib_regression(tmp_path):
    """Documents *why* is_accessible_dir exists: the old bare pattern
    raises on the interpreters Studio ships on (3.12+)."""
    parent = tmp_path / "ollama"
    parent.mkdir()
    os.chmod(parent, 0o000)
    try:
        with pytest.raises(PermissionError):
            # This is the pre-fix expression from get_recommended_folders.
            Path(parent / ".ollama" / "models").is_dir()
    finally:
        os.chmod(parent, 0o755)
