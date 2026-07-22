# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Regression test for the /recommended-folders (and /browse-folders) 500
caused by an unreadable model directory, e.g. a stock root-owned
``ollama`` install at ``/usr/share/ollama/.ollama/models``.

Root cause: ``routes.models`` folder-scan helpers probed candidates with a
bare ``Path(p).is_dir()``. On Python <= 3.11 that returned ``False`` for an
unreadable path; on Python >= 3.12 ``is_dir()`` propagates
``PermissionError`` (EACCES), so the endpoint 500-ed through the whole
middleware stack instead of skipping the directory. Probes now go through
the module-level ``_safe_is_dir`` helper.

``routes.models`` pulls the full backend dep tree (fastapi, structlog, the
models package, ...), so rather than stand up the app we extract the real
``_safe_is_dir`` from the source file and exercise it in isolation —
dependency-free while still running the shipped code.

Run:
    python -m pytest studio/backend/tests/test_recommended_folders_permission.py -v
"""

import ast
import os
import sys
from pathlib import Path

import pytest

_backend_root = Path(__file__).resolve().parent.parent
_models_src = _backend_root / "routes" / "models.py"


def _load_safe_is_dir():
    """Return the real ``_safe_is_dir`` from routes/models.py without
    importing the dependency-laden module."""
    tree = ast.parse(_models_src.read_text())
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_safe_is_dir"
    )
    module = ast.Module(body = [fn], type_ignores = [])
    ns: dict = {"Path": Path, "os": os}
    exec(compile(module, f"<extracted {_models_src}>", "exec"), ns)
    return ns["_safe_is_dir"]


safe_is_dir = _load_safe_is_dir()

# The superuser bypasses permission bits, so the chmod-000 setup below
# would not deny access when running as root.
_skip_as_root = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason = "root bypasses filesystem permission bits",
)


def test_helper_exists_in_source():
    # Guard against a refactor silently dropping the helper the fix needs.
    assert callable(safe_is_dir)


def test_readable_dir_is_true(tmp_path):
    assert safe_is_dir(tmp_path) is True


def test_missing_path_is_false(tmp_path):
    assert safe_is_dir(tmp_path / "does-not-exist") is False


def test_file_is_false(tmp_path):
    f = tmp_path / "weights.gguf"
    f.write_bytes(b"x")
    assert safe_is_dir(f) is False


@_skip_as_root
def test_mode000_dir_itself_is_still_a_dir(tmp_path):
    """A mode-000 directory is still stat-able via its (traversable)
    parent, so _safe_is_dir reports True without raising. Filtering out
    dirs we can't *read* is the caller's separate os.access(R_OK|X_OK)
    check, not this helper's job."""
    locked = tmp_path / "locked"
    locked.mkdir()
    os.chmod(locked, 0o000)
    try:
        assert safe_is_dir(locked) is True  # must not raise
    finally:
        os.chmod(locked, 0o755)


@_skip_as_root
def test_path_under_unreadable_parent_returns_false_not_raises(tmp_path):
    """The production scenario: stat()-ing a child of a mode-700 system
    directory, e.g. ``/usr/share/ollama/.ollama/models``."""
    parent = tmp_path / "ollama"
    parent.mkdir()
    os.chmod(parent, 0o000)
    try:
        assert safe_is_dir(parent / ".ollama" / "models") is False
    finally:
        os.chmod(parent, 0o755)


@_skip_as_root
@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason = "is_dir() only propagates PermissionError on Python >= 3.12",
)
def test_demonstrates_the_underlying_stdlib_regression(tmp_path):
    """Documents *why* _safe_is_dir exists: the old bare pattern raises on
    the interpreters Unsloth ships on (3.12+)."""
    parent = tmp_path / "ollama"
    parent.mkdir()
    os.chmod(parent, 0o000)
    try:
        with pytest.raises(PermissionError):
            Path(parent / ".ollama" / "models").is_dir()  # pre-fix expr
    finally:
        os.chmod(parent, 0o755)
