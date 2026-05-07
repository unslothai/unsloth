# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import sys
import types
from pathlib import Path

import pytest
from fastapi import HTTPException

# Keep this test runnable in lightweight environments where optional logging
# deps are not installed.
if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

import routes.models as models_route


def test_resolve_browse_target_returns_allowed_directory(tmp_path):
    allowed = tmp_path / "allowed"
    target = allowed / "models" / "nested"
    target.mkdir(parents = True)

    resolved = models_route._resolve_browse_target(str(target), [allowed])

    assert resolved == target.resolve()


def test_resolve_browse_target_rejects_outside_allowlist(tmp_path):
    allowed = tmp_path / "allowed"
    disallowed = tmp_path / "disallowed"
    allowed.mkdir()
    disallowed.mkdir()

    with pytest.raises(HTTPException) as exc_info:
        models_route._resolve_browse_target(str(disallowed), [allowed])

    assert exc_info.value.status_code == 403


def test_resolve_browse_target_rejects_file_path(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    model_file = allowed / "model.gguf"
    model_file.write_text("gguf")

    with pytest.raises(HTTPException) as exc_info:
        models_route._resolve_browse_target(str(model_file), [allowed])

    assert exc_info.value.status_code == 400


def test_resolve_browse_target_allows_symlink_into_other_allowed_root(tmp_path):
    home_root = tmp_path / "home"
    scan_root = tmp_path / "scan"
    target = scan_root / "nested"
    home_root.mkdir()
    target.mkdir(parents = True)
    (home_root / "scan-link").symlink_to(scan_root, target_is_directory = True)

    resolved = models_route._resolve_browse_target(
        str(home_root / "scan-link" / "nested"),
        [home_root, scan_root],
    )

    assert resolved == target.resolve()


@pytest.mark.skipif(os.altsep is not None, reason = "POSIX-only path semantics")
def test_resolve_browse_target_allows_backslash_in_posix_segment(tmp_path):
    allowed = tmp_path / "allowed"
    target = allowed / r"dir\name"
    target.mkdir(parents = True)

    resolved = models_route._resolve_browse_target(str(target), [allowed])

    assert resolved == target.resolve()
