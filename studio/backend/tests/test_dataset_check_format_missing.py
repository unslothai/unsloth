# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A deleted upload must report itself, without swallowing other failures."""

import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from routes import datasets as datasets_route  # noqa: E402


def _check(dataset_name: str) -> HTTPException:
    request = datasets_route.CheckFormatRequest(dataset_name = dataset_name)
    with pytest.raises(HTTPException) as exc:
        datasets_route.check_format(request, current_subject = "test")
    return exc.value


def test_missing_upload_reports_404():
    from utils.paths import dataset_uploads_root

    missing = dataset_uploads_root() / "deleted-upload-test-fixture.jsonl"
    assert not missing.exists()
    error = _check(str(missing))
    assert error.status_code == 404
    assert "no longer exists" in error.detail


def test_corrupt_local_file_keeps_its_own_error(monkeypatch):
    from utils.paths import dataset_uploads_root

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    corrupt = dataset_uploads_root() / "corrupt-test-fixture.jsonl"
    corrupt.parent.mkdir(parents = True, exist_ok = True)
    corrupt.write_text("{not valid json at all\n")
    try:
        assert _check(str(corrupt)).status_code != 404
    finally:
        corrupt.unlink(missing_ok = True)


def test_hub_repo_id_never_reports_a_deleted_file(monkeypatch):
    # A repo id under an "uploads" namespace must keep its own Hub error.
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    assert _check("uploads/private-or-unreachable").status_code != 404
