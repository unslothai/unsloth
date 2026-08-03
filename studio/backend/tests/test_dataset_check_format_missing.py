# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A deleted upload must report itself, not fail as a repo lookup."""

import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from routes import datasets as datasets_route  # noqa: E402


def _expect_missing_file_404(dataset_name: str) -> None:
    request = datasets_route.CheckFormatRequest(dataset_name = dataset_name)
    with pytest.raises(HTTPException) as exc:
        datasets_route.check_format(request, current_subject = "test")
    assert exc.value.status_code == 404
    assert "no longer exists" in exc.value.detail


def test_missing_absolute_upload_reports_404():
    from utils.paths import dataset_uploads_root

    missing = dataset_uploads_root() / "deleted-upload-test-fixture.jsonl"
    assert not missing.exists()
    _expect_missing_file_404(str(missing))


# Relative refs can also be Hub repo ids, so they only become the missing-file
# error once the Hub lookup fails; offline mode makes that failure deterministic.
def test_missing_relative_upload_reports_404(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    _expect_missing_file_404("uploads/deleted-upload-test-fixture.jsonl")


def test_missing_relative_recipe_reports_404(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    _expect_missing_file_404("recipes/deleted-recipe-test-fixture/parquet-files")


@pytest.mark.parametrize(
    "dataset_name, expected",
    [
        ("/tmp/some/upload.jsonl", True),
        ("uploads/foo.jsonl", True),
        ("assets/datasets/uploads/foo.jsonl", True),
        ("recipes/foo/parquet-files", True),
        ("unsloth/LaTeX_OCR", False),
        ("squad", False),
    ],
)
def test_local_reference_detection(dataset_name, expected):
    assert datasets_route._is_local_dataset_reference(dataset_name) is expected


def test_corrupt_local_file_keeps_its_own_error(monkeypatch):
    from utils.paths import dataset_uploads_root

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    corrupt = dataset_uploads_root() / "corrupt-test-fixture.jsonl"
    corrupt.parent.mkdir(parents = True, exist_ok = True)
    corrupt.write_text("{not valid json at all\n")
    try:
        request = datasets_route.CheckFormatRequest(dataset_name = str(corrupt))
        with pytest.raises(HTTPException) as exc:
            datasets_route.check_format(request, current_subject = "test")
        assert exc.value.status_code != 404
    finally:
        corrupt.unlink(missing_ok = True)
