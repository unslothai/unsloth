# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for training dataset upload limits and cleanup."""

import asyncio
import sys
from pathlib import Path
from typing import cast

import pytest
from fastapi import HTTPException, UploadFile

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from routes import datasets as datasets_route  # noqa: E402


class FakeUploadFile:
    def __init__(self, filename: str, chunks: list[bytes]):
        self.filename = filename
        self._chunks = list(chunks)

    async def read(self, _size: int = -1) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


@pytest.fixture(autouse = True)
def isolate_upload_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets_route, "DATASET_UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(datasets_route, "get_upload_limit_bytes", lambda: 1024 * 1024)
    monkeypatch.setattr(datasets_route, "get_upload_limit_label", lambda: "1MB")
    return tmp_path


def test_dataset_upload_under_configured_cap_succeeds(isolate_upload_dir):
    upload = FakeUploadFile("sample.csv", [b"a,b\n1,2\n"])
    response = asyncio.run(
        datasets_route.upload_dataset(
            cast(UploadFile, upload), current_subject = "test-user"
        )
    )
    stored = Path(response.stored_path)
    assert response.filename == "sample.csv"
    assert stored.exists()
    assert stored.parent == isolate_upload_dir
    assert stored.read_bytes() == b"a,b\n1,2\n"


def test_dataset_upload_over_configured_cap_removes_partial_file(isolate_upload_dir):
    upload = FakeUploadFile(
        "sample.csv",
        [b"x" * (1024 * 1024), b"y"],
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            datasets_route.upload_dataset(
                cast(UploadFile, upload), current_subject = "test-user"
            )
        )
    assert exc.value.status_code == 413
    assert "Maximum is 1MB" in exc.value.detail
    assert list(isolate_upload_dir.iterdir()) == []
