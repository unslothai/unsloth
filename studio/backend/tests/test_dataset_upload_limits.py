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
    monkeypatch.setattr(datasets_route.local, "DATASET_UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(datasets_route.local, "get_upload_limit_mb", lambda: 1)
    return tmp_path


def test_legacy_dataset_routes_are_documented_as_deprecated_aliases():
    routes = {route.path: route for route in datasets_route.router.routes}

    for path in (
        "/upload",
        "/local",
        "/download-progress",
        "/check-format",
        "/ai-assist-mapping",
    ):
        assert routes[path].deprecated is True


def test_legacy_format_alias_preserves_body_token(monkeypatch):
    captured = {}

    def check_format(request, token):
        captured.update(request = request, token = token)
        return datasets_route.CheckFormatResponse(
            requires_manual_mapping = False,
            detected_format = "alpaca",
            columns = ["instruction", "output"],
        )

    monkeypatch.setattr(datasets_route.formatting, "check_format_response", check_format)
    request = datasets_route.CheckFormatRequest(
        dataset_name = "org/data",
        hf_token = "body-token",
        split = "validation",
    )

    datasets_route.check_format(
        request,
        hf_token = "header-token",
        current_subject = "test-user",
    )

    assert captured["token"] == "body-token"
    assert captured["request"].train_split == "validation"


def test_legacy_ai_assist_alias_preserves_body_token(monkeypatch):
    captured = {}

    def ai_assist(request, token):
        captured.update(request = request, token = token)
        return datasets_route.AiAssistMappingResponse(success = True)

    monkeypatch.setattr(
        datasets_route.formatting,
        "ai_assist_mapping_response",
        ai_assist,
    )
    request = datasets_route.AiAssistMappingRequest(
        columns = ["text"],
        samples = [{"text": "hello"}],
        hf_token = "body-token",
    )

    datasets_route.ai_assist_mapping(
        request,
        hf_token = "header-token",
        current_subject = "test-user",
    )

    assert captured["token"] == "body-token"
    assert captured["request"].columns == ["text"]


def test_legacy_local_alias_preserves_recipe_only_response(monkeypatch):
    result = datasets_route.local.LocalDatasetsResponse(
        datasets = [
            datasets_route.local.LocalDatasetItem(
                id = "recipe_one",
                label = "Recipe One",
                path = "/datasets/recipe_one",
                source = "recipe",
            ),
            datasets_route.local.LocalDatasetItem(
                id = "upload.jsonl",
                label = "upload.jsonl",
                path = "/uploads/upload.jsonl",
                source = "upload",
            ),
        ]
    )
    monkeypatch.setattr(
        datasets_route.local,
        "list_local_datasets_response",
        lambda: result,
    )

    response = datasets_route.list_local_datasets(current_subject = "test-user")

    assert [item.id for item in response.datasets] == ["recipe_one"]
    assert not hasattr(response.datasets[0], "source")


def test_dataset_upload_under_configured_cap_succeeds(isolate_upload_dir):
    upload = FakeUploadFile("sample.csv", [b"a,b\n1,2\n"])
    response = asyncio.run(
        datasets_route.upload_dataset(cast(UploadFile, upload), current_subject = "test-user")
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
            datasets_route.upload_dataset(cast(UploadFile, upload), current_subject = "test-user")
        )
    assert exc.value.status_code == 413
    assert "Maximum is 1MB" in exc.value.detail
    assert list(isolate_upload_dir.iterdir()) == []


def test_cancelled_dataset_upload_removes_partial_file(isolate_upload_dir):
    class CancelledUploadFile(FakeUploadFile):
        async def read(self, size: int = -1) -> bytes:
            if self._chunks:
                return await super().read(size)
            raise asyncio.CancelledError

    upload = CancelledUploadFile("sample.csv", [b"partial"])
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            datasets_route.upload_dataset(
                cast(UploadFile, upload),
                current_subject = "test-user",
            )
        )

    assert list(isolate_upload_dir.iterdir()) == []


def test_hub_upload_path_has_multipart_streaming_headroom():
    source = (_BACKEND_ROOT / "main.py").read_text(encoding = "utf-8")

    prefixes = source.split("_DATASET_UPLOAD_PASSTHROUGH_PREFIXES =", 1)[1].split(")", 1)[0]
    assert '"/api/datasets/upload"' in prefixes
    assert '"/api/hub/datasets/upload"' in prefixes
