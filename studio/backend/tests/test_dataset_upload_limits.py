# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for training dataset upload limits and cleanup."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi import HTTPException, UploadFile

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from routes import datasets as datasets_route  # noqa: E402
from utils.datasets import llm_assist as legacy_llm_assist  # noqa: E402


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


@pytest.mark.parametrize(
    ("hf_token", "expected_token"),
    ((None, False), ("request-token", "request-token")),
)
def test_training_dataset_preview_uses_only_request_hf_token(
    monkeypatch, tmp_path, hf_token, expected_token
):
    api_tokens = []
    load_tokens = []

    class _Api:
        def list_repo_files(self, *_args, **kwargs):
            api_tokens.append(kwargs.get("token"))
            return ["train.jsonl"]

    class _Dataset(list):
        @classmethod
        def from_list(cls, rows):
            return cls(rows)

    def _load_dataset(**kwargs):
        load_tokens.append(kwargs.get("token"))
        return _Dataset([{"text": "hello"}])

    monkeypatch.setenv("HF_TOKEN", "operator-secret-token")
    monkeypatch.setattr(datasets_route, "resolve_dataset_path", lambda _name: tmp_path / "missing")
    monkeypatch.setattr(
        datasets_route,
        "check_dataset_format",
        lambda *_args, **_kwargs: {
            "requires_manual_mapping": True,
            "detected_format": "text",
            "columns": ["text"],
        },
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi = _Api))
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(Dataset = _Dataset, load_dataset = _load_dataset),
    )

    datasets_route.check_format(
        datasets_route.CheckFormatRequest(dataset_name = "Org/Data", hf_token = hf_token),
        current_subject = "test-user",
    )

    assert api_tokens == [expected_token]
    assert load_tokens == [expected_token]


@pytest.mark.parametrize(
    ("hf_token", "expected_token"),
    ((None, False), ("  request-token\n", "request-token")),
)
def test_legacy_dataset_card_uses_only_request_hf_token(monkeypatch, hf_token, expected_token):
    captured_tokens = []

    class _DatasetCard:
        @classmethod
        def load(cls, *_args, **kwargs):
            captured_tokens.append(kwargs["token"])
            return SimpleNamespace(text = "", data = None)

    monkeypatch.setenv("HF_TOKEN", "operator-secret-token")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(DatasetCard = _DatasetCard),
    )

    assert legacy_llm_assist.fetch_hf_dataset_card("Org/Data", hf_token) == ("", {})
    assert captured_tokens == [expected_token]
