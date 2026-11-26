# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from hub.routes import datasets as datasets_route


@pytest.fixture
def isolate_upload_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets_route.local, "DATASET_UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(datasets_route.local, "get_upload_limit_mb", lambda: 1)
    return tmp_path


def test_pandas_json_upload_keeps_row_indices_aligned_in_preview_and_training(isolate_upload_dir):
    from datasets import load_dataset

    source = b'{"instruction":{"B":"second","A":"first"},"output":{"A":"1","B":"2"}}'
    response = asyncio.run(
        datasets_route.upload_dataset(
            UploadFile(BytesIO(source), filename = "pandas.json"),
            native_path_lease = None,
            current_subject = "test-user",
            json_columns = True,
        )
    )
    path = Path(response.stored_path)
    preview, _ = datasets_route.local._load_local_preview_slice(
        dataset_path = path,
        train_split = "train",
        preview_size = 2,
    )
    expected = [{"instruction": "second", "output": "2"}, {"instruction": "first", "output": "1"}]
    assert preview.to_list() == expected
    assert load_dataset("json", data_files = str(path), split = "train").to_list() == expected


def test_ordinary_json_upload_preserves_file_contents(isolate_upload_dir):
    source = b'[{"instruction":{"question":"hello"},"output":{"answer":"yes"}}]'
    response = asyncio.run(
        datasets_route.upload_dataset(
            UploadFile(BytesIO(source), filename = "rows.json"),
            native_path_lease = None,
            current_subject = "test-user",
        )
    )
    assert Path(response.stored_path).read_bytes() == source


def test_pandas_default_column_name_is_preserved(isolate_upload_dir):
    from datasets import load_dataset

    response = asyncio.run(
        datasets_route.upload_dataset(
            UploadFile(BytesIO(b'{"0":{"1":"hello","2":"hi"}}'), filename = "pandas.json"),
            native_path_lease = None,
            current_subject = "test-user",
            json_columns = True,
        )
    )
    path = Path(response.stored_path)
    preview, _ = datasets_route.local._load_local_preview_slice(
        dataset_path = path,
        train_split = "train",
        preview_size = 2,
    )
    expected = [{"0": "hello"}, {"0": "hi"}]
    assert preview.to_list() == expected
    assert load_dataset("json", data_files = str(path), split = "train").to_list() == expected


@pytest.mark.parametrize("declared", [True, False])
def test_pandas_input_limit_leaves_no_partial_upload(isolate_upload_dir, monkeypatch, declared):
    source = b'{"0":{"0":"' + b"a" * 1_050_000 + b'"}}'
    monkeypatch.setattr(datasets_route.local, "get_upload_limit_mb", lambda: 2)
    monkeypatch.setattr(datasets_route.local, "_PANDAS_JSON_MAX_MB", 1)
    with pytest.raises(HTTPException, match = "upload too large") as exc:
        asyncio.run(
            datasets_route.upload_dataset(
                UploadFile(
                    BytesIO(source), filename = "pandas.json", size = len(source) if declared else None
                ),
                native_path_lease = None,
                current_subject = "test-user",
                json_columns = True,
            )
        )
    assert exc.value.status_code == 413
    assert list(isolate_upload_dir.iterdir()) == []


@pytest.mark.parametrize(
    "source",
    [
        b'[{"instruction":"hello","output":"yes"}]',
        b'{"instruction":{"A":"hello"},"output":{"B":"yes"}}',
        b'{"instruction":{"A":"hello","A":"again"},"output":{"A":"yes"}}',
    ],
)
def test_invalid_pandas_json_upload_publishes_nothing(isolate_upload_dir, source):
    with pytest.raises(HTTPException, match = "Invalid pandas column JSON") as exc:
        asyncio.run(
            datasets_route.upload_dataset(
                UploadFile(BytesIO(source), filename = "pandas.json"),
                native_path_lease = None,
                current_subject = "test-user",
                json_columns = True,
            )
        )
    assert exc.value.status_code == 400
    assert list(isolate_upload_dir.iterdir()) == []
