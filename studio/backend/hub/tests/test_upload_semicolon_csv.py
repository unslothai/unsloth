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


@pytest.mark.parametrize("prefix", [b"", b"\xef\xbb\xbfsep=;\r\n"])
def test_semicolon_csv_upload_matches_preview_and_training(isolate_upload_dir, prefix):
    from datasets import load_dataset
    from utils.datasets.cells import csv_as_text_kwargs

    source = (
        prefix + b'instruction;"hint, context";output\r\n"1, 2 or 3?";"One; two";"two\nlines"\r\n'
    )
    response = asyncio.run(
        datasets_route.upload_dataset(
            UploadFile(BytesIO(source), filename = "excel.csv"),
            native_path_lease = None,
            current_subject = "test-user",
            csv_delimiter = ";",
        )
    )
    path = Path(response.stored_path)
    preview, _ = datasets_route.local._load_local_preview_slice(
        dataset_path = path,
        train_split = "train",
        preview_size = 2,
    )
    expected = [{"instruction": "1, 2 or 3?", "hint, context": "One; two", "output": "two\nlines"}]
    assert preview.to_list() == expected
    assert (
        load_dataset(
            "csv", data_files = str(path), split = "train", **csv_as_text_kwargs([path])
        ).to_list()
        == expected
    )


def test_default_csv_upload_keeps_semicolons_in_standard_csv(isolate_upload_dir):
    source = b'instruction,output\r\n"say a;b",a;b\r\n'
    response = asyncio.run(
        datasets_route.upload_dataset(
            UploadFile(BytesIO(source), filename = "standard.csv"),
            native_path_lease = None,
            current_subject = "test-user",
        )
    )
    assert Path(response.stored_path).read_bytes() == source


@pytest.mark.parametrize(
    "source",
    [
        b'instruction;output\r\n"open quote;answer\r\n',
        b"instruction;output\r\nfirst;answer;extra field\r\n",
        b"instruction;output\r\n" + b"valid;row\r\n" * 1024 + b"wide;row;extra\r\n",
        b"instruction;output\r\n" + b"valid;row\r\n" * 2048 + b"wide;row;extra\r\n",
    ],
)
def test_invalid_semicolon_csv_upload_publishes_nothing(isolate_upload_dir, source):
    with pytest.raises(HTTPException, match = "Invalid semicolon CSV") as exc:
        asyncio.run(
            datasets_route.upload_dataset(
                UploadFile(BytesIO(source), filename = "excel.csv"),
                native_path_lease = None,
                current_subject = "test-user",
                csv_delimiter = ";",
            )
        )
    assert exc.value.status_code == 400
    assert list(isolate_upload_dir.iterdir()) == []


def test_semicolon_csv_keeps_long_fields_and_pads_short_rows(isolate_upload_dir):
    import csv

    long_field = b"word" * 50_000 + b';, ""quote""\r\nnext line'
    source = (
        b'\xef\xbb\xbfsep=;\r\ninstruction;output\r\n"' + long_field + b'";yes\r\n   \r\nshort\r\n'
    )
    response = asyncio.run(
        datasets_route.upload_dataset(
            UploadFile(BytesIO(source), filename = "excel.csv"),
            native_path_lease = None,
            current_subject = "test-user",
            csv_delimiter = ";",
        )
    )
    expected = b'instruction,output\r\n"' + long_field + b'",yes\r\nshort,\r\n'
    assert Path(response.stored_path).read_bytes() == expected
    assert csv.field_size_limit() >= 131_072


@pytest.mark.parametrize("encoding", ["cp1252", "cp1251"])
def test_semicolon_csv_rejects_ansi_export_without_guessing(isolate_upload_dir, encoding):
    source = "frage;antwort\r\nGröße;Привет\r\n".encode(encoding, errors = "replace")
    with pytest.raises(HTTPException, match = "save it from Excel as CSV UTF-8") as exc:
        asyncio.run(
            datasets_route.upload_dataset(
                UploadFile(BytesIO(source), filename = "excel.csv"),
                native_path_lease = None,
                current_subject = "test-user",
                csv_delimiter = ";",
            )
        )
    assert exc.value.status_code == 400
    assert list(isolate_upload_dir.iterdir()) == []


def test_semicolon_csv_padding_cannot_exceed_upload_limit(isolate_upload_dir):
    header = ";".join(f"c{i}" for i in range(20_000)).encode()
    source = header + b"\r\n" + b"x\r\n" * 200
    with pytest.raises(HTTPException, match = "upload too large") as exc:
        asyncio.run(
            datasets_route.upload_dataset(
                UploadFile(BytesIO(source), filename = "excel.csv"),
                native_path_lease = None,
                current_subject = "test-user",
                csv_delimiter = ";",
            )
        )
    assert exc.value.status_code == 413
    assert list(isolate_upload_dir.iterdir()) == []
