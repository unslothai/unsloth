# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A local CSV preview parses only the preview rows, and shows exactly what the default 10k-row chunk showed;
its total is the file's exact record count, or None where the full load would fail."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import datasets  # noqa: E402
import pytest  # noqa: E402
from datasets.packaged_modules.csv.csv import Csv  # noqa: E402

from hub.services.datasets import local  # noqa: E402
from utils.datasets.cells import csv_as_text_kwargs  # noqa: E402

PREVIEW = 10
HEADER = "instruction,input,output\n"


def _rows(count: int, start: int = 0) -> str:
    return "".join(f"q{i},in{i},a{i}\n" for i in range(start, start + count))


def _preview(path: Path):
    return local._load_local_preview_slice(
        dataset_path = path, train_split = "train", preview_size = PREVIEW
    )


def _full_load_rows(path: Path):
    """Rows the training load reads (pandas, the CSV builder's parser); None where it raises."""
    import pandas as pd
    try:
        return len(pd.read_csv(path, keep_default_na = False, na_values = [""])) or None
    except Exception:
        return None


def _default_chunk_preview(path: Path):
    return local._stream_file_preview_slice(
        path, PREVIEW, dates_as_written = True, **csv_as_text_kwargs([path])
    )


@pytest.fixture
def parsed_tables(monkeypatch):
    sizes: list[int] = []
    cast = Csv._cast_table

    def record(self, table):
        sizes.append(table.num_rows)
        return cast(self, table)

    monkeypatch.setattr(Csv, "_cast_table", record)
    return sizes


@pytest.fixture
def load_calls(monkeypatch):
    calls: list[dict] = []
    load = datasets.load_dataset

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return load(*args, **kwargs)

    monkeypatch.setattr(datasets, "load_dataset", spy)
    return calls


def test_csv_preview_parses_only_the_preview_rows(tmp_path, parsed_tables, load_calls):
    path = tmp_path / "data.csv"
    path.write_text(HEADER + _rows(500))

    preview, total = _preview(path)

    assert [call.get("chunksize") for call in load_calls] == [PREVIEW]
    assert parsed_tables == [PREVIEW]
    # The count is a separate bounded pandas scan, so the preview still parsed only PREVIEW rows.
    assert len(preview) == PREVIEW and total == 500


CASES = {
    "plain": HEADER + _rows(50),
    "fewer_than_preview": HEADER + _rows(3),
    "exactly_preview": HEADER + _rows(PREVIEW),
    "quoted_newlines": HEADER
    + "".join(f'"line1\nline2 {i}","a, b","say ""hi"" {i}"\n' for i in range(30)),
    "bom": "﻿" + HEADER + _rows(30),
    "crlf": (HEADER + _rows(30)).replace("\n", "\r\n"),
    "empty_and_na_cells": HEADER
    + "".join(f"q{i},{'' if i % 2 else 'NA'},{'null' if i % 3 else ''}\n" for i in range(30)),
    "numbers_stay_as_written": "question,answer,score\n"
    + "".join(f"q{i},{i:05d},{i * 0.5}\n" for i in range(30))
    + "q,not a number,x\n",
    "short_row": HEADER + _rows(3) + "only_one\n" + _rows(20, 3),
    "blank_lines": HEADER + "q0,a,b\n\n\nq1,c,d\n" + _rows(20, 2),
    "duplicate_headers": "text,text,label\n" + "".join(f"t{i},u{i},{i % 2}\n" for i in range(30)),
    "semicolons": "instruction;output\n" + "".join(f"q{i};a{i}\n" for i in range(30)),
    "malformed_after_the_first_10k_rows": HEADER + _rows(10_000) + "x,y,z,extra\n" + _rows(5),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_csv_preview_matches_the_default_chunk(tmp_path, name):
    path = tmp_path / f"{name}.csv"
    path.write_bytes(CASES[name].encode("utf-8"))

    expected, _ = _default_chunk_preview(path)
    preview, total = _preview(path)

    assert preview.features == expected.features
    assert preview.to_list() == expected.to_list()
    assert total == _full_load_rows(path)


@pytest.mark.parametrize("line", [4, 50, 9_999])
def test_a_malformed_row_in_the_first_10k_rows_still_fails(tmp_path, line):
    path = tmp_path / "bad.csv"
    path.write_text(HEADER + _rows(line - 1) + "x,y,z,extra\n" + _rows(20, line))

    with pytest.raises(Exception) as expected:
        _default_chunk_preview(path)
    with pytest.raises(type(expected.value)) as raised:
        _preview(path)
    assert str(raised.value) == str(expected.value)


def test_json_preview_takes_no_chunk_size(tmp_path, load_calls):
    path = tmp_path / "data.jsonl"
    path.write_text("".join(f'{{"text": "t{i}"}}\n' for i in range(30)))

    preview, _ = _preview(path)

    assert preview["text"] == [f"t{i}" for i in range(PREVIEW)]
    assert all("chunksize" not in call for call in load_calls)


def test_csv_preview_of_a_header_only_file_still_reports_empty(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text(HEADER)

    with pytest.raises(local.HTTPException) as raised:
        _preview(path)
    assert raised.value.status_code == 400


def test_training_csv_kwargs_keep_the_default_chunk(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(HEADER + _rows(5))

    assert "chunksize" not in csv_as_text_kwargs([path])
