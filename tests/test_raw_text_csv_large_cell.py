# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""A CSV text cell over the csv module's 128 KB default limit loads instead of raising."""

import csv
import sys
import threading
import types
from pathlib import Path

RAW_TEXT_PATH = Path(__file__).parents[1] / "unsloth" / "dataprep" / "raw_text.py"


def _load_raw_text():
    sys.modules.setdefault("datasets", types.SimpleNamespace(Dataset = object))
    module = types.ModuleType("unsloth_raw_text_under_test")
    exec(
        compile(RAW_TEXT_PATH.read_text(encoding = "utf-8"), str(RAW_TEXT_PATH), "exec"),
        module.__dict__,
    )
    return module


def _write_csv(path, cells):
    with open(path, "w", newline = "", encoding = "utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "text"])
        for index, cell in enumerate(cells):
            writer.writerow([index, cell])


def test_cell_over_default_csv_limit_is_read(tmp_path):
    loader = _load_raw_text().RawTextDataLoader(tokenizer = object())
    long_cell = "word " * 40_000
    assert len(long_cell) > csv.field_size_limit()
    path = tmp_path / "book.csv"
    _write_csv(path, ["short", long_cell])
    assert loader._read_file_by_format(str(path), "csv_text_column") == "short\n\n" + long_cell


def test_csv_field_limit_is_restored(tmp_path):
    loader = _load_raw_text().RawTextDataLoader(tokenizer = object())
    path = tmp_path / "data.csv"
    _write_csv(path, ["hello"])
    before = csv.field_size_limit()
    loader._read_file_by_format(str(path), "csv_text_column")
    assert csv.field_size_limit() == before


def test_overlapping_reads_keep_the_raised_limit(tmp_path):
    module = _load_raw_text()
    a_started, b_started, a_done = threading.Event(), threading.Event(), threading.Event()

    class Loader(module.RawTextDataLoader):
        def _extract_text_from_csv_row(self, row):
            name = threading.current_thread().name
            if name == "A" and not a_started.is_set():
                a_started.set()
                b_started.wait(1)
            if name == "B" and not b_started.is_set():
                b_started.set()
                a_done.wait(1)
            return super()._extract_text_from_csv_row(row)

    paths = {"A": tmp_path / "a.csv", "B": tmp_path / "b.csv"}
    for path in paths.values():
        _write_csv(path, ["short", "word " * 40_000])
    before = csv.field_size_limit()
    results = {}

    def read():
        name = threading.current_thread().name
        try:
            Loader(tokenizer = object())._read_file_by_format(str(paths[name]), "csv_text_column")
            results[name] = "ok"
        except Exception as error:
            results[name] = repr(error)
        if name == "A":
            a_done.set()

    thread_a = threading.Thread(target = read, name = "A")
    thread_a.start()
    a_started.wait(2)
    thread_b = threading.Thread(target = read, name = "B")
    thread_b.start()
    thread_a.join()
    thread_b.join()
    assert results == {"A": "ok", "B": "ok"}
    assert csv.field_size_limit() == before
