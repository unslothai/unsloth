"""A CSV text cell over the csv module's 128 KB default limit loads instead of raising."""

import csv
import sys
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
