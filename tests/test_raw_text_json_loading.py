"""Regression test for .json parsing in unsloth/dataprep/raw_text.py.

Both .json and .jsonl map to the "json_lines" handler, which used to parse the
file one line at a time. A real .json file is a single JSON document (commonly
a top-level list of records), so every line failed json.loads, the whole
document was dropped, and the handler returned "" (load_from_file then rejected
the valid file as "empty"). The handler now parses the file as one JSON value
first and falls back to line-by-line for true .jsonl.

raw_text.py's only third-party import is `datasets`, so we stub it and exec the
module directly, with no `import unsloth` (which needs a GPU / unsloth_zoo).
"""

import json
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


def test_json_document_is_parsed_whole(tmp_path):
    loader = _load_raw_text().RawTextDataLoader(tokenizer = object())
    path = tmp_path / "data.json"
    path.write_text(
        json.dumps([{"text": "hello world"}, {"text": "second sample"}], indent = 2), encoding = "utf-8"
    )
    assert loader._read_file_by_format(str(path), "json_lines") == "hello world\n\nsecond sample"


def test_jsonl_is_still_parsed_line_by_line(tmp_path):
    loader = _load_raw_text().RawTextDataLoader(tokenizer = object())
    path = tmp_path / "data.jsonl"
    path.write_text('{"text": "a"}\n{"text": "b"}\n', encoding = "utf-8")
    assert loader._read_file_by_format(str(path), "json_lines") == "a\n\nb"


def test_jsonl_is_never_materialized(tmp_path):
    """A .jsonl file must keep streaming, whole-document parsing is only for .json."""
    real_open = open

    class _StreamOnlyFile:
        """File wrapper that fails the test if the whole file is pulled into memory."""

        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            self.handle.close()
            return False

        def __iter__(self):
            return iter(self.handle)

        def read(self, *args, **kwargs):
            raise AssertionError(".jsonl was read whole instead of streamed line by line")

        def seek(self, *args, **kwargs):
            raise AssertionError(".jsonl was re-read instead of streamed line by line")

    module = _load_raw_text()
    module.open = lambda *args, **kwargs: _StreamOnlyFile(real_open(*args, **kwargs))

    path = tmp_path / "big.jsonl"
    path.write_text('{"text": "a"}\n\n{"text": "b"}\nnot json at all\n', encoding = "utf-8")
    loader = module.RawTextDataLoader(tokenizer = object())
    assert loader._read_file_by_format(str(path), "json_lines") == "a\n\nb"


def test_json_holding_json_lines_still_falls_back(tmp_path):
    """A .json file that actually holds JSON Lines still parses, via the per-line fallback."""
    loader = _load_raw_text().RawTextDataLoader(tokenizer = object())
    path = tmp_path / "mislabelled.json"
    path.write_text('{"text": "a"}\n{"text": "b"}\n', encoding = "utf-8")
    assert loader._read_file_by_format(str(path), "json_lines") == "a\n\nb"


def test_utf8_bom_json_document_is_parsed(tmp_path):
    """Windows tooling prefixes a UTF-8 BOM; it must not sink the whole document."""
    loader = _load_raw_text().RawTextDataLoader(tokenizer = object())
    path = tmp_path / "bom.json"
    path.write_text(
        json.dumps([{"text": "hello world"}, {"text": "second sample"}], indent = 2),
        encoding = "utf-8-sig",
    )
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert loader._read_file_by_format(str(path), "json_lines") == "hello world\n\nsecond sample"


def test_utf8_bom_jsonl_keeps_first_record(tmp_path):
    """A BOM must not silently drop the first .jsonl record."""
    loader = _load_raw_text().RawTextDataLoader(tokenizer = object())
    path = tmp_path / "bom.jsonl"
    path.write_text('{"text": "a"}\n{"text": "b"}\n', encoding = "utf-8-sig")
    assert loader._read_file_by_format(str(path), "json_lines") == "a\n\nb"


def test_utf8_bom_json_holding_json_lines_falls_back(tmp_path):
    """The per-line fallback re-reads from byte 0, so the BOM must be stripped again."""
    loader = _load_raw_text().RawTextDataLoader(tokenizer = object())
    path = tmp_path / "bom_mislabelled.json"
    path.write_text('{"text": "a"}\n{"text": "b"}\n', encoding = "utf-8-sig")
    assert loader._read_file_by_format(str(path), "json_lines") == "a\n\nb"


def test_utf8_bom_plain_text_and_csv(tmp_path):
    """The BOM also leaks into .txt training text and the first .csv column name."""
    loader = _load_raw_text().RawTextDataLoader(tokenizer = object())
    txt = tmp_path / "bom.txt"
    txt.write_text("hello", encoding = "utf-8-sig")
    assert loader._read_file_by_format(str(txt), "plain_text") == "hello"

    csv_path = tmp_path / "bom.csv"
    csv_path.write_text("text,other\nhello,x\n", encoding = "utf-8-sig")
    assert loader._read_file_by_format(str(csv_path), "csv_text_column") == "hello"
