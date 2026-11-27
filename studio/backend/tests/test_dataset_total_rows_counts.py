# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""check-format's ``total_rows`` for a Hub id (datasets server ``/size``) and an uploaded JSON / JSONL /
CSV file (bounded scan): exact or None, never an error and never a long wait."""

from __future__ import annotations

import json
import time

import httpx
import pytest
from datasets import Dataset

from hub.schemas.datasets import CheckFormatRequest
from hub.services.datasets import formatting, row_counts
from hub.services.datasets.local import _load_local_preview_slice

SERVER = "https://datasets-server.example"


def _size_body(splits, partial = False):
    return {"size": {"splits": splits}, "partial": partial}


def _split(
    config,
    split,
    rows,
    estimated = None,
):
    return {"config": config, "split": split, "num_rows": rows, "estimated_num_rows": estimated}


class _Response:
    def __init__(
        self,
        body,
        status = 200,
    ):
        self._body, self.status_code = body, status

    def json(self):
        return self._body


@pytest.fixture
def hub(monkeypatch):
    """Route ``/size`` to a fake; ``calls`` records each request's params."""
    row_counts._hub_rows_cache.clear()
    state = {"body": _size_body([_split("default", "train", 1000)]), "calls": [], "delay": 0.0}

    def fake_get(
        url,
        params = None,
        headers = None,
        timeout = None,
    ):
        state["calls"].append(
            {"url": url, "params": params, "headers": headers, "timeout": timeout}
        )
        if state["delay"]:
            time.sleep(state["delay"])
        if isinstance(state["body"], Exception):
            raise state["body"]
        return _Response(state["body"], state.get("status", 200))

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(row_counts, "_hub_rows_source", lambda: SERVER)
    yield state
    row_counts._hub_rows_cache.clear()


def test_hub_rows_reads_split_num_rows(hub):
    assert row_counts.start_hub_split_rows("org/ds", None, "train", "hf_tok")() == 1000
    call = hub["calls"][0]
    assert call["url"] == f"{SERVER}/size"
    assert call["params"] == {"dataset": "org/ds"}
    assert call["headers"] == {"Authorization": "Bearer hf_tok"}
    assert call["timeout"] <= row_counts.HUB_ROWS_TIMEOUT_S


def test_hub_rows_picks_subset_and_split(hub):
    hub["body"] = _size_body(
        [
            _split("en", "train", 50),
            _split("en", "test", 7),
            _split("fr", "train", 60),
        ]
    )
    assert row_counts.start_hub_split_rows("org/ds", "fr", "train", None)() == 60
    assert hub["calls"][0]["params"] == {"dataset": "org/ds", "config": "fr"}
    assert hub["calls"][0]["headers"] is None
    assert row_counts.start_hub_split_rows("org/ds", "en", "test", None)() == 7
    # No subset and several named configs: load_dataset would refuse, so no count.
    assert row_counts.start_hub_split_rows("org/ds2", None, "train", None)() is None
    assert row_counts.start_hub_split_rows("org/ds2", None, "validation", None)() is None


def test_hub_rows_partial_uses_estimate_or_none(hub):
    hub["body"] = _size_body([_split("default", "train", 100, estimated = 5_000_000)], partial = True)
    assert row_counts.start_hub_split_rows("org/big", None, "train", None)() == 5_000_000
    hub["body"] = _size_body([_split("default", "train", 100)], partial = True)
    assert row_counts.start_hub_split_rows("org/big2", None, "train", None)() is None


@pytest.mark.parametrize(
    "body",
    [
        httpx.ConnectTimeout("slow"),
        RuntimeError("boom"),
        {"error": "not ready"},
        _size_body([_split("default", "train", 0)]),
        _size_body([_split("default", "train", True)]),
        ["not", "a", "dict"],
    ],
)
def test_hub_rows_failure_is_none(hub, body):
    hub["body"] = body
    assert row_counts.start_hub_split_rows("org/ds", None, "train", None)() is None
    assert row_counts._hub_rows_cache == {}


def test_hub_rows_http_error_is_none(hub):
    hub["status"] = 401
    assert row_counts.start_hub_split_rows("org/private", None, "train", None)() is None


def test_hub_rows_slow_server_does_not_block(hub):
    hub["delay"] = 1.0
    started = time.monotonic()
    assert row_counts.start_hub_split_rows("org/ds", None, "train", None, timeout = 0.1)() is None
    assert time.monotonic() - started < 0.5


def test_hub_rows_cache_hit_skips_request(hub):
    assert row_counts.start_hub_split_rows("org/ds", None, "train", None)() == 1000
    assert row_counts.start_hub_split_rows("org/ds", None, "train", "other")() == 1000
    assert len(hub["calls"]) == 1
    # A different split or config is its own entry.
    assert row_counts.start_hub_split_rows("org/ds", "default", "train", None)() == 1000
    assert len(hub["calls"]) == 2


def test_hub_rows_cache_is_bounded(hub, monkeypatch):
    monkeypatch.setattr(row_counts, "_HUB_ROWS_CACHE_MAX", 3)
    for index in range(5):
        row_counts.start_hub_split_rows(f"org/ds{index}", None, "train", None)()
    assert len(row_counts._hub_rows_cache) == 3


def test_hub_rows_unusable_split_or_source_skips_network(hub, monkeypatch):
    assert row_counts.start_hub_split_rows("org/ds", None, "train[:10]", None)() is None
    monkeypatch.setattr(row_counts, "_hub_rows_source", lambda: None)
    assert row_counts.start_hub_split_rows("org/ds", None, "train", None)() is None
    assert hub["calls"] == []


def test_check_format_fills_total_rows_for_hub_id(hub, monkeypatch):
    preview = Dataset.from_list([{"text": f"row {i}"} for i in range(10)])
    monkeypatch.setattr(formatting, "load_preview_slice", lambda *a, **k: (preview, None))
    monkeypatch.setattr(formatting, "account_hf_token", lambda token: token)
    response = formatting.check_format_response(
        CheckFormatRequest(dataset_name = "mlabonne/guanaco-llama2-1k")
    )
    assert response.total_rows == 1000


def test_check_format_keeps_preview_total_and_skips_local(hub, monkeypatch):
    preview = Dataset.from_list([{"text": f"row {i}"} for i in range(10)])
    monkeypatch.setattr(formatting, "load_preview_slice", lambda *a, **k: (preview, 42))
    monkeypatch.setattr(formatting, "account_hf_token", lambda token: token)
    response = formatting.check_format_response(
        CheckFormatRequest(dataset_name = "org/ds", prefer_local_cache = True)
    )
    assert response.total_rows == 42
    assert hub["calls"] == []


def test_check_format_hub_failure_leaves_none(hub, monkeypatch):
    hub["body"] = httpx.ReadTimeout("slow")
    preview = Dataset.from_list([{"text": f"row {i}"} for i in range(10)])
    monkeypatch.setattr(formatting, "load_preview_slice", lambda *a, **k: (preview, None))
    monkeypatch.setattr(formatting, "account_hf_token", lambda token: token)
    response = formatting.check_format_response(CheckFormatRequest(dataset_name = "org/ds"))
    assert response.total_rows is None


def _write(path, text):
    path.write_text(text, encoding = "utf-8")
    return path


def test_local_jsonl_count(tmp_path):
    rows = [{"text": f"row {i}"} for i in range(25)]
    path = _write(tmp_path / "d.jsonl", "\n".join(json.dumps(r) for r in rows) + "\n\n  \n")
    assert row_counts.count_local_file_rows(path) == 25
    preview, total = _load_local_preview_slice(
        dataset_path = path, train_split = "train", preview_size = 10
    )
    assert len(preview) == 10 and total == 25


def test_local_json_array_and_lines(tmp_path):
    rows = [{"text": f"row {i}"} for i in range(13)]
    array = _write(tmp_path / "a.json", json.dumps(rows, indent = 2))
    assert row_counts.count_local_file_rows(array) == 13
    lines = _write(tmp_path / "l.json", "\n".join(json.dumps(r) for r in rows))
    assert row_counts.count_local_file_rows(lines) == 13
    # A pretty-printed object spans lines: no count rather than a wrong one.
    pretty = _write(tmp_path / "p.json", json.dumps({"text": ["a", "b"]}, indent = 2))
    assert row_counts.count_local_file_rows(pretty) is None
    preview, total = _load_local_preview_slice(
        dataset_path = array, train_split = "train", preview_size = 10
    )
    assert len(preview) == 10 and total == 13


def test_local_csv_count_with_quoted_newlines(tmp_path):
    body = "text,label\n" + "".join(f'"line {i}\nstill row {i}, quoted",{i}\n' for i in range(17))
    path = _write(tmp_path / "d.csv", body + "\n")
    assert row_counts.count_local_file_rows(path) == 17
    preview, total = _load_local_preview_slice(
        dataset_path = path, train_split = "train", preview_size = 10
    )
    assert len(preview) == 10 and total == 17


def test_local_counts_respect_caps(tmp_path, monkeypatch):
    rows = [{"text": f"row {i}"} for i in range(30)]
    jsonl = _write(tmp_path / "d.jsonl", "\n".join(json.dumps(r) for r in rows))
    array = _write(tmp_path / "a.json", json.dumps(rows))
    csv = _write(tmp_path / "d.csv", "text\n" + "\n".join(f"r{i}" for i in range(30)))
    monkeypatch.setattr(row_counts, "LINE_SCAN_MAX_BYTES", 10)
    monkeypatch.setattr(row_counts, "JSON_ARRAY_MAX_BYTES", 10)
    monkeypatch.setattr(row_counts, "CSV_PARSE_MAX_BYTES", 10)
    assert row_counts.count_local_file_rows(jsonl) is None
    assert row_counts.count_local_file_rows(array) is None
    assert row_counts.count_local_file_rows(csv) is None
    _, total = _load_local_preview_slice(dataset_path = jsonl, train_split = "train", preview_size = 5)
    assert total is None


def test_local_count_failures_are_none(tmp_path):
    assert row_counts.count_local_file_rows(tmp_path / "missing.jsonl") is None
    assert row_counts.count_local_file_rows(_write(tmp_path / "e.csv", "")) is None
    assert row_counts.count_local_file_rows(_write(tmp_path / "s.json", "[1, 2]")) is None
    assert row_counts.count_local_file_rows(_write(tmp_path / "x.txt", "a\nb\n")) is None


def test_local_dir_of_several_files_is_not_counted(tmp_path):
    for name in ("a.jsonl", "b.jsonl"):
        _write(tmp_path / name, "\n".join(json.dumps({"text": str(i)}) for i in range(12)))
    _, total = _load_local_preview_slice(dataset_path = tmp_path, train_split = "train", preview_size = 5)
    assert total is None


def test_local_json_over_the_cap_is_never_read_whole(tmp_path, monkeypatch):
    # A minified single-line array past the array cap: classified from a bounded prefix, never parsed.
    monkeypatch.setattr(row_counts, "JSON_ARRAY_MAX_BYTES", 1000)
    monkeypatch.setattr(row_counts, "FIRST_LINE_MAX_BYTES", 64)
    big = tmp_path / "big.json"
    big.write_text(json.dumps([{"text": "x" * 50} for _ in range(50)]), encoding = "utf-8")
    reads = []
    real_open = open

    def tracking_open(
        path,
        mode = "r",
        *args,
        **kwargs,
    ):
        handle = real_open(path, mode, *args, **kwargs)
        real_readline = handle.readline

        def readline(limit = -1):
            reads.append(limit)
            return real_readline(limit)

        try:
            handle.readline = readline
        except AttributeError:
            pass
        return handle

    monkeypatch.setattr("builtins.open", tracking_open)
    monkeypatch.setattr(json, "load", lambda *a, **k: pytest.fail("parsed an over-cap array"))
    assert row_counts.count_local_file_rows(big) is None
    assert reads and all(0 < limit <= 65 for limit in reads)

    # Past every cap: rejected from the size alone.
    monkeypatch.setattr(row_counts, "LINE_SCAN_MAX_BYTES", 1000)
    reads.clear()
    assert row_counts.count_local_file_rows(big) is None
    assert reads == []


def test_local_json_lines_with_a_huge_first_row_give_none(tmp_path, monkeypatch):
    monkeypatch.setattr(row_counts, "FIRST_LINE_MAX_BYTES", 64)
    path = tmp_path / "rows.json"
    path.write_text(
        json.dumps({"text": "y" * 200}) + "\n" + json.dumps({"text": "z"}) + "\n", encoding = "utf-8"
    )
    assert row_counts.count_local_file_rows(path) is None
