# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Manifest files saved with a UTF-8 BOM must not abort or corrupt the audio-path rewrite."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest

# Load by path to skip heavy package __init__ chains (structlog, torch).
_BACKEND = Path(__file__).resolve().parents[1]


def _load(mod_name, rel_path):
    spec = importlib.util.spec_from_file_location(mod_name, _BACKEND / rel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s3_dataset = _load("s3_dataset_bom", "core/training/s3_dataset.py")

_BOM = "\ufeff"
_MANIFEST_ROWS = {"audio": "clips/a.wav"}, {"audio": "clips/b.wav"}


def _resolve(value):
    return f"/dl/{value}" if isinstance(value, str) and value.endswith(".wav") else None


@pytest.mark.parametrize("extension", [".jsonl", ".json"])
def test_jsonl_manifest_with_utf8_bom_rewrites_every_row(tmp_path, extension):
    manifest = tmp_path / f"data{extension}"
    manifest.write_text(
        _BOM + "".join(json.dumps(row) + "\n" for row in _MANIFEST_ROWS),
        encoding = "utf-8",
    )
    s3_dataset._rewrite_json_manifest(str(manifest), _resolve)
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding = "utf-8").splitlines()
        if line.strip()
    ]
    assert rows == [
        {"audio": "/dl/clips/a.wav"},
        {"audio": "/dl/clips/b.wav"},
    ]


def test_json_manifest_with_utf8_bom_rewrites_the_record(tmp_path):
    manifest = tmp_path / "data.json"
    manifest.write_text(_BOM + json.dumps({"audio": "clips/a.wav"}), encoding = "utf-8")
    s3_dataset._rewrite_json_manifest(str(manifest), _resolve)
    assert json.loads(manifest.read_text(encoding = "utf-8")) == {
        "audio": "/dl/clips/a.wav"
    }


def test_csv_manifest_with_utf8_bom_rewrites_the_first_cell(tmp_path):
    manifest = tmp_path / "data.csv"
    manifest.write_text(_BOM + "clips/a.wav,clips/b.wav\n", encoding = "utf-8")
    s3_dataset._rewrite_csv_manifest(str(manifest), _resolve)
    with open(manifest, encoding = "utf-8", newline = "") as f:
        rows = list(csv.reader(f))
    assert rows == [["/dl/clips/a.wav", "/dl/clips/b.wav"]]
