# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Offset walking and batched member writes must leave the export archive byte for byte unchanged."""

from __future__ import annotations

import io
import random
import sys
import zipfile
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from utils import debug_log_export, debug_log_sources
from utils.log_redaction import redact_log_text


class _Clock:
    """One tick per call: the deadline cuts at the same record every run."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        self.now += 1.0
        return self.now


def _write_logs(tmp_path: Path, bodies: list[bytes]) -> list[debug_log_sources.LogSource]:
    sources = []
    for index, body in enumerate(bodies):
        path = tmp_path / f"f{index}.log"
        path.write_bytes(body)
        sources.append(
            debug_log_sources.LogSource(
                id = f"s{index}",
                family = f"fam{index}",
                label = path.name,
                realpath = str(path),
                size_bytes = len(body),
                modified_at = 0,
                is_current = index == 0,
            )
        )
    return sources


def _mixed_body(seed: int, size: int) -> bytes:
    rng = random.Random(seed)
    pieces = [
        b"plain line",
        b"HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz12",
        b"caf\xc3\xa9",
        b"\xff\xfe invalid",
        b"\r",
        b"x" * 700,
    ]
    parts = []
    total = 0
    while total < size:
        record = b"".join(rng.choice(pieces) for _ in range(rng.randint(0, 6)))
        record += rng.choice([b"\n", b"\r\n", b"\n\n"])
        parts.append(record)
        total += len(record)
    return b"".join(parts) + b"tail without newline"


def _reference_lines(body: bytes) -> list[str]:
    lines = body.split(b"\n")
    out = []
    for raw in lines:
        if len(raw) > debug_log_export.MAX_RECORD_BYTES:
            out.append(debug_log_export.OVERSIZED_MARKER)
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            out.append(debug_log_export.UNREADABLE_MARKER)
            continue
        out.append(redact_log_text(text.rstrip("\r")))
    if lines[-1] == b"":
        out.pop()
    return out


def _build(monkeypatch, sources) -> bytes:
    monkeypatch.setattr(debug_log_sources, "list_sources", lambda: sources)
    handle = debug_log_export.build_log_archive()
    try:
        return handle.read()
    finally:
        handle.close()


@pytest.mark.parametrize("chunk", [7, 1000, 64 * 1024, 64 * 1024 + 1])
def test_records_split_across_read_chunks_match_a_whole_file_split(monkeypatch, tmp_path, chunk):
    monkeypatch.setattr(debug_log_export, "_READ_CHUNK_BYTES", chunk)
    body = _mixed_body(chunk, 200_000)
    data = _build(monkeypatch, _write_logs(tmp_path, [body]))
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        text = archive.read("fam0/f0.log").decode("utf-8")
    assert text == "\n".join(_reference_lines(body)) + "\n"


@pytest.mark.parametrize("seed", range(6))
def test_write_batch_size_does_not_change_archive_bytes(monkeypatch, tmp_path, seed):
    rng = random.Random(seed)
    monkeypatch.setattr(debug_log_export, "_READ_CHUNK_BYTES", rng.choice([1000, 64 * 1024]))
    monkeypatch.setattr(debug_log_export, "MAX_TOTAL_SOURCE_BYTES", rng.choice([10**9, 150_000]))
    bodies = [_mixed_body(seed * 10 + index, rng.randint(1, 120_000)) for index in range(3)]
    sources = _write_logs(tmp_path, bodies)
    archives = []
    for batch in (1, debug_log_export.STREAM_CHUNK_BYTES, 10**12):
        monkeypatch.setattr(debug_log_export, "STREAM_CHUNK_BYTES", batch)
        monkeypatch.setattr(debug_log_export, "time", _Clock())
        monkeypatch.setattr(debug_log_export, "MAX_BUILD_SECONDS", rng.choice([10**9, 900.0]))
        archives.append(_build(monkeypatch, sources))
    assert archives[0] == archives[1] == archives[2]


def test_deadline_cuts_after_the_same_record_regardless_of_batching(monkeypatch, tmp_path):
    body = b"".join(b"line %d\n" % index for index in range(5000))
    sources = _write_logs(tmp_path, [body])
    monkeypatch.setattr(debug_log_export, "MAX_BUILD_SECONDS", 2000.0)
    monkeypatch.setattr(debug_log_export, "time", _Clock())
    data = _build(monkeypatch, sources)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        lines = archive.read("fam0/f0.log").decode().splitlines()
    assert lines[-1] == debug_log_export.TRUNCATED_MARKER
    assert lines[:-1] == [f"line {index}" for index in range(len(lines) - 1)]
    assert 1000 < len(lines) < 5000


def test_records_yielded_before_a_read_error_are_all_written(monkeypatch, tmp_path):
    body = b"".join(b"record %d\n" % index for index in range(3000))
    sources = _write_logs(tmp_path, [body])
    original = debug_log_export._redact_record
    calls = []

    def failing(raw: bytes) -> str:
        calls.append(raw)
        if len(calls) == 2500:
            raise OSError(5, "boom")
        return original(raw)

    monkeypatch.setattr(debug_log_export, "_redact_record", failing)
    data = _build(monkeypatch, sources)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert archive.testzip() is None
        lines = archive.read("fam0/f0.log").decode().splitlines()
        warnings = archive.read(debug_log_export.WARNINGS_MEMBER).decode()
    assert lines == [f"record {index}" for index in range(2499)]
    assert "fam0/f0.log: OSError (errno 5)" in warnings
