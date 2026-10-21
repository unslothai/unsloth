# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An `.arrow` data file is validated before its rows are read: lengths and offsets that do not fit the file's
buffers are a ValueError, never a read past them."""

from __future__ import annotations

import io
import struct

import pytest

from hub.services.datasets.local import _stream_file_preview_slice

VALUES = b"ABCDEFGH"
ROWS = 15
WINDOW = 8192


@pytest.fixture(autouse = True)
def guarded():
    try:
        from utils.security import dataset_media
    except ImportError:  # a tree without the guard
        return
    dataset_media.install()
    dataset_media.install()


def _stream_bytes() -> bytes:
    import pyarrow

    offsets = [0] + [len(VALUES)] * ROWS
    column = pyarrow.LargeListArray.from_arrays(
        pyarrow.array(offsets, pyarrow.int64()),
        pyarrow.array(list(VALUES), pyarrow.uint8()),
    )
    batch = pyarrow.record_batch([column], names = ["v"])
    sink = io.BytesIO()
    with pyarrow.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue()


def _malformed(raw: bytes) -> bytes:
    """The same stream claiming WINDOW values over its 8-byte buffer, with offsets around it."""
    crafted = bytearray(raw)
    nodes = crafted.index(struct.pack("<4q", ROWS, 0, len(VALUES), 0))
    crafted[nodes + 16 : nodes + 24] = struct.pack("<q", WINDOW)
    offsets = [0] + [len(VALUES)] * ROWS
    start = crafted.index(struct.pack("<16q", *offsets), nodes)
    crafted[start : start + 128] = struct.pack("<16q", -WINDOW, 0, *[WINDOW] * (ROWS - 1))
    return bytes(crafted)


def test_malformed_arrow_stream_is_refused(tmp_path):
    path = tmp_path / "train.arrow"
    path.write_bytes(_malformed(_stream_bytes()))
    with pytest.raises(ValueError):
        _stream_file_preview_slice(path, 10)


def test_malformed_arrow_file_is_refused_by_a_full_load(tmp_path):
    from datasets import load_dataset

    path = tmp_path / "train.arrow"
    path.write_bytes(_malformed(_stream_bytes()))
    with pytest.raises(Exception) as exc:
        load_dataset("arrow", data_files = str(path), split = "train", cache_dir = str(tmp_path / "c"))
    chain = [exc.value, exc.value.__cause__, exc.value.__context__]
    assert any(isinstance(error, ValueError) for error in chain)


def test_intact_arrow_stream_previews(tmp_path):
    path = tmp_path / "validation.arrow"
    path.write_bytes(_stream_bytes())
    dataset, _ = _stream_file_preview_slice(path, 10)
    assert bytes(dataset[0]["v"]) == VALUES


def test_datasets_cache_files_skip_the_builder(tmp_path, monkeypatch):
    from datasets import load_dataset
    from datasets.packaged_modules.arrow.arrow import Arrow

    path = tmp_path / "validation.arrow"
    path.write_bytes(_stream_bytes())
    kwargs = {"data_files": str(path), "split": "train", "cache_dir": str(tmp_path / "c")}
    assert bytes(load_dataset("arrow", **kwargs)[0]["v"]) == VALUES

    def no_cast(self, pa_table):
        raise AssertionError("a cached dataset was re-read through the arrow builder")

    monkeypatch.setattr(Arrow, "_cast_table", no_cast)
    assert bytes(load_dataset("arrow", **kwargs)[0]["v"]) == VALUES
