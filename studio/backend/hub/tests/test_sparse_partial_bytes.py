# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The all-sparse partial: a file sized before a single byte of it is transferred.

A parallel Range writer sets the ``.incomplete`` to its final length up front and then fills
blocks out of order. In between the file is its full logical size with nothing allocated behind
it, and crediting ``st_size`` there reads "0 B left" on a download that has not moved.
"""

import os
from pathlib import Path

import pytest

from hub.utils import hf_cache_state
from hub.utils.hf_cache_state import blob_bytes_present


_SIZE = 8 * 1024 * 1024


def _sized_but_unwritten(path: Path) -> Path:
    """Full length, no blocks: what a preallocating writer leaves before its first chunk."""
    with open(path, "wb") as handle:
        handle.truncate(_SIZE)
    return path


def _needs_sparse_support(path: Path) -> None:
    if not hf_cache_state._holds_no_data(_sized_but_unwritten(path.parent / ".probe")):
        pytest.skip("filesystem does not report sparse extents")


def test_a_sized_but_unwritten_partial_is_worth_nothing(tmp_path):
    blob = _sized_but_unwritten(tmp_path / "a.incomplete")
    _needs_sparse_support(blob)
    assert os.stat(blob).st_size == _SIZE
    assert blob_bytes_present(blob) == 0


def test_the_first_chunk_to_land_is_counted(tmp_path):
    """The zero is about emptiness, not about being sparse -- a written extent still counts."""
    blob = _sized_but_unwritten(tmp_path / "b.incomplete")
    _needs_sparse_support(blob)
    with open(blob, "r+b") as handle:
        handle.seek(_SIZE - 4096)
        handle.write(b"x" * 4096)
    present = blob_bytes_present(blob)
    assert 0 < present <= _SIZE


def test_a_filesystem_that_reports_no_blocks_still_falls_back_to_size(tmp_path, monkeypatch):
    """Where the field is simply never populated, the logical size remains the best guess."""
    blob = _sized_but_unwritten(tmp_path / "c.incomplete")
    monkeypatch.setattr(hf_cache_state, "_holds_no_data", lambda _path: False)
    assert blob_bytes_present(blob) == _SIZE


def test_a_whole_file_is_still_measured_by_its_blocks(tmp_path):
    """The dense case is unchanged: bytes actually written are what is present."""
    blob = tmp_path / "d.incomplete"
    blob.write_bytes(b"x" * 65536)
    _needs_sparse_support(blob)
    assert blob_bytes_present(blob) == 65536
