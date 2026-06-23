#!/usr/bin/env python3
"""Regression test: SyntheticDataKit.chunk_data must not drop a single-chunk document."""

import os
import tempfile
from types import SimpleNamespace

from unsloth.dataprep.synthetic import SyntheticDataKit


class _MockTokenizer:
    def __call__(self, text, add_special_tokens = False):
        return SimpleNamespace(input_ids = list(range(len(text.split()))))

    def decode(self, token_ids):
        return " ".join(f"w{i}" for i in token_ids)


def _make_kit(max_seq_length = 2048, max_generation_tokens = 512, overlap = 64):
    kit = SyntheticDataKit.__new__(SyntheticDataKit)
    kit.tokenizer = _MockTokenizer()
    kit.max_seq_length = max_seq_length
    kit.max_generation_tokens = max_generation_tokens
    kit.overlap = overlap
    return kit


def _chunk(text):
    """Returns (chunk_filenames, chunk_contents); reads content before cleanup."""
    kit = _make_kit()
    with tempfile.NamedTemporaryFile("w", suffix = ".txt", delete = False) as f:
        f.write(text)
        path = f.name
    created = []
    try:
        created = kit.chunk_data(filename = path)
        contents = []
        for fn in created:
            with open(fn, encoding = "utf-8") as fh:
                contents.append(fh.read())
        return list(created), contents
    finally:
        os.unlink(path)
        for fn in created:
            if os.path.exists(fn):
                os.unlink(fn)


def test_chunk_data_keeps_single_chunk_document():
    # A short document fits in one chunk (n_chunks == 1) and must still produce
    # one output file rather than silently vanishing.
    out, contents = _chunk("word " * 50)
    assert len(out) == 1, f"single-chunk doc should yield 1 file, got {len(out)}"
    assert contents[0] != "", "the chunk file must contain the document text"


def test_chunk_data_still_splits_long_document():
    # A long document (n_chunks > 1) must still produce multiple chunks.
    out, _ = _chunk("word " * 5000)
    assert len(out) > 1, f"long doc should yield multiple chunks, got {len(out)}"


if __name__ == "__main__":
    test_chunk_data_keeps_single_chunk_document()
    test_chunk_data_still_splits_long_document()
    print("OK")
