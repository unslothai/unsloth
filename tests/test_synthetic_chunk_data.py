#!/usr/bin/env python3
"""Regression tests for SyntheticDataKit.chunk_data: short-document handling and overlap validation."""

import os
import tempfile
from types import SimpleNamespace

from unsloth.dataprep.synthetic import SyntheticDataKit


class _MockTokenizer:
    def __call__(
        self,
        text,
        add_special_tokens = False,
    ):
        return SimpleNamespace(input_ids = list(range(len(text.split()))))

    def decode(self, token_ids):
        return " ".join(f"w{i}" for i in token_ids)


def _make_kit(
    max_seq_length = 2048,
    max_generation_tokens = 512,
    overlap = 64,
):
    kit = SyntheticDataKit.__new__(SyntheticDataKit)
    kit.tokenizer = _MockTokenizer()
    kit.max_seq_length = max_seq_length
    kit.max_generation_tokens = max_generation_tokens
    kit.overlap = overlap
    return kit


def _chunk(text, kit = None):
    """Returns (chunk_filenames, chunk_contents); reads content before cleanup."""
    if kit is None:
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


def test_chunk_data_empty_document_yields_no_chunks():
    # An empty document must not produce an (empty) chunk file.
    out, _ = _chunk("")
    assert out == [], f"empty doc should yield no files, got {len(out)}"


def test_chunk_data_short_document_is_not_split_into_fragments():
    # A document shorter than the overlap previously reached the multi-chunk path
    # (n_chunks >= 3) where linspace produced negative start indices, slicing the
    # wrong tail tokens. It must be emitted as one chunk covering the whole document.
    kit = _make_kit(
        max_seq_length = 2048, max_generation_tokens = 920, overlap = 64
    )  # max_tokens = 80
    out, contents = _chunk(
        "word " * 50, kit = kit
    )  # 50 tokens < overlap (would be 4 chunks)
    assert len(out) == 1, f"sub-overlap doc should yield 1 chunk, got {len(out)}"
    assert contents[0].split() == [
        f"w{i}" for i in range(50)
    ], f"chunk must cover the whole document, not a fragment; got: {contents[0]!r}"


def test_chunk_data_rejects_overlap_not_smaller_than_chunk():
    # If overlap >= chunk size the stride is non-positive, which would divide by zero
    # or emit one oversized chunk. The config must be rejected with a clear error.
    kit = _make_kit(
        max_seq_length = 2048, max_generation_tokens = 950, overlap = 64
    )  # max_tokens = 20
    with tempfile.NamedTemporaryFile("w", suffix = ".txt", delete = False) as f:
        f.write("word " * 50)
        path = f.name
    try:
        try:
            kit.chunk_data(filename = path)
            raise AssertionError("expected RuntimeError when overlap >= chunk size")
        except RuntimeError as e:
            assert "overlap" in str(e), f"error should mention overlap, got: {e}"
    finally:
        os.unlink(path)


def test_chunk_data_uninitialized_error_names_real_class():
    # Without max_seq_length the guard tells the user which method to call first.
    # The message must name the real class (SyntheticDataKit) so copying it works;
    # a misspelling would raise NameError when the user follows it verbatim.
    kit = SyntheticDataKit.__new__(SyntheticDataKit)
    kit.tokenizer = _MockTokenizer()  # max_seq_length intentionally unset
    with tempfile.NamedTemporaryFile("w", suffix = ".txt", delete = False) as f:
        f.write("word " * 50)
        path = f.name
    try:
        try:
            kit.chunk_data(filename = path)
            raise AssertionError("expected RuntimeError when max_seq_length is unset")
        except RuntimeError as e:
            msg = str(e)
            assert (
                "SyntheticDataKit.from_pretrained" in msg
            ), f"error must name SyntheticDataKit.from_pretrained, got: {msg}"
            assert (
                "SynthetidDataKit" not in msg
            ), f"error must not misspell the class name, got: {msg}"
    finally:
        os.unlink(path)


def test_chunk_data_chunks_do_not_exceed_max_tokens():
    # Every chunk must fit within max_tokens. The old multi-chunk path emitted one
    # fewer, oversized chunk, and a doc just over the threshold came back unsplit.
    kit = _make_kit(max_seq_length = 2048, max_generation_tokens = 760, overlap = 64)
    max_tokens = 2048 - 760 * 2 - 128  # 400

    for n_words in (500, 2000):
        out, contents = _chunk("word " * n_words, kit = kit)
        assert (
            len(out) >= 2
        ), f"a {n_words}-token doc (> max_tokens={max_tokens}) must be split, got {len(out)}"
        for content in contents:
            n_tokens = len(content.split())
            assert (
                n_tokens <= max_tokens
            ), f"chunk has {n_tokens} tokens, exceeding max_tokens={max_tokens}"


def test_chunk_data_does_not_over_split():
    # n_chunks must be the minimum count: ceil((length - overlap) / stride), not
    # ceil(length / stride) which over-splits just past a stride multiple. At 673
    # tokens (max_tokens=400, overlap=64) the tight count gives 2 chunks (~369+368).
    kit = _make_kit(max_seq_length = 2048, max_generation_tokens = 760, overlap = 64)
    max_tokens = 2048 - 760 * 2 - 128  # 400
    out, contents = _chunk("word " * 673, kit = kit)
    assert (
        len(out) == 2
    ), f"673-token doc should yield the minimal 2 chunks, got {len(out)}"
    for content in contents:
        n_tokens = len(content.split())
        assert (
            n_tokens <= max_tokens
        ), f"chunk has {n_tokens} tokens, exceeding max_tokens={max_tokens}"


if __name__ == "__main__":
    test_chunk_data_keeps_single_chunk_document()
    test_chunk_data_still_splits_long_document()
    test_chunk_data_empty_document_yields_no_chunks()
    test_chunk_data_short_document_is_not_split_into_fragments()
    test_chunk_data_rejects_overlap_not_smaller_than_chunk()
    test_chunk_data_uninitialized_error_names_real_class()
    test_chunk_data_chunks_do_not_exceed_max_tokens()
    test_chunk_data_does_not_over_split()
    print("OK")
