# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Generated Unicode and separator cases preserve text and token bounds."""

import pytest

pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st

from core.rag.chunking import chunk_pages
from core.rag.parsers import parse_text


@pytest.mark.parametrize("counter", ["words", "characters", "utf8"])
@settings(max_examples = 250, deadline = None, derandomize = True, database = None)
@given(
    text = st.text(alphabet = "abcdef0123 \r\n\t#.| café東京漢字\U0001f600", min_size = 0, max_size = 2000),
    limit = st.integers(min_value = 8, max_value = 128),
    overlap = st.integers(min_value = 0, max_value = 64),
)
def test_generated_chunks_preserve_every_nonspace_character(counter, text, limit, overlap):
    counters = {
        "words": lambda value: len(value.split()),
        "characters": len,
        "utf8": lambda value: len(value.encode("utf-8")),
    }
    count = counters[counter]
    chunks = chunk_pages(
        parse_text(text), max_tokens = limit, overlap = min(overlap, limit - 1), count = count
    )
    covered = set()
    for index, chunk in enumerate(chunks):
        assert chunk.chunk_index == index
        assert chunk.text == text[chunk.page_char_start : chunk.page_char_end]
        assert chunk.token_count == count(chunk.text) <= limit
        covered.update(range(chunk.page_char_start, chunk.page_char_end))
    assert all(i in covered for i, char in enumerate(text) if not char.isspace())
