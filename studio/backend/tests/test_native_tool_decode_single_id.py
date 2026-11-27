# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""transformers 5.24's TextStreamer decodes one bare id after a newline; NativeToolTokenDecoder iterated it."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from core.inference.native_tool_tokens import NativeToolTokenDecoder
from test_mlx_incremental_decode import _byte_level  # noqa: E402


@pytest.fixture(scope = "module")
def decoder():
    tokenizer = _byte_level()
    decoder = NativeToolTokenDecoder(tokenizer)
    assert decoder.dropped_ids()
    return tokenizer, decoder


def test_decode_takes_a_single_id(decoder):
    tokenizer, native = decoder
    ids = tokenizer.encode("hello world\n", add_special_tokens = False)
    for token_id in ids:
        assert native.decode(token_id) == native.decode([token_id])


def test_decode_takes_a_zero_dim_tensor(decoder):
    tokenizer, native = decoder
    token_id = tokenizer.encode("hello", add_special_tokens = False)[-1]
    assert native.decode(torch.tensor(token_id)) == native.decode([token_id])


def test_single_special_id_is_still_suppressed(decoder):
    tokenizer, native = decoder
    eot = tokenizer.convert_tokens_to_ids("<|eot|>")
    assert native.decode(eot) == ""


def test_newline_flush_as_transformers_5_24_runs_it(decoder):
    """The 5.24 newline branch of TextStreamer.put, verbatim, over the native decoder."""
    tokenizer, native = decoder
    token_cache = tokenizer.encode("first line\n", add_special_tokens = False)
    text = native.decode(token_cache)
    assert text.endswith("\n")
    token_cache = token_cache[-1:]
    print_len = len(native.decode(token_cache[-1]))
    assert print_len == len(native.decode(token_cache))
