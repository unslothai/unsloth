# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""IncrementalTextIteratorStreamer emits exactly upstream TextIteratorStreamer's chunks while
decoding only the ids since the last split. In-memory tokenizers, no download."""

import random
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

from transformers import TextIteratorStreamer

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import core.inference.incremental_streamer as incremental_streamer
from core.inference.incremental_streamer import IncrementalTextIteratorStreamer
from core.inference.native_tool_tokens import NativeToolTokenDecoder
from test_mlx_incremental_decode import (  # noqa: E402
    _CORPUS,
    _TOKENIZERS,
    _byte_level,
    _counting,
    _streams,
)


class _Recording:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunks = []

    def on_finalized_text(
        self,
        text,
        stream_end = False,
    ):
        self.chunks.append((text, stream_end))


class _Upstream(_Recording, TextIteratorStreamer):
    pass


class _Incremental(_Recording, IncrementalTextIteratorStreamer):
    pass


def _drive(cls, source, ids, *, skip_prompt, sizes, generations, **decode_kwargs):
    streamer = cls(source, skip_prompt = skip_prompt, **decode_kwargs)
    for _ in range(generations):
        streamer.put(torch.tensor([ids[:3] or [0]]))
        index = 0
        for size in sizes:
            if index >= len(ids):
                break
            streamer.put(torch.tensor(ids[index : index + size]))
            index += size
        streamer.end()
    return streamer


@pytest.fixture(scope = "module", params = sorted(_TOKENIZERS))
def tokenizer(request):
    return _TOKENIZERS[request.param]()


@pytest.mark.parametrize("mode", ["plain", "keep-special", "native-tool", "reasoning"])
def test_chunks_equal_upstream(tokenizer, mode):
    rng = random.Random(mode)
    newline_ids = {i for i in range(len(tokenizer)) if "\n" in tokenizer.decode([i])}
    streams = dict(_streams(tokenizer, seed = 2))
    for name, ids in list(streams.items()):
        streams[f"{name}-one-line"] = [i for i in ids if i not in newline_ids]
    for name, ids in streams.items():
        if mode == "plain":
            make, kwargs = (lambda: tokenizer), {"skip_special_tokens": True}
        elif mode == "keep-special":
            make, kwargs = (lambda: tokenizer), {"skip_special_tokens": False}
        elif mode == "native-tool":
            make, kwargs = (lambda: NativeToolTokenDecoder(tokenizer)), {}
        else:
            make = lambda: NativeToolTokenDecoder(
                tokenizer, preserved_tokens = ("<think>", "</think>")
            )
            kwargs = {"skip_special_tokens": False}
        sizes = [1 if rng.random() < 0.9 else rng.randint(2, 5) for _ in ids]
        for skip_prompt in (True, False):
            options = dict(skip_prompt = skip_prompt, sizes = sizes, generations = 2, **kwargs)
            expected = _drive(_Upstream, make(), ids, **options).chunks
            streamer = _drive(_Incremental, make(), ids, **options)
            assert streamer.chunks == expected, (name, skip_prompt)
            if name.endswith("-one-line") and len(ids) > 2 * incremental_streamer._MIN_LINE_IDS:
                assert streamer._decoder, name


def test_work_per_token_stays_bounded():
    tokenizer = _byte_level(cleanup = True)
    ids = tokenizer.encode(" ".join(_CORPUS[:1]) * 120, add_special_tokens = False)
    assert len(ids) > 2000 and "\n" not in tokenizer.decode(ids)
    counter = _counting(tokenizer)
    streamer = _Incremental(tokenizer, skip_prompt = False, skip_special_tokens = True)
    for token_id in ids:
        streamer.put(torch.tensor([token_id]))
    # Upstream re-decodes the whole line every token: about len(ids) ** 2 / 2 ids.
    assert counter.decoded <= len(ids) * 32


def test_decoder_resolved_once_and_only_for_long_lines(monkeypatch):
    tokenizer = _byte_level(cleanup = False)
    calls = []
    real = incremental_streamer._line_decoder
    monkeypatch.setattr(
        incremental_streamer,
        "_line_decoder",
        lambda *args: calls.append(1) or real(*args),
    )
    word = tokenizer.encode(" word", add_special_tokens = False)
    newline = tokenizer.encode("\n", add_special_tokens = False)
    short_line = word * 8 + newline
    long_line = word * (2 * incremental_streamer._MIN_LINE_IDS) + newline
    streamer = _Incremental(tokenizer, skip_prompt = False)
    for token_id in short_line * 20:
        streamer.put(torch.tensor([token_id]))
    assert calls == []
    for token_id in long_line * 20:
        streamer.put(torch.tensor([token_id]))
    streamer.end()
    assert calls == [1]
    upstream = _Upstream(tokenizer, skip_prompt = False)
    for token_id in short_line * 20 + long_line * 20:
        upstream.put(torch.tensor([token_id]))
    upstream.end()
    assert streamer.chunks == upstream.chunks
