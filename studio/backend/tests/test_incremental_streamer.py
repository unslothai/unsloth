# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""IncrementalTextIteratorStreamer emits exactly transformers 5.24's TextIteratorStreamer chunks on
every installed version while decoding only the ids since the last split. In-memory tokenizers, plus
real tokenizer files (no weights) for the indent check, which skips offline."""

import random
import sys
import unicodedata
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

import transformers
from packaging.version import Version
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


class _Reference(_Recording, TextIteratorStreamer):
    """transformers 5.24's ``TextStreamer.put``, whole-line decode per token, for older installs."""

    def put(self, value):
        if len(value.shape) > 1:
            value = value[0]
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return
        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = self.token_cache[-1:]
            self.print_len = len(self.tokenizer.decode(self.token_cache[-1], **self.decode_kwargs))
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        elif all(
            char.isspace() or unicodedata.category(char)[0] in "LMN"
            for char in text[self.print_len :]
        ):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)
        self.on_finalized_text(printable_text)


_UPSTREAM_HAS_524_PUT = Version(transformers.__version__).release >= (5, 24)


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
            expected = _drive(_Reference, make(), ids, **options).chunks
            if _UPSTREAM_HAS_524_PUT:
                assert _drive(_Upstream, make(), ids, **options).chunks == expected, name
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
    upstream = _Reference(tokenizer, skip_prompt = False)
    for token_id in short_line * 20 + long_line * 20:
        upstream.put(torch.tensor([token_id]))
    upstream.end()
    assert streamer.chunks == upstream.chunks


# Llama / Mistral decode with Strip(" ", 1, 0), Gemma with a bare Replace, Qwen and Llama 3 ByteLevel.
_REAL_TOKENIZERS = [
    "unsloth/llama-2-7b-chat",
    "unsloth/mistral-7b-instruct-v0.3",
    "unsloth/gemma-2-2b-it",
    "unsloth/Qwen2.5-0.5B-Instruct",
    "unsloth/Llama-3.2-1B-Instruct",
]
_LONG = " ".join(f"item_{i}" for i in range(48))
_INDENTED = (
    "Here is code:\n```python\ndef f(x):\n    if x:\n        return [i for i in range(x)]\n"
    f"    values = [{_LONG}]\n        nested = 1\n\n\tdone = True\n```\nDone."
)


def _tokenizer_is_whole(tokenizer):
    """A cache caught mid-download (config.json and tokenizer_config.json, no tokenizer.json /
    tokenizer.model yet) still loads with local_files_only, as a vocabulary of only the added
    tokens: every word becomes <unk>, so Gemma decodes to bare newlines and Llama 2 to ''."""
    probe = "def f(x):\n    if x:"
    ids = tokenizer.encode(probe, add_special_tokens = False)
    return tokenizer.decode(ids, skip_special_tokens = True).strip() == probe.strip()


@pytest.fixture(scope = "module", params = _REAL_TOKENIZERS)
def real_tokenizer(request, allow_outbound_network, tmp_path_factory):
    from filelock import FileLock
    from transformers import AutoTokenizer

    repo = request.param
    # One lock per repo for every xdist worker (they share the Hub cache): a worker reading the
    # cache while another is still downloading into it gets the partial vocabulary above.
    lock = tmp_path_factory.getbasetemp().parent / f"tokenizer-{repo.replace('/', '--')}.lock"
    with FileLock(str(lock)):
        try:
            tokenizer = AutoTokenizer.from_pretrained(repo, local_files_only = True)
            if _tokenizer_is_whole(tokenizer):
                return tokenizer
        except Exception:  # noqa: BLE001 -- not cached yet
            pass
        try:
            with allow_outbound_network():
                tokenizer = AutoTokenizer.from_pretrained(repo)
        except Exception as e:  # noqa: BLE001 -- offline or hub unavailable
            # Skips when this machine has no Hub access: the tokenizer files come from the Hub.
            pytest.skip(reason = f"tokenizer {repo} not downloadable (no Hub access): {e}")
    assert _tokenizer_is_whole(
        tokenizer
    ), f"{repo}: the downloaded tokenizer does not round-trip plain text"
    return tokenizer


def test_partial_tokenizer_cache_is_not_taken_for_a_tokenizer():
    """The vocabulary a half-downloaded cache yields: only special tokens, so text decodes to ''."""

    class _AddedTokensOnly:
        def encode(
            self,
            text,
            add_special_tokens = False,
        ):
            return [0 if ch != "\n" else 1 for ch in text]

        def decode(
            self,
            ids,
            skip_special_tokens = True,
        ):
            return "".join("\n" if i == 1 else "" for i in ids)

    class _Whole:
        def encode(
            self,
            text,
            add_special_tokens = False,
        ):
            return [ord(ch) for ch in text]

        def decode(
            self,
            ids,
            skip_special_tokens = True,
        ):
            return "".join(map(chr, ids))

    assert not _tokenizer_is_whole(_AddedTokensOnly())
    assert _tokenizer_is_whole(_Whole())


@pytest.mark.parametrize("mode", ["plain", "native-tool"])
def test_streamed_text_keeps_indentation(real_tokenizer, mode):
    """Every chunk boundary: printed text is a prefix of decoding all ids so far, and at the
    end equals it, so no line loses the indent space SentencePiece strips at decode start."""
    source = real_tokenizer if mode == "plain" else NativeToolTokenDecoder(real_tokenizer)
    ids = real_tokenizer.encode(_INDENTED, add_special_tokens = False)
    full = source.decode(ids, skip_special_tokens = True)
    assert "\n    if x:" in full and "\n        nested" in full
    # The values line is long enough for the incremental decoder to attach.
    assert (
        max(
            len(real_tokenizer.encode(line, add_special_tokens = False))
            for line in _INDENTED.split("\n")
        )
        > incremental_streamer._MIN_LINE_IDS
    )
    rng = random.Random(0)
    for _ in range(3):
        streamer = _Incremental(source, skip_prompt = False, skip_special_tokens = True)
        index = 0
        while index < len(ids):
            size = 1 if rng.random() < 0.8 else rng.randint(2, 6)
            streamer.put(torch.tensor(ids[index : index + size]))
            index += size
            printed = "".join(text for text, _ in streamer.chunks)
            assert source.decode(ids[:index], skip_special_tokens = True).startswith(printed)
        streamer.end()
        assert "".join(text for text, _ in streamer.chunks) == full
