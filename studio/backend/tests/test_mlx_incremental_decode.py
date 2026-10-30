# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The MLX reply snapshots decode only the ids since the last safe split, and still equal a full
re-decode at every step. Built on small in-memory tokenizers, so no MLX and no download."""

import random
import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from core.inference.incremental_decode import IncrementalDecode, split_profile
from core.inference.native_tool_tokens import NativeToolTokenDecoder

_CORPUS = [
    "The quick brown fox isn't here. It's gone , they 've said ! Don 't ' quote ' me ? Yes .",
    "def f(x):\n    return {'a': [1, 2, 3]}  # comment\n\n\tif x  >  0 :\n\t\tpass\r\n",
    "😀🎉👩‍👩‍👧‍👦 🇯🇵 🏳️‍🌈 café naïve résumé 𝔘𝔫𝔦𝔠𝔬𝔡𝔢",
    "日本語のテキストと中文混合，한국어 문장도。龘靐齉",
    "a    b\t\t\tc\n\n\n   \n  d     e" + " " * 40 + "f",
    '<think>\nreasoning</think>\n\nAnswer: 42 <tool_call>{"name": "f"}</tool_call>',
]
_ADDED = ["<think>", "</think>", "<tool_call>", "</tool_call>"]


def _fast(backend, cleanup, **specials):
    fast = PreTrainedTokenizerFast(
        tokenizer_object = backend,
        clean_up_tokenization_spaces = cleanup,
        **specials,
    )
    fast.add_tokens(_ADDED)
    return fast


def _byte_level(cleanup = False):
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size = 700,
        initial_alphabet = pre_tokenizers.ByteLevel.alphabet(),
        special_tokens = ["<|eot|>"],
    )
    backend.train_from_iterator(_CORPUS * 4, trainer)
    return _fast(backend, cleanup, eos_token = "<|eot|>")


def _sentencepiece(decoder, cleanup = False):
    """A Llama-style piece vocab with <0xNN> byte fallback, so pieces outside it arrive as bytes."""
    pieces = ["<unk>", "<s>", "</s>"] + [f"<0x{value:02X}>" for value in range(256)]
    pieces += sorted({ch for line in _CORPUS[:2] for ch in line.replace(" ", "▁")})
    pieces += ["▁", "▁the", "▁a", "n't", "'s", "▁.", "▁,", "▁'", "▁n", "ve", "re", "▁▁", "日本"]
    vocab = {piece: index for index, piece in enumerate(dict.fromkeys(pieces))}
    backend = Tokenizer(models.BPE(vocab = vocab, merges = [], unk_token = "<unk>", byte_fallback = True))
    backend.normalizer = normalizers.Sequence(
        [normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")]
    )
    backend.decoder = decoder
    return _fast(backend, cleanup, bos_token = "<s>", eos_token = "</s>", unk_token = "<unk>")


def _llama_spm_decoder():
    return decoders.Sequence(
        [
            decoders.Replace("▁", " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
            decoders.Strip(" ", 1, 0),
        ]
    )


def _gemma_spm_decoder():
    return decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse()])


_TOKENIZERS = {
    "bytelevel": lambda: _byte_level(),
    "bytelevel-cleanup": lambda: _byte_level(cleanup = True),
    "llama-spm": lambda: _sentencepiece(_llama_spm_decoder()),
    "llama-spm-cleanup": lambda: _sentencepiece(_llama_spm_decoder(), cleanup = True),
    "gemma-spm": lambda: _sentencepiece(_gemma_spm_decoder()),
    "metaspace": lambda: _sentencepiece(
        decoders.Metaspace(replacement = "▁", prepend_scheme = "always")
    ),
}
_DECODE_KWARGS = [
    {"skip_special_tokens": True},
    {"skip_special_tokens": True, "clean_up_tokenization_spaces": False},
    {"skip_special_tokens": True, "clean_up_tokenization_spaces": True},
    {"skip_special_tokens": False, "spaces_between_special_tokens": False},
]


@pytest.fixture(scope = "module", params = sorted(_TOKENIZERS))
def tokenizer(request):
    return _TOKENIZERS[request.param]()


class _Wrapper:
    """mlx-lm's TokenizerWrapper forwards decode to the Hugging Face tokenizer."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


def _streams(tokenizer, seed = 0):
    rng = random.Random(seed)
    size = len(tokenizer)
    text = "\n".join(_CORPUS)
    encoded = tokenizer.encode(text * 3, add_special_tokens = False)
    special = sorted(tokenizer.added_tokens_decoder)
    byte_ids = [
        i for i in range(size) if (tokenizer.convert_ids_to_tokens(i) or "").startswith("<0x")
    ]
    tricky = tokenizer.encode(" n't 's 've 're ' . , ? ! \n 😀 日本", add_special_tokens = False)
    pool = byte_ids + special + tricky or list(range(size))
    streams = {
        "text": encoded,
        "text-from-mid-character": encoded[1:],
        "emoji": tokenizer.encode(_CORPUS[2] * 6, add_special_tokens = False),
    }
    for index in range(4):
        # Ids past the vocabulary too: a padded embedding can sample one, and decode drops it.
        streams[f"random-{index}"] = [rng.randrange(size + 8) for _ in range(300)]
        streams[f"adversarial-{index}"] = [
            rng.choice(pool) if rng.random() < 0.6 else rng.randrange(size) for _ in range(300)
        ]
    return streams


@pytest.mark.parametrize(
    "decode_kwargs", _DECODE_KWARGS, ids = lambda kw: ",".join(f"{k}={v}" for k, v in kw.items())
)
def test_every_snapshot_equals_a_full_redecode(tokenizer, decode_kwargs):
    assert split_profile(tokenizer) is not None
    for name, ids in _streams(tokenizer).items():
        for source in (tokenizer, _Wrapper(tokenizer)):
            incremental = IncrementalDecode(source, decode_kwargs)
            for count in range(1, len(ids) + 1):
                expected = tokenizer.decode(ids[:count], **decode_kwargs)
                assert incremental.feed(ids[count - 1]) == expected, (name, count)
            # The split path stayed on: a fallback to full decodes would also match.
            assert incremental._profile is not None, name


def test_native_decoder_snapshots_equal_its_full_decode(tokenizer):
    decoder = NativeToolTokenDecoder(tokenizer, preserved_tokens = {"<think>", "</think>"})
    for name, ids in _streams(tokenizer, seed = 1).items():
        incremental = decoder.incremental()
        for count in range(1, len(ids) + 1):
            assert incremental.feed(ids[count - 1]) == decoder.decode(ids[:count]), (name, count)


class _CountingBackend:
    """Counts the ids each Rust decode receives, the cost a full re-decode pays per token."""

    def __init__(self, inner):
        self._inner = inner
        self.decoded = 0

    def decode(self, ids, *args, **kwargs):
        self.decoded += len(ids)
        return self._inner.decode(ids, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _counting(tokenizer):
    counter = _CountingBackend(tokenizer._tokenizer)
    tokenizer._tokenizer = counter
    return counter


def test_work_per_token_stays_bounded():
    tokenizer = _byte_level(cleanup = True)
    counter = _counting(tokenizer)
    ids = tokenizer.encode("\n".join(_CORPUS[:2]) * 60, add_special_tokens = False)
    incremental = IncrementalDecode(tokenizer, {"skip_special_tokens": True})
    for token_id in ids:
        incremental.feed(token_id)
    counter.decoded = 0
    for token_id in ids[:200]:
        incremental.feed(token_id)
    # A full re-decode would pass every id so far, len(ids) + 100 on average, each step.
    assert len(ids) > 2000
    assert counter.decoded <= 200 * 32


@pytest.mark.parametrize(
    "backend",
    [
        decoders.WordPiece(),
        decoders.Sequence([decoders.Fuse(), decoders.Replace("ab", "x")]),
        decoders.Sequence([decoders.Fuse(), decoders.Strip(" ", 2, 0)]),
    ],
    ids = ["wordpiece", "multi-character-replace", "strip-two"],
)
def test_an_unknown_decoder_keeps_the_full_decode(backend):
    tokenizer = _sentencepiece(backend)
    assert split_profile(tokenizer) is None
    ids = _streams(tokenizer)["adversarial-0"]
    incremental = IncrementalDecode(tokenizer, {"skip_special_tokens": True})
    for count in range(1, len(ids) + 1):
        assert incremental.feed(ids[count - 1]) == tokenizer.decode(
            ids[:count], skip_special_tokens = True
        )


@pytest.mark.parametrize("scheme", ["always", "first"])
def test_a_metaspace_piece_ending_in_its_replacement_never_splits_a_cleanup_pattern(scheme):
    # Metaspace drops the replacement char of a run's first id only: "a▁" decodes alone
    # to "a" but to "a " mid-reply, so cleanup's " ." spans a split taken after it.
    vocab = {"<unk>": 0, "x": 1, "a▁": 2, ".": 3, "'s": 4, "▁b": 5}
    backend = Tokenizer(models.WordLevel(vocab, unk_token = "<unk>"))
    backend.decoder = decoders.Metaspace(prepend_scheme = scheme)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = backend, unk_token = "<unk>", clean_up_tokenization_spaces = True
    )
    assert split_profile(tokenizer) is not None
    for ids in ([1, 2, 2, 3, 1], [5, 2, 2, 4, 2, 3, 5], [1, 2, 3, 1, 2, 2, 3]):
        for kwargs in ({}, {"skip_special_tokens": True}, {"clean_up_tokenization_spaces": True}):
            incremental = IncrementalDecode(_Wrapper(tokenizer), kwargs)
            for count in range(1, len(ids) + 1):
                assert incremental.feed(ids[count - 1]) == tokenizer.decode(ids[:count], **kwargs)


def test_an_overridden_decode_keeps_the_full_decode():
    tokenizer = _byte_level()

    class _Override(type(tokenizer)):
        def _decode(self, token_ids, **kwargs):
            return super()._decode(token_ids, **kwargs).upper()

    tokenizer.__class__ = _Override
    assert split_profile(tokenizer) is None
    assert split_profile(SimpleNamespace(decode = lambda ids, **_k: "")) is None


# End to end through the MLX text path, with mlx_lm faked and a real tokenizer.


class _Resp:
    def __init__(self, text, token):
        self.text = text
        self.token = token


def _text_backend(monkeypatch, tokenizer, ids):
    from core.inference import mlx_inference

    mlx_lm = types.ModuleType("mlx_lm")
    sample_utils = types.ModuleType("mlx_lm.sample_utils")
    sample_utils.make_sampler = lambda **_k: object()
    mlx_lm.sample_utils = sample_utils
    mlx_lm.stream_generate = lambda *_a, **_k: iter([_Resp("", token) for token in ids])
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample_utils)
    monkeypatch.setattr(mlx_inference, "_mlx_sampling_processors", lambda **_k: None)

    @contextmanager
    def _adapter_state(_model, _state):
        yield

    monkeypatch.setattr(mlx_inference, "_temporary_mlx_adapter_state", _adapter_state)
    backend = mlx_inference.MLXInferenceBackend()
    backend._model = SimpleNamespace(config = {"model_type": "llama"})
    backend._tokenizer = _Wrapper(tokenizer)
    backend._served_context = 1 << 20
    monkeypatch.setattr(
        backend,
        "_render_text_prompt",
        lambda *_a, **_k: SimpleNamespace(prompt = "P", reasoning_channel_markers = None),
        raising = False,
    )
    monkeypatch.setattr(
        backend, "_prepare_prompt_cache", lambda _p, _a: ([1, 2], None, None, [1, 2], 0)
    )
    return backend


@pytest.mark.parametrize(
    "tools", [None, [{"type": "function", "function": {"name": "f"}}]], ids = ["plain", "tools"]
)
def test_the_mlx_text_stream_yields_full_redecode_snapshots_with_bounded_work(monkeypatch, tools):
    tokenizer = _byte_level(cleanup = True)
    ids = tokenizer.encode("\n".join(_CORPUS) * 30, add_special_tokens = False)
    expected = []
    if tools:
        decoder = NativeToolTokenDecoder(tokenizer)
        full = lambda count: decoder.decode(ids[:count])
    else:
        full = lambda count: tokenizer.decode(ids[:count], skip_special_tokens = True)
    for count in range(1, len(ids) + 1):
        expected.append(full(count).rstrip("\ufffd"))
    counter = _counting(tokenizer)
    backend = _text_backend(monkeypatch, tokenizer, ids)
    snapshots = list(
        backend._generate_text(
            [{"role": "user", "content": "hi"}], 0.0, 1.0, 0, 0.0, len(ids), 1.0, None, tools = tools
        )
    )
    assert snapshots[: len(expected)] == expected
    # Re-decoding every id each step would pass about len(ids) ** 2 / 2 ids.
    assert len(ids) > 1500
    assert counter.decoded <= 32 * len(ids)


def test_a_row_stream_snapshots_equal_a_full_redecode():
    from core.inference import mlx_inference

    tokenizer = _sentencepiece(_llama_spm_decoder(), cleanup = True)
    ids = _streams(tokenizer)["text"]
    row = mlx_inference._RowStream(
        decode = mlx_inference._ReplyText(tokenizer, {"skip_special_tokens": True})
    )
    for count in range(1, len(ids) + 1):
        expected = tokenizer.decode(ids[:count], skip_special_tokens = True).rstrip("\ufffd")
        assert row.feed(token = ids[count - 1]) == [expected]
