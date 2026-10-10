# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import copy
import glob
import os
import pickle

import pytest
from tokenizers import (
    Regex,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from transformers import PreTrainedTokenizerFast

import unsloth.tokenizers_v1 as tv1

RC_DIR = os.path.expanduser(os.environ.get(tv1._PATH_ENV) or tv1._DEFAULT_DIR)
HAS_RC = bool(glob.glob(os.path.join(RC_DIR, "tokenizers", "tokenizers*")))
needs_rc = pytest.mark.skipif(
    not HAS_RC, reason = f"tokenizers release candidate not installed in {RC_DIR}"
)

CORPUS = [
    "Hello world! This is a tiny corpus for a test tokenizer.",
    "def f(x):\n    return x ** 2  # code",
    "The quick brown fox jumps over the lazy dog. 😁 café 中文",
    "Numbers 0 12 345 6789 and   spaces\tand\nnewlines",
] * 25
TEXTS = [
    "Hello world!",
    "  spaced  out  ",
    "<s> special inside </s> text",
    "Unseen 🚀 emoji and Zürich",
    "",
    "def g(y):\n\treturn y",
]


# Qwen2's pretokenizer; the RC implements real models' patterns, not arbitrary regex.
QWEN2_PATTERN = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def _byte_level(pattern = QWEN2_PATTERN):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(pattern), behavior = "isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space = False, use_regex = False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size = 400,
        special_tokens = ["<unk>", "<s>", "</s>", "<pad>"],
        initial_alphabet = pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(CORPUS, trainer = trainer)
    tok.post_processor = processors.TemplateProcessing(
        single = "<s> $A", pair = "<s> $A </s> $B", special_tokens = [("<s>", 1), ("</s>", 2)]
    )
    return tok


def _unigram():
    tok = Tokenizer(models.Unigram())
    tok.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Replace(" ", "▁")])
    tok.pre_tokenizer = pre_tokenizers.Metaspace()
    tok.decoder = decoders.Metaspace()
    trainer = trainers.UnigramTrainer(
        vocab_size = 300, special_tokens = ["<unk>", "<s>", "</s>", "<pad>"], unk_token = "<unk>"
    )
    tok.train_from_iterator(CORPUS, trainer = trainer)
    return tok


def _hf(backend):
    return PreTrainedTokenizerFast(
        tokenizer_object = backend,
        bos_token = "<s>",
        eos_token = "</s>",
        unk_token = "<unk>",
        pad_token = "<pad>",
    )


@pytest.fixture
def rc_on(monkeypatch):
    monkeypatch.setenv(tv1._ENV, "1")
    monkeypatch.setenv(tv1._VERIFY_ENV, "1")
    monkeypatch.setattr(tv1, "_v1_tried", False)
    monkeypatch.setattr(tv1, "_v1_module", None)
    monkeypatch.setattr(tv1, "_snapshots", {})


def _pair(build):
    # Unigram training is not deterministic: clone one trained backend instead of training twice.
    backend = build()
    plain = _hf(Tokenizer.from_str(backend.to_str()))
    fast = tv1.enable_tokenizers_v1(_hf(Tokenizer.from_str(backend.to_str())))
    assert isinstance(fast._tokenizer, tv1.TokenizersV1Backend)
    return plain, fast


def test_off_by_default_changes_nothing(monkeypatch):
    monkeypatch.delenv(tv1._ENV, raising = False)
    tok = _hf(_byte_level())
    backend = tok._tokenizer
    assert tv1.enable_tokenizers_v1(tok) is tok
    assert tok._tokenizer is backend


def test_missing_rc_keeps_0x(monkeypatch, tmp_path):
    monkeypatch.setenv(tv1._ENV, "1")
    monkeypatch.setenv(tv1._PATH_ENV, str(tmp_path))
    monkeypatch.setattr(tv1, "_v1_tried", False)
    monkeypatch.setattr(tv1, "_v1_module", None)
    tok = _hf(_byte_level())
    backend = tok._tokenizer
    tv1.enable_tokenizers_v1(tok)
    assert tok._tokenizer is backend


@needs_rc
@pytest.mark.parametrize("build", [_byte_level, _unigram], ids = ["bytelevel_bpe", "unigram"])
def test_plain_calls_take_rc_and_match(rc_on, build):
    plain, fast = _pair(build)
    for kwargs in ({}, {"add_special_tokens": False}, {"split_special_tokens": True}):
        want, got = plain(TEXTS, **kwargs), fast(TEXTS, **kwargs)
        assert got["input_ids"] == want["input_ids"]
        assert got["attention_mask"] == want["attention_mask"]
        assert all(isinstance(e, tv1._CompatEncoding) for e in got.encodings)
    assert fast(TEXTS[0])["input_ids"] == plain(TEXTS[0])["input_ids"]
    assert fast.encode(TEXTS[3]) == plain.encode(TEXTS[3])
    assert fast(TEXTS, return_tensors = "pt", padding = True)["input_ids"].tolist() == (
        plain(TEXTS, return_tensors = "pt", padding = True)["input_ids"].tolist()
    )


@needs_rc
@pytest.mark.parametrize("side", ["right", "left"])
def test_padding_and_truncation_take_rc_and_match(rc_on, side):
    plain, fast = _pair(_byte_level)
    for tok in (plain, fast):
        tok.padding_side = tok.truncation_side = side
    cases = [
        {"padding": True},
        {"padding": "max_length", "max_length": 24, "pad_to_multiple_of": 8},
        {"truncation": True, "max_length": 4},
        # unsloth_zoo's dataset tokenizer call
        {"truncation": True, "max_length": 6, "add_special_tokens": False},
        {"padding": True, "truncation": True, "max_length": 5, "return_tensors": "pt"},
    ]
    for kwargs in cases:
        want, got = plain(TEXTS, **kwargs), fast(TEXTS, **kwargs)
        assert {k: v.tolist() if hasattr(v, "tolist") else v for k, v in got.items()} == (
            {k: v.tolist() if hasattr(v, "tolist") else v for k, v in want.items()}
        )
        assert all(isinstance(e, tv1._CompatEncoding) for e in got.encodings)


@needs_rc
def test_max_length_below_special_tokens_matches_0x(rc_on):
    # 0.x leaves such a row untruncated; the RC would keep only the special tokens.
    plain, fast = _pair(_byte_level)
    for kwargs in (
        {"truncation": True, "max_length": 0},
        {"truncation": True, "max_length": 0, "add_special_tokens": False},
    ):
        assert dict(fast(TEXTS, **kwargs)) == dict(plain(TEXTS, **kwargs))


@needs_rc
def test_overflow_and_fields_the_rc_lacks_come_from_0x(rc_on):
    plain, fast = _pair(_byte_level)
    for kwargs in (
        {"return_offsets_mapping": True, "return_special_tokens_mask": True},
        {"padding": True, "return_offsets_mapping": True, "return_special_tokens_mask": True},
        {"truncation": True, "max_length": 4, "stride": 1, "return_overflowing_tokens": True},
    ):
        want, got = plain(TEXTS, **kwargs), fast(TEXTS, **kwargs)
        assert dict(got) == dict(want)
        for i in range(len(want["input_ids"])):
            assert got.word_ids(i) == want.word_ids(i)
            assert got.tokens(i) == want.tokens(i)


@needs_rc
def test_pairs_and_presplit_words_fall_back_to_0x(rc_on):
    plain, fast = _pair(_byte_level)
    got = fast(TEXTS[:2], TEXTS[2:4])
    assert dict(got) == dict(plain(TEXTS[:2], TEXTS[2:4]))
    assert not any(isinstance(e, tv1._CompatEncoding) for e in got.encodings)
    words = [["Hello", "world"], ["a", "b", "c"]]
    got = fast(words, is_split_into_words = True)
    assert dict(got) == dict(plain(words, is_split_into_words = True))
    assert not any(isinstance(e, tv1._CompatEncoding) for e in got.encodings)


@needs_rc
def test_added_tokens_rebuild_the_rc_copy(rc_on):
    plain, fast = _pair(_byte_level)
    fast("warm up")
    for tok in (plain, fast):
        tok.add_tokens(["<new_token>"])
        tok.add_special_tokens({"additional_special_tokens": ["<|special|>"]})
    text = ["a <new_token> b <|special|> c"]
    got = fast(text)
    assert got["input_ids"] == plain(text)["input_ids"]
    assert isinstance(got.encodings[0], tv1._CompatEncoding)


@needs_rc
def test_probe_mismatch_keeps_0x(rc_on, monkeypatch):
    plain, fast = _pair(_byte_level)

    def _refuse(base, text):
        raise ValueError("forced mismatch")

    monkeypatch.setattr(tv1, "_build_snapshot", _refuse)
    got = fast(TEXTS)
    assert got["input_ids"] == plain(TEXTS)["input_ids"]
    assert not any(isinstance(e, tv1._CompatEncoding) for e in got.encodings)
    assert fast._tokenizer._unsloth_state.disabled


@needs_rc
def test_tokenizer_the_rc_cannot_load_stays_on_0x(rc_on):
    build = lambda: _byte_level(r"\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+|\s+")
    plain, fast = _pair(build)
    got = fast(TEXTS)
    assert got["input_ids"] == plain(TEXTS)["input_ids"]
    assert not any(isinstance(e, tv1._CompatEncoding) for e in got.encodings)
    assert fast._tokenizer._unsloth_state.disabled


@needs_rc
def test_copy_pickle_save_and_idempotence(rc_on, tmp_path):
    plain, fast = _pair(_byte_level)
    assert tv1.enable_tokenizers_v1(fast)._tokenizer is fast._tokenizer
    for clone in (copy.deepcopy(fast), pickle.loads(pickle.dumps(fast))):
        assert isinstance(clone._tokenizer, tv1.TokenizersV1Backend)
        assert clone(TEXTS)["input_ids"] == plain(TEXTS)["input_ids"]
    encodings = fast(TEXTS).encodings
    assert [e.ids for e in pickle.loads(pickle.dumps(encodings))] == [
        e.ids for e in plain(TEXTS).encodings
    ]
    fast.save_pretrained(tmp_path / "fast")
    plain.save_pretrained(tmp_path / "plain")
    assert (tmp_path / "fast" / "tokenizer.json").read_text() == (
        tmp_path / "plain" / "tokenizer.json"
    ).read_text()


@needs_rc
def test_positional_arguments_follow_0x(rc_on):
    plain, fast = _pair(_byte_level)
    for args in ((False,), (False, False)):
        want = plain._tokenizer.encode_batch(TEXTS, *args)
        got = fast._tokenizer.encode_batch(TEXTS, *args)
        assert [e.ids for e in got] == [e.ids for e in want]
    assert (
        fast._tokenizer.encode(TEXTS[0], None, False, False).ids
        == plain._tokenizer.encode(TEXTS[0], None, False, False).ids
    )
    fast_out = fast._tokenizer.encode_batch_fast(TEXTS)
    assert not any(isinstance(e, tv1._CompatEncoding) for e in fast_out)


@needs_rc
def test_encoding_mutation_and_padded_overflow_match_0x(rc_on):
    plain, fast = _pair(_byte_level)
    want, got = plain._tokenizer.encode(TEXTS[3]), fast._tokenizer.encode(TEXTS[3])
    assert isinstance(got, tv1._CompatEncoding)
    for e in (want, got):
        e.pad(32, pad_id = 3, pad_token = "<pad>")
    assert (got.ids, got.attention_mask, len(got)) == (want.ids, want.attention_mask, len(want))
    kwargs = {
        "padding": True,
        "truncation": True,
        "max_length": 6,
        "stride": 1,
        "return_overflowing_tokens": True,
    }
    for kw in (kwargs, dict(kwargs, padding = "max_length", pad_to_multiple_of = 3)):
        assert dict(fast(TEXTS, **kw)) == dict(plain(TEXTS, **kw))


@needs_rc
def test_mutations_outside_the_wrapper_are_seen(rc_on):
    plain, fast = _pair(_byte_level)
    raw = fast._tokenizer._unsloth_base
    fast(TEXTS)
    new = processors.TemplateProcessing(single = "$A </s>", special_tokens = [("</s>", 2)])
    raw.post_processor = new
    plain._tokenizer.post_processor = new
    got = fast(TEXTS)
    assert got["input_ids"] == plain(TEXTS)["input_ids"]
    assert all(isinstance(e, tv1._CompatEncoding) for e in got.encodings)


@needs_rc
def test_custom_component_and_rc_failure_fall_back(rc_on):
    plain, fast = _pair(_byte_level)
    fast(TEXTS)
    snapshot = fast._tokenizer._unsloth_state.snapshot

    class _Raises:
        def encode_batch(self, *args, **kwargs):
            raise RuntimeError("rc failure")

    snapshot_v1, snapshot.v1 = snapshot.v1, _Raises()
    try:
        got = fast(TEXTS)
    finally:
        snapshot.v1 = snapshot_v1
    assert got["input_ids"] == plain(TEXTS)["input_ids"]
    assert not any(isinstance(e, tv1._CompatEncoding) for e in got.encodings)

    class _Split:
        def pre_tokenize(self, pretok):
            pretok.split(lambda i, s: [s])

    custom = _hf(_byte_level())
    custom._tokenizer.pre_tokenizer = pre_tokenizers.PreTokenizer.custom(_Split())
    reference = custom(TEXTS)["input_ids"]
    tv1.enable_tokenizers_v1(custom)
    assert custom(TEXTS)["input_ids"] == reference
    assert custom._tokenizer._unsloth_state.disabled


@needs_rc
@pytest.mark.skipif(not hasattr(os, "fork"), reason = "os.fork is POSIX only")
def test_fork_while_another_thread_holds_a_lock(rc_on, monkeypatch):
    import threading
    import time

    # Verify mode would materialize in the parent, so the child would never touch the lock.
    monkeypatch.delenv(tv1._VERIFY_ENV)
    plain, fast = _pair(_byte_level)
    got = fast(TEXTS)
    want_offsets = plain(TEXTS).encodings[0].offsets
    snapshot = fast._tokenizer._unsloth_state.snapshot
    held, release = threading.Event(), threading.Event()

    def hold():
        with snapshot.lock:
            held.set()
            release.wait()

    thread = threading.Thread(target = hold)
    thread.start()
    held.wait()
    pid = os.fork()
    if pid == 0:
        os._exit(0 if got.encodings[0].offsets == want_offsets else 1)
    release.set()
    thread.join()
    deadline = time.time() + 60
    while True:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            break
        if time.time() > deadline:
            os.kill(pid, 9)
            pytest.fail("child hung on a lock inherited from another thread")
        time.sleep(0.1)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


CHAT_TEMPLATE = (
    "{% for m in messages %}{{ bos_token }}[{{ m['role'] }}] {{ m['content'] }}{{ eos_token }}{% endfor %}"
    "{% if add_generation_prompt %}[assistant] {% endif %}"
)
CONVERSATION = [
    {"role": "user", "content": "Hello, y'all! 😁 What is 12*7?"},
    {"role": "assistant", "content": "It is 84."},
]


@needs_rc
def test_tokenizer_api_surface_unchanged(rc_on):
    plain, fast = _pair(_byte_level)
    for tok in (plain, fast):
        tok.chat_template = CHAT_TEMPLATE
    assert type(fast) is type(plain)
    assert set(dir(fast._tokenizer)) == set(dir(plain._tokenizer))
    for name in (a for a in dir(plain) if not a.startswith("_")):
        value = getattr(plain, name)
        if not callable(value) and name != "backend_tokenizer":
            other = getattr(fast, name)
            assert other == value or repr(other) == repr(value), name
    cases = [
        {"tokenize": False},
        {"tokenize": True},
        {"tokenize": True, "return_dict": True},
        {"tokenize": True, "return_dict": True, "return_tensors": "pt"},
        {"tokenize": True, "add_generation_prompt": True},
        {"tokenize": True, "truncation": True, "max_length": 6},
    ]
    for kwargs in cases:
        want, got = (
            plain.apply_chat_template(CONVERSATION, **kwargs),
            fast.apply_chat_template(CONVERSATION, **kwargs),
        )
        norm = (
            lambda x: {k: v.tolist() if hasattr(v, "tolist") else v for k, v in x.items()}
            if hasattr(x, "keys")
            else x
        )
        assert norm(got) == norm(want), kwargs
    batch = [CONVERSATION, CONVERSATION[:1]]
    want = plain.apply_chat_template(batch, tokenize = True, padding = True, return_dict = True)
    got = fast.apply_chat_template(batch, tokenize = True, padding = True, return_dict = True)
    assert dict(got) == dict(want)
    assert all(isinstance(e, tv1._CompatEncoding) for e in got.encodings)


def _map(
    ds,
    tok,
    mutate = False,
):
    def fn(rows):
        if mutate:
            tok.add_tokens(["<worker_token>"])
        out = tok(rows["text"])
        return {
            "input_ids": out["input_ids"],
            "rc": [isinstance(e, tv1._CompatEncoding) for e in out.encodings],
        }

    return ds.map(fn, batched = True, batch_size = 10, num_proc = 2, load_from_cache_file = False)


@needs_rc
def test_dataset_map_multiprocess(rc_on):
    # Building an RC tokenizer in a child forked after a build deadlocks; workers must reuse.
    datasets = pytest.importorskip("datasets")
    ds = datasets.Dataset.from_dict({"text": CORPUS})
    plain, used = _pair(_byte_level)
    used("encoded in the parent before the fork")
    plain_u, unused = _pair(_unigram)
    for tok, ref in ((used, plain), (unused, plain_u)):
        got = _map(ds, tok)
        assert got["input_ids"] == _map(ds, ref)["input_ids"]
        assert all(got["rc"])
    got = _map(ds, used, mutate = True)
    plain.add_tokens(["<worker_token>"])
    assert got["input_ids"] == _map(ds, plain)["input_ids"]
    import multiprocess

    if multiprocess.get_start_method() == "fork":
        # Forked after a build: the worker cannot rebuild, so it stays on 0.x. Spawned workers rebuild.
        assert not any(got["rc"])


def test_decorator_only_touches_model_tokenizer_pairs(monkeypatch):
    monkeypatch.delenv(tv1._ENV, raising = False)
    sentinel = object()
    assert tv1.tokenizers_v1_on_return(lambda: sentinel)() is sentinel
    tok = _hf(_byte_level())
    backend = tok._tokenizer
    model, out = tv1.tokenizers_v1_on_return(lambda: ("model", tok))()
    assert out is tok and tok._tokenizer is backend
