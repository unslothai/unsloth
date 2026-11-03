# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import gc
import os
import weakref

from core.rag import embeddings
from core.rag.chunking import chunk_pages
from core.rag.parsers import Page


class _Tok:
    def __init__(self, per_word = 1):
        self.per_word = per_word
        self.calls = []

    def encode(
        self,
        text,
        add_special_tokens = False,
    ):
        self.calls.append(text)
        return [0] * (self.per_word * len(text.split()) + len(text) % 3)


class _Model:
    def __init__(self, tok):
        self.tokenizer = tok


def test_a_repeated_text_is_tokenized_once(monkeypatch):
    tok = _Tok()
    monkeypatch.setattr(embeddings, "_get", lambda model_name = None: _Model(tok))
    count = embeddings._st_token_counter("org/embedder")
    assert [count("a b c"), count("a b c"), count("d e"), count("a b c")] == [5, 5, 2, 5]
    assert tok.calls == ["a b c", "d e"]


def test_switching_the_model_between_calls_recounts(monkeypatch):
    """model_name=None follows the configured model on every call, so a count memoized
    for the old tokenizer must not answer for the new one."""
    old, new = _Tok(1), _Tok(5)
    models = {"current": _Model(old)}
    looked_up = []

    def _get(model_name = None):
        looked_up.append(model_name)
        return models["current"]

    monkeypatch.setattr(embeddings, "_get", _get)
    count = embeddings._st_token_counter(None)
    assert count("one two three") == 3 + len("one two three") % 3
    models["current"] = _Model(new)
    assert count("one two three") == 15 + len("one two three") % 3
    models["current"] = _Model(old)
    assert count("one two three") == 3 + len("one two three") % 3
    assert len(old.calls) == 2 and len(new.calls) == 1
    assert looked_up == [None, None, None]


def test_the_memo_does_not_keep_an_unloaded_tokenizer_alive(monkeypatch):
    holder = {"model": _Model(_Tok())}
    monkeypatch.setattr(embeddings, "_get", lambda model_name = None: holder["model"])
    count = embeddings._st_token_counter("org/embedder")
    count("alpha beta")
    gone = weakref.ref(holder["model"].tokenizer)
    holder["model"] = _Model(_Tok())
    gc.collect()
    assert gone() is None
    assert count("alpha beta") == 2 + len("alpha beta") % 3


def test_a_memoized_count_leaves_the_same_environment(monkeypatch):
    tok = _Tok()
    monkeypatch.setattr(embeddings, "_get", lambda model_name = None: _Model(tok))
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    count = embeddings._st_token_counter("org/embedder")
    count("x y")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    assert count("x y") == 2 + len("x y") % 3
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    assert len(tok.calls) == 1


def test_a_tokenizer_without_weakref_support_is_counted_every_time(monkeypatch):
    class _SlotTok:
        __slots__ = ("calls",)

        def __init__(self):
            self.calls = 0

        def encode(
            self,
            text,
            add_special_tokens = False,
        ):
            self.calls += 1
            return [0] * len(text.split())

    tok = _SlotTok()
    monkeypatch.setattr(embeddings, "_get", lambda model_name = None: _Model(tok))
    count = embeddings._st_token_counter("org/embedder")
    assert [count("a b"), count("a b")] == [2, 2]
    assert tok.calls == 2


def test_chunks_are_unchanged_and_repeats_are_not_retokenized(monkeypatch):
    """Text with no spaces (as in CJK) is split per character, repeating the same pieces."""

    class _CharTok(_Tok):
        def encode(
            self,
            text,
            add_special_tokens = False,
        ):
            self.calls.append(text)
            return [0] * len(text.strip())

    tok = _CharTok()
    monkeypatch.setattr(embeddings, "_get", lambda model_name = None: _Model(tok))
    text = ("".join(chr(0x4E00 + (i * 7) % 40) for i in range(300)) + "\n") * 3
    pages = [Page(text = text, page_number = 1, char_count = len(text))]
    expected = chunk_pages(
        pages, max_tokens = 48, overlap = 8, count = lambda t: len(_CharTok().encode(t))
    )
    got = chunk_pages(pages, max_tokens = 48, overlap = 8, count = embeddings._st_token_counter("m"))
    assert got == expected
    assert len(got) > 10
    assert len(tok.calls) == len(set(tok.calls)) < 200
