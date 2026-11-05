# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The llama-server token counter counts many texts per /tokenize POST, and auto-inject encodes its query once."""

import json
import math
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag import embeddings, store  # noqa: E402
from core.rag.chunking import Chunk, Page, chunk_pages  # noqa: E402
from core.rag.embed_llama_server import LlamaServerBackend  # noqa: E402


def _tokens(text):
    # Stand-in tokenizer: one token per word and per punctuation mark, never negative ids.
    return [hash(w) & 0xFFFF for w in text.replace(".", " . ").replace(",", " , ").split()]


class _Server:
    """Answers /tokenize the way llama-server's tokenize_mixed does: strings tokenized alone, ints kept."""

    def __init__(
        self,
        refuse_arrays = False,
        drop_markers = False,
    ):
        self.posts = []
        self.refuse_arrays = refuse_arrays
        self.drop_markers = drop_markers

    def __call__(self, path, payload, **_kwargs):
        assert path == "/tokenize" and payload["add_special"] is False
        content = payload["content"]
        self.posts.append(content)
        if isinstance(content, str):
            return {"tokens": _tokens(content)}
        if self.refuse_arrays:
            request = httpx.Request("POST", "http://x/tokenize")
            response = httpx.Response(400, request = request, text = "bad")
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise RuntimeError("llama-server embedder POST /tokenize -> 400: bad") from e
        out = []
        for part in content:
            if isinstance(part, str):
                out.extend(_tokens(part))
            elif not self.drop_markers:
                out.append(part)
        return {"tokens": out}


def _counter(monkeypatch, server):
    b = LlamaServerBackend()
    monkeypatch.setattr(b, "_post", server)
    return b.token_counter()


def _pages(n = 12):
    para = "Alpha bravo, charlie delta. Echo foxtrot golf hotel india juliet kilo lima.\n\n"
    return [
        Page(text = (para * (3 + i % 5)) + "x " * (40 * (i % 3)), page_number = i + 1, char_count = 0)
        for i in range(n)
    ]


def _rows(chunks):
    return [
        (
            c.text,
            c.token_count,
            c.page_number,
            c.source_page_index,
            c.chunk_index,
            c.page_char_start,
            c.page_char_end,
            c.whole_document_token_count,
        )
        for c in chunks
    ]


def test_batched_counts_equal_single_counts(monkeypatch):
    server = _Server()
    count = _counter(monkeypatch, server)
    texts = ["a b c", "", "d, e.", "a b c", "one", " "]
    assert count.many(texts) == [len(_tokens(t)) for t in texts]
    # One array POST for the five distinct texts, then the memo answers singles and repeats.
    assert len(server.posts) == 1 and server.posts[0][1] == -1
    assert count("d, e.") == 4 and count.many(["one", "a b c"]) == [1, 3]
    assert len(server.posts) == 1


def test_lone_text_and_singles_keep_the_string_request(monkeypatch):
    server = _Server()
    count = _counter(monkeypatch, server)
    assert count.many(["hello world"]) == [2]
    assert count("x y z") == 3
    assert server.posts == ["hello world", "x y z"]


@pytest.mark.parametrize("max_tokens,overlap", [(8, 0), (16, 4), (64, 16), (512, 64)])
def test_chunks_and_posts_vs_one_text_per_post(monkeypatch, max_tokens, overlap):
    single = _Server()
    plain = _counter(monkeypatch, single)
    one_at_a_time = lambda text: plain(text)  # noqa: E731 - no batch form: the old path
    batched_server = _Server()
    batched = _counter(monkeypatch, batched_server)
    pages = _pages()
    expected = _rows(
        chunk_pages(pages, max_tokens = max_tokens, overlap = overlap, count = one_at_a_time)
    )
    assert (
        _rows(chunk_pages(pages, max_tokens = max_tokens, overlap = overlap, count = batched)) == expected
    )
    assert len(batched_server.posts) < len(single.posts) / 4


@pytest.mark.parametrize("server", [_Server(refuse_arrays = True), _Server(drop_markers = True)])
def test_a_server_that_cannot_batch_falls_back_to_single_posts(monkeypatch, server):
    count = _counter(monkeypatch, server)
    texts = ["a b", "c d e", "f"]
    assert count.many(texts) == [2, 3, 1]
    assert server.posts[1:] == texts
    # Later batches go straight to single posts.
    assert count.many(["g h", "i"]) == [2, 1]
    assert server.posts[-2:] == ["g h", "i"]


def test_a_failed_batch_post_still_raises(monkeypatch):
    def broken(path, payload, **_kwargs):
        raise RuntimeError("llama-server embedder POST /tokenize failed after retry")

    count = _counter(monkeypatch, broken)
    with pytest.raises(RuntimeError, match = "failed after retry"):
        count.many(["a", "b"])


def test_batches_are_bounded(monkeypatch):
    from core.rag import embed_llama_server as mod

    monkeypatch.setattr(mod, "_TOKENIZE_BATCH", 3)
    monkeypatch.setattr(mod, "_TOKENIZE_BATCH_CHARS", 10)
    server = _Server()
    count = _counter(monkeypatch, server)
    texts = [f"w{i}" for i in range(7)] + ["a long text of many chars"]
    assert count.many(texts) == [1] * 7 + [6]
    sizes = [1 if isinstance(p, str) else (len(p) + 1) // 2 for p in server.posts]
    assert sizes == [3, 3, 1, 1]


def test_shared_token_counter_exposes_the_batch_form(monkeypatch):
    server = _Server()
    b = LlamaServerBackend()
    monkeypatch.setattr(b, "_post", server)
    monkeypatch.setattr(embeddings, "_get_backend", lambda model_name = None: b)
    count = embeddings.token_counter("org/embedder-GGUF")
    assert count.many(["a b", "c"]) == [2, 1] and len(server.posts) == 1


VOCAB = ["alpha", "bravo", "summary", "thread"]


def _embed(text):
    v = [float(text.lower().count(w)) + 0.01 for w in VOCAB]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def test_autoinject_encodes_its_query_once(rag_conn, monkeypatch):
    from core.inference import tools
    from storage import rag_db

    calls = []

    def encode(
        texts,
        *,
        model_name = None,
        normalize = True,
    ):
        calls.append(list(texts))
        return [_embed(t) for t in texts]

    monkeypatch.setattr(embeddings, "encode", encode)
    monkeypatch.setattr(embeddings, "dim", lambda model_name = None: len(VOCAB))
    monkeypatch.setattr(rag_db, "RAG_AVAILABLE", True, raising = False)
    monkeypatch.setattr(rag_db, "rag_available", lambda: True, raising = False)
    text = "alpha bravo thread notes"
    store.create_document(
        rag_conn, scope = "thread_T", filename = "t.pdf", sha256 = "dt", document_id = "dt"
    )
    chunk = Chunk(
        text = text,
        token_count = 4,
        page_number = None,
        source_page_index = 0,
        chunk_index = 0,
        page_char_start = 0,
        page_char_end = len(text),
    )
    store.add_chunks(rag_conn, "thread_T", "dt", [chunk], [_embed(text)])
    rag_conn.execute("UPDATE documents SET status='completed', num_chunks=1 WHERE id='dt'")
    rag_conn.commit()
    # The floored search finds nothing strong, so the unfloored thread search asks the same query again.
    out = tools.build_rag_autoinject(
        [{"role": "user", "content": "summarize this"}],
        {
            "thread_id": "T",
            "autoinject": True,
            "autoinject_min_score": 0.99,
            "mode": "hybrid",
            "context_length": 100,
        },
    )
    assert out is not None and "t.pdf" in json.dumps(out)
    assert calls == [["summarize this"]]
