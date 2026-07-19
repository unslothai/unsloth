# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Embedder concurrency tests: the fast tokenizer isn't thread-safe, so encode
and token counting must be serialized (else threads panic "Already borrowed")."""

import os
import threading
import time

import numpy as np
import pytest

from core.rag import config, embeddings


@pytest.fixture(autouse = True)
def _pin_st_backend(monkeypatch):
    # Tests patch ST internals (_get), so force the ST backend.
    monkeypatch.setattr(config, "EMBED_BACKEND", "sentence-transformers")
    embeddings._reset_backend()
    yield
    embeddings._reset_backend()


class _ConcurrencyProbe:
    """Records whether two callers were in the guarded body at once."""

    def __init__(self):
        self.inside = 0
        self.saw_overlap = False
        self._g = threading.Lock()

    def enter(self):
        with self._g:
            self.inside += 1
            if self.inside > 1:
                self.saw_overlap = True
        time.sleep(0.005)  # widen the race window
        with self._g:
            self.inside -= 1


class _FakeModel:
    def __init__(self, probe):
        self._probe = probe
        self.tokenizer = _FakeTokenizer(probe)

    def encode(self, texts, **_kw):
        self._probe.enter()
        return np.zeros((len(texts), 4), dtype = np.float32)


class _FakeTokenizer:
    def __init__(self, probe):
        self._probe = probe

    def encode(self, text, **_kw):
        self._probe.enter()
        return list(range(len(text.split())))


def _hammer(fn, n = 8):
    errors: list[Exception] = []

    def worker():
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target = worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def test_encode_is_serialized(monkeypatch):
    probe = _ConcurrencyProbe()
    monkeypatch.setattr(embeddings, "_get", lambda model_name = None: _FakeModel(probe))
    errors = _hammer(lambda: embeddings.encode(["alpha beta", "gamma"]))
    assert errors == []
    assert probe.saw_overlap is False  # compute lock serialized encode()


def test_token_counter_is_serialized(monkeypatch):
    probe = _ConcurrencyProbe()
    monkeypatch.setattr(embeddings, "_get", lambda model_name = None: _FakeModel(probe))
    count = embeddings.token_counter()
    errors = _hammer(lambda: count("one two three four"))
    assert errors == []
    assert probe.saw_overlap is False  # counting shares the tokenizer lock


def test_encode_enables_parallelism_only_during_call(monkeypatch):
    seen = {}

    class _M:
        tokenizer = None

        def encode(self, texts, **_kw):
            seen["during"] = os.environ.get("TOKENIZERS_PARALLELISM")
            return np.zeros((len(texts), 4), dtype = np.float32)

    monkeypatch.setattr(embeddings, "_get", lambda model_name = None: _M())
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    embeddings.encode(["alpha", "beta"])
    assert seen["during"] == "true"  # rayon batch tokenization enabled in-call
    assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"  # restored after


def test_token_counter_enables_parallelism_only_during_call(monkeypatch):
    seen = {}

    class _Tok:
        def encode(self, text, **_kw):
            seen["during"] = os.environ.get("TOKENIZERS_PARALLELISM")
            return list(range(len(text.split())))

    class _M:
        tokenizer = _Tok()

    monkeypatch.setattr(embeddings, "_get", lambda model_name = None: _M())
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    count = embeddings.token_counter()
    count("alpha beta gamma")
    assert seen["during"] == "true"  # rayon enabled in-call, like _st_encode
    assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"  # restored after


class _SentinelLlamaBackend:
    """Stand-in for LlamaServerBackend; never spawns a real server."""


def _force_st_load_failure(monkeypatch):
    """Make the ST warm-probe raise."""

    def _boom(model_name = None):
        raise RuntimeError("torch is broken on this machine")

    monkeypatch.setattr(embeddings, "_get", _boom)


def _patch_llama_backend(monkeypatch, *, binary):
    from core.inference.llama_cpp import LlamaCppBackend
    from core.rag import embed_llama_server

    monkeypatch.setattr(LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: binary))
    monkeypatch.setattr(embed_llama_server, "LlamaServerBackend", _SentinelLlamaBackend)


def test_st_failure_falls_back_to_llama_server(monkeypatch):
    # ST can't load but llama-server is available -> use it.
    _force_st_load_failure(monkeypatch)
    _patch_llama_backend(monkeypatch, binary = "/fake/llama-server")
    embeddings._reset_backend()
    backend = embeddings._get_backend()
    assert isinstance(backend, _SentinelLlamaBackend)


def test_st_failure_without_llama_binary_reraises(monkeypatch):
    # No llama-server binary -> surface the failure, don't degrade to nothing.
    _force_st_load_failure(monkeypatch)
    _patch_llama_backend(monkeypatch, binary = None)
    embeddings._reset_backend()
    with pytest.raises(RuntimeError, match = "torch is broken"):
        embeddings._get_backend()


def test_st_success_keeps_sentence_transformers(monkeypatch):
    # Clean ST probe -> ST backend stays selected, no fallback.
    monkeypatch.setattr(embeddings, "_get", lambda model_name = None: object())
    _patch_llama_backend(monkeypatch, binary = "/fake/llama-server")
    embeddings._reset_backend()
    backend = embeddings._get_backend()
    assert isinstance(backend, embeddings._SentenceTransformersBackend)


class _BoomOnEncodeModel:
    """Loads fine (init probe passes) but raises when encoding."""

    tokenizer = None

    def encode(self, texts, **_kw):
        raise RuntimeError("CUDA error during encode")


def test_st_encode_runtime_failure_switches_to_llama(monkeypatch):
    # encode() blows up mid-run -> switch to llama-server and stay switched.
    monkeypatch.setattr(embeddings, "_get", lambda model_name = None: _BoomOnEncodeModel())
    _patch_llama_backend(monkeypatch, binary = "/fake/llama-server")
    calls = {}

    def _sentinel_encode(
        self,
        texts,
        *,
        model_name = None,
        normalize = True,
    ):
        calls["used"] = True
        return np.zeros((len(texts), 4), dtype = np.float32)

    monkeypatch.setattr(_SentinelLlamaBackend, "encode", _sentinel_encode, raising = False)
    embeddings._reset_backend()

    out = embeddings.encode(["alpha", "beta"])
    assert calls.get("used") is True  # retried on the llama fallback
    assert out.shape == (2, 4)
    # Switch is process-wide: later calls keep using llama, not ST.
    assert isinstance(embeddings._get_backend(), _SentinelLlamaBackend)


def test_st_encode_failure_without_llama_binary_reraises(monkeypatch):
    # No llama-server binary -> surface the encode error.
    monkeypatch.setattr(embeddings, "_get", lambda model_name = None: _BoomOnEncodeModel())
    _patch_llama_backend(monkeypatch, binary = None)
    embeddings._reset_backend()
    with pytest.raises(RuntimeError, match = "CUDA error during encode"):
        embeddings.encode(["alpha", "beta"])


def test_get_resolves_default_casing_before_loading(monkeypatch):
    # _get() must resolve a repo id to the exact cache casing before constructing
    # SentenceTransformer: a default differing only by case is NOT persist-normalized by
    # /settings, so without resolving here the offline load would miss the cache dir.
    # Verify both the loader and the security gate receive the resolved name.
    import sys
    import types

    captured = {}

    class _FakeST:
        def __init__(self, name, **kwargs):
            captured["load_name"] = name
            captured["local_files_only"] = kwargs.get("local_files_only")

    fake_st = types.ModuleType("sentence_transformers")
    fake_st.SentenceTransformer = _FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_st)

    monkeypatch.setattr(config, "effective_embedding_model", lambda: "baai/bge-m3")
    monkeypatch.setattr(
        "utils.models.resolve_st_cached_repo_id_case",
        lambda repo_id: "BAAI/bge-m3" if repo_id == "baai/bge-m3" else repo_id,
    )
    monkeypatch.setattr(embeddings, "_device", lambda: "cpu")
    monkeypatch.setattr(embeddings, "_install_torchao_stub_once", lambda: None)
    guarded = {}
    monkeypatch.setattr(
        embeddings, "_guard_model_security", lambda name, local_only: guarded.update(name = name)
    )

    monkeypatch.setattr(embeddings, "_model", None, raising = False)
    monkeypatch.setattr(embeddings, "_name", None, raising = False)
    embeddings._get()

    assert captured["load_name"] == "BAAI/bge-m3"  # resolved to the cache casing
    assert guarded["name"] == "BAAI/bge-m3"  # gate scans the same resolved name
