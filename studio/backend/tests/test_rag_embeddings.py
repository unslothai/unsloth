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


class _RecordingBackend:
    """A backend that records unload() and reports whether it killed a GPU process."""

    def __init__(self, killed_gpu_process = False):
        self.unloaded = False
        self._killed = killed_gpu_process

    def unload(self):
        self.unloaded = True
        return self._killed


def test_unload_noop_when_no_backend_built():
    # The common no-RAG path: load_model calls unload() on every GGUF load, and
    # with no backend resolved it must be a safe no-op (no build, no error) and
    # report no GPU-process kill so the caller skips the VRAM-settle wait.
    embeddings._reset_backend()
    assert embeddings._backend is None
    assert embeddings.unload() is False  # must not raise or build a backend
    assert embeddings._backend is None


def test_unload_dispatches_and_reports_gpu_kill():
    # unload() frees whatever backend is live and propagates whether an external
    # GPU process was killed (True for llama-server) so load_model waits for the
    # driver's async reclaim; False for the in-process ST drop.
    st_like = _RecordingBackend(killed_gpu_process = False)
    embeddings._backend = st_like
    try:
        assert embeddings.unload() is False
    finally:
        embeddings._reset_backend()
    assert st_like.unloaded is True

    llama_like = _RecordingBackend(killed_gpu_process = True)
    embeddings._backend = llama_like
    try:
        assert embeddings.unload() is True
    finally:
        embeddings._reset_backend()
    assert llama_like.unloaded is True


def test_st_unload_drops_cached_model(monkeypatch):
    # _st_unload clears the cached ST model so its VRAM is released; the next
    # encode/warm reloads lazily.
    monkeypatch.setattr(embeddings, "_model", object(), raising = False)
    monkeypatch.setattr(embeddings, "_name", "some-model", raising = False)
    embeddings._st_unload()
    assert embeddings._model is None
    assert embeddings._name is None


def test_st_unload_noop_when_nothing_loaded(monkeypatch):
    # No model loaded -> _st_unload must not touch torch (it would otherwise
    # create a CUDA context on a process that never embedded).
    monkeypatch.setattr(embeddings, "_model", None, raising = False)
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)  # any torch use would raise
    embeddings._st_unload()  # must early-return before importing torch
    assert embeddings._model is None


def test_llama_unload_skips_cpu_only_server(monkeypatch):
    # A CPU-only embedder holds no VRAM, so a model load must leave it running
    # (killing it would abort ingestion for no GPU-probe gain). A GPU child is
    # killed so its VRAM is reclaimed.
    from core.rag.embed_llama_server import LlamaServerBackend

    b = LlamaServerBackend()
    monkeypatch.setattr(b, "_process_alive", lambda: True)
    killed = {"n": 0}
    monkeypatch.setattr(b, "_kill_process", lambda: killed.__setitem__("n", killed["n"] + 1))

    b._started_on_gpu = False
    assert b.unload() is False  # CPU-only: leave it running
    assert killed["n"] == 0

    b._started_on_gpu = True
    assert b.unload() is True  # GPU child: kill to free VRAM
    assert killed["n"] == 1


def test_llama_post_no_respawn_when_unloaded_midflight(monkeypatch):
    # If a model load unloads us mid-POST (epoch moves), the kill is deliberate, so
    # the dropped connection must NOT respawn the embedder back into model VRAM.
    import httpx

    from core.rag.embed_llama_server import LlamaServerBackend

    b = LlamaServerBackend()
    monkeypatch.setattr(b, "_ensure_ready", lambda: None)
    restarts = {"n": 0}
    monkeypatch.setattr(b, "_restart", lambda: restarts.__setitem__("n", restarts["n"] + 1))

    def _post_killed_by_unload(url, json):
        b._unload_epoch += 1  # a concurrent unload() killed the child
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(b._client, "post", _post_killed_by_unload)
    with pytest.raises(RuntimeError, match = "failed after retry"):
        b._post("/v1/embeddings", {})
    assert restarts["n"] == 0


def test_llama_post_respawns_on_transient_drop(monkeypatch):
    # A dropped connection with no unload (the reaper killed us) must still respawn
    # once and retry, the pre-existing self-heal the epoch guard must not break.
    import httpx

    from core.rag.embed_llama_server import LlamaServerBackend

    b = LlamaServerBackend()
    monkeypatch.setattr(b, "_ensure_ready", lambda: None)
    monkeypatch.setattr(b, "_restart", lambda: None)
    calls = {"n": 0}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True}

    def _post(url, json):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("connection refused")
        return _Resp()

    monkeypatch.setattr(b._client, "post", _post)
    assert b._post("/v1/embeddings", {}) == {"ok": True}
    assert calls["n"] == 2  # respawned and retried


def test_llama_deliberate_post_works_after_unload(monkeypatch):
    # Codex #1: after a GPU unload, ingestion calls token_counter() (a /tokenize
    # POST) before any encode. That deliberate call must respawn and succeed, not
    # be blocked by a stale unload guard.
    from core.rag.embed_llama_server import LlamaServerBackend

    b = LlamaServerBackend()
    b._started_on_gpu = True
    monkeypatch.setattr(b, "_process_alive", lambda: True)
    monkeypatch.setattr(b, "_kill_process", lambda: None)
    assert b.unload() is True

    monkeypatch.setattr(b, "_ensure_ready", lambda: None)

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"tokens": [1, 2, 3]}

    monkeypatch.setattr(b._client, "post", lambda url, json: _Resp())
    assert b._post("/tokenize", {}) == {"tokens": [1, 2, 3]}
