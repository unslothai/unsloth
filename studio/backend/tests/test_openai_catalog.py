# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GET /v1/models lists the full server catalog (loaded + locally available)."""

import asyncio
import json
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import routes.inference as inf  # noqa: E402


class _Info:
    def __init__(
        self,
        id,
        display_name,
        model_id = None,
    ):
        self.id = id
        self.display_name = display_name
        self.model_id = model_id


class _FakeLlama:
    is_loaded = True
    model_identifier = "/srv/models/Qwen3-Q4.gguf"
    context_length = 4096
    max_context_length = None
    native_context_length = None

    def __init__(self, loaded = True):
        self.is_loaded = loaded


class _FakeUnsloth:
    active_model_name = None
    models: dict = {}
    context_length = None
    max_seq_length = None


def test_catalog_lists_loaded_and_available(monkeypatch):
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: _FakeLlama())
    monkeypatch.setattr(inf, "get_inference_backend", lambda: _FakeUnsloth())

    async def _fake_catalog():
        return [
            _Info("/data/models/Qwen3-Q4.gguf", "Qwen3-Q4"),  # same as loaded -> dedup
            _Info("/data/models/Llama-8B-Q8.gguf", "Llama-8B-Q8"),  # available, not loaded
            _Info("models--org--Foo", "Foo", model_id = "org/Foo"),  # hf cache repo id
        ]

    monkeypatch.setattr(inf, "_cached_local_catalog", _fake_catalog)

    data = asyncio.run(inf._openai_catalog_objects())
    ids = {m["id"]: m for m in data}

    # Loaded model is present, marked loaded, and keeps context fields.
    assert ids["Qwen3-Q4"]["loaded"] is True
    assert ids["Qwen3-Q4"]["context_length"] == 4096
    # Available-but-not-loaded models are listed too.
    assert ids["Llama-8B-Q8"]["loaded"] is False
    assert ids["org/Foo"]["loaded"] is False
    # The loaded gguf and the on-disk copy collapse to one clean id.
    assert [m["id"] for m in data].count("Qwen3-Q4") == 1
    # No absolute paths or .gguf suffixes leak anywhere.
    blob = json.dumps(data)
    assert ".gguf" not in blob
    assert "/srv/" not in blob
    assert "/data/" not in blob


def test_path_loaded_model_not_relisted_under_alias(monkeypatch):
    # An LM Studio/Ollama row carries a model_id alias but is loaded by its path,
    # so the loaded entry is keyed under public_model_id(path). The matching
    # catalog row must dedup against that path-derived key, not reappear as an
    # unloaded duplicate under its alias.
    monkeypatch.setattr(
        inf, "get_llama_cpp_backend", lambda: _FakeLlama()
    )  # loaded id == /srv/models/Qwen3-Q4.gguf -> "Qwen3-Q4"
    monkeypatch.setattr(inf, "get_inference_backend", lambda: _FakeUnsloth())

    async def _fake_catalog():
        return [
            # Same physical model as loaded (path basename "Qwen3-Q4") but exposed
            # by the scanner with an Ollama-style alias.
            _Info("/srv/models/Qwen3-Q4.gguf", "Qwen3-Q4", model_id = "ollama/qwen3:q4"),
            _Info("/data/models/Other.gguf", "Other", model_id = "ollama/other:tag"),
        ]

    monkeypatch.setattr(inf, "_cached_local_catalog", _fake_catalog)

    data = asyncio.run(inf._openai_catalog_objects())
    ids = [m["id"] for m in data]
    loaded = {m["id"]: m["loaded"] for m in data}
    # The loaded model appears exactly once, under its loaded id, marked loaded.
    assert ids.count("Qwen3-Q4") == 1
    assert loaded["Qwen3-Q4"] is True
    # Its alias is NOT re-listed as an unloaded duplicate.
    assert "ollama/qwen3:q4" not in ids
    # A genuinely-distinct unloaded alias still shows.
    assert "ollama/other:tag" in ids
    assert loaded["ollama/other:tag"] is False


def test_distinct_unloaded_models_sharing_a_basename_both_appear(monkeypatch):
    # The path-dedup must fire only against LOADED entries. Two different local
    # files with the same basename, where the later one has a model_id alias,
    # must NOT collapse: the path_id collision is between two unloaded rows, not
    # against a loaded model.
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: _FakeLlama(loaded = False))
    monkeypatch.setattr(inf, "get_inference_backend", lambda: _FakeUnsloth())

    async def _fake_catalog():
        return [
            _Info("/models/tiny.gguf", "tiny"),                              # cid -> "tiny"
            _Info("/ollama/tiny.gguf", "tiny", model_id = "ollama/tiny:q4"),  # same basename, aliased
        ]

    monkeypatch.setattr(inf, "_cached_local_catalog", _fake_catalog)

    data = asyncio.run(inf._openai_catalog_objects())
    ids = [m["id"] for m in data]
    assert "tiny" in ids
    assert "ollama/tiny:q4" in ids  # not wrongly dropped by a basename collision
    assert len(data) == 2


def test_retrieve_loaded_model_skips_catalog_scan(monkeypatch):
    # Retrieving a loaded id must resolve from the loaded set alone, never paying
    # for the filesystem scan that _cached_local_catalog drives.
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: _FakeLlama())
    monkeypatch.setattr(inf, "get_inference_backend", lambda: _FakeUnsloth())

    async def _boom():
        raise AssertionError("catalog scan must not run for a loaded id")

    monkeypatch.setattr(inf, "_cached_local_catalog", _boom)

    model = asyncio.run(inf.openai_retrieve_model("Qwen3-Q4", current_subject = "t"))
    assert model["id"] == "Qwen3-Q4"
    assert model["loaded"] is True


def test_cached_local_catalog_offloads_and_caches(monkeypatch):
    # The filesystem scan must run off the event loop (asyncio.to_thread) and be
    # cached, so a burst of /v1/models calls does not re-scan or block.
    calls = {"scan": 0, "threaded": 0}

    def _fake_collect(_root):
        calls["scan"] += 1
        return [_Info("/data/models/A.gguf", "A")]

    import routes.models as models_mod

    monkeypatch.setattr(models_mod, "collect_local_models", _fake_collect)

    real_to_thread = inf.asyncio.to_thread

    async def _counting_to_thread(fn, *a, **k):
        calls["threaded"] += 1
        return await real_to_thread(fn, *a, **k)

    monkeypatch.setattr(inf.asyncio, "to_thread", _counting_to_thread)
    # Fresh cache for a deterministic count.
    monkeypatch.setattr(inf, "_CATALOG_CACHE", {"at": 0.0, "models": []})

    async def _run():
        first = await inf._cached_local_catalog()
        second = await inf._cached_local_catalog()  # within TTL -> cached
        return first, second

    first, second = asyncio.run(_run())
    assert [i.id for i in first] == ["/data/models/A.gguf"]
    assert second is first or [i.id for i in second] == [i.id for i in first]
    assert calls["scan"] == 1  # cached: scanned once for two calls
    assert calls["threaded"] == 1  # offloaded to a worker thread
