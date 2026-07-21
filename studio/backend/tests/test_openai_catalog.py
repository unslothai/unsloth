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
from core.inference import local_model_resolver as resolver  # noqa: E402


class _Info:
    def __init__(
        self,
        id,
        display_name,
        model_id = None,
        is_gguf = True,
    ):
        self.id = id
        self.display_name = display_name
        self.model_id = model_id
        self.is_gguf = is_gguf  # drives the files-based GGUF check in the test


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
            _Info(
                "/data/models/Llama-8B-Q8.gguf", "Llama-8B-Q8"
            ),  # available, not loaded
            # HF-cache GGUF: model_format is unset for these, so a files-based check
            # (not model_format) must still list it.
            _Info("models--org--Foo", "Foo", model_id = "org/Foo"),
            # Non-GGUF (safetensors) can't be served via /v1: must NOT be advertised.
            _Info("/data/models/Mistral-7B", "Mistral-7B", is_gguf = False),
        ]

    monkeypatch.setattr(inf, "_cached_local_catalog", _fake_catalog)
    # GGUF-ness is read from the on-disk files; drive it off each info's flag here.
    monkeypatch.setattr(resolver, "info_has_local_gguf", lambda info: info.is_gguf)

    data = asyncio.run(inf._openai_catalog_objects())
    ids = {m["id"]: m for m in data}

    # Loaded model is present, marked loaded, and keeps context fields.
    assert ids["Qwen3-Q4"]["loaded"] is True
    assert ids["Qwen3-Q4"]["context_length"] == 4096
    # Available-but-not-loaded GGUF models are listed too.
    assert ids["Llama-8B-Q8"]["loaded"] is False
    # The HF-cache GGUF is listed despite model_format being unset.
    assert ids["org/Foo"]["loaded"] is False
    # The non-GGUF model is filtered out (/v1 can never serve it).
    assert "Mistral-7B" not in ids
    # The loaded gguf and the on-disk copy collapse to one clean id.
    assert [m["id"] for m in data].count("Qwen3-Q4") == 1
    # No absolute paths or .gguf suffixes leak anywhere.
    blob = json.dumps(data)
    assert ".gguf" not in blob
    assert "/srv/" not in blob
    assert "/data/" not in blob


def test_catalog_lock_is_per_loop():
    # Codex P2: a module-level asyncio.Lock ties its waiters to the loop that first
    # awaited it, so a second event loop awaiting it in a multi-loop process can
    # hang. The catalog lock must be per-loop (distinct lock per running loop), and
    # the old shared _CATALOG_LOCK must be gone so it can't be reintroduced.
    async def _get():
        return inf._catalog_lock()

    a = asyncio.run(_get())
    b = asyncio.run(_get())  # a fresh event loop
    assert a is not b
    assert not hasattr(inf, "_CATALOG_LOCK")


def test_empty_and_errored_scans_are_cached(monkeypatch):
    # Cache validity is keyed on the timestamp, not list contents, so an empty
    # (fresh install / no local models) or errored scan is still cached for the
    # TTL instead of rescanning the filesystem on every /v1/models poll.
    import routes.models as models_mod
    for outcome in ("empty", "error"):
        calls = {"n": 0}

        def _scan(_root, _outcome = outcome):
            calls["n"] += 1
            if _outcome == "error":
                raise RuntimeError("scan blew up")
            return []

        monkeypatch.setattr(models_mod, "collect_local_models", _scan)
        monkeypatch.setattr(inf, "_CATALOG_CACHE", {"at": 0.0, "models": []})

        async def _run():
            return [await inf._cached_local_catalog() for _ in range(3)]

        results = asyncio.run(_run())
        assert results == [[], [], []], outcome
        assert calls["n"] == 1, f"{outcome} scan ran {calls['n']}x (TTL not honored)"


def test_catalog_ttl_starts_after_scan_completes(monkeypatch):
    # The cache timestamp must be taken AFTER the scan, not before it. A scan that
    # outlives the TTL would otherwise leave the cache born-expired, so the next
    # caller rescans instead of reusing the just-computed catalog.
    import routes.models as models_mod

    clock = {"t": 1000.0}
    monkeypatch.setattr(inf.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(inf, "_CATALOG_CACHE", {"at": 0.0, "models": []})

    calls = {"n": 0}

    def _slow_scan(_root):
        calls["n"] += 1
        clock["t"] += inf._CATALOG_TTL_S + 10  # the scan itself outlives the TTL
        return [_Info("/m/A.gguf", "A")]

    monkeypatch.setattr(models_mod, "collect_local_models", _slow_scan)

    async def _run():
        first = await inf._cached_local_catalog()
        second = await inf._cached_local_catalog()  # clock unchanged since scan end
        return first, second

    first, second = asyncio.run(_run())
    assert [i.id for i in first] == ["/m/A.gguf"]
    assert (
        calls["n"] == 1
    ), "TTL started before the scan -> cache born expired, rescanned"


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
