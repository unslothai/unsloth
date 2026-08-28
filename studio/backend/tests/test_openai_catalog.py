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
            _Info("/data/models/Llama-8B-Q8.gguf", "Llama-8B-Q8"),  # available, not loaded
            # HF-cache GGUF: model_format is unset for these, so a files-based check
            # (not model_format) must still list it.
            _Info("models--org--Foo", "Foo", model_id = "org/Foo"),
            # Non-GGUF (safetensors/MLX): the orchestrator serves it, so it is listed too.
            _Info("/data/models/Mistral-7B", "Mistral-7B", is_gguf = False),
        ]

    monkeypatch.setattr(inf, "_cached_local_catalog", _fake_catalog)
    # Format and the quant labels come from one on-disk scan; drive both off the flag.
    monkeypatch.setattr(
        resolver,
        "local_servable_model",
        lambda info: (True, ("Q8_0",)) if info.is_gguf else (False, ()),
    )

    data = asyncio.run(inf._openai_catalog_objects())
    ids = {m["id"]: m for m in data}

    # Loaded model is present, marked loaded, and keeps context fields.
    assert ids["Qwen3-Q4"]["loaded"] is True
    assert ids["Qwen3-Q4"]["context_length"] == 4096
    # Not-loaded GGUFs are listed too, with the quant a client appends to pin them.
    assert ids["Llama-8B-Q8"]["loaded"] is False
    assert ids["Llama-8B-Q8"]["quant"] == "Q8_0"
    # The HF-cache GGUF is listed despite model_format being unset.
    assert ids["org/Foo"]["loaded"] is False
    # The non-GGUF model is listed so an API client can switch to it, with no quant to pin.
    assert ids["Mistral-7B"]["loaded"] is False
    assert "quant" not in ids["Mistral-7B"]
    # The loaded gguf and the on-disk copy collapse to one clean id.
    assert [m["id"] for m in data].count("Qwen3-Q4") == 1
    # No absolute paths or .gguf suffixes leak anywhere.
    blob = json.dumps(data)
    assert ".gguf" not in blob
    assert "/srv/" not in blob
    assert "/data/" not in blob


def test_a_resident_non_gguf_model_is_marked_loaded_and_stays_quantless(monkeypatch):
    # llama-only residency lists the model that is serving as unloaded, and the ungated
    # hf_variant read stamps a stale quant on this row.
    class _Orchestrator:
        active_model_name = "/data/models/Mistral-7B"
        models: dict = {}
        context_length = None
        max_seq_length = None

    llama = _FakeLlama(loaded = False)
    llama.hf_variant = "Q4_K_M"
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: llama)
    monkeypatch.setattr(inf, "get_inference_backend", lambda: _Orchestrator())

    async def _fake_catalog():
        return [_Info("/data/models/Mistral-7B", "Mistral-7B", is_gguf = False)]

    monkeypatch.setattr(inf, "_cached_local_catalog", _fake_catalog)
    monkeypatch.setattr(resolver, "local_servable_model", lambda info: (False, ()))
    ids = {m["id"]: m for m in asyncio.run(inf._openai_catalog_objects())}
    assert ids["Mistral-7B"]["loaded"] is True
    assert "quant" not in ids["Mistral-7B"]


def test_a_manually_loaded_non_gguf_model_has_one_catalog_row(monkeypatch):
    class _Orchestrator:
        active_model_name = "/srv/lmstudio/mlx-community/Qwen3-8B-4bit"
        models: dict = {}
        context_length = 8192
        max_seq_length = None

    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: _FakeLlama(loaded = False))
    monkeypatch.setattr(inf, "get_inference_backend", lambda: _Orchestrator())

    info = _Info(
        "/srv/lmstudio/mlx-community/Qwen3-8B-4bit",
        "Qwen3-8B-4bit",
        model_id = "mlx-community/Qwen3-8B-4bit",
        is_gguf = False,
    )
    info.path = info.id

    async def _fake_catalog():
        return [info]

    monkeypatch.setattr(inf, "_cached_local_catalog", _fake_catalog)
    monkeypatch.setattr(resolver, "local_servable_model", lambda _info: (False, ()))
    data = asyncio.run(inf._openai_catalog_objects())

    assert data == [
        {
            "id": "Qwen3-8B-4bit",
            "object": "model",
            "created": data[0]["created"],
            "owned_by": "unsloth-studio",
            "context_length": 8192,
            "loaded": True,
        }
    ]


def test_non_gguf_catalog_dedupe_keeps_a_distinct_same_basename_path(monkeypatch):
    class _Orchestrator:
        active_model_name = "/srv/a/publisher/model"
        models: dict = {}
        context_length = None
        max_seq_length = None

    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: _FakeLlama(loaded = False))
    monkeypatch.setattr(inf, "get_inference_backend", lambda: _Orchestrator())

    resident = _Info("/srv/a/publisher/model", "model", model_id = "publisher-a/model", is_gguf = False)
    available = _Info(
        "/srv/b/publisher/model", "model", model_id = "publisher-b/model", is_gguf = False
    )
    resident.path = resident.id
    available.path = available.id

    async def _fake_catalog():
        return [resident, available]

    monkeypatch.setattr(inf, "_cached_local_catalog", _fake_catalog)
    monkeypatch.setattr(resolver, "local_servable_model", lambda _info: (False, ()))
    ids = {entry["id"]: entry for entry in asyncio.run(inf._openai_catalog_objects())}

    assert set(ids) == {"model", "publisher-b/model"}
    assert ids["model"]["loaded"] is True
    assert ids["publisher-b/model"]["loaded"] is False


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
    assert calls["n"] == 1, "TTL started before the scan -> cache born expired, rescanned"


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


def test_monitor_active_model_is_a_public_id_not_a_host_path(monkeypatch):
    # The settings UI renders this and --secure serves it publicly, so never a load path.
    class _Llama:
        is_loaded = True
        model_identifier = "/home/me/.cache/huggingface/hub/models--org--A-GGUF/snapshots/abc"
        hf_variant = "UD-Q4_K_XL"
        _openai_advertised_id = "org/A-GGUF"

    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: _Llama())
    assert inf._monitor_active_model() == "org/A-GGUF:UD-Q4_K_XL"


def test_monitor_active_model_cleans_a_path_with_no_advertised_id(monkeypatch):
    class _Llama:
        is_loaded = True
        model_identifier = "/data/models/Llama-8B-Q8.gguf"
        hf_variant = None
        _openai_advertised_id = None

    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: _Llama())
    label = inf._monitor_active_model()
    assert "/" not in label and ".gguf" not in label


def test_monitor_reports_nothing_once_a_non_gguf_model_is_unloaded(monkeypatch):
    # An auto-switch load records the requested repo id on the orchestrator, and an
    # unload clears active_model_name without clearing that alias. Reading the alias
    # ungated made the monitor report a ready model with nothing loaded.
    class _Orchestrator:
        active_model_name = None
        _openai_advertised_id = "unsloth/Qwen3-MLX"

    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: _FakeLlama(loaded = False))
    monkeypatch.setattr(inf, "_peek_inference_backend", lambda: _Orchestrator())
    assert inf._monitor_active_model() is None


def test_a_non_gguf_model_reports_one_id_across_every_v1_surface(monkeypatch):
    # /v1/models, GET /v1/models/{id} and the chat-completions response body must all
    # name an auto-switched model the same way, or a client cannot round-trip the id.
    class _Orchestrator:
        active_model_name = "/srv/lmstudio/mlx-community/Qwen3-8B-4bit"
        _openai_advertised_id = "mlx-community/Qwen3-8B-4bit"
        models: dict = {}
        context_length = None
        max_seq_length = None

    orchestrator = _Orchestrator()
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: _FakeLlama(loaded = False))
    monkeypatch.setattr(inf, "get_inference_backend", lambda: orchestrator)

    listed = inf._openai_model_objects()[0]["id"]
    retrieved = asyncio.run(
        inf.openai_retrieve_model(orchestrator.active_model_name, current_subject = "t")
    )
    assert listed == "mlx-community/Qwen3-8B-4bit"
    assert retrieved["id"] == listed and retrieved["loaded"] is True
    assert inf._orchestrator_public_model_id(orchestrator) == listed


def test_lifecycle_label_recovers_the_repo_id_from_an_hf_cache_path():
    # An auto-switch load gets the snapshot dir, whose basename is a commit sha.
    snap = "/home/me/.cache/huggingface/hub/models--unsloth--gemma-4-E4B-it-GGUF/snapshots/bfc15c3"
    assert (
        inf._lifecycle_model_label(snap, "UD-Q4_K_XL") == "unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL"
    )


def test_lifecycle_model_label_is_path_free():
    label = inf._lifecycle_model_label("/data/models/Llama-8B-Q8.gguf", "Q8_0")
    assert "/" not in label and ".gguf" not in label
    assert inf._lifecycle_model_label("org/A-GGUF", "Q4_K_M") == "org/A-GGUF:Q4_K_M"
    # An id that already carries a quant is not double-suffixed.
    assert inf._lifecycle_model_label("org/A-GGUF:Q4_K_M", "Q8_0") == "org/A-GGUF:Q4_K_M"


def test_a_standalone_gguf_does_not_advertise_a_quant_that_stops_resolving(monkeypatch):
    # llama.cpp reads hf_variant off the filename, but the resolver stores standalone files
    # with no quants, so a pinned "<stem>:<quant>" would 404 once it is not resident.
    from core.inference.local_model_resolver import _LocalGgufEntry

    standalone = _LocalGgufEntry("Qwen3-Q4", "/srv/models/Qwen3-Q4.gguf", ())
    repo = _LocalGgufEntry("org/Foo", "/hf/models--org--Foo/snapshots/a", ("Q4_K_M",))
    monkeypatch.setattr(resolver, "_scan", (1.0, {"qwen3-q4": standalone, "org/foo": repo}))
    monkeypatch.setattr(inf, "get_inference_backend", lambda: _FakeUnsloth())

    llama = _FakeLlama()
    llama.hf_variant = "Q4_K_M"
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: llama)
    assert "quant" not in inf._openai_model_objects()[0]

    # The same quant on a repo the resolver does list stays advertised.
    llama.model_identifier = "org/Foo"
    assert inf._openai_model_objects()[0]["quant"] == "Q4_K_M"

    # A cold index cannot prove the reference either, and publishing on no proof is
    # exactly what hands out the pin that later fails to resolve.
    monkeypatch.setattr(resolver, "_scan", (0.0, {}))
    # Stub the walk: a real multi-root scan inside the cold-wait budget makes this
    # test time out into a 503 under load instead of asserting what it is here for.
    monkeypatch.setattr(resolver, "_build_index", lambda: {})
    monkeypatch.setattr(resolver, "warm_index_soon", lambda: None)
    assert "quant" not in inf._openai_model_objects()[0]


def test_a_loaded_alias_advertises_the_quant_that_is_actually_loaded(monkeypatch):
    # Marking the alias loaded while still publishing the preferred on-disk quant said
    # alias:Q4 was loaded while Q8 was serving, and pinning that 404s with switching off.
    monkeypatch.setattr(inf, "get_inference_backend", lambda: _FakeUnsloth())
    llama = _FakeLlama()
    llama.hf_variant = "Q8_0"
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: llama)

    alias = _Info("/srv/models", "Qwen3", model_id = "publisher/Qwen3")
    alias.path = "/srv/models"  # holds the resident /srv/models/Qwen3-Q4.gguf

    async def _fake_catalog():
        return [alias]

    monkeypatch.setattr(inf, "_cached_local_catalog", _fake_catalog)
    monkeypatch.setattr(resolver, "local_servable_model", lambda info: (True, ("Q4_K_M", "Q8_0")))
    ids = {m["id"]: m for m in asyncio.run(inf._openai_catalog_objects())}
    assert ids["publisher/Qwen3"]["loaded"] is True
    assert ids["publisher/Qwen3"]["quant"] == "Q8_0"


def test_a_nested_model_directory_is_not_the_resident_one(monkeypatch):
    # Two indexed models can nest (/models/A holding A, /models/A/sub/B holding B). A
    # plain prefix test made loading B mark A resident, so a request for A was answered
    # with B's weights. The innermost indexed model owns the file.
    outer = _Info("/models/A", "A", model_id = "publisher/A")
    outer.path = "/models/A"
    inner = _Info("/models/A/sub/B", "B", model_id = "publisher/B")
    inner.path = "/models/A/sub/B"
    monkeypatch.setitem(inf._CATALOG_CACHE, "models", [outer, inner])

    llama = _FakeLlama()
    llama.gguf_path = "/models/A/sub/B/model-Q4_K_M.gguf"
    llama.model_identifier = llama.gguf_path
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: llama)
    monkeypatch.setattr(inf, "get_inference_backend", lambda: _FakeUnsloth())

    assert inf._resolves_to_resident("/models/A/sub/B") is True
    assert inf._resolves_to_resident("/models/A") is False
    # With nothing indexed there is no nesting to tell apart, so the directory-to-file
    # match this exists for must still hold.
    monkeypatch.setitem(inf._CATALOG_CACHE, "models", [])
    assert inf._resolves_to_resident("/models/A") is True


def test_a_transformers_model_does_not_mark_a_gguf_alias_loaded(monkeypatch):
    # Every entry in this loop is advertised as GGUF with a GGUF quant. A Transformers
    # model live from a directory that also holds GGUF exports is not one, and marking
    # the alias loaded had the examples pin a quant nothing can serve with switching off.
    unsloth = _FakeUnsloth()
    unsloth.active_model_name = "/srv/models"
    monkeypatch.setattr(inf, "get_inference_backend", lambda: unsloth)
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: _FakeLlama(loaded = False))

    alias = _Info("/srv/models", "Qwen3", model_id = "publisher/Qwen3")
    alias.path = "/srv/models"  # also holds /srv/models/Qwen3-Q4.gguf

    async def _fake_catalog():
        return [alias]

    monkeypatch.setattr(inf, "_cached_local_catalog", _fake_catalog)
    monkeypatch.setattr(resolver, "local_servable_model", lambda info: (True, ("Q4_K_M",)))
    ids = {m["id"]: m for m in asyncio.run(inf._openai_catalog_objects())}
    assert ids["publisher/Qwen3"]["loaded"] is False


def test_an_alias_for_the_resident_weights_is_not_listed_as_unloaded(monkeypatch):
    # A GGUF loaded by absolute path keys the resident entry by basename, so an id-only dedup
    # would emit the alias again marked not loaded.
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: _FakeLlama())
    monkeypatch.setattr(inf, "get_inference_backend", lambda: _FakeUnsloth())

    alias = _Info("/srv/models", "Qwen3", model_id = "publisher/Qwen3")
    alias.path = "/srv/models"  # holds the resident /srv/models/Qwen3-Q4.gguf

    async def _fake_catalog():
        return [alias]

    monkeypatch.setattr(inf, "_cached_local_catalog", _fake_catalog)
    monkeypatch.setattr(resolver, "local_servable_model", lambda info: (True, ("Q4_K_M",)))
    ids = {m["id"]: m for m in asyncio.run(inf._openai_catalog_objects())}
    assert ids["publisher/Qwen3"]["loaded"] is True
