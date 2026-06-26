# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in OpenAI /v1 model auto-switch: resolver, hook, and settings coercion.

No GPU or llama-server: the backend and the load route are mocked, mirroring
tests/test_gguf_completion_usage.py.
"""

import asyncio

import pytest

import routes.inference as inference_route
from models.inference import LoadRequest
from core.inference import local_model_resolver as resolver
from utils import openai_auto_switch_settings as settings


class _FakeBackend:
    def __init__(
        self,
        loaded_id = None,
        hf_variant = None,
        advertised_id = None,
    ):
        self.model_identifier = loaded_id
        self.is_loaded = loaded_id is not None
        self.hf_variant = hf_variant
        self._openai_advertised_id = advertised_id


class _LoadRecorder:
    """Stand-in for the load route: records calls and simulates a load."""

    def __init__(
        self,
        backend,
        fail = False,
    ):
        self.backend = backend
        self.calls = []
        self.fail = fail

    async def __call__(
        self,
        request,
        fastapi_request,
        current_subject = None,
    ):
        self.calls.append(request)
        if self.fail:
            from fastapi import HTTPException
            raise HTTPException(status_code = 503, detail = "load failed")
        self.backend.model_identifier = request.model_path
        self.backend.is_loaded = True
        # Mirror _load_model_impl: a load advertises its own id until the
        # auto-switch caller overwrites it with the repo id.
        self.backend._openai_advertised_id = None
        return None


def _wire(monkeypatch, *, enabled, resolves_to, backend, recorder):
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: enabled)
    monkeypatch.setattr(resolver, "resolve_local_gguf", lambda _m: resolves_to)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    # Auto-switch loads via _load_model_impl (the /load route holds the lifecycle
    # gate that auto-switch already owns, so it calls the impl directly).
    monkeypatch.setattr(inference_route, "_load_model_impl", recorder)
    monkeypatch.setattr(inference_route, "_auto_switch_waiters", {})
    monkeypatch.setattr(inference_route, "_auto_switch_request_waiters", {})


def _run_hook(model = "some/model"):
    asyncio.run(inference_route._maybe_auto_switch_model(model, object(), "tester"))


def test_flag_off_never_loads(monkeypatch):
    backend = _FakeBackend("unsloth/A-GGUF")
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = False,
        resolves_to = ("unsloth/B-GGUF", None, "unsloth/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    _run_hook("unsloth/B-GGUF")
    assert rec.calls == []


def test_unknown_model_falls_through(monkeypatch):
    backend = _FakeBackend("unsloth/A-GGUF")
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = True, resolves_to = None, backend = backend, recorder = rec)
    _run_hook("gpt-4o-mini")
    assert rec.calls == []


def test_already_loaded_does_not_reload(monkeypatch):
    backend = _FakeBackend("unsloth/A-GGUF")
    rec = _LoadRecorder(backend)
    # Case-insensitive match against the loaded identifier.
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("unsloth/a-gguf", None, "unsloth/a-gguf"),
        backend = backend,
        recorder = rec,
    )
    _run_hook("unsloth/A-GGUF")
    assert rec.calls == []


def test_known_unloaded_model_switches_once(monkeypatch):
    backend = _FakeBackend("unsloth/A-GGUF")
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("unsloth/B-GGUF", "Q4_K_M", "unsloth/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    _run_hook("unsloth/B-GGUF:Q4_K_M")
    assert len(rec.calls) == 1
    req = rec.calls[0]
    assert isinstance(req, LoadRequest)
    assert req.model_path == "unsloth/B-GGUF"
    assert req.gguf_variant == "Q4_K_M"
    assert backend.model_identifier == "unsloth/B-GGUF"


def test_concurrent_same_target_loads_once(monkeypatch):
    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("unsloth/B-GGUF", None, "unsloth/B-GGUF"),
        backend = backend,
        recorder = rec,
    )

    async def _race():
        await asyncio.gather(
            inference_route._maybe_auto_switch_model("unsloth/B-GGUF", object(), "t"),
            inference_route._maybe_auto_switch_model("unsloth/B-GGUF", object(), "t"),
        )

    asyncio.run(_race())
    assert len(rec.calls) == 1


def test_load_failure_propagates(monkeypatch):
    from fastapi import HTTPException

    backend = _FakeBackend("unsloth/A-GGUF")
    rec = _LoadRecorder(backend, fail = True)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("unsloth/B-GGUF", None, "unsloth/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    with pytest.raises(HTTPException):
        _run_hook("unsloth/B-GGUF")


def test_same_repo_different_variant_switches(monkeypatch):
    # Q4_K_M loaded, Q8_0 requested: a different quant must trigger a reload.
    backend = _FakeBackend("unsloth/B-GGUF", hf_variant = "Q4_K_M")
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("unsloth/B-GGUF", "Q8_0", "unsloth/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    _run_hook("unsloth/B-GGUF:Q8_0")
    assert len(rec.calls) == 1
    assert rec.calls[0].gguf_variant == "Q8_0"


def test_same_repo_same_variant_does_not_reload(monkeypatch):
    backend = _FakeBackend("unsloth/B-GGUF", hf_variant = "Q4_K_M")
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("unsloth/B-GGUF", "q4_k_m", "unsloth/B-GGUF"),  # case-insensitive
        backend = backend,
        recorder = rec,
    )
    _run_hook("unsloth/B-GGUF:Q4_K_M")
    assert rec.calls == []


def test_responses_endpoint_wires_auto_switch_before_dispatch():
    # The /v1/responses endpoint must invoke the auto-switch hook before either
    # dispatcher so streaming requests switch too. Asserted on the source, which
    # is immune to test-ordering effects on the shared inference module.
    import inspect

    src = inspect.getsource(inference_route.openai_responses)
    assert "_maybe_auto_switch_model" in src
    hook_at = src.index("_maybe_auto_switch_model")
    assert hook_at < src.index("_responses_stream")
    assert hook_at < src.index("_responses_non_streaming")


def test_embeddings_endpoint_wires_auto_switch_before_loaded_check():
    # /v1/embeddings is model-bearing too, so it must auto-switch before the
    # loaded-state gate. Asserted on the source for order-independence.
    import inspect

    src = inspect.getsource(inference_route.openai_embeddings)
    assert "_auto_switch_from_request_body" in src
    assert src.index("_auto_switch_from_request_body") < src.index("is_loaded")


def test_count_tokens_endpoint_wires_auto_switch_before_loaded_check():
    # The Anthropic token-count endpoint must count with the requested model.
    import inspect

    src = inspect.getsource(inference_route.anthropic_count_tokens)
    assert "_maybe_auto_switch_model" in src
    assert src.index("_maybe_auto_switch_model") < src.index("is_loaded")


# ── resolver ────────────────────────────────────────────────────────


def test_local_gguf_entry_filters_non_gguf_and_recurses(tmp_path):
    from types import SimpleNamespace

    # Transformers/safetensors folder: not a GGUF, must be rejected.
    tf = tmp_path / "tf-model"
    tf.mkdir()
    (tf / "config.json").write_text("{}")
    (tf / "model.safetensors").write_text("x")
    assert resolver._local_gguf_entry("tf", SimpleNamespace(path = str(tf))) is None

    # Standalone .gguf file: an entry with no quant sub-selection.
    bare = tmp_path / "x.gguf"
    bare.write_text("x")
    e = resolver._local_gguf_entry("x", SimpleNamespace(path = str(bare)))
    assert e is not None and e.variants == ()

    # HF-cache snapshots with a quant subdir (the nested layout the previous
    # shallow glob missed): must still be detected.
    repo = tmp_path / "models--org--repo"
    (repo / "snapshots" / "abc" / "BF16").mkdir(parents = True)
    (repo / "snapshots" / "abc" / "BF16" / "model-BF16.gguf").write_text("x")
    e2 = resolver._local_gguf_entry("org/repo", SimpleNamespace(path = str(repo)))
    assert e2 is not None and e2.variants


def _entry(loader_id, *variants):
    # load_path == loader_id for tests; production stores a concrete local path.
    return resolver._LocalGgufEntry(loader_id, loader_id, tuple(variants))


def test_resolver_matches_and_splits_variant(monkeypatch):
    monkeypatch.setattr(
        resolver,
        "_build_index",
        lambda: {"unsloth/b-gguf": _entry("unsloth/B-GGUF", "UD-Q5_K_XL", "Q4_K_M")},
    )
    resolver._scan = (0.0, {})  # force a rescan
    # A requested variant present on disk resolves (case-insensitive).
    assert resolver.resolve_local_gguf("unsloth/B-GGUF:ud-q5_k_xl") == (
        "unsloth/B-GGUF",
        "UD-Q5_K_XL",
        "unsloth/B-GGUF",
    )
    # A bare id resolves to a concrete local quant, never a remote one.
    assert resolver.resolve_local_gguf("unsloth/B-GGUF") == (
        "unsloth/B-GGUF",
        "UD-Q5_K_XL",
        "unsloth/B-GGUF",
    )
    # A variant that is not on disk must not resolve (no remote download).
    assert resolver.resolve_local_gguf("unsloth/B-GGUF:Q8_0") is None
    assert resolver.resolve_local_gguf("totally/unknown") is None
    assert resolver.resolve_local_gguf("") is None


def test_resolver_failsafe_on_internal_error(monkeypatch):
    # Resolution is best-effort: any internal failure must fall through to None
    # so the request still serves the loaded model instead of 500-ing. The hook
    # calls resolve_local_gguf without its own guard, so the guard lives here.
    def boom():
        raise RuntimeError("scan blew up")

    monkeypatch.setattr(resolver, "_build_index", boom)
    resolver._scan = (0.0, {})
    assert resolver.resolve_local_gguf("unsloth/B-GGUF") is None


def test_resolver_nonstring_model_is_failsafe():
    # /v1/completions and /v1/embeddings pass body.get("model") straight through,
    # so a non-string must not raise on .strip().
    assert resolver.resolve_local_gguf(123) is None
    assert resolver.resolve_local_gguf({"a": 1}) is None
    assert resolver.resolve_local_gguf(None) is None


def test_resolver_exact_id_with_colon_wins(monkeypatch):
    # A local id that itself contains a colon (e.g. a Windows path) must match
    # exactly rather than being split at the drive-letter colon.
    win = r"C:\models\foo.gguf"
    monkeypatch.setattr(resolver, "_build_index", lambda: {win.lower(): _entry(win)})
    resolver._scan = (0.0, {})
    assert resolver.resolve_local_gguf(win) == (win, None, win)


# ── settings coercion ───────────────────────────────────────────────


def test_setting_coercion():
    assert settings._coerce_bool("on") is True
    assert settings._coerce_bool("off") is False
    assert settings._coerce_bool("garbage") is None
    assert settings._coerce_int("5") == 5
    assert settings._coerce_int(-3) == 0
    assert settings._coerce_int("nope") is None


# ── idle keep-warm ──────────────────────────────────────────────────


def test_idle_loop_does_not_unload_freshly_loaded_model(monkeypatch):
    # Server idle far longer than the TTL, then a model is loaded: the load
    # transition stamps activity so the next poll must not unload it.
    import time
    from core.inference import llama_keepwarm as kw

    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 1)
    kw._inflight = 0
    kw._last_active = time.monotonic() - 3600

    unloads = []
    backend = _FakeBackend("unsloth/Fresh-GGUF")
    backend.unload_model = lambda: unloads.append(1)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)

    async def _drive():
        task = asyncio.create_task(kw.idle_unload_loop(poll_seconds = 0.01))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_drive())
    assert unloads == []


def test_audio_generate_is_tracked_as_inference_path():
    # Direct GGUF TTS uses the llama backend and can outlive the idle TTL, so
    # the keep-warm middleware must count it as in-flight inference.
    from core.inference.llama_keepwarm import _is_inference_path

    assert _is_inference_path("/api/inference/audio/generate") is True
    assert _is_inference_path("/v1/chat/completions") is True
    assert _is_inference_path("/api/inference/models/list") is False


def test_idle_loop_does_not_unload_while_request_inflight(monkeypatch):
    # An in-flight request (inflight > 0) must protect the model from unload
    # even when it has been idle by wall-clock past the TTL.
    import time
    from core.inference import llama_keepwarm as kw

    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 0.01)
    monkeypatch.setattr(kw, "_inflight", 1)
    monkeypatch.setattr(kw, "_last_active", time.monotonic() - 3600)

    unloads = []
    backend = _FakeBackend("unsloth/Active-GGUF")
    backend.unload_model = lambda: unloads.append(1)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)

    async def _drive():
        task = asyncio.create_task(kw.idle_unload_loop(poll_seconds = 0.01))
        await asyncio.sleep(0.08)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_drive())
    assert unloads == []


# ── per-model launch overrides ──────────────────────────────────────


def test_auto_switch_applies_model_override(monkeypatch):
    # A configured model loads with its saved launch flags, not bare defaults.
    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("unsloth/B-GGUF", "Q4_K_M", "unsloth/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    monkeypatch.setattr(
        settings,
        "get_model_override",
        lambda model_id: {"llama_extra_args": ["--n-gpu-layers", "20"], "max_seq_length": 4096},
    )

    _run_hook("unsloth/B-GGUF")
    assert len(rec.calls) == 1
    req = rec.calls[0]
    assert req.model_path == "unsloth/B-GGUF"
    assert req.gguf_variant == "Q4_K_M"
    assert req.llama_extra_args == ["--n-gpu-layers", "20"]
    assert req.max_seq_length == 4096


def test_auto_switch_applies_partial_override(monkeypatch):
    # Only llama_extra_args is configured: it is applied, max_seq_length stays default.
    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("unsloth/B-GGUF", "Q4_K_M", "unsloth/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    monkeypatch.setattr(
        settings, "get_model_override", lambda model_id: {"llama_extra_args": ["--flash-attn"]}
    )

    _run_hook("unsloth/B-GGUF")
    req = rec.calls[0]
    assert req.llama_extra_args == ["--flash-attn"]
    assert req.max_seq_length == 0  # untouched default


def _mock_override_store(monkeypatch):
    """Back the override read + atomic-merge write with an in-memory dict."""
    import storage.studio_db as db

    store = {}

    def _merge_entry(key, entry_key, entry_value):
        current = dict(store.get(key) or {})
        if entry_value:
            current[entry_key] = entry_value
        else:
            current.pop(entry_key, None)
        store[key] = current
        return current

    monkeypatch.setattr(db, "upsert_app_setting_map_entry", _merge_entry)
    monkeypatch.setattr(db, "get_app_setting", lambda k, default = None: store.get(k, default))
    settings._cache.clear()
    return store


def test_model_override_roundtrip(monkeypatch):
    _mock_override_store(monkeypatch)

    settings.set_model_override(
        "unsloth/B-GGUF", llama_extra_args = ["--n-gpu-layers", "20"], max_seq_length = 4096
    )
    assert settings.get_model_override("unsloth/B-GGUF") == {
        "llama_extra_args": ["--n-gpu-layers", "20"],
        "max_seq_length": 4096,
    }
    # An override with no fields removes the entry rather than storing an empty one.
    settings.set_model_override("unsloth/B-GGUF", llama_extra_args = [], max_seq_length = None)
    assert settings.get_model_override("unsloth/B-GGUF") == {}
    assert settings.get_model_overrides() == {}


def test_override_route_rejects_managed_flag_and_removes(monkeypatch):
    import routes.settings as settings_route
    from fastapi import HTTPException

    _mock_override_store(monkeypatch)

    # A managed/denylisted llama-server flag is rejected with 400, not 500.
    bad = settings_route.ModelOverridePayload(
        model_id = "unsloth/B-GGUF", llama_extra_args = ["--port", "1234"]
    )
    with pytest.raises(HTTPException) as excinfo:
        settings_route.update_openai_auto_switch_override(bad, "tester")
    assert excinfo.value.status_code == 400

    # A valid override is stored, then an empty payload removes it through the route.
    ok = settings_route.ModelOverridePayload(
        model_id = "unsloth/B-GGUF", llama_extra_args = ["--flash-attn"], max_seq_length = 4096
    )
    resp = settings_route.update_openai_auto_switch_override(ok, "tester")
    assert resp.overrides["unsloth/B-GGUF"]["max_seq_length"] == 4096
    assert "llama_extra_args" in resp.overrides["unsloth/B-GGUF"]

    empty = settings_route.ModelOverridePayload(model_id = "unsloth/B-GGUF")
    resp2 = settings_route.update_openai_auto_switch_override(empty, "tester")
    assert "unsloth/B-GGUF" not in resp2.overrides


# ── /v1/models discovery ────────────────────────────────────────────


def test_list_switch_eligible_ids(monkeypatch):
    # Several index keys map to the same entry; the listing is the distinct,
    # SORTED set of loader ids (insertion order B,A,C below differs from sorted).
    eb = resolver._LocalGgufEntry("unsloth/B-GGUF", "/p/B", ("Q4_K_M",))
    ea = resolver._LocalGgufEntry("unsloth/A-GGUF", "/p/A", ())
    ec = resolver._LocalGgufEntry("unsloth/C-GGUF", "/p/C", ())
    monkeypatch.setattr(
        resolver,
        "_build_index",
        lambda: {"unsloth/b-gguf": eb, "b-gguf": eb, "unsloth/a-gguf": ea, "unsloth/c-gguf": ec},
    )
    resolver._scan = (0.0, {})
    assert resolver.list_switch_eligible_ids() == [
        "unsloth/A-GGUF",
        "unsloth/B-GGUF",
        "unsloth/C-GGUF",
    ]


def test_v1_models_lists_eligible_only_when_enabled(monkeypatch):
    # Loaded model B (rich fields); eligible models A and B.
    monkeypatch.setattr(
        inference_route,
        "_openai_model_objects",
        lambda: [
            {
                "id": "unsloth/B-GGUF",
                "object": "model",
                "created": 1,
                "owned_by": "local",
                "context_length": 8192,
            }
        ],
    )
    monkeypatch.setattr(
        resolver, "list_switch_eligible_ids", lambda: ["unsloth/A-GGUF", "unsloth/B-GGUF"]
    )

    # Off: drop-in behavior unchanged — only the loaded model is listed.
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: False)
    data = asyncio.run(inference_route._all_openai_model_objects())
    assert [m["id"] for m in data] == ["unsloth/B-GGUF"]

    # On: loaded model first (keeps rich fields), then eligible extras with B
    # deduped and listed minimally.
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    data = asyncio.run(inference_route._all_openai_model_objects())
    assert [m["id"] for m in data] == ["unsloth/B-GGUF", "unsloth/A-GGUF"]
    loaded = next(m for m in data if m["id"] == "unsloth/B-GGUF")
    extra = next(m for m in data if m["id"] == "unsloth/A-GGUF")
    assert loaded["context_length"] == 8192
    assert "context_length" not in extra


def test_v1_models_retrieve_eligible_only_when_enabled(monkeypatch):
    from fastapi import HTTPException

    monkeypatch.setattr(
        inference_route,
        "_openai_model_objects",
        lambda: [{"id": "unsloth/B-GGUF", "object": "model", "created": 1, "owned_by": "local"}],
    )
    monkeypatch.setattr(
        resolver, "list_switch_eligible_ids", lambda: ["unsloth/A-GGUF", "unsloth/B-GGUF"]
    )

    # Off: an eligible-but-unloaded id is not retrievable (unchanged drop-in behavior).
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: False)
    with pytest.raises(HTTPException) as off:
        asyncio.run(inference_route.openai_retrieve_model("unsloth/A-GGUF", "tester"))
    assert off.value.status_code == 404

    # On: an eligible id returns a minimal object echoing the requested id...
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    obj = asyncio.run(inference_route.openai_retrieve_model("unsloth/A-GGUF", "tester"))
    assert obj["id"] == "unsloth/A-GGUF" and obj["object"] == "model"
    # ...but a truly unknown id still 404s.
    with pytest.raises(HTTPException) as unknown:
        asyncio.run(inference_route.openai_retrieve_model("totally/unknown", "tester"))
    assert unknown.value.status_code == 404


def test_v1_models_retrieve_is_case_insensitive(monkeypatch):
    # The resolver lowercases its index, so a retrieve that differs only in case
    # from a listed id must still hit (200), not 404. Guards the .lower() compare
    # in openai_retrieve_model against a silent revert.
    monkeypatch.setattr(
        inference_route,
        "_openai_model_objects",
        lambda: [{"id": "unsloth/B-GGUF", "object": "model", "created": 1, "owned_by": "local"}],
    )
    monkeypatch.setattr(
        resolver, "list_switch_eligible_ids", lambda: ["unsloth/A-GGUF", "unsloth/B-GGUF"]
    )
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)

    # An eligible id retrieved with different casing still resolves.
    obj = asyncio.run(inference_route.openai_retrieve_model("unsloth/a-gguf", "tester"))
    assert obj["id"] == "unsloth/A-GGUF"
    # The loaded model is also case-insensitively retrievable.
    loaded = asyncio.run(inference_route.openai_retrieve_model("UNSLOTH/B-GGUF", "tester"))
    assert loaded["id"] == "unsloth/B-GGUF"


# ── hardening: hidden models, idle/enabled coupling, count_tokens keep-warm ──


def test_index_excludes_hidden_models(tmp_path, monkeypatch):
    # The llama.cpp validation probe and RAG embedding weights are hidden from
    # Studio's pickers; they must never become auto-switch targets.
    from types import SimpleNamespace
    import routes.models as models_route

    normal = tmp_path / "normal-Q4_K_M.gguf"
    normal.write_bytes(b"x" * 32)
    probe = tmp_path / "stories260K.gguf"  # llama.cpp install-validation probe
    probe.write_bytes(b"x" * 32)

    def _info(mid, path):
        return SimpleNamespace(id = mid, path = str(path), model_id = mid, display_name = mid)

    monkeypatch.setattr(
        models_route,
        "_scan_models_dir",
        lambda *a, **k: [_info("org/Normal-GGUF", normal), _info("ggml-org/models", probe)],
    )
    monkeypatch.setattr(models_route, "_scan_hf_cache", lambda *a, **k: [])
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: tmp_path)
    resolver._scan = (0.0, {})

    index = resolver._index()
    assert "org/normal-gguf" in index  # keys are normalized to lowercase
    assert "ggml-org/models" not in index
    # And the hidden probe cannot be auto-switched to by name.
    resolver._scan = (0.0, {})
    assert resolver.resolve_local_gguf("ggml-org/models") is None


def test_idle_disabled_when_auto_switch_off(monkeypatch):
    # "Off means unchanged": a stored idle TTL must report 0 while auto-switch is
    # off, so the idle loop and keep-warm middleware can never unload the model.
    store = {settings.AUTO_UNLOAD_IDLE_SETTING_KEY: 60}
    monkeypatch.setattr(settings, "_cached_setting", lambda k, d = None: store.get(k, d))
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: False)
    assert settings.get_auto_unload_idle_seconds() == 0
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    assert settings.get_auto_unload_idle_seconds() == 60


def test_count_tokens_is_tracked_as_inference_path():
    # count_tokens counts via the loaded tokenizer, so idle-unload must not pull
    # the model out from under it; it has to be a tracked in-flight path.
    from core.inference.llama_keepwarm import _is_inference_path

    assert _is_inference_path("/v1/messages/count_tokens") is True
    assert _is_inference_path("/api/inference/messages/count_tokens") is True
    assert _is_inference_path("/v1/messages") is True


# ── review follow-ups: bare-id reuse, responses order, in-flight tracking ──


def test_bare_id_tolerates_any_loaded_variant(monkeypatch):
    # Repo already loaded as Q4_K_M; a BARE request for the same repo (resolver
    # picks the largest local quant, Q8_0) must NOT reload a different quant.
    backend = _FakeBackend("unsloth/B-GGUF", hf_variant = "Q4_K_M")
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("unsloth/B-GGUF", "Q8_0", "unsloth/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    _run_hook("unsloth/B-GGUF")  # bare, no :VARIANT
    assert rec.calls == []
    # An explicit :VARIANT request still honors the quant (reloads to Q8_0).
    rec2 = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("unsloth/B-GGUF", "Q8_0", "unsloth/B-GGUF"),
        backend = backend,
        recorder = rec2,
    )
    _run_hook("unsloth/B-GGUF:Q8_0")
    assert len(rec2.calls) == 1


def test_responses_hook_runs_after_input_validation():
    # A request that 400s on empty input must not have triggered a model load,
    # so the auto-switch hook must come after the input-validation guard.
    import inspect

    src = inspect.getsource(inference_route.openai_responses)
    assert "No input provided" in src
    assert src.index("No input provided") < src.index("_maybe_auto_switch_model")


def test_keepwarm_tracks_inflight_when_enabled_even_if_idle_zero(monkeypatch):
    # In-flight must be counted whenever auto-switch is on, even with idle TTL 0,
    # so enabling idle mid-stream cannot unload an in-flight request.
    from core.inference import llama_keepwarm as kw

    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    kw._inflight = 0
    seen = {}

    async def app(scope, receive, send):
        seen["inflight"] = kw._inflight
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})

    async def drive():
        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(_m):
            pass

        scope = {"type": "http", "path": "/v1/chat/completions", "method": "POST", "headers": []}
        await kw.LlamaKeepWarmMiddleware(app)(scope, receive, send)

    asyncio.run(drive())
    assert seen["inflight"] == 1  # counted despite idle TTL being 0
    assert kw._inflight == 0  # balanced after completion


# ── review follow-ups: OFF-state body, swap guard, alias reload, always-track ──


def _bad_body_request():
    import json as _json
    class _BadReq:
        async def json(self):
            raise _json.JSONDecodeError("expecting value", "", 0)

    return _BadReq()


def test_completions_malformed_body_503_not_500_when_unloaded(monkeypatch):
    # OFF + nothing loaded + unparseable body must still 503 (pre-feature
    # behavior), not 500 from the early body read.
    from fastapi import HTTPException

    backend = _FakeBackend(None)
    _wire(
        monkeypatch,
        enabled = False,
        resolves_to = None,
        backend = backend,
        recorder = _LoadRecorder(backend),
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_completions(_bad_body_request(), "tester"))
    assert exc.value.status_code == 503


def test_embeddings_malformed_body_503_not_500_when_unloaded(monkeypatch):
    from fastapi import HTTPException

    backend = _FakeBackend(None)
    _wire(
        monkeypatch,
        enabled = False,
        resolves_to = None,
        backend = backend,
        recorder = _LoadRecorder(backend),
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_embeddings(_bad_body_request(), "tester"))
    assert exc.value.status_code == 503


def test_non_string_model_falls_through_without_error(monkeypatch):
    # A non-string model (e.g. {"model": 123} on a raw-body endpoint) must be
    # treated as absent, never raising in the membership checks, even when a stash
    # exists from idle-unload.
    from core.inference import llama_keepwarm as kw

    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = True, resolves_to = None, backend = backend, recorder = rec)
    monkeypatch.setattr(kw, "_last_unloaded_model", ("unsloth/A-GGUF", None))
    asyncio.run(inference_route._maybe_auto_switch_model(123, object(), "tester"))
    assert rec.calls == []  # no load, no TypeError


def test_anthropic_validates_max_tokens_before_auto_switch():
    # An Anthropic request missing max_tokens must 400 before the hook runs, so an
    # invalid request never triggers a model load. Asserted on the source order.
    import inspect

    src = inspect.getsource(inference_route.anthropic_messages)
    assert "_maybe_auto_switch_model" in src
    assert src.index("max_tokens: field required") < src.index("_maybe_auto_switch_model")


def test_alias_reloads_model_freed_by_idle_unload_with_quant(monkeypatch):
    # After idle-unload frees the model, an unknown/alias name (resolves to None)
    # reloads what was freed, including the exact quant, instead of 503-ing.
    from core.inference import llama_keepwarm as kw

    backend = _FakeBackend(None)  # idle-unload emptied the backend
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = True, resolves_to = None, backend = backend, recorder = rec)
    monkeypatch.setattr(kw, "_inflight", 0)
    monkeypatch.setattr(kw, "_last_unloaded_model", ("unsloth/A-GGUF", "Q4_K_M"))
    _run_hook("gpt-4o-mini")
    assert len(rec.calls) == 1
    assert rec.calls[0].model_path == "unsloth/A-GGUF"
    assert rec.calls[0].gguf_variant == "Q4_K_M"  # exact freed quant restored


def test_alias_does_not_reload_when_model_already_loaded(monkeypatch):
    # The reload only triggers on an empty backend; with something loaded, an
    # unknown name still falls through (drop-in) without resurrecting the stash.
    from core.inference import llama_keepwarm as kw

    backend = _FakeBackend("unsloth/B-GGUF")
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = True, resolves_to = None, backend = backend, recorder = rec)
    monkeypatch.setattr(kw, "_last_unloaded_model", ("unsloth/A-GGUF", None))
    _run_hook("gpt-4o-mini")
    assert rec.calls == []


def test_idle_loop_does_not_unload_while_request_pending(monkeypatch):
    # A request that has marked itself pending (waiting on the unload gate) but not
    # yet started must keep the idle loop from unloading the model.
    from core.inference import llama_keepwarm as kw

    monkeypatch.setattr(kw, "_inflight", 0)
    monkeypatch.setattr(kw, "_pending", 0)
    monkeypatch.setattr(kw, "_last_active", 0.0)  # far past any TTL
    kw._note_pending()
    try:
        assert kw._is_idle(1.0) is False  # pending request blocks unload
    finally:
        kw._note_unpending()
    assert kw._is_idle(1.0) is True  # cleared once it is no longer pending


def test_keepwarm_tracks_inflight_even_when_auto_switch_off(monkeypatch):
    # A stream that starts while the feature is OFF must still be counted, so
    # enabling idle-unload mid-stream cannot unload it.
    from core.inference import llama_keepwarm as kw

    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: False)
    monkeypatch.setattr(kw, "_inflight", 0)
    seen = {}

    async def app(scope, receive, send):
        seen["inflight"] = kw._inflight
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})

    async def drive():
        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(_m):
            pass

        scope = {"type": "http", "path": "/v1/chat/completions", "method": "POST", "headers": []}
        await kw.LlamaKeepWarmMiddleware(app)(scope, receive, send)

    asyncio.run(drive())
    assert seen["inflight"] == 1  # tracked despite the feature being off
    assert kw._inflight == 0


def test_build_index_covers_legacy_default_lmstudio_and_custom_roots(monkeypatch, tmp_path):
    # _build_index must scan the same roots the model picker lists, else a model
    # the UI shows is silently served as the loaded one. Verify each is consulted.
    from pathlib import Path
    import routes.models as models_route
    from utils import paths as upaths
    import storage.studio_db as studio_db

    scanned = []
    monkeypatch.setattr(
        models_route,
        "_scan_models_dir",
        lambda d, limit = None: scanned.append(("models", str(Path(d).resolve()))) or [],
    )
    monkeypatch.setattr(
        models_route,
        "_scan_hf_cache",
        lambda d: scanned.append(("hf", str(Path(d).resolve()))) or [],
    )
    monkeypatch.setattr(
        models_route,
        "_scan_lmstudio_dir",
        lambda d: scanned.append(("lm", str(Path(d).resolve()))) or [],
    )
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: tmp_path / "active")
    monkeypatch.setattr(models_route, "_is_hidden_model", lambda *a, **k: False)
    monkeypatch.setattr(upaths, "legacy_hf_cache_dir", lambda: tmp_path / "legacy")
    monkeypatch.setattr(upaths, "hf_default_cache_dir", lambda: tmp_path / "default")
    monkeypatch.setattr(upaths, "lmstudio_model_dirs", lambda: [tmp_path / "lmstudio"])
    monkeypatch.setattr(
        studio_db, "list_scan_folders", lambda: [{"path": str(tmp_path / "custom")}]
    )
    for sub in ("active", "legacy", "default", "lmstudio", "custom"):
        (tmp_path / sub).mkdir()

    resolver._build_index()

    hf = {p for k, p in scanned if k == "hf"}
    lm = {p for k, p in scanned if k == "lm"}
    assert str((tmp_path / "legacy").resolve()) in hf
    assert str((tmp_path / "default").resolve()) in hf
    assert str((tmp_path / "custom").resolve()) in hf
    assert str((tmp_path / "lmstudio").resolve()) in lm


# ── gemini round: list-body 400, non-POST not tracked ──


def _json_body_request(payload):
    class _Req:
        async def json(self):
            return payload

    return _Req()


def test_completions_list_body_is_400_not_500(monkeypatch):
    # A valid JSON non-dict body (e.g. a list) on a loaded backend is a clean 400,
    # not a 500 from body.get(...).
    from fastapi import HTTPException

    backend = _FakeBackend("unsloth/A-GGUF")  # loaded
    _wire(
        monkeypatch,
        enabled = False,
        resolves_to = None,
        backend = backend,
        recorder = _LoadRecorder(backend),
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_completions(_json_body_request([]), "tester"))
    assert exc.value.status_code == 400


def test_embeddings_list_body_is_400_not_500(monkeypatch):
    from fastapi import HTTPException

    backend = _FakeBackend("unsloth/A-GGUF")
    _wire(
        monkeypatch,
        enabled = False,
        resolves_to = None,
        backend = backend,
        recorder = _LoadRecorder(backend),
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_embeddings(_json_body_request([]), "tester"))
    assert exc.value.status_code == 400


def test_middleware_ignores_non_post(monkeypatch):
    # CORS preflight (OPTIONS) on an inference path must not be tracked as in-flight.
    from core.inference import llama_keepwarm as kw

    monkeypatch.setattr(kw, "_inflight", 0)
    seen = {}

    async def app(scope, receive, send):
        seen["inflight"] = kw._inflight
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def drive():
        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(_m):
            pass

        scope = {"type": "http", "path": "/v1/chat/completions", "method": "OPTIONS", "headers": []}
        await kw.LlamaKeepWarmMiddleware(app)(scope, receive, send)

    asyncio.run(drive())
    assert seen["inflight"] == 0  # OPTIONS not counted
    assert kw._inflight == 0


# ── review round 4: swap guard, idle variant identity, load-by-path, stash clear ──


def test_auto_switch_refuses_when_another_inference_is_active(monkeypatch):
    # A cross-model swap must 409 (not kill) while another inference request is in
    # flight; the requesting call itself is excluded from the count.
    from fastapi import HTTPException
    from core.inference import llama_keepwarm as kw

    backend = _FakeBackend("org/A-GGUF", hf_variant = "Q4_K_M")
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("/p/B", "Q8_0", "org/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    monkeypatch.setattr(kw, "_inflight", 2)  # this request + another active one
    monkeypatch.setattr(kw, "_pending", 0)
    with pytest.raises(HTTPException) as exc:
        _run_hook("org/B-GGUF:Q8_0")
    assert exc.value.status_code == 409
    assert rec.calls == []


def test_auto_switch_swaps_when_only_caller_is_active(monkeypatch):
    # Only the caller is in flight: nothing else to protect, so the swap proceeds.
    from core.inference import llama_keepwarm as kw

    backend = _FakeBackend("org/A-GGUF")
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("/p/B", None, "org/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    monkeypatch.setattr(kw, "_inflight", 1)
    monkeypatch.setattr(kw, "_pending", 0)
    _run_hook("org/B-GGUF")
    assert len(rec.calls) == 1
    assert rec.calls[0].model_path == "/p/B"  # concrete local path, not the repo id


def test_idle_loop_resets_timer_for_same_repo_different_variant(monkeypatch):
    # Same repo, different quant counts as a fresh model: the idle timer resets, so
    # the new variant is not unloaded before one TTL of its own.
    import time
    from core.inference import llama_keepwarm as kw

    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 0.05)
    monkeypatch.setattr(kw, "_inflight", 0)
    monkeypatch.setattr(kw, "_pending", 0)

    unloads = []
    backend = _FakeBackend("org/model-GGUF", hf_variant = "Q4_K_M")
    backend.unload_model = lambda: unloads.append(1)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)

    async def _drive():
        task = asyncio.create_task(kw.idle_unload_loop(poll_seconds = 0.01))
        await asyncio.sleep(0.03)
        assert unloads == []
        kw._last_active = time.monotonic() - 60  # force idle
        backend.hf_variant = "Q8_0"  # same id, new quant -> fresh identity
        await asyncio.sleep(0.03)
        assert unloads == []  # timer reset by the variant change, not unloaded
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_drive())


def test_generate_stream_is_tracked_as_inference_path():
    from core.inference.llama_keepwarm import _is_inference_path

    assert _is_inference_path("/api/inference/generate/stream") is True
    assert _is_inference_path("/api/inference/audio/generate") is True
    assert _is_inference_path("/v1/responses") is True


def test_successful_manual_load_clears_last_unloaded_stash():
    from core.inference import llama_keepwarm as kw

    kw._set_last_unloaded(("org/A-GGUF", "Q4_K_M"))
    assert kw.get_last_unloaded_model() == ("org/A-GGUF", "Q4_K_M")
    kw.note_model_loaded()
    assert kw.get_last_unloaded_model() is None


def test_hf_cache_entry_loads_from_local_snapshot_path(tmp_path):
    # An HF-cache repo resolves to its on-disk snapshot dir, so /load takes the
    # local branch (no repo-id download). loader_id stays the repo id.
    from types import SimpleNamespace

    repo = tmp_path / "models--org--Repo"
    snap = repo / "snapshots" / "abc123"
    snap.mkdir(parents = True)
    (snap / "model-Q4_K_M.gguf").write_bytes(b"GGUF stub")

    entry = resolver._local_gguf_entry("org/Repo", SimpleNamespace(id = "org/Repo", path = str(repo)))
    assert entry is not None
    assert entry.loader_id == "org/Repo"  # advertised id unchanged
    assert "snapshots" in entry.load_path  # loads from the concrete snapshot dir
    assert entry.load_path != "org/Repo"  # never the bare repo id
    assert entry.variants  # quant detected on disk


# ── review round 5: concurrent-swap, repo-id identity, /v1/models id, gate, 503 ──


def test_already_loaded_by_repo_id_is_not_reswapped(monkeypatch):
    # A model loaded normally has model_identifier == repo id, but the resolver
    # returns the concrete load path. A request for that repo must count as already
    # serving (no reload, no 409) even with another inference active.
    from core.inference import llama_keepwarm as kw

    backend = _FakeBackend("org/Repo-GGUF", hf_variant = "Q4_K_M")
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("/cache/models--org--Repo-GGUF/snapshots/abc", "Q4_K_M", "org/Repo-GGUF"),
        backend = backend,
        recorder = rec,
    )
    monkeypatch.setattr(kw, "_inflight", 2)
    monkeypatch.setattr(kw, "_pending", 0)
    _run_hook("org/Repo-GGUF:Q4_K_M")  # exact quant
    _run_hook("org/Repo-GGUF")  # bare id
    assert rec.calls == []


def test_auto_switch_advertises_repo_id_after_load(monkeypatch):
    # After a load-by-path, the backend advertises the repo id (override key), not
    # the concrete path, so /v1/models and the idle stash stay name-based.
    backend = _FakeBackend("org/A-GGUF")
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("/p/B-snapshot", "Q8_0", "org/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    _run_hook("org/B-GGUF:Q8_0")
    assert rec.calls[0].model_path == "/p/B-snapshot"  # loaded by concrete path
    assert backend._openai_advertised_id == "org/B-GGUF"  # advertised by repo id


def test_concurrent_same_target_requests_load_once(monkeypatch):
    # Two concurrent requests for the same unloaded model must load once, not each
    # 409 the other. Simulate the second request already waiting (registered) while
    # the first runs the hook with _inflight counting both.
    from core.inference import llama_keepwarm as kw

    backend = _FakeBackend("org/A-GGUF")
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("/p/B", "Q8_0", "org/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    monkeypatch.setattr(kw, "_inflight", 2)  # both same-target requests counted
    monkeypatch.setattr(kw, "_pending", 0)
    inference_route._note_switch_waiter(inference_route._switch_key("org/B-GGUF", "Q8_0"), 1)
    _run_hook("org/B-GGUF:Q8_0")
    assert len(rec.calls) == 1  # loads once, no 409


def test_swap_still_refused_when_other_request_targets_different_model(monkeypatch):
    # A concurrent request heading to a different target still blocks the swap: the
    # same-target exclusion must not swallow a genuinely conflicting request.
    from fastapi import HTTPException
    from core.inference import llama_keepwarm as kw

    backend = _FakeBackend("org/A-GGUF")
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("/p/B", "Q8_0", "org/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    monkeypatch.setattr(kw, "_inflight", 2)
    monkeypatch.setattr(kw, "_pending", 0)
    inference_route._note_switch_waiter(inference_route._switch_key("org/C-GGUF", "Q4_K_M"), 1)
    with pytest.raises(HTTPException) as exc:
        _run_hook("org/B-GGUF:Q8_0")
    assert exc.value.status_code == 409
    assert rec.calls == []


def test_v1_models_advertises_repo_id_not_load_path(monkeypatch):
    # /v1/models must report the advertised repo id, never the host load path.
    from types import SimpleNamespace

    llama = _FakeBackend("/cache/models--org--Repo/snapshots/abc")
    llama._openai_advertised_id = "org/Repo-GGUF"
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: llama)
    monkeypatch.setattr(
        inference_route, "get_inference_backend", lambda: SimpleNamespace(active_model_name = None)
    )
    objects = inference_route._openai_model_objects()
    assert [o["id"] for o in objects] == ["org/Repo-GGUF"]


def test_idle_alias_reload_preserves_override_via_advertised_id(monkeypatch):
    # The idle stash carries (load_path, quant, advertised_id). An alias reload must
    # look up the override by the advertised repo id, not the concrete load path,
    # so the user's saved launch flags survive the unload/reload.
    from core.inference import llama_keepwarm as kw

    backend = _FakeBackend(None)  # idle-unload emptied the slot
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = True, resolves_to = None, backend = backend, recorder = rec)
    monkeypatch.setattr(kw, "_inflight", 0)
    monkeypatch.setattr(kw, "_last_unloaded_model", ("/cache/snap/A", "Q4_K_M", "org/A-GGUF"))
    overrides = {"org/A-GGUF": {"max_seq_length": 8192}}
    monkeypatch.setattr(settings, "get_model_override", lambda mid: overrides.get(mid, {}))
    _run_hook("gpt-4o-mini")
    assert rec.calls[0].model_path == "/cache/snap/A"  # reloads the freed path
    assert rec.calls[0].gguf_variant == "Q4_K_M"
    assert rec.calls[0].max_seq_length == 8192  # override keyed by repo id, not path


def test_load_route_holds_lifecycle_gate(monkeypatch):
    # Lock the manual /load gate against silent revert: the route must wrap the
    # load in inference_lifecycle_gate so idle-unload can't fire mid-load.
    import inspect

    src = inspect.getsource(inference_route.load_model)
    assert "inference_lifecycle_gate" in src
    assert "_load_model_impl" in src


def _anthropic_payload(max_tokens = None):
    from models.inference import AnthropicMessagesRequest, AnthropicMessage
    return AnthropicMessagesRequest(
        model = "claude-x",
        max_tokens = max_tokens,
        messages = [AnthropicMessage(role = "user", content = "hi")],
    )


def test_anthropic_503_when_unloaded_and_auto_switch_off(monkeypatch):
    # Default-off parity: unloaded backend + auto-switch off 503s before the
    # max_tokens 400, exactly as the pre-feature endpoint did.
    from fastapi import HTTPException

    backend = _FakeBackend(None)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: False)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.anthropic_messages(_anthropic_payload(), object(), "tester"))
    assert exc.value.status_code == 503


def test_anthropic_400_when_auto_switch_on_and_max_tokens_missing(monkeypatch):
    # With auto-switch on, request-shape validation runs first: a missing
    # max_tokens still 400s before any load is attempted.
    from fastapi import HTTPException

    backend = _FakeBackend(None)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.anthropic_messages(_anthropic_payload(), object(), "tester"))
    assert exc.value.status_code == 400


# ── review round 6: concurrency ordering, external untrack, unload gate, ids ──


def test_pending_same_target_request_does_not_force_409(monkeypatch):
    # A second same-target request blocked in the middleware (pending, not yet
    # generating) must not make the first request 409: pending is excluded.
    from core.inference import llama_keepwarm as kw

    backend = _FakeBackend("org/A-GGUF")
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("/p/B", "Q8_0", "org/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    monkeypatch.setattr(kw, "_inflight", 1)  # just the caller
    monkeypatch.setattr(kw, "_pending", 1)  # second request blocked in middleware
    _run_hook("org/B-GGUF:Q8_0")
    assert len(rec.calls) == 1  # loads once, no 409


def test_concurrent_same_target_loads_once_while_other_still_resolving(monkeypatch):
    # The real middleware counts a concurrent same-model request as in-flight
    # before it resolves and registers a target waiter. The raw-request waiter,
    # registered before resolve, must still exclude it so the first request loads.
    from core.inference import llama_keepwarm as kw

    backend = _FakeBackend("org/A-GGUF")
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("/p/B", "Q8_0", "org/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    monkeypatch.setattr(kw, "_inflight", 2)  # caller + a still-resolving twin
    monkeypatch.setattr(kw, "_pending", 0)
    # The twin has only registered its raw requested model (not yet a target waiter).
    inference_route._note_request_waiter(inference_route._request_waiter_key("org/B-GGUF:Q8_0"), 1)
    _run_hook("org/B-GGUF:Q8_0")
    assert len(rec.calls) == 1  # loads once, no 409


def test_external_untrack_decrements_inflight_and_is_idempotent():
    from core.inference import llama_keepwarm as kw

    kw._inflight = 2
    scope = {"type": "http"}
    kw.untrack_current_request(scope)
    assert kw._inflight == 1
    assert scope.get(kw._UNTRACKED_SCOPE_KEY) is True
    kw.untrack_current_request(scope)  # idempotent: no further decrement
    assert kw._inflight == 1
    kw._inflight = 0


def test_manual_unload_interrupts_even_while_inference_active(monkeypatch):
    # A manual /unload is a deliberate action: it tears down immediately even with
    # a request in flight (only the automatic idle loop defers). No 409.
    from core.inference import llama_keepwarm as kw
    from models.inference import UnloadRequest

    backend = _FakeBackend("org/A-GGUF")
    backend.is_active = True
    backend.unload_model = lambda: setattr(backend, "is_loaded", False)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(inference_route, "is_registered_native_path_label", lambda *a: False)
    monkeypatch.setattr(kw, "_inflight", 1)  # another request streaming
    monkeypatch.setattr(kw, "_pending", 0)
    resp = asyncio.run(
        inference_route.unload_model(UnloadRequest(model_path = "org/A-GGUF"), "tester")
    )
    assert resp.status == "unloaded"
    assert not backend.is_loaded  # torn down despite the active request


def test_auto_switch_refuses_when_unsloth_stream_active(monkeypatch):
    # The GGUF slot is empty but an Unsloth model is streaming (counted in-flight).
    # _load_model_impl would unload it, so auto-switch must 409, not only when a
    # GGUF is loaded.
    from fastapi import HTTPException
    from core.inference import llama_keepwarm as kw

    backend = _FakeBackend(None)  # no GGUF loaded
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("/p/B", "Q8_0", "org/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    monkeypatch.setattr(kw, "_inflight", 2)  # an Unsloth stream + this request
    monkeypatch.setattr(kw, "_pending", 0)
    with pytest.raises(HTTPException) as exc:
        _run_hook("org/B-GGUF:Q8_0")
    assert exc.value.status_code == 409
    assert rec.calls == []  # the active Unsloth model is not torn down


def test_public_model_id_prefers_advertised_over_path():
    backend = _FakeBackend("/cache/models--org--Repo/snapshots/abc")
    backend._openai_advertised_id = "org/Repo-GGUF"
    assert inference_route._llama_public_model_id(backend) == "org/Repo-GGUF"
    backend._openai_advertised_id = None
    # No advertised id: falls back to the identifier, then the explicit fallback.
    assert (
        inference_route._llama_public_model_id(backend) == "/cache/models--org--Repo/snapshots/abc"
    )
    backend.model_identifier = None
    assert inference_route._llama_public_model_id(backend, "req") == "req"


def test_chat_validates_non_system_message_before_auto_switch():
    # A system-only chat must be rejected before the hook so an invalid request
    # never swaps the resident model. Asserted on source order.
    import inspect
    src = inspect.getsource(inference_route.openai_chat_completions)
    assert src.index("At least one non-system message is required.") < src.index(
        "_maybe_auto_switch_model"
    )


def test_chat_untracks_external_provider_before_proxy():
    # The external-provider branch must untrack the request before proxying so its
    # stream can't block a concurrent local auto-switch.
    import inspect
    src = inspect.getsource(inference_route.openai_chat_completions)
    assert src.index("untrack_current_request") < src.index("_proxy_to_external_provider")


# ── round 7: API-initiated training defers to active inference, UI does not ──


def test_authenticated_via_api_key_detects_key_vs_session():
    from fastapi.security import HTTPAuthorizationCredentials
    from auth.authentication import authenticated_via_api_key, API_KEY_PREFIX

    key = HTTPAuthorizationCredentials(scheme = "Bearer", credentials = API_KEY_PREFIX + "abc")
    jwt = HTTPAuthorizationCredentials(scheme = "Bearer", credentials = "eyJhbGciOiJ.session")
    assert asyncio.run(authenticated_via_api_key(key)) is True
    assert asyncio.run(authenticated_via_api_key(jwt)) is False


def _training_request():
    from models.training import TrainingStartRequest
    return TrainingStartRequest(
        model_name = "unsloth/test", training_type = "LoRA/QLoRA", format_type = "alpaca"
    )


def test_api_training_refused_while_inference_active(monkeypatch):
    # API-key caller: training is refused with 409 while a request streams, so it
    # can't free VRAM by unloading the chat model out from under the stream.
    from fastapi import HTTPException
    from core.inference import llama_keepwarm as kw
    import routes.training as training_route

    monkeypatch.setattr(kw, "_inflight", 1)
    monkeypatch.setattr(kw, "_pending", 0)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            training_route.start_training(
                _training_request(), current_subject = "t", via_api_key = True
            )
        )
    assert exc.value.status_code == 409


def test_ui_training_not_blocked_by_active_inference(monkeypatch):
    # UI (session auth) caller: the API guard is skipped, so training proceeds past
    # it even with inference active (here it hits the normal already-active path).
    from types import SimpleNamespace
    from core.inference import llama_keepwarm as kw
    import routes.training as training_route

    monkeypatch.setattr(kw, "_inflight", 1)
    monkeypatch.setattr(kw, "_pending", 0)
    fake = SimpleNamespace(is_training_active = lambda: True, current_job_id = "job-1")
    monkeypatch.setattr(training_route, "get_training_backend", lambda: fake)
    resp = asyncio.run(
        training_route.start_training(_training_request(), current_subject = "t", via_api_key = False)
    )
    assert resp.status == "error" and "already" in (resp.error or "").lower()


# ── UNSLOTH_MODEL_IDLE_TTL env override (borrowed from PR 6517) ──


def test_env_idle_ttl_standalone_when_no_stored_value(monkeypatch):
    # With nothing stored, the env var enables idle-unload even while auto-switch
    # is off (headless/ops default), and the UI reader reflects it.
    monkeypatch.setattr(settings, "_cached_setting", lambda k, d = None: d)  # nothing stored
    monkeypatch.setenv("UNSLOTH_MODEL_IDLE_TTL", "600")
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: False)
    assert settings.get_auto_unload_idle_seconds() == 600
    assert settings.get_stored_auto_unload_idle_seconds() == 600


def test_stored_idle_value_overrides_env_and_stays_gated(monkeypatch):
    # An explicit stored value wins over the env default and remains gated on the
    # auto-switch toggle.
    store = {settings.AUTO_UNLOAD_IDLE_SETTING_KEY: 30}
    monkeypatch.setattr(settings, "_cached_setting", lambda k, d = None: store.get(k, d))
    monkeypatch.setenv("UNSLOTH_MODEL_IDLE_TTL", "600")
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    assert settings.get_auto_unload_idle_seconds() == 30  # stored wins, not env
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: False)
    assert settings.get_auto_unload_idle_seconds() == 0  # explicit value still gated off


def test_env_idle_ttl_invalid_is_ignored(monkeypatch):
    monkeypatch.setattr(settings, "_cached_setting", lambda k, d = None: d)
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: False)
    monkeypatch.setenv("UNSLOTH_MODEL_IDLE_TTL", "not-a-number")
    assert settings.get_auto_unload_idle_seconds() == 0
    monkeypatch.delenv("UNSLOTH_MODEL_IDLE_TTL", raising = False)
    assert settings.get_auto_unload_idle_seconds() == 0


# ── codex/gemini round: standalone-idle reload, path-as-id, embeddings input, retrieve id ──


def test_env_idle_standalone_reloads_freed_model_with_auto_switch_off(monkeypatch):
    # C3: a standalone UNSLOTH_MODEL_IDLE_TTL (auto-switch OFF) freed the model on
    # idle; the next request must restore exactly what was freed even though the
    # resolver never runs while auto-switch is off.
    from core.inference import llama_keepwarm as kw

    backend = _FakeBackend(None)  # idle-unload emptied the slot
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = False,
        resolves_to = ("/p/B", "Q8_0", "org/B-GGUF"),  # would switch if resolver ran
        backend = backend,
        recorder = rec,
    )
    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 600)  # standalone env TTL
    monkeypatch.setattr(kw, "_inflight", 0)
    monkeypatch.setattr(kw, "_last_unloaded_model", ("/cache/snap/A", "Q4_K_M", "org/A-GGUF"))
    _run_hook("org/B-GGUF")
    # Resolver skipped (auto-switch off), so only the stash reload runs: the freed A
    # is restored, not the resolves_to target B.
    assert len(rec.calls) == 1
    assert rec.calls[0].model_path == "/cache/snap/A"
    assert rec.calls[0].gguf_variant == "Q4_K_M"


def test_no_stash_reload_when_idle_off_and_auto_switch_off(monkeypatch):
    # C3 guard: with both auto-switch and idle-unload off the hook is a pure no-op
    # and must not resurrect a stashed model (that path only serves the idle feature).
    from core.inference import llama_keepwarm as kw

    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = False, resolves_to = None, backend = backend, recorder = rec)
    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 0)
    monkeypatch.setattr(kw, "_inflight", 0)
    monkeypatch.setattr(kw, "_last_unloaded_model", ("/cache/snap/A", "Q4_K_M", "org/A-GGUF"))
    _run_hook("org/B-GGUF")
    assert rec.calls == []


def test_stash_reload_skipped_while_unsloth_model_active(monkeypatch):
    # An Unsloth/Transformers model loaded after an idle-unload leaves the GGUF slot
    # empty but is the live model; an unknown /v1 name must NOT resurrect the stale
    # GGUF stash (that reload would tear the active Unsloth model down).
    from types import SimpleNamespace
    from core.inference import llama_keepwarm as kw

    backend = _FakeBackend(None)  # GGUF slot empty
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = True, resolves_to = None, backend = backend, recorder = rec)
    monkeypatch.setattr(kw, "_inflight", 0)
    monkeypatch.setattr(kw, "_last_unloaded_model", ("/cache/snap/A", "Q4_K_M", "org/A-GGUF"))
    # An Unsloth model is the live backend.
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: SimpleNamespace(active_model_name = "unsloth/Qwen3-8B"),
    )
    _run_hook("gpt-4o-mini")
    assert rec.calls == []  # stale GGUF not reloaded over the active Unsloth model


def test_is_abs_path_id_distinguishes_path_from_repo_id():
    assert resolver._is_abs_path_id("/abs/path/model.gguf") is True
    assert resolver._is_abs_path_id("org/Repo-GGUF") is False
    assert resolver._is_abs_path_id("Repo") is False


def test_advertised_loader_id_prefers_alias_over_abs_path():
    # C1: the ./models and LM Studio scanners report the on-disk path as info.id.
    from types import SimpleNamespace

    f = resolver._advertised_loader_id
    # An absolute-path id falls back to the first non-path alias.
    assert (
        f(SimpleNamespace(id = "/home/me/models/x", model_id = "org/X-GGUF", display_name = "X"))
        == "org/X-GGUF"
    )
    # No alias available: keep the path (still resolvable by it).
    assert (
        f(SimpleNamespace(id = "/home/me/models/x", model_id = None, display_name = None))
        == "/home/me/models/x"
    )
    # A normal repo id is advertised as-is.
    assert (
        f(SimpleNamespace(id = "org/X-GGUF", model_id = "org/X-GGUF", display_name = "X")) == "org/X-GGUF"
    )


def test_index_advertises_alias_not_filesystem_path(tmp_path, monkeypatch):
    # C1 end-to-end: a scanner that reports the path as the id must not advertise the
    # host path in /v1/models, yet the model stays resolvable by that path too.
    from types import SimpleNamespace
    import routes.models as models_route

    gguf = tmp_path / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"x" * 32)
    info = SimpleNamespace(
        id = str(gguf),  # scanner uses the on-disk path as the id
        path = str(gguf),
        model_id = "org/Repo-GGUF",
        display_name = "Repo",
    )
    monkeypatch.setattr(models_route, "_scan_models_dir", lambda *a, **k: [info])
    monkeypatch.setattr(models_route, "_scan_hf_cache", lambda *a, **k: [])
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(models_route, "_is_hidden_model", lambda *a, **k: False)
    resolver._scan = (0.0, {})

    # The advertised id is the alias, never the absolute path.
    assert resolver.list_switch_eligible_ids() == ["org/Repo-GGUF"]
    # But the model is still resolvable by its on-disk path (an indexed alias).
    resolver._scan = (0.0, {})
    assert resolver.resolve_local_gguf(str(gguf)) is not None


def test_embeddings_input_present_helper():
    f = inference_route._embeddings_input_present
    assert f({"input": "hi"}) is True
    assert f({"input": ["a", "b"]}) is True
    assert f({"input": [1, 2, 3]}) is True
    assert f({}) is False
    assert f({"input": ""}) is False
    assert f({"input": []}) is False


def test_embeddings_rejects_missing_input_before_switch(monkeypatch):
    # C2: with auto-switch on, an embeddings request carrying no input must 400
    # before the hook, so an invalid request never swaps the resident model.
    from fastapi import HTTPException

    backend = _FakeBackend("org/A-GGUF")  # loaded
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("/p/B", "Q8_0", "org/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            inference_route.openai_embeddings(_json_body_request({"model": "org/B-GGUF"}), "tester")
        )
    assert exc.value.status_code == 400
    assert rec.calls == []  # no model switch happened


def test_retrieve_model_tolerates_non_string_id(monkeypatch):
    # G2: a model object with a non-string id (defensive) must be skipped rather
    # than crashing the .lower() compare; a valid id is still found, unknown 404s.
    from fastapi import HTTPException

    async def _objs():
        return [{"id": 123, "object": "model"}, {"id": "org/B-GGUF", "object": "model"}]

    monkeypatch.setattr(inference_route, "_all_openai_model_objects", _objs)
    obj = asyncio.run(inference_route.openai_retrieve_model("org/B-GGUF", "tester"))
    assert obj["id"] == "org/B-GGUF"
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_retrieve_model("123", "tester"))
    assert exc.value.status_code == 404


# ── 10-reviewer round: automatic-load validation asymmetry, audio, preview, idle timer ──


def _stash(monkeypatch, *, idle = 600):
    """Common setup for the standalone-idle reload paths: feature off, idle TTL on,
    an idle-freed model in the stash, nothing loaded, no in-flight requests."""
    from core.inference import llama_keepwarm as kw

    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: idle)
    monkeypatch.setattr(kw, "_inflight", 0)
    monkeypatch.setattr(kw, "_last_unloaded_model", ("/cache/snap/A", "Q4_K_M", "org/A-GGUF"))


def test_completions_prompt_present_helper():
    f = inference_route._completions_prompt_present
    assert f({"prompt": "hi"}) is True
    assert f({"prompt": ["a", "b"]}) is True
    assert f({}) is False
    assert f({"prompt": ""}) is False
    assert f({"prompt": []}) is False


def test_completions_rejects_missing_prompt_before_switch(monkeypatch):
    # #1: /v1/completions had no prompt pre-check, so a malformed request naming a
    # different downloaded GGUF loaded it before failing. Now it 400s first.
    from fastapi import HTTPException

    backend = _FakeBackend("org/A-GGUF")
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("/p/B", "Q8_0", "org/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            inference_route.openai_completions(
                _json_body_request({"model": "org/B-GGUF"}), "tester"
            )
        )
    assert exc.value.status_code == 400
    assert rec.calls == []  # no switch before rejection


def test_chat_system_only_rejected_before_idle_reload(monkeypatch):
    # #4: the chat pre-load guard only checked auto-switch; a standalone idle TTL
    # could still reload a system-only chat before the 400. Now it 400s first.
    from fastapi import HTTPException
    from models.inference import ChatCompletionRequest

    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = False, resolves_to = None, backend = backend, recorder = rec)
    _stash(monkeypatch)
    payload = ChatCompletionRequest(model = "x", messages = [{"role": "system", "content": "sys"}])
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))
    assert exc.value.status_code == 400
    assert rec.calls == []  # no reload before rejection


def test_embeddings_missing_input_rejected_before_idle_reload(monkeypatch):
    # #5: same gap on /v1/embeddings; the missing-input 400 must fire under a
    # standalone idle TTL too, not only when auto-switch is on.
    from fastapi import HTTPException

    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = False, resolves_to = None, backend = backend, recorder = rec)
    _stash(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_embeddings(_json_body_request({"model": "x"}), "tester"))
    assert exc.value.status_code == 400
    assert rec.calls == []  # no reload before rejection


def test_messages_does_not_503_before_reload_hook_when_idle_on(monkeypatch):
    # #3: /v1/messages 503'd before the reload hook when auto-switch was off, so a
    # standalone idle TTL could never restore the freed model. The early 503 now
    # defers to any automatic-load trigger, so the reload hook runs.
    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = False, resolves_to = None, backend = backend, recorder = rec)
    _stash(monkeypatch)
    # The handler proceeds past the hook to real generation (no llama-server here),
    # so tolerate the downstream failure; the reload having run is the assertion.
    try:
        asyncio.run(
            inference_route.anthropic_messages(
                _anthropic_payload(max_tokens = 16), object(), "tester"
            )
        )
    except Exception:
        pass
    assert len(rec.calls) == 1
    assert rec.calls[0].model_path == "/cache/snap/A"


def test_messages_503_gated_on_automatic_load_predicate():
    # Lock the #3 fix at the source: the early 503 must check the shared predicate.
    import inspect
    src = inspect.getsource(inference_route.anthropic_messages)
    assert "_automatic_model_load_may_run" in src


def test_raw_body_without_model_reloads_freed_model(monkeypatch):
    # #6: a raw completions/embeddings body that omits `model` passed None, which
    # skipped the idle-stash reload and 503'd. A non-empty sentinel now lets the
    # reload run while still resolving as unknown.
    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = False, resolves_to = None, backend = backend, recorder = rec)
    _stash(monkeypatch)
    body = asyncio.run(
        inference_route._auto_switch_from_request_body(
            _json_body_request({"prompt": "hi"}), "tester"
        )
    )
    assert body == {"prompt": "hi"}
    assert len(rec.calls) == 1
    assert rec.calls[0].model_path == "/cache/snap/A"
    assert rec.calls[0].gguf_variant == "Q4_K_M"


def test_audio_generate_reloads_idle_freed_model(monkeypatch):
    # #2: /audio/generate is keep-warm-tracked but had no reload hook, so an
    # idle-freed audio GGUF stayed unloaded. The hook now restores it.
    from models.inference import ChatCompletionRequest

    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = False, resolves_to = None, backend = backend, recorder = rec)
    _stash(monkeypatch)
    payload = ChatCompletionRequest(model = "x", messages = [{"role": "user", "content": "say hi"}])
    # Falls through to the non-audio backend path (no real model) after the reload;
    # tolerate that downstream failure, the reload having run is the assertion.
    try:
        asyncio.run(inference_route.generate_audio(payload, object(), "tester"))
    except Exception:
        pass
    assert len(rec.calls) == 1
    assert rec.calls[0].model_path == "/cache/snap/A"


def test_audio_generate_does_not_reload_on_invalid_request(monkeypatch):
    # The audio reload hook must run after message validation, so an empty request
    # never triggers a reload.
    from fastapi import HTTPException
    from models.inference import ChatCompletionRequest

    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = False, resolves_to = None, backend = backend, recorder = rec)
    _stash(monkeypatch)
    payload = ChatCompletionRequest(model = "x", messages = [])
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.generate_audio(payload, object(), "tester"))
    assert exc.value.status_code == 400
    assert rec.calls == []


def test_preview_scope_disables_auto_switch(monkeypatch):
    # #7: the public preview route delegates to the chat handler; a caller-supplied
    # model must not switch away from the pinned checkpoint. The scope opt-out flag
    # makes the hook a no-op.
    backend = _FakeBackend("org/A-GGUF")
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("/p/B", "Q8_0", "org/B-GGUF"),
        backend = backend,
        recorder = rec,
    )

    class _Req:
        def __init__(self):
            self.scope = {}

    req = _Req()
    inference_route.disable_openai_auto_switch_for_request(req.scope)
    asyncio.run(inference_route._maybe_auto_switch_model("org/B-GGUF", req, "tester"))
    assert rec.calls == []  # preview opt-out suppressed the switch

    # Control: a fresh request without the flag would switch.
    req2 = _Req()
    asyncio.run(inference_route._maybe_auto_switch_model("org/B-GGUF", req2, "tester"))
    assert len(rec.calls) == 1


def test_preview_chat_is_tracked_as_inference_path():
    # #8: long preview streams use the same backend; the keep-warm middleware must
    # count them so the idle loop can't unload mid-response.
    from core.inference.llama_keepwarm import _is_inference_path

    assert _is_inference_path("/p/my-run/v1/chat/completions") is True
    assert _is_inference_path("/p/my-run/ckpt-100/v1/chat/completions") is True
    assert _is_inference_path("/p/my-run/v1/models") is False


def test_untrack_does_not_reset_idle_timer():
    # #9: external-provider traffic was keeping the local GGUF warm forever because
    # untrack stamped _last_active. It must decrement in-flight without restamping.
    import time
    from core.inference import llama_keepwarm as kw

    kw._inflight = 1
    kw._last_active = time.monotonic() - 3600
    before = kw._last_active
    scope = {"type": "http"}
    kw.untrack_current_request(scope)
    assert kw._inflight == 0
    assert kw._last_active == before  # idle timer not reset by an untracked request
    kw._inflight = 0


def test_note_start_does_not_reset_idle_timer():
    # The start stamp was removed so an external request that is later untracked
    # cannot reset the timer at start either; in-flight count still protects it.
    import time
    from core.inference import llama_keepwarm as kw

    kw._inflight = 0
    kw._pending = 0
    kw._last_active = time.monotonic() - 3600
    before = kw._last_active
    kw._note_start()
    try:
        assert kw._inflight == 1
        assert kw._last_active == before  # start no longer stamps activity
        assert kw._is_idle(1.0) is False  # but in-flight still blocks unload
    finally:
        kw._note_end()  # restores _last_active stamp on completion


# ── codex review (merge round): reload-only sentinel, Anthropic tool validation ──


def test_omitted_model_does_not_resolve_to_a_named_gguf(monkeypatch):
    # Codex P2: a raw-body request that omits `model` must never run the resolver,
    # so a downloaded GGUF literally named "default" can't be switched to. The
    # resolver here would switch to B if it ran; it must not.
    backend = _FakeBackend("org/A-GGUF")  # a model is already loaded
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("/p/B", "Q8_0", "org/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    body = asyncio.run(
        inference_route._auto_switch_from_request_body(
            _json_body_request({"prompt": "hi"}), "tester"
        )
    )
    assert body == {"prompt": "hi"}
    assert rec.calls == []  # resolver skipped (would have switched to B otherwise)


def test_omitted_model_still_reloads_idle_freed_model(monkeypatch):
    # The reload-only sentinel must still restore an idle-freed model (the round-9
    # behavior), it just never runs the resolver.
    from core.inference import llama_keepwarm as kw

    backend = _FakeBackend(None)  # idle-unload emptied the slot
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = False, resolves_to = None, backend = backend, recorder = rec)
    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 600)
    monkeypatch.setattr(kw, "_inflight", 0)
    monkeypatch.setattr(kw, "_last_unloaded_model", ("/cache/snap/A", "Q4_K_M", "org/A-GGUF"))
    asyncio.run(
        inference_route._auto_switch_from_request_body(
            _json_body_request({"prompt": "hi"}), "tester"
        )
    )
    assert len(rec.calls) == 1
    assert rec.calls[0].model_path == "/cache/snap/A"


def _anthropic_payload_with_tools(tools, max_tokens = 16):
    from models.inference import AnthropicMessagesRequest, AnthropicMessage
    return AnthropicMessagesRequest(
        model = "org/B-GGUF",
        max_tokens = max_tokens,
        messages = [AnthropicMessage(role = "user", content = "hi")],
        tools = tools,
    )


def test_anthropic_invalid_tool_rejected_before_switch(monkeypatch):
    # Codex P2: a malformed client tool (no input_schema, no server-tool type) must
    # 400 before the auto-switch hook, so an invalid request never evicts the model.
    from fastapi import HTTPException

    backend = _FakeBackend("org/A-GGUF")
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("/p/B", "Q8_0", "org/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    payload = _anthropic_payload_with_tools([{"name": "broken"}])  # missing input_schema
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.anthropic_messages(payload, object(), "tester"))
    assert exc.value.status_code == 400
    assert rec.calls == []  # rejected before the model load


def test_anthropic_validates_tools_before_auto_switch():
    # Lock the order at the source: tool-shape validation precedes the hook.
    import inspect
    src = inspect.getsource(inference_route.anthropic_messages)
    assert src.index("missing required field 'input_schema'") < src.index(
        "_maybe_auto_switch_model"
    )


# ── codex review (round 2): schema-default model, Responses tool validation ──


def _chat_msg(text = "hi"):
    from models.inference import ChatMessage
    return ChatMessage(role = "user", content = text)


def _responses_payload(*, tools = None, set_model = True):
    from models.inference import ResponsesRequest

    kwargs = dict(input = "hi")
    if set_model:
        kwargs["model"] = "org/B-GGUF"
    if tools is not None:
        kwargs["tools"] = tools
    return ResponsesRequest(**kwargs)


def test_switch_model_for_payload_only_switches_when_explicit():
    # Codex P2: an omitted `model` (pydantic fills "default") must be reload-only;
    # an explicitly set model -- including a literal "default" -- is honored.
    from models.inference import ChatCompletionRequest

    omitted = ChatCompletionRequest(messages = [_chat_msg()])
    assert inference_route._switch_model_for_payload(omitted) == inference_route._RELOAD_ONLY_MODEL
    explicit_default = ChatCompletionRequest(model = "default", messages = [_chat_msg()])
    assert inference_route._switch_model_for_payload(explicit_default) == "default"
    explicit = ChatCompletionRequest(model = "org/B-GGUF", messages = [_chat_msg()])
    assert inference_route._switch_model_for_payload(explicit) == "org/B-GGUF"


def test_omitted_schema_model_skips_resolver(monkeypatch):
    # End to end: a schema request omitting `model` must not run the resolver, so a
    # GGUF named "default" is never swapped to; an explicit model still switches.
    from models.inference import ChatCompletionRequest

    backend = _FakeBackend("org/A-GGUF")
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("org/B-GGUF", "Q8_0", "org/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    omitted = ChatCompletionRequest(messages = [_chat_msg()])
    asyncio.run(
        inference_route._maybe_auto_switch_model(
            inference_route._switch_model_for_payload(omitted), object(), "tester"
        )
    )
    assert rec.calls == []  # resolver skipped
    explicit = ChatCompletionRequest(model = "org/B-GGUF", messages = [_chat_msg()])
    asyncio.run(
        inference_route._maybe_auto_switch_model(
            inference_route._switch_model_for_payload(explicit), object(), "tester"
        )
    )
    assert len(rec.calls) == 1  # explicit model still switches


def test_build_chat_request_propagates_omitted_model():
    # _build_chat_request must not turn an omitted Responses model into an explicit
    # "default", or the non-streaming chat re-check would switch on it.
    omitted = _responses_payload(set_model = False)
    chat_req = inference_route._build_chat_request(omitted, [_chat_msg()], stream = False)
    assert "model" not in chat_req.model_fields_set
    explicit = _responses_payload(set_model = True)
    chat_req2 = inference_route._build_chat_request(explicit, [_chat_msg()], stream = False)
    assert "model" in chat_req2.model_fields_set


def test_responses_invalid_function_tool_rejected_before_switch(monkeypatch):
    # Codex P2: a malformed function tool (no name) must 400 before the hook, so an
    # invalid /v1/responses request never switches or evicts the loaded model.
    from fastapi import HTTPException

    backend = _FakeBackend("org/A-GGUF")
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("org/B-GGUF", "Q8_0", "org/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    payload = _responses_payload(tools = [{"type": "function", "parameters": {}}])
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_responses(payload, object(), "tester"))
    assert exc.value.status_code == 400
    assert rec.calls == []  # rejected before the model load


def test_responses_valid_and_builtin_tools_pass_validation(monkeypatch):
    # A well-formed function tool and a built-in (non-function) tool must pass the
    # pre-switch check. Stub the hook so the test stops right after validation.
    class _Reached(Exception):
        pass

    async def _boom(*a, **k):
        raise _Reached()

    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _boom)
    payload = _responses_payload(
        tools = [{"type": "function", "name": "ok", "parameters": {}}, {"type": "web_search"}]
    )
    with pytest.raises(_Reached):
        asyncio.run(inference_route.openai_responses(payload, object(), "tester"))


def test_responses_validates_tools_before_auto_switch():
    # Lock the order at the source: tool validation precedes the switch hook.
    import inspect
    src = inspect.getsource(inference_route.openai_responses)
    assert src.index("each function tool must have a 'name'") < src.index(
        "_maybe_auto_switch_model"
    )


# ── codex review (round 3): process-wide swap gate across event loops ──


def test_swap_acquires_process_gate_before_load():
    # Lock in the structure: the process-wide gate is acquired before the load and
    # always released, so a cross-loop swap can't reach _load_model_impl unguarded.
    import inspect

    src = inspect.getsource(inference_route._maybe_auto_switch_model)
    assert src.index("_acquire_swap_gate") < src.index("_load_model_impl")
    assert "_auto_switch_process_lock.release()" in src


# ── codex review (round 4): validate modality + tool-confirmation before switch ──


def _chat_request(**kw):
    from models.inference import ChatCompletionRequest, ChatMessage
    kw.setdefault("messages", [ChatMessage(role = "user", content = "hi")])
    return ChatCompletionRequest(**kw)


def test_chat_confirm_without_stream_rejected_before_switch(monkeypatch):
    # Codex P2: confirm_tool_calls=true + stream=false + local tools is an invalid
    # shape; it must 400 before the switch hook so it can't evict the resident model.
    from fastapi import HTTPException

    backend = _FakeBackend("org/A-GGUF")
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("org/B-GGUF", "Q8_0", "org/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    payload = _chat_request(
        model = "org/B-GGUF", enable_tools = True, confirm_tool_calls = True, stream = False
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))
    assert exc.value.status_code == 400
    assert rec.calls == []


def test_chat_confirm_with_bypass_permissions_reaches_hook(monkeypatch):
    # bypass_permissions suppresses the confirm gate, so the pre-check must not fire;
    # the request should reach the switch hook (stubbed here to a sentinel).
    class _Reached(Exception):
        pass

    async def _boom(*a, **k):
        raise _Reached()

    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _boom)
    payload = _chat_request(
        model = "org/B-GGUF",
        enable_tools = True,
        confirm_tool_calls = True,
        stream = False,
        bypass_permissions = True,
    )
    with pytest.raises(_Reached):
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))


def test_require_vision_rejects_text_target_before_switch(monkeypatch):
    # Codex P2: an image request naming a different text-only GGUF must 400 before
    # the swap, so the resident vision model is not evicted for a rejected request.
    from fastapi import HTTPException

    backend = _FakeBackend("org/A-GGUF")
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("/local/B.gguf", "Q8_0", "org/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    monkeypatch.setattr(inference_route, "_target_is_vision", lambda _p: False)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            inference_route._maybe_auto_switch_model(
                "org/B-GGUF", object(), "t", require_vision = True
            )
        )
    assert exc.value.status_code == 400
    assert rec.calls == []  # rejected before the load


def test_require_vision_allows_vision_target(monkeypatch):
    backend = _FakeBackend("org/A-GGUF")
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("/local/B.gguf", "Q8_0", "org/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    monkeypatch.setattr(inference_route, "_target_is_vision", lambda _p: True)
    asyncio.run(
        inference_route._maybe_auto_switch_model("org/B-GGUF", object(), "t", require_vision = True)
    )
    assert len(rec.calls) == 1  # vision target still switches


def test_require_vision_ignores_reload_stash(monkeypatch):
    # The reload-stash path restores the model the request was already using; the
    # modality check applies only to an explicit resolver target, not a restore.
    from core.inference import llama_keepwarm as kw

    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = False, resolves_to = None, backend = backend, recorder = rec)
    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 600)
    monkeypatch.setattr(kw, "_inflight", 0)
    monkeypatch.setattr(kw, "_last_unloaded_model", ("/cache/snap/A", "Q4_K_M", "org/A-GGUF"))
    monkeypatch.setattr(
        inference_route, "_target_is_vision", lambda _p: False
    )  # would reject if used
    asyncio.run(
        inference_route._maybe_auto_switch_model("org/B-GGUF", object(), "t", require_vision = True)
    )
    assert len(rec.calls) == 1
    assert rec.calls[0].model_path == "/cache/snap/A"  # restored despite require_vision


def test_chat_validates_confirm_and_modality_before_switch():
    # Lock the order at the source: confirm-shape rejection precedes the hook, and
    # the hook rejects a non-vision target before the load.
    import inspect

    src = inspect.getsource(inference_route.openai_chat_completions)
    assert src.index("confirm_tool_calls requires stream=true") < src.index(
        "_maybe_auto_switch_model"
    )
    assert "require_vision" in src
    hook = inspect.getsource(inference_route._maybe_auto_switch_model)
    assert hook.index("require_vision") < hook.index("_load_model_impl")
    assert "does not support vision" in hook


def test_auto_switch_serializes_across_event_loops(monkeypatch):
    # Codex P2: the per-loop asyncio lock can't serialize two swaps on different
    # event loops in one process. The process-wide gate must, so the two slow loads
    # never overlap on the single model slot.
    import threading

    backend = _FakeBackend("org/A-GGUF")
    state = {"cur": 0, "max": 0}
    loaded: list = []
    slock = threading.Lock()

    async def _slow_load(
        request,
        fastapi_request,
        current_subject = None,
    ):
        with slock:
            state["cur"] += 1
            state["max"] = max(state["max"], state["cur"])
        await asyncio.sleep(0.1)  # widen the window so an unguarded race would overlap
        with slock:
            state["cur"] -= 1
            loaded.append(request.model_path)
        backend.model_identifier = request.model_path
        backend.is_loaded = True
        backend._openai_advertised_id = None

    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    monkeypatch.setattr(resolver, "resolve_local_gguf", lambda m: (m, "Q8_0", m))
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(inference_route, "_load_model_impl", _slow_load)
    monkeypatch.setattr(inference_route, "_auto_switch_waiters", {})
    monkeypatch.setattr(inference_route, "_auto_switch_request_waiters", {})

    barrier = threading.Barrier(2)

    def _run(model):
        barrier.wait()  # release both threads together so they truly race
        asyncio.run(inference_route._maybe_auto_switch_model(model, object(), "t"))

    threads = [
        threading.Thread(target = _run, args = ("org/B-GGUF",)),
        threading.Thread(target = _run, args = ("org/C-GGUF",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert state["max"] == 1  # the gate serialized the two cross-loop swaps
    assert sorted(loaded) == ["org/B-GGUF", "org/C-GGUF"]  # both still swapped


def test_acquire_swap_gate_is_cancellation_safe():
    # A waiter cancelled while waiting for the gate (client disconnect mid-swap)
    # must not leak it: after the holder releases, a fresh acquire still succeeds.
    # The to_thread(acquire) approach would leak here -- its worker thread keeps
    # acquiring after cancel, so the gate is taken but never released.
    async def main():
        await inference_route._acquire_swap_gate()  # this loop holds the gate
        try:

            async def waiter():
                await inference_route._acquire_swap_gate()

            t = asyncio.create_task(waiter())
            await asyncio.sleep(0.05)  # let it spin waiting on the held gate
            t.cancel()
            with pytest.raises(asyncio.CancelledError):
                await t
        finally:
            inference_route._auto_switch_process_lock.release()
        # Gate is free again (the cancelled waiter never acquired it).
        await asyncio.wait_for(inference_route._acquire_swap_gate(), timeout = 1)
        inference_route._auto_switch_process_lock.release()

    asyncio.run(asyncio.wait_for(main(), timeout = 5))
