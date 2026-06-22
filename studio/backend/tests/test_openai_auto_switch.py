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
    ):
        self.model_identifier = loaded_id
        self.is_loaded = loaded_id is not None
        self.hf_variant = hf_variant


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
        return None


def _wire(monkeypatch, *, enabled, resolves_to, backend, recorder):
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: enabled)
    monkeypatch.setattr(resolver, "resolve_local_gguf", lambda _m: resolves_to)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(inference_route, "load_model", recorder)


def _run_hook(model = "some/model"):
    asyncio.run(inference_route._maybe_auto_switch_model(model, object(), "tester"))


def test_flag_off_never_loads(monkeypatch):
    backend = _FakeBackend("unsloth/A-GGUF")
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = False,
        resolves_to = ("unsloth/B-GGUF", None),
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
        resolves_to = ("unsloth/a-gguf", None),
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
        resolves_to = ("unsloth/B-GGUF", "Q4_K_M"),
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
        resolves_to = ("unsloth/B-GGUF", None),
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
        resolves_to = ("unsloth/B-GGUF", None),
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
        resolves_to = ("unsloth/B-GGUF", "Q8_0"),
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
        resolves_to = ("unsloth/B-GGUF", "q4_k_m"),  # case-insensitive
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
    return resolver._LocalGgufEntry(loader_id, tuple(variants))


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
    )
    # A bare id resolves to a concrete local quant, never a remote one.
    assert resolver.resolve_local_gguf("unsloth/B-GGUF") == ("unsloth/B-GGUF", "UD-Q5_K_XL")
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
    assert resolver.resolve_local_gguf(win) == (win, None)


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
        resolves_to = ("unsloth/B-GGUF", "Q4_K_M"),
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
        resolves_to = ("unsloth/B-GGUF", "Q4_K_M"),
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
    eb = resolver._LocalGgufEntry("unsloth/B-GGUF", ("Q4_K_M",))
    ea = resolver._LocalGgufEntry("unsloth/A-GGUF", ())
    ec = resolver._LocalGgufEntry("unsloth/C-GGUF", ())
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
        resolves_to = ("unsloth/B-GGUF", "Q8_0"),
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
        resolves_to = ("unsloth/B-GGUF", "Q8_0"),
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
    _wire(monkeypatch, enabled = False, resolves_to = None, backend = backend, recorder = _LoadRecorder(backend))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_completions(_bad_body_request(), "tester"))
    assert exc.value.status_code == 503


def test_embeddings_malformed_body_503_not_500_when_unloaded(monkeypatch):
    from fastapi import HTTPException

    backend = _FakeBackend(None)
    _wire(monkeypatch, enabled = False, resolves_to = None, backend = backend, recorder = _LoadRecorder(backend))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_embeddings(_bad_body_request(), "tester"))
    assert exc.value.status_code == 503


def test_swap_refused_while_another_stream_inflight(monkeypatch):
    # A cross-model swap must not kill a stream still running on the loaded model:
    # with another request in flight, decline with 409 instead of loading.
    from fastapi import HTTPException
    from core.inference import llama_keepwarm as kw

    backend = _FakeBackend("unsloth/A-GGUF")
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = True, resolves_to = ("unsloth/B-GGUF", None), backend = backend, recorder = rec)
    monkeypatch.setattr(kw, "_inflight", 2)  # a stream on A + this request
    with pytest.raises(HTTPException) as exc:
        _run_hook("unsloth/B-GGUF")
    assert exc.value.status_code == 409
    assert rec.calls == []  # the in-flight stream was never killed


def test_swap_proceeds_when_no_other_stream(monkeypatch):
    # Only this request is in flight: nothing to protect, so the swap proceeds.
    from core.inference import llama_keepwarm as kw

    backend = _FakeBackend("unsloth/A-GGUF")
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = True, resolves_to = ("unsloth/B-GGUF", None), backend = backend, recorder = rec)
    monkeypatch.setattr(kw, "_inflight", 1)
    _run_hook("unsloth/B-GGUF")
    assert len(rec.calls) == 1


def test_alias_reloads_model_freed_by_idle_unload(monkeypatch):
    # After idle-unload frees the model, an unknown/alias name (resolves to None)
    # reloads what was freed instead of 503-ing on an empty backend.
    from core.inference import llama_keepwarm as kw

    backend = _FakeBackend(None)  # idle-unload emptied the backend
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = True, resolves_to = None, backend = backend, recorder = rec)
    monkeypatch.setattr(kw, "_inflight", 0)
    monkeypatch.setattr(kw, "_last_unloaded_model", "unsloth/A-GGUF")
    _run_hook("gpt-4o-mini")
    assert len(rec.calls) == 1
    assert rec.calls[0].model_path == "unsloth/A-GGUF"


def test_alias_does_not_reload_when_model_already_loaded(monkeypatch):
    # The reload only triggers on an empty backend; with something loaded, an
    # unknown name still falls through (drop-in) without resurrecting the stash.
    from core.inference import llama_keepwarm as kw

    backend = _FakeBackend("unsloth/B-GGUF")
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = True, resolves_to = None, backend = backend, recorder = rec)
    monkeypatch.setattr(kw, "_last_unloaded_model", "unsloth/A-GGUF")
    _run_hook("gpt-4o-mini")
    assert rec.calls == []


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
    monkeypatch.setattr(models_route, "_scan_models_dir", lambda d, limit = None: scanned.append(("models", str(Path(d).resolve()))) or [])
    monkeypatch.setattr(models_route, "_scan_hf_cache", lambda d: scanned.append(("hf", str(Path(d).resolve()))) or [])
    monkeypatch.setattr(models_route, "_scan_lmstudio_dir", lambda d: scanned.append(("lm", str(Path(d).resolve()))) or [])
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: tmp_path / "active")
    monkeypatch.setattr(models_route, "_is_hidden_model", lambda *a, **k: False)
    monkeypatch.setattr(upaths, "legacy_hf_cache_dir", lambda: tmp_path / "legacy")
    monkeypatch.setattr(upaths, "hf_default_cache_dir", lambda: tmp_path / "default")
    monkeypatch.setattr(upaths, "lmstudio_model_dirs", lambda: [tmp_path / "lmstudio"])
    monkeypatch.setattr(studio_db, "list_scan_folders", lambda: [{"path": str(tmp_path / "custom")}])
    for sub in ("active", "legacy", "default", "lmstudio", "custom"):
        (tmp_path / sub).mkdir()

    resolver._build_index()

    hf = {p for k, p in scanned if k == "hf"}
    lm = {p for k, p in scanned if k == "lm"}
    assert str((tmp_path / "legacy").resolve()) in hf
    assert str((tmp_path / "default").resolve()) in hf
    assert str((tmp_path / "custom").resolve()) in hf
    assert str((tmp_path / "lmstudio").resolve()) in lm
