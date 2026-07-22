# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in OpenAI /v1 model auto-switch: resolver, hook, and settings coercion.

No GPU or llama-server: the backend and the load route are mocked, mirroring
tests/test_gguf_completion_usage.py.
"""

import asyncio
import os

import pytest

import routes.inference as inference_route
from models.inference import LoadRequest
from core.inference import local_model_resolver as resolver
from utils import openai_auto_switch_settings as settings


class _FakeBackend:
    effective_parallel_slots = 1
    _slot_save_binary = None
    _gguf_path = None

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

    def save_slots_for_resume(self, should_abort = None):
        return None

    def restore_slots_for_resume(self, manifest):
        return None

    def _slot_launch_fingerprint(self):
        return ((), None, None, 1)

    def _gguf_file_identity(self, path):
        try:
            st = os.stat(path)
        except OSError:
            return None
        return ((st.st_size, st.st_mtime_ns),)


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
        *,
        current_request_counted = False,
    ):
        # Mirror the production load boundary before recording any replacement.
        await inference_route._wait_for_model_switch_idle(
            current_request_counted = current_request_counted
        )
        self.calls.append(request)
        if self.fail:
            from fastapi import HTTPException
            raise HTTPException(status_code = 503, detail = "load failed")
        self.backend.model_identifier = request.model_path
        self.backend.hf_variant = getattr(request, "gguf_variant", None)
        self.backend._gguf_path = request.model_path
        self.backend.is_loaded = True
        # Mirror _load_model_impl: a load advertises its own id until the
        # auto-switch caller overwrites it with the repo id.
        self.backend._openai_advertised_id = None
        from core.inference import llama_keepwarm as kw

        kw.note_model_loaded(self.backend)
        return None


def _wire(monkeypatch, *, enabled, resolves_to, backend, recorder):
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: enabled)
    monkeypatch.setattr(resolver, "resolve_local_gguf", lambda _m: resolves_to)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    # Auto-switch loads via _load_model_impl (the /load route holds the lifecycle
    # gate that auto-switch already owns, so it calls the impl directly).
    monkeypatch.setattr(inference_route, "_load_model_impl", recorder)
    monkeypatch.setattr(inference_route, "_auto_switch_waiters", {})


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


def test_openai_compat_routes_bound_to_handlers_with_auth():
    # Inserting a helper between a @router.post decorator and its handler silently
    # rebinds the route to the helper and drops its auth dependency (this happened to
    # /messages/count_tokens). The source-inspection tests above miss it because they
    # call the handler directly. Lock the path -> (handler, auth) mapping at the route
    # level so any decorator/handler split is caught.
    expected = {
        ("POST", "/chat/completions"): "openai_chat_completions",
        ("POST", "/completions"): "openai_completions",
        ("POST", "/embeddings"): "openai_embeddings",
        ("POST", "/responses"): "openai_responses",
        ("POST", "/messages"): "anthropic_messages",
        ("POST", "/messages/count_tokens"): "anthropic_count_tokens",
        ("POST", "/audio/generate"): "generate_audio",
        ("GET", "/models"): "openai_list_models",
        ("GET", "/models/{model_id:path}"): "openai_retrieve_model",
    }
    seen = {}
    for r in inference_route.router.routes:
        path = getattr(r, "path", None)
        endpoint = getattr(r, "endpoint", None)
        if path is None or endpoint is None:
            continue
        for method in getattr(r, "methods", None) or ():
            seen[(method, path)] = r
    for key, handler in expected.items():
        assert key in seen, f"route {key} is not registered"
        route = seen[key]
        assert (
            route.endpoint.__name__ == handler
        ), f"{key} bound to {route.endpoint.__name__}, expected {handler}"
        deps = [d.call.__name__ for d in route.dependant.dependencies]
        assert "get_current_subject" in deps, f"{key} lost its auth dependency"


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


def test_local_gguf_entry_rejects_standalone_mmproj(tmp_path):
    # Codex P2: _scan_models_dir's standalone-.gguf pass emits an entry for a
    # bare mmproj projector (it only filters mmproj inside directory scans). A
    # projector is not a servable model, so the resolver must reject it or
    # /v1/models advertises it and a switch could load it over the real weights.
    from types import SimpleNamespace

    proj = tmp_path / "mmproj-F16.gguf"
    proj.write_text("x")
    assert resolver._local_gguf_entry("p", SimpleNamespace(path = str(proj))) is None
    assert resolver.info_has_local_gguf(SimpleNamespace(id = str(proj), path = str(proj))) is False


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


def test_idle_loop_unloads_after_ttl_and_stashes_for_reload(monkeypatch):
    # The headline behavior (the other idle tests only cover the negative paths):
    # with nothing in flight and the TTL elapsed, the loop frees the GGUF exactly
    # once and records its identity so a later alias request can reload that variant.
    import time
    from core.inference import llama_keepwarm as kw

    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 0.005)
    kw._inflight = 0
    kw._pending = 0
    kw._last_active = time.monotonic() - 3600
    kw._last_unloaded_model = None

    unloads = []
    backend = _FakeBackend("unsloth/Idle-GGUF", hf_variant = "Q4_K_M")

    def _unload():
        unloads.append(1)
        backend.is_loaded = False  # a real unload clears the slot

    backend.unload_model = _unload
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)

    async def _drive():
        task = asyncio.create_task(kw.idle_unload_loop(poll_seconds = 0.02))
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_drive())
    assert unloads == [1]  # freed once, not repeatedly
    stash = kw.get_last_unloaded_model()
    assert stash is not None and stash[0] == "unsloth/Idle-GGUF" and stash[1] == "Q4_K_M"


def test_idle_loop_deletes_saved_kv_when_unload_fails(monkeypatch, tmp_path):
    import time
    from core.inference import llama_keepwarm as kw

    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 0.005)
    monkeypatch.setattr(settings, "get_auto_unload_keep_kv", lambda: True)
    kw._inflight = 0
    kw._pending = 0
    kw._last_active = time.monotonic() - 3600
    kw._last_unloaded_model = None
    kw._kv_resume = None

    saved = tmp_path / "resume-abc-slot0.bin"
    backend = _FakeBackend("unsloth/Idle-GGUF")
    manifests = []

    def _save(should_abort = None):
        if manifests:
            return None
        saved.write_bytes(b"kv")
        manifest = {"dir": str(tmp_path), "slots": [{"id": 0, "filename": saved.name}]}
        manifests.append(manifest)
        return manifest

    def _unload():
        raise RuntimeError("cuda teardown failed")

    backend.save_slots_for_resume = _save
    backend.unload_model = _unload
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)

    async def _drive():
        task = asyncio.create_task(kw.idle_unload_loop(poll_seconds = 0.01))
        for _ in range(200):
            await asyncio.sleep(0.01)
            if manifests and not saved.exists():
                break
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_drive())
    assert manifests and not saved.exists()
    assert kw._kv_resume is None


def test_disabling_idle_unload_purges_saved_kv(monkeypatch, tmp_path):
    # PUT leaves keep-KV on but makes idle unload inactive: saved KV must go too.
    import routes.settings as settings_route
    from core.inference import llama_keepwarm as kw

    saved = tmp_path / "resume-abc-slot0.bin"
    saved.write_bytes(b"kv")
    kw._kv_resume = {
        "identity": ("m", None, "m"),
        "dir": str(tmp_path),
        "slots": [{"id": 0, "filename": saved.name}],
    }
    monkeypatch.setattr(settings_route, "set_openai_auto_switch", lambda *a: (False, 300, True))
    monkeypatch.setattr(settings_route, "get_auto_unload_idle_seconds", lambda: 0)

    payload = settings_route.OpenAIAutoSwitchPayload(enabled = False)
    resp = settings_route.update_openai_auto_switch(payload, "tester")
    assert resp.idle_unload_active is False and resp.auto_unload_keep_kv is True
    assert kw._kv_resume is None and not saved.exists()


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


def test_model_override_rejects_zero_max_seq_length():
    # 0 is not a valid sequence length and the setter drops a falsy value, so the
    # payload must reject it at the boundary instead of accepting then discarding it.
    import pydantic
    import routes.settings as settings_route

    with pytest.raises(pydantic.ValidationError):
        settings_route.ModelOverridePayload(model_id = "x", max_seq_length = 0)
    assert settings_route.ModelOverridePayload(model_id = "x", max_seq_length = 1).max_seq_length == 1


def test_update_openai_auto_switch_writes_both_keys_in_one_transaction(monkeypatch):
    # The PUT must persist enabled + idle in a single upsert so a settings write can't
    # leave one key updated and the other stale.
    import routes.settings as settings_route
    import storage.studio_db as db
    from utils.openai_auto_switch_settings import (
        AUTO_UNLOAD_IDLE_SETTING_KEY,
        OPENAI_AUTO_SWITCH_SETTING_KEY,
    )

    calls = []

    def _capture(mapping):
        calls.append(dict(mapping))
        return {}

    monkeypatch.setattr(db, "upsert_app_settings", _capture)
    settings._cache.clear()

    payload = settings_route.OpenAIAutoSwitchPayload(enabled = True, auto_unload_idle_seconds = 120)
    resp = settings_route.update_openai_auto_switch(payload, "tester")
    assert resp.enabled is True and resp.auto_unload_idle_seconds == 120
    assert len(calls) == 1  # one transaction, not two
    written = calls[0]
    assert written.get(OPENAI_AUTO_SWITCH_SETTING_KEY) is True
    assert written.get(AUTO_UNLOAD_IDLE_SETTING_KEY) == 120


def test_settings_report_idle_unload_active_when_env_backed(monkeypatch):
    # Codex P2: with UNSLOTH_MODEL_IDLE_TTL driving idle-unload while the toggle is
    # off, the settings response must report idle_unload_active so the UI shows the
    # feature as active via env rather than "needs enable".
    import routes.settings as settings_route

    monkeypatch.setattr(settings_route, "get_openai_auto_switch_enabled", lambda: False)
    monkeypatch.setattr(settings_route, "get_stored_auto_unload_idle_seconds", lambda: 600)
    monkeypatch.setattr(
        settings_route, "get_auto_unload_idle_seconds", lambda: 600
    )  # effective > 0
    resp = settings_route.get_openai_auto_switch("tester")
    assert resp.enabled is False and resp.idle_unload_active is True
    # Effective TTL 0 (off, nothing env-backed) -> not active.
    monkeypatch.setattr(settings_route, "get_auto_unload_idle_seconds", lambda: 0)
    assert settings_route.get_openai_auto_switch("tester").idle_unload_active is False


# ── /v1/models discovery ────────────────────────────────────────────


def test_v1_models_retrieve_is_case_insensitive(monkeypatch):
    # The resolver lowercases its index, so a retrieve that differs only in case
    # from a catalog id must still hit (200), not 404. Guards the .lower() compare
    # in openai_retrieve_model against a silent revert. (The full local catalog is
    # main's #6519; only the loaded fast-path is exact, the catalog loop is lenient.)
    from fastapi import HTTPException

    monkeypatch.setattr(inference_route, "_openai_model_objects", lambda: [])  # nothing loaded

    async def _catalog():
        return [
            {"id": "unsloth/A-GGUF", "object": "model", "created": 1, "owned_by": "local"},
            {"id": "unsloth/B-GGUF", "object": "model", "created": 1, "owned_by": "local"},
        ]

    monkeypatch.setattr(inference_route, "_openai_catalog_objects", _catalog)

    # A catalog id retrieved with different casing still resolves.
    obj = asyncio.run(inference_route.openai_retrieve_model("unsloth/a-gguf", "tester"))
    assert obj["id"] == "unsloth/A-GGUF"
    # A truly unknown id still 404s.
    with pytest.raises(HTTPException) as unknown:
        asyncio.run(inference_route.openai_retrieve_model("totally/unknown", "tester"))
    assert unknown.value.status_code == 404


# ── hardening: hidden models, idle/enabled coupling, count_tokens keep-warm ──


def test_index_excludes_hidden_models(tmp_path, monkeypatch):
    # The llama.cpp validation probe and RAG embedding weights are hidden from
    # Unsloth's pickers; they must never become auto-switch targets.
    from types import SimpleNamespace
    import routes.models as models_route

    normal = tmp_path / "normal-Q4_K_M.gguf"
    normal.write_bytes(b"x" * 32)
    probe = tmp_path / "stories260K.gguf"  # llama.cpp install-validation probe
    probe.write_bytes(b"x" * 32)
    embedder = tmp_path / "embedding-Q8_0.gguf"
    embedder.write_bytes(b"x" * 32)
    local_default_embedder = tmp_path / "bge-small-en-v1.5-F16.gguf"
    local_default_embedder.write_bytes(b"x" * 32)

    def _info(mid, path):
        return SimpleNamespace(id = mid, path = str(path), model_id = mid, display_name = mid)

    monkeypatch.setattr(
        models_route,
        "_scan_models_dir",
        lambda *a, **k: [
            _info("org/Normal-GGUF", normal),
            _info("ggml-org/models", probe),
            SimpleNamespace(
                id = str(embedder),
                path = str(embedder),
                model_id = "unsloth/bge-small-en-v1.5-GGUF",
                display_name = "embedding-Q8_0",
            ),
            SimpleNamespace(
                id = str(local_default_embedder),
                path = str(local_default_embedder),
                model_id = None,
                display_name = local_default_embedder.name,
            ),
        ],
    )
    monkeypatch.setattr(models_route, "_scan_hf_cache", lambda *a, **k: [])
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: tmp_path)
    resolver._scan = (0.0, {})

    index = resolver._index()
    assert "org/normal-gguf" in index  # keys are normalized to lowercase
    assert "ggml-org/models" not in index
    assert "unsloth/bge-small-en-v1.5-gguf" not in index
    assert str(local_default_embedder).lower() not in index
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


def test_responses_system_only_rejected_before_switch(monkeypatch):
    # Codex P2: instructions-only input normalises to a lone system message, which
    # passes the empty-input check; it must 400 before the switch so an invalid
    # Responses request can't evict the resident model.
    from fastapi import HTTPException
    from models.inference import ResponsesRequest

    async def _boom(*a, **k):
        raise AssertionError("must not switch a system-only Responses request")

    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _boom)
    payload = ResponsesRequest(model = "org/B-GGUF", instructions = "be helpful", input = "")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_responses(payload, object(), "tester"))
    assert exc.value.status_code == 400


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


def test_auto_switch_waits_for_another_inference_to_finish(monkeypatch):
    # A cross-model swap queues while another request is generating, then loads
    # after that request drains. The requesting call itself is excluded.
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

    async def _drive():
        task = asyncio.create_task(
            inference_route._maybe_auto_switch_model("org/B-GGUF:Q8_0", object(), "tester")
        )
        await asyncio.sleep(0.05)
        assert rec.calls == []
        kw._note_end()  # the other generation finishes; this request remains counted
        await asyncio.wait_for(task, timeout = 1)

    asyncio.run(_drive())
    assert len(rec.calls) == 1


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


def test_already_serving_by_path_records_advertised_alias(monkeypatch):
    # Codex P2: a model loaded by local path and requested via an advertised alias
    # that resolves to the same path is already serving (no reload), but /v1/models
    # and responses would report the path basename and list the alias as loaded:false
    # unless the alias is recorded as the advertised id on the already-serving return.
    path = "/cache/models--org--Repo-GGUF/snapshots/abc"
    backend = _FakeBackend(path, hf_variant = "Q4_K_M")  # loaded by path, no advertised id
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = (path, "Q4_K_M", "org/Repo-GGUF"),
        backend = backend,
        recorder = rec,
    )
    assert backend._openai_advertised_id is None
    _run_hook("org/Repo-GGUF:Q4_K_M")
    assert rec.calls == []  # already serving -> no reload
    assert backend._openai_advertised_id == "org/Repo-GGUF"  # alias now recorded


def test_streaming_responses_uses_advertised_id_helper():
    # Codex P2: streamed /v1/responses envelopes must derive the model id from
    # _llama_public_model_id (which prefers _openai_advertised_id), not the raw
    # model_identifier. After an auto-switch to a cached HF GGUF the identifier is
    # the snapshot path while the repo id lives in _openai_advertised_id, so the raw
    # form would stream a snapshot basename while /v1/models, chat, and non-streaming
    # responses report the repo id.
    import inspect

    src = inspect.getsource(inference_route._responses_stream)
    assert "_clean_model = _llama_public_model_id(llama_backend" in src
    assert 'public_model_id(getattr(llama_backend, "model_identifier"' not in src


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
    assert len(rec.calls) == 1


def test_queued_different_target_does_not_deadlock_current_swap(monkeypatch):
    # A concurrent request already queued for another target is not generating,
    # so it must not prevent the current serialized swap from proceeding.
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
    _run_hook("org/B-GGUF:Q8_0")
    assert len(rec.calls) == 1


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


def test_model_replacements_recheck_sidecar_swap_before_either_backend_is_unloaded():
    # Both replacement directions drain active inference, then recheck whether a
    # sidecar install reserved the lifecycle gate during that wait. Exact-model
    # reuse exits earlier, so an already-loaded model never waits on unrelated inference.
    import inspect

    src = inspect.getsource(inference_route._load_model_impl)
    gguf_wait = src.index("await _wait_for_model_switch_idle", src.index("if config.is_gguf:"))
    gguf_sidecar_check = src.index("_raise_if_sidecar_swap_in_progress()", gguf_wait)
    unload_unsloth = src.index("unsloth_backend.unload_model", gguf_wait)
    standard_wait = src.index("await _wait_for_model_switch_idle", gguf_wait + 1)
    standard_sidecar_check = src.index("_raise_if_sidecar_swap_in_progress()", standard_wait)
    unload_gguf = src.index("llama_backend.unload_model()", standard_wait)
    already_loaded = src.index('status = "already_loaded"')

    assert already_loaded < gguf_wait < gguf_sidecar_check < unload_unsloth
    assert standard_wait < standard_sidecar_check < unload_gguf


def test_switch_waiter_deregisters_before_swap_gate_release():
    # A waiter left registered after the swap gate is released would let a swap on
    # another event loop count the finished request as still queued, pass the drain
    # early, and unload the model that request is about to generate against.
    import inspect

    src = inspect.getsource(inference_route._maybe_auto_switch_model)
    deregister = src.index("_note_switch_waiter(key, -1)")
    release = src.index("_auto_switch_process_lock.release()")
    assert deregister < release


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


def test_pending_same_target_request_does_not_block_swap(monkeypatch):
    # A second same-target request blocked in the middleware (pending, not yet
    # generating) must not block the first request: pending is excluded.
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
    assert len(rec.calls) == 1


def test_swap_waits_until_concurrent_request_finishes_resolving(monkeypatch):
    # The real middleware counts a concurrent same-model request as in-flight
    # before it resolves and registers a target waiter. Treat it as active until
    # its target is known, then recognize it as another queued switch request.
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
    # The twin is still resolving, so it is counted in-flight but has not joined
    # the concrete target queue yet.

    async def _drive():
        task = asyncio.create_task(
            inference_route._maybe_auto_switch_model("org/B-GGUF:Q8_0", object(), "tester")
        )
        await asyncio.sleep(0.05)
        assert rec.calls == []
        inference_route._note_switch_waiter(inference_route._switch_key("org/B-GGUF", "Q8_0"), 1)
        await asyncio.wait_for(task, timeout = 1)

    asyncio.run(_drive())
    assert len(rec.calls) == 1


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


def test_auto_switch_waits_when_unsloth_stream_active(monkeypatch):
    # The GGUF slot is empty but an Unsloth model is streaming (counted in-flight).
    # The replacement waits for it just as it does for a GGUF generation.
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

    async def _drive():
        task = asyncio.create_task(
            inference_route._maybe_auto_switch_model("org/B-GGUF:Q8_0", object(), "tester")
        )
        await asyncio.sleep(0.05)
        assert rec.calls == []
        kw._note_end()
        await asyncio.wait_for(task, timeout = 1)

    asyncio.run(_drive())
    assert len(rec.calls) == 1


def test_public_model_id_prefers_advertised_over_path():
    backend = _FakeBackend("/cache/models--org--Repo/snapshots/abc/model.gguf")
    backend._openai_advertised_id = "org/Repo-GGUF"
    # The advertised repo id from an auto-switch load wins.
    assert inference_route._llama_public_model_id(backend) == "org/Repo-GGUF"
    backend._openai_advertised_id = None
    # No advertised id: the identifier is cleaned to a public id (delegates to
    # public_model_id), never the raw on-disk .gguf path.
    cleaned = inference_route._llama_public_model_id(backend)
    assert cleaned and "/cache/" not in cleaned and not cleaned.endswith(".gguf")
    # An already-clean repo id passes through unchanged.
    backend.model_identifier = "org/Repo-GGUF"
    assert inference_route._llama_public_model_id(backend) == "org/Repo-GGUF"
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
    store = {settings.AUTO_UNLOAD_IDLE_SETTING_KEY: 90}
    monkeypatch.setattr(settings, "_cached_setting", lambda k, d = None: store.get(k, d))
    monkeypatch.setenv("UNSLOTH_MODEL_IDLE_TTL", "600")
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    assert settings.get_auto_unload_idle_seconds() == 90  # stored wins, not env
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
    # No alias available: strip the path to a public id so a host path is never advertised.
    assert (
        f(
            SimpleNamespace(
                id = "/home/me/models/Qwen3-8B-Q4_K_M.gguf", model_id = None, display_name = None
            )
        )
        == "Qwen3-8B-Q4_K_M"
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
    from storage import studio_db
    import utils.paths as paths

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
    monkeypatch.setattr(paths, "lmstudio_model_dirs", lambda: [])
    monkeypatch.setattr(studio_db, "list_scan_folders", lambda: [])
    resolver._scan = (0.0, {})

    # The advertised id is the alias, never the absolute path.
    advertised = sorted({entry.loader_id for entry in resolver._index().values()})
    assert advertised == ["org/Repo-GGUF"]
    # But the model is still resolvable by its on-disk path (an indexed alias).
    resolver._scan = (0.0, {})
    assert resolver.resolve_local_gguf(str(gguf)) is not None


def test_build_index_survives_a_failing_scanner(tmp_path, monkeypatch):
    # gemini: one bad scanner (e.g. a permission error on ./models) must drop only
    # that source, not abort the whole index and lose what the others found.
    from types import SimpleNamespace
    import routes.models as models_route
    import utils.paths as paths

    def _boom(*a, **k):
        raise OSError("permission denied")

    lm_info = SimpleNamespace(
        id = "org/Repo-GGUF", path = "/lm/Repo", model_id = "org/Repo-GGUF", display_name = "Repo"
    )
    monkeypatch.setattr(models_route, "_scan_models_dir", _boom)  # ./models blows up
    monkeypatch.setattr(models_route, "_scan_hf_cache", lambda *a, **k: [])
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(models_route, "_is_hidden_model", lambda *a, **k: False)
    monkeypatch.setattr(models_route, "_scan_lmstudio_dir", lambda *a, **k: [lm_info])
    monkeypatch.setattr(paths, "legacy_hf_cache_dir", lambda: None)
    monkeypatch.setattr(paths, "hf_default_cache_dir", lambda: None)
    monkeypatch.setattr(paths, "lmstudio_model_dirs", lambda: [tmp_path])
    # The on-disk GGUF check is covered elsewhere; here a found info becomes an entry.
    monkeypatch.setattr(
        resolver,
        "_local_gguf_entry",
        lambda loader_id, info: resolver._LocalGgufEntry(loader_id, "/lm/Repo", ()),
    )
    resolver._scan = (0.0, {})
    index = resolver._build_index()
    assert any(e.loader_id == "org/Repo-GGUF" for e in index.values())


def test_info_has_local_gguf_reads_files_not_model_format(tmp_path):
    # Codex: HF-cache GGUF snapshots leave model_format unset, so /v1/models must
    # decide GGUF-ness from the on-disk files. A standalone .gguf (no model_format)
    # is servable; a safetensors-only dir is not.
    from types import SimpleNamespace

    gguf = tmp_path / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"x" * 32)
    assert resolver.info_has_local_gguf(SimpleNamespace(id = str(gguf), path = str(gguf))) is True

    st = tmp_path / "safetensors_model"
    st.mkdir()
    (st / "model.safetensors").write_bytes(b"x" * 32)
    assert resolver.info_has_local_gguf(SimpleNamespace(id = str(st), path = str(st))) is False


def test_info_has_local_gguf_excludes_ollama_links(tmp_path):
    # Codex P2: Ollama entries come from a scanner _build_index skips, so their
    # advertised ids never resolve; the catalog must not report them as servable.
    from types import SimpleNamespace

    links = tmp_path / ".studio_links"
    links.mkdir()
    ollama_gguf = links / "model-Q4_K_M.gguf"
    ollama_gguf.write_bytes(b"x" * 32)
    assert (
        resolver.info_has_local_gguf(SimpleNamespace(id = "ollama/foo:latest", path = str(ollama_gguf)))
        is False
    )
    # The same GGUF outside an ollama-link dir is still servable.
    plain = tmp_path / "model-Q4_K_M.gguf"
    plain.write_bytes(b"x" * 32)
    assert resolver.info_has_local_gguf(SimpleNamespace(id = str(plain), path = str(plain))) is True


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

    monkeypatch.setattr(inference_route, "_openai_model_objects", lambda: [])  # nothing loaded
    monkeypatch.setattr(inference_route, "_openai_catalog_objects", _objs)
    obj = asyncio.run(inference_route.openai_retrieve_model("org/B-GGUF", "tester"))
    assert obj["id"] == "org/B-GGUF"
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_retrieve_model("123", "tester"))
    assert exc.value.status_code == 404


def test_retrieve_model_resolves_raw_path_to_advertised_id(monkeypatch):
    # Codex P2: a client caching the legacy absolute .gguf path must still retrieve
    # a loaded auto-switch model. Its /v1/models entry is keyed by the advertised
    # repo id (identifier = snapshot path), so the raw-path fallback must map the raw
    # id to that advertised id, not public_model_id(path), or a loaded model 404s.
    from types import SimpleNamespace

    raw_path = "/cache/models--org--B-GGUF/snapshots/abc/model.gguf"
    llama = SimpleNamespace(
        is_loaded = True, model_identifier = raw_path, _openai_advertised_id = "org/B-GGUF"
    )
    infer = SimpleNamespace(active_model_name = None)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: llama)
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: infer)
    monkeypatch.setattr(
        inference_route,
        "_openai_model_objects",
        lambda: [{"id": "org/B-GGUF", "object": "model"}],
    )

    async def _empty():
        return []

    monkeypatch.setattr(inference_route, "_openai_catalog_objects", _empty)
    obj = asyncio.run(inference_route.openai_retrieve_model(raw_path, "tester"))
    assert obj["id"] == "org/B-GGUF" and obj["loaded"] is True


def test_chat_streaming_n_gt_1_rejected_before_switch(monkeypatch):
    # Codex P2: only the non-streaming GGUF path returns multiple choices, so
    # stream=true + n>1 is invalid on every local serving path. Both fields are
    # known pre-switch, so it must 400 before the switch rather than loading model B.
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
    payload = _chat_request(model = "org/B-GGUF", stream = True, n = 2)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))
    assert exc.value.status_code == 400
    assert rec.calls == []


def test_resolver_cache_stamped_after_slow_build(monkeypatch):
    # Codex P2: the cache must be stamped AFTER _build_index. A scan slower than the
    # TTL would otherwise store an already-expired cache and rebuild every request.
    import core.inference.local_model_resolver as r

    clock = {"t": 1000.0}
    monkeypatch.setattr(r.time, "monotonic", lambda: clock["t"])
    calls = {"n": 0}

    def _slow_build():
        calls["n"] += 1
        clock["t"] += r._CACHE_TTL_S + 10.0  # the scan itself outlasts the TTL
        return {}

    monkeypatch.setattr(r, "_build_index", _slow_build)
    r._scan = (0.0, {})
    r._index()  # builds once, stamps post-scan
    r._index()  # immediately after: must reuse the cache, not rebuild
    assert calls["n"] == 1


def test_keepwarm_does_not_stamp_activity_on_401(monkeypatch):
    # Codex P2: the keep-warm middleware runs before auth, so a 401 must decrement
    # the in-flight count without stamping activity, or unauthenticated probes would
    # keep the model warm and block idle-unload.
    import core.inference.llama_keepwarm as kw

    monkeypatch.setattr(kw, "_inflight", 0)
    monkeypatch.setattr(kw, "_pending", 0)
    monkeypatch.setattr(kw, "_last_active", 100.0)

    async def _recv():
        return {"type": "http.request"}

    async def _run(status_code):
        async def _app(scope, receive, send):
            await send({"type": "http.response.start", "status": status_code, "headers": []})
            await send({"type": "http.response.body", "body": b"x", "more_body": False})

        sent = []

        async def _send(m):
            sent.append(m)

        mw = kw.LlamaKeepWarmMiddleware(_app)
        await mw({"type": "http", "method": "POST", "path": "/v1/chat/completions"}, _recv, _send)

    asyncio.run(_run(401))
    assert kw._inflight == 0  # balanced (start then untracked end)
    assert kw._last_active == 100.0  # activity NOT stamped for an auth failure
    # A served (200) request still stamps activity.
    asyncio.run(_run(200))
    assert kw._inflight == 0
    assert kw._last_active != 100.0


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
    # Lock the order at the source: tool-shape validation precedes the hook, for
    # both /messages and /messages/count_tokens (shared helper).
    import inspect
    for fn in (inference_route.anthropic_messages, inference_route.anthropic_count_tokens):
        src = inspect.getsource(fn)
        assert src.index("_validate_anthropic_client_tools") < src.index("_maybe_auto_switch_model")


def test_anthropic_mixed_tools_rejected_before_switch(monkeypatch):
    # Codex P2: combining an Anthropic server tool (type) with a custom client tool
    # (input_schema) is unsupported and must 400 before the switch, so the request
    # can't evict the loaded model only to be rejected after the load.
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
    payload = _anthropic_payload_with_tools(
        [
            {"type": "web_search_20250305"},  # server tool
            {"name": "my_func", "input_schema": {"type": "object"}},  # client tool
        ]
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.anthropic_messages(payload, object(), "tester"))
    assert exc.value.status_code == 400
    assert rec.calls == []  # rejected before the model load


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


def test_responses_forcing_tool_choice_without_name_rejected_before_switch(monkeypatch):
    # Codex P2: a forcing-function tool_choice with no name (Responses shape
    # {"type": "function"}) must 400 before the switch, so the streaming path can't
    # forward a bad choice and an invalid request can't evict the model.
    from fastapi import HTTPException
    from models.inference import ResponsesRequest

    async def _boom(*a, **k):
        raise AssertionError("must not switch on an invalid tool_choice")

    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _boom)
    payload = ResponsesRequest(model = "org/B-GGUF", input = "hi", tool_choice = {"type": "function"})
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_responses(payload, object(), "tester"))
    assert exc.value.status_code == 400
    # A named forcing choice is accepted (reaches the switch, which is mocked to raise).
    ok = ResponsesRequest(
        model = "org/B-GGUF", input = "hi", tool_choice = {"type": "function", "name": "f"}
    )
    with pytest.raises(AssertionError):
        asyncio.run(inference_route.openai_responses(ok, object(), "tester"))


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


def test_chat_audio_input_guards_target_before_switch(monkeypatch):
    # Codex P2: a chat request carrying audio_base64 must guard the target before the
    # switch -- audio rides the same companion mmproj as vision -- so a text-only
    # target can't be loaded and evict the working audio model. Assert the handler
    # flags require_vision so the hook's multimodal probe runs.
    class _Reached(Exception):
        pass

    captured = {}

    async def _capture(
        model,
        request,
        subject,
        *,
        require_vision = False,
    ):
        captured["require_vision"] = require_vision
        raise _Reached()

    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _capture)
    payload = _chat_request(model = "org/B-GGUF", audio_base64 = "AAAA")
    with pytest.raises(_Reached):
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))
    assert captured["require_vision"] is True


def test_completions_rejects_object_prompt_before_switch(monkeypatch):
    # Codex P2: an object prompt like {"prompt": {}} is a deterministic client error
    # (only a string or array is valid). It must 400 before the switch so a bad shape
    # can't load the named GGUF only to be rejected by llama-server after eviction.
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
                _json_body_request({"model": "org/B-GGUF", "prompt": {}}), "tester"
            )
        )
    assert exc.value.status_code == 400
    assert rec.calls == []  # no switch before rejection


def test_embeddings_rejects_object_input_before_switch(monkeypatch):
    # Codex P2: an object input like {"input": {}} is a deterministic client error
    # (only a string or array is valid); reject before the switch, like completions.
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
            inference_route.openai_embeddings(
                _json_body_request({"model": "org/B-GGUF", "input": {}}), "tester"
            )
        )
    assert exc.value.status_code == 400
    assert rec.calls == []


def test_chat_oversized_audio_rejected_before_switch(monkeypatch):
    # Codex P2: the audio size cap is a cheap, target-independent length check, so an
    # oversized upload must 413 before the switch rather than loading a GGUF first.
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
    big = "A" * (inference_route._MAX_AUDIO_B64_CHARS + 1)
    payload = _chat_request(model = "org/B-GGUF", audio_base64 = big)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))
    assert exc.value.status_code == 413
    assert rec.calls == []


def test_chat_confirm_without_stream_mcp_rejected_before_switch(monkeypatch):
    # Codex P2: mcp_enabled opens the local tool loop on its own, so confirm+no-stream
    # +mcp is the same invalid shape as confirm+no-stream+tools and must 400 before
    # the switch. The old guard only checked explicit tool fields and missed it.
    import state.tool_policy as _tp
    from fastapi import HTTPException

    monkeypatch.setattr(_tp, "get_tool_policy", lambda: None)  # no CLI --disable-tools
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
        model = "org/B-GGUF", mcp_enabled = True, confirm_tool_calls = True, stream = False
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))
    assert exc.value.status_code == 400
    assert rec.calls == []


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
    assert "does not support the image or audio input" in hook


def test_messages_have_image_helper():
    from models.inference import ChatMessage, ImageContentPart, ImageUrl, TextContentPart

    f = inference_route._messages_have_image
    text_only = [
        ChatMessage(role = "user", content = "hi"),
        ChatMessage(role = "user", content = [TextContentPart(type = "text", text = "hi")]),
    ]
    assert f(text_only) is False
    img = ImageContentPart(type = "image_url", image_url = ImageUrl(url = "data:image/png;base64,AAAA"))
    assert f([ChatMessage(role = "user", content = [img])]) is True


def test_anthropic_request_has_image_helper():
    from types import SimpleNamespace

    f = inference_route._anthropic_request_has_image
    text = SimpleNamespace(messages = [SimpleNamespace(content = "hi")])
    assert f(text) is False
    text_block = SimpleNamespace(
        messages = [SimpleNamespace(content = [{"type": "text", "text": "hi"}])]
    )
    assert f(text_block) is False
    dict_img = SimpleNamespace(messages = [SimpleNamespace(content = [{"type": "image"}])])
    assert f(dict_img) is True
    typed_img = SimpleNamespace(messages = [SimpleNamespace(content = [SimpleNamespace(type = "image")])])
    assert f(typed_img) is True


def test_responses_and_anthropic_wire_require_vision_from_images():
    # P2: the modality guard must fire on /v1/responses and /v1/messages too, so an
    # image request can't evict a vision model for a text-only target. Lock the wiring
    # at the source: each hook derives require_vision from the request's images.
    import inspect

    responses_src = inspect.getsource(inference_route.openai_responses)
    assert "require_vision = _messages_have_image(" in responses_src
    anthropic_src = inspect.getsource(inference_route.anthropic_messages)
    assert "require_vision = _anthropic_request_has_image(" in anthropic_src
    # /messages/count_tokens shares the /messages translation, so it needs the same
    # guard: an image count must not evict a vision model for a text-only target.
    count_src = inspect.getsource(inference_route.anthropic_count_tokens)
    assert "require_vision = _anthropic_request_has_image(" in count_src


# ── codex review (round 5): count_tokens tools, tool_choice, process-wide gate ──


def test_count_tokens_rejects_malformed_tool_before_switch(monkeypatch):
    # Codex P2: /v1/messages/count_tokens must reject a malformed tool before the
    # switch, like /messages, so a count request can't evict the loaded model.
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
    payload = _anthropic_payload_with_tools([{"name": "broken"}])  # no input_schema/type
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.anthropic_count_tokens(payload, object(), "tester"))
    assert exc.value.status_code == 400
    assert rec.calls == []


def test_count_tokens_forwards_vision_guard_to_switch(monkeypatch):
    # Codex P2: an image /v1/messages/count_tokens naming a text-only GGUF must
    # carry the same require_vision guard as /messages, so it can't evict a loaded
    # vision model for a swap that can't serve the request.
    class _Reached(Exception):
        pass

    captured = {}

    async def _capture(
        model,
        request,
        subject,
        *,
        require_vision = False,
    ):
        captured["require_vision"] = require_vision
        raise _Reached()

    monkeypatch.setattr(inference_route, "_anthropic_request_has_image", lambda p: True)
    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _capture)
    payload = _anthropic_payload_with_tools(None)  # no tools -> tool validation passes
    with pytest.raises(_Reached):
        asyncio.run(inference_route.anthropic_count_tokens(payload, object(), "tester"))
    assert captured["require_vision"] is True


def test_audio_generate_is_reload_only(monkeypatch):
    # Codex P2: /audio/generate must not switch to a client-named GGUF. A local
    # GGUF's audio-input capability is not a cheap pre-load probe (the mmproj signal
    # can't tell an audio projector from a vision one), so resolving the client model
    # could evict the working audio model for a target that then fails the audio
    # check. Only the idle-stash restore runs: the hook gets the reload-only sentinel.
    from models.inference import ChatCompletionRequest

    class _Reached(Exception):
        pass

    captured = {}

    async def _capture(
        model,
        request,
        subject,
        *,
        require_vision = False,
    ):
        captured["model"] = model
        raise _Reached()

    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _capture)
    payload = ChatCompletionRequest(
        model = "org/B-GGUF", messages = [{"role": "user", "content": "say hi"}]
    )
    with pytest.raises(_Reached):
        asyncio.run(inference_route.generate_audio(payload, object(), "tester"))
    assert captured["model"] == inference_route._RELOAD_ONLY_MODEL


def test_note_model_unloaded_clears_reload_stash(monkeypatch):
    # Codex P2: a deliberate unload must drop the idle reload stash so the next /v1
    # request can't resurrect the just-unloaded model. (The idle loop unloads via the
    # backend directly, so clearing on the route never fights keep-warm.)
    import core.inference.llama_keepwarm as kw

    kw._set_last_unloaded(("org/A-GGUF", "Q4_K_M"))
    assert kw.get_last_unloaded_model() == ("org/A-GGUF", "Q4_K_M")
    kw.note_model_unloaded()
    assert kw.get_last_unloaded_model() is None


def test_unload_route_clears_reload_stash(monkeypatch):
    # The /unload route must clear the stash on both the GGUF and non-GGUF branches.
    import inspect
    src = inspect.getsource(inference_route.unload_model)
    assert src.count("note_model_unloaded()") >= 2


def test_non_gguf_load_clears_reload_stash():
    # A non-GGUF (Transformers/Unsloth) load must clear the stash like the GGUF
    # branch, so it never lingers until the idle poll (or forever, idle-unload off).
    import inspect

    src = inspect.getsource(inference_route._load_model_impl)
    assert src.count("note_model_loaded()") >= 1  # non-GGUF branch
    assert "to_thread(note_model_loaded, llama_backend)" in src  # GGUF branch


def test_chat_rejects_malformed_tool_choice_before_switch(monkeypatch):
    # Codex P2: a forcing object with no function name must 400 before the switch.
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
    payload = _chat_request(model = "org/B-GGUF", tool_choice = {"type": "function", "function": {}})
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))
    assert exc.value.status_code == 400
    assert rec.calls == []


def test_chat_valid_tool_choice_reaches_hook(monkeypatch):
    # A well-formed forcing object must pass the pre-check and reach the hook.
    class _Reached(Exception):
        pass

    async def _boom(*a, **k):
        raise _Reached()

    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _boom)
    payload = _chat_request(
        model = "org/B-GGUF", tool_choice = {"type": "function", "function": {"name": "ok"}}
    )
    with pytest.raises(_Reached):
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))


def test_lifecycle_gate_serializes_across_loops():
    # Codex P2: the lifecycle gate must be process-wide so a swap on one loop blocks
    # inference starting on another. Two loops must never hold the gate at once.
    import threading
    from core.inference import llama_keepwarm as kw

    state = {"cur": 0, "max": 0}
    slock = threading.Lock()

    async def _use():
        async with kw._unload_gate():
            with slock:
                state["cur"] += 1
                state["max"] = max(state["max"], state["cur"])
            await asyncio.sleep(0.05)
            with slock:
                state["cur"] -= 1

    barrier = threading.Barrier(2)

    def _run():
        barrier.wait()
        asyncio.run(_use())

    threads = [threading.Thread(target = _run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert state["max"] == 1  # never held on two loops at once


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
        *,
        current_request_counted = False,
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


def test_no_model_loaded_detail_appends_hint_only_when_off(monkeypatch):
    # The "no model loaded" errors point at the opt-in auto-switch toggle so a
    # request naming a listed-but-unloaded model is self-explanatory -- but only
    # when it's off. With it on the name simply didn't resolve, so no hint.
    base = "No GGUF model loaded. Load a GGUF model first."

    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: False)
    off = inference_route._no_model_loaded_detail(base)
    assert off.startswith(base)
    assert "Model auto-switch" in off and "Settings > API" in off

    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    assert inference_route._no_model_loaded_detail(base) == base


def _run_responses_stream_no_model(monkeypatch, *, enabled, active_model_name):
    # Drive _responses_stream's GGUF-not-loaded guard: llama backend unloaded,
    # inference backend maybe holding a non-GGUF model. Returns the 400 detail.
    from fastapi import HTTPException
    from models.inference import ResponsesRequest, ChatMessage

    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: enabled)
    monkeypatch.setattr(
        inference_route, "get_llama_cpp_backend", lambda: _FakeBackend(loaded_id = None)
    )
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: type("_B", (), {"active_model_name": active_model_name})(),
    )
    payload = ResponsesRequest(model = "unsloth/Qwen3.5-4B-GGUF", stream = True)
    messages = [ChatMessage(role = "user", content = "hi")]
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route._responses_stream(payload, messages, None))
    assert exc.value.status_code == 400
    return exc.value.detail


def test_responses_stream_hint_matches_toggle_regardless_of_active_model(monkeypatch):
    # Streaming /v1/responses shares the GGUF-only 400 with the other "no model
    # loaded" sites, so the auto-switch hint attaches whenever the toggle is
    # off -- including while a non-GGUF model is active, since auto-switch
    # evicts it to load a resolved GGUF (_maybe_auto_switch_model's resolver
    # branch has no active-model guard, unlike its reload-stash branch). Only
    # the toggle being on suppresses it.
    hinted = _run_responses_stream_no_model(monkeypatch, enabled = False, active_model_name = None)
    assert "Model auto-switch" in hinted

    on = _run_responses_stream_no_model(monkeypatch, enabled = True, active_model_name = None)
    assert "Model auto-switch" not in on

    non_gguf_loaded = _run_responses_stream_no_model(
        monkeypatch, enabled = False, active_model_name = "unsloth/Llama-3.2-1B-Instruct"
    )
    assert "Model auto-switch" in non_gguf_loaded


# ── idle-unload KV persistence (slot save/restore) ──────────────────


def _seed_kv_manifest(
    tmp_path,
    identity = ("unsloth/A-GGUF", "Q4_K_M", "unsloth/A-GGUF"),
    gguf = None,
):
    if gguf is None:
        gguf_file = tmp_path / "model.gguf"
        gguf_file.write_bytes(b"gguf")
        gguf = str(gguf_file)
    st = os.stat(gguf)
    state_file = tmp_path / "resume-abc-slot0.bin"
    state_file.write_bytes(b"kv")
    return state_file, {
        "identity": identity,
        "dir": str(tmp_path),
        "binary": ("/bin/llama-server", 111),
        "gguf": gguf,
        "gguf_stat": ((st.st_size, st.st_mtime_ns),),
        "launch": ((), None, None, 1),
        "slots": [{"id": 0, "filename": state_file.name, "n_saved": 42}],
    }


def _drive_idle_loop(
    kw,
    poll_seconds = 0.02,
    run_for = 0.2,
):
    async def _drive():
        task = asyncio.create_task(kw.idle_unload_loop(poll_seconds = poll_seconds))
        await asyncio.sleep(run_for)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_drive())


def test_idle_unload_saves_slots_before_unload_and_stashes_manifest(monkeypatch, tmp_path):
    import time
    from core.inference import llama_keepwarm as kw

    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 0.005)
    monkeypatch.setattr(settings, "get_auto_unload_keep_kv", lambda: True)
    kw._inflight = 0
    kw._pending = 0
    kw._last_active = time.monotonic() - 3600
    kw._last_unloaded_model = None
    kw._kv_resume = None

    events = []
    backend = _FakeBackend("unsloth/Idle-GGUF", hf_variant = "Q4_K_M")
    manifest = {
        "dir": str(tmp_path),
        "binary": ("bin", 1),
        "slots": [{"id": 0, "filename": "f.bin", "n_saved": 42}],
    }

    def _save(should_abort = None):
        events.append("save")
        return manifest

    def _unload():
        events.append("unload")
        backend.is_loaded = False

    backend.save_slots_for_resume = _save
    backend.unload_model = _unload
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)

    _drive_idle_loop(kw)
    # KV must be saved while the server is still alive, then exactly one unload.
    assert events == ["save", "unload"]
    assert kw.get_last_unloaded_model()[:2] == ("unsloth/Idle-GGUF", "Q4_K_M")
    resume = kw.take_kv_resume()
    assert resume is not None
    assert resume["identity"][:2] == ("unsloth/Idle-GGUF", "Q4_K_M")
    assert resume["slots"][0]["filename"] == "f.bin"


def test_idle_save_failure_still_unloads_plain(monkeypatch):
    import time
    from core.inference import llama_keepwarm as kw

    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 0.005)
    monkeypatch.setattr(settings, "get_auto_unload_keep_kv", lambda: True)
    kw._inflight = 0
    kw._pending = 0
    kw._last_active = time.monotonic() - 3600
    kw._last_unloaded_model = None
    kw._kv_resume = None

    unloads = []
    backend = _FakeBackend("unsloth/Idle-GGUF", hf_variant = "Q4_K_M")

    def _save(should_abort = None):
        raise RuntimeError("slot save exploded")

    def _unload():
        unloads.append(1)
        backend.is_loaded = False

    backend.save_slots_for_resume = _save
    backend.unload_model = _unload
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)

    _drive_idle_loop(kw)
    assert unloads == [1]  # the save failure must not skip the unload
    assert kw.get_last_unloaded_model() is not None
    assert kw.take_kv_resume() is None


def test_keep_kv_setting_off_skips_save(monkeypatch):
    import time
    from core.inference import llama_keepwarm as kw

    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 0.005)
    monkeypatch.setattr(settings, "get_auto_unload_keep_kv", lambda: False)
    kw._inflight = 0
    kw._pending = 0
    kw._last_active = time.monotonic() - 3600
    kw._last_unloaded_model = None
    kw._kv_resume = None

    saves, unloads = [], []
    backend = _FakeBackend("unsloth/Idle-GGUF")

    def _unload():
        unloads.append(1)
        backend.is_loaded = False

    backend.save_slots_for_resume = lambda *a, **k: saves.append(1)
    backend.unload_model = _unload
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)

    _drive_idle_loop(kw)
    assert saves == []
    assert unloads == [1]
    assert kw.take_kv_resume() is None


def test_keep_kv_disabled_mid_save_discards_manifest(monkeypatch, tmp_path):
    import time
    from core.inference import llama_keepwarm as kw

    keep = {"on": True}
    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 0.005)
    monkeypatch.setattr(settings, "get_auto_unload_keep_kv", lambda: keep["on"])
    kw._inflight = 0
    kw._pending = 0
    kw._last_active = time.monotonic() - 3600
    kw._last_unloaded_model = None
    kw._kv_resume = None

    unloads = []
    backend = _FakeBackend("unsloth/Idle-GGUF", hf_variant = "Q4_K_M")
    state_file = tmp_path / "resume-mid-slot0.bin"
    state_file.write_bytes(b"kv")
    manifest = {
        "dir": str(tmp_path),
        "binary": ("bin", 1),
        "slots": [{"id": 0, "filename": state_file.name, "n_saved": 1}],
    }

    def _save(should_abort = None):
        keep["on"] = False  # user flips the toggle while the save runs
        return manifest

    def _unload():
        unloads.append(1)
        backend.is_loaded = False

    backend.save_slots_for_resume = _save
    backend.unload_model = _unload
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)

    _drive_idle_loop(kw)
    assert unloads == [1]  # still unloads; only the stash is dropped
    assert kw.take_kv_resume() is None
    assert not state_file.exists()


def test_idle_ttl_disabled_mid_save_skips_unload(monkeypatch, tmp_path):
    import time
    from core.inference import llama_keepwarm as kw

    ttl = {"v": 0.005}
    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: ttl["v"])
    monkeypatch.setattr(settings, "get_auto_unload_keep_kv", lambda: True)
    kw._inflight = 0
    kw._pending = 0
    kw._last_active = time.monotonic() - 3600
    kw._last_unloaded_model = None
    kw._kv_resume = None

    unloads = []
    backend = _FakeBackend("unsloth/Idle-GGUF", hf_variant = "Q4_K_M")
    state_file = tmp_path / "resume-mid-slot0.bin"
    state_file.write_bytes(b"kv")
    manifest = {
        "dir": str(tmp_path),
        "binary": ("bin", 1),
        "slots": [{"id": 0, "filename": state_file.name, "n_saved": 1}],
    }

    def _save(should_abort = None):
        ttl["v"] = 0  # user turns idle unload off while the save runs
        return manifest

    backend.save_slots_for_resume = _save
    backend.unload_model = lambda: unloads.append(1)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)

    _drive_idle_loop(kw)
    assert unloads == []  # the unload was cancelled by the setting change
    assert kw.take_kv_resume() is None
    assert not state_file.exists()


def test_alias_reload_restores_slots_and_deletes_files(monkeypatch, tmp_path):
    from core.inference import llama_keepwarm as kw

    backend = _FakeBackend(None)  # idle-unload emptied the backend
    backend._slot_save_binary = ("/bin/llama-server", 111)
    restored = []
    backend.restore_slots_for_resume = lambda manifest: restored.append(manifest)

    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = True, resolves_to = None, backend = backend, recorder = rec)
    monkeypatch.setattr(kw, "_inflight", 0)
    state_file, manifest = _seed_kv_manifest(tmp_path)
    monkeypatch.setattr(kw, "_last_unloaded_model", (manifest["gguf"], "Q4_K_M"))
    monkeypatch.setattr(kw, "_kv_resume", manifest)

    _run_hook("gpt-4o-mini")
    assert len(rec.calls) == 1
    assert len(restored) == 1  # same model + binary: restore ran
    assert not state_file.exists()  # state file deleted after the restore
    assert kw._kv_resume is None


def test_no_restore_when_different_model_loads(monkeypatch, tmp_path):
    from core.inference import llama_keepwarm as kw

    backend = _FakeBackend(None)
    backend._slot_save_binary = ("/bin/llama-server", 111)
    restored = []
    backend.restore_slots_for_resume = lambda manifest: restored.append(manifest)
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("unsloth/B-GGUF", None, "unsloth/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    monkeypatch.setattr(kw, "_inflight", 0)
    state_file, manifest = _seed_kv_manifest(tmp_path)  # manifest is for model A
    monkeypatch.setattr(kw, "_kv_resume", manifest)

    _run_hook("unsloth/B-GGUF")
    assert len(rec.calls) == 1
    assert restored == []  # different model: never restored
    assert not state_file.exists()  # but the stale files are gone
    assert kw._kv_resume is None


def test_restore_skipped_when_binary_changed(monkeypatch, tmp_path):
    from core.inference import llama_keepwarm as kw

    state_file, manifest = _seed_kv_manifest(tmp_path)
    backend = _FakeBackend("unsloth/A-GGUF", hf_variant = "Q4_K_M")
    backend._gguf_path = manifest["gguf"]
    backend._slot_save_binary = ("/bin/llama-server", 222)  # newer mtime
    restored = []
    backend.restore_slots_for_resume = lambda manifest: restored.append(manifest)

    kw.restore_kv_resume(backend, manifest)
    assert restored == []
    assert not state_file.exists()


def test_restore_skipped_when_launch_config_changed(tmp_path):
    from core.inference import llama_keepwarm as kw

    state_file, manifest = _seed_kv_manifest(tmp_path)
    backend = _FakeBackend("unsloth/A-GGUF", hf_variant = "Q4_K_M")
    backend._gguf_path = manifest["gguf"]
    backend._slot_save_binary = ("/bin/llama-server", 111)
    backend._slot_launch_fingerprint = lambda: (("--rope-freq-scale", "0.5"), None, None, 1)
    restored = []
    backend.restore_slots_for_resume = lambda manifest: restored.append(manifest)

    kw.restore_kv_resume(backend, manifest)
    assert restored == []
    assert not state_file.exists()


def test_restore_skipped_when_gguf_rewritten_in_place(tmp_path):
    from core.inference import llama_keepwarm as kw

    state_file, manifest = _seed_kv_manifest(tmp_path)
    with open(manifest["gguf"], "wb") as fh:
        fh.write(b"different weights")  # same path, new content
    backend = _FakeBackend("unsloth/A-GGUF", hf_variant = "Q4_K_M")
    backend._gguf_path = manifest["gguf"]
    backend._slot_save_binary = ("/bin/llama-server", 111)
    restored = []
    backend.restore_slots_for_resume = lambda manifest: restored.append(manifest)

    kw.restore_kv_resume(backend, manifest)
    assert restored == []
    assert not state_file.exists()


def test_note_model_unloaded_purges_manifest_and_files(tmp_path):
    from core.inference import llama_keepwarm as kw

    state_file, manifest = _seed_kv_manifest(tmp_path)
    kw._set_last_unloaded(("org/A-GGUF", "Q4_K_M"))
    kw._set_kv_resume(manifest)
    kw.note_model_unloaded()
    assert kw.get_last_unloaded_model() is None
    assert kw.take_kv_resume() is None
    assert not state_file.exists()


def test_note_model_loaded_purges_manifest_and_files(tmp_path):
    from core.inference import llama_keepwarm as kw

    state_file, manifest = _seed_kv_manifest(tmp_path)
    kw._set_last_unloaded(("org/A-GGUF", "Q4_K_M"))
    kw._set_kv_resume(manifest)
    kw.note_model_loaded()
    assert kw.get_last_unloaded_model() is None
    assert kw.take_kv_resume() is None
    assert not state_file.exists()


def test_new_idle_save_purges_previous_manifest_files(tmp_path):
    from core.inference import llama_keepwarm as kw

    old_file, old_manifest = _seed_kv_manifest(tmp_path)
    kw._set_kv_resume(old_manifest)
    new_file = tmp_path / "resume-def-slot0.bin"
    new_file.write_bytes(b"kv2")
    kw._set_kv_resume(
        {
            "identity": ("unsloth/B-GGUF", None, "unsloth/B-GGUF"),
            "dir": str(tmp_path),
            "binary": ("/bin/llama-server", 111),
            "slots": [{"id": 0, "filename": new_file.name, "n_saved": 7}],
        }
    )
    assert not old_file.exists()  # replaced manifest's files purged
    assert new_file.exists()
    assert kw.take_kv_resume()["slots"][0]["filename"] == new_file.name


def test_sweep_slot_save_dir_removes_only_resume_files(monkeypatch, tmp_path):
    from core.inference import llama_keepwarm as kw
    from utils.paths import storage_roots

    monkeypatch.setattr(storage_roots, "llama_slot_cache_root", lambda: tmp_path)
    stale = tmp_path / "resume-old-slot0.bin"
    stale.write_bytes(b"kv")
    other = tmp_path / "unrelated.txt"
    other.write_text("keep")
    kw.sweep_slot_save_dir()
    assert not stale.exists()
    assert other.exists()


def test_keep_kv_setting_roundtrip_and_default(monkeypatch):
    import storage.studio_db as db

    store = {}
    monkeypatch.setattr(db, "upsert_app_settings", lambda m: store.update(m))
    monkeypatch.setattr(settings, "_cached_setting", lambda k, d = None: store.get(k, d))

    assert settings.get_auto_unload_keep_kv() is True  # default when never stored
    assert settings.set_openai_auto_switch(True, 60, False)[2] is False
    assert store[settings.AUTO_UNLOAD_KEEP_KV_SETTING_KEY] is False
    assert settings.get_auto_unload_keep_kv() is False
    # None leaves the stored value untouched (older clients can't reset it).
    assert settings.set_openai_auto_switch(True, 60, None)[2] is False
    assert store[settings.AUTO_UNLOAD_KEEP_KV_SETTING_KEY] is False
    with pytest.raises(ValueError, match = "true or false"):
        settings.set_openai_auto_switch(True, 60, "garbage")


def test_stale_stash_cleanup_waits_for_lifecycle_gate(monkeypatch, tmp_path):
    # The loop's stale-stash purge must wait on the gate a mid-reload holds.
    import time
    from core.inference import llama_keepwarm as kw

    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 3600)
    kw._inflight = 0
    kw._pending = 0
    kw._last_active = time.monotonic()
    backend = _FakeBackend("unsloth/New-GGUF")
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    state_file, manifest = _seed_kv_manifest(tmp_path)
    kw._kv_resume = manifest
    kw._last_unloaded_model = ("unsloth/A-GGUF", "Q4_K_M")

    assert kw._lifecycle_lock.acquire(blocking = False)  # simulate in-flight reload
    try:
        _drive_idle_loop(kw)
        assert kw._kv_resume is manifest  # purge deferred while the gate is held
        assert state_file.exists()
    finally:
        kw._lifecycle_lock.release()
    _drive_idle_loop(kw)
    assert kw._kv_resume is None  # gate freed: genuinely stale stash purged
    assert not state_file.exists()


def test_put_route_disabling_keep_kv_purges_saved_state(monkeypatch, tmp_path):
    import routes.settings as settings_route
    import storage.studio_db as db
    from core.inference import llama_keepwarm as kw

    store = {}
    monkeypatch.setattr(db, "upsert_app_settings", lambda m: store.update(m))
    monkeypatch.setattr(settings, "_cached_setting", lambda k, d = None: store.get(k, d))
    state_file, manifest = _seed_kv_manifest(tmp_path)
    monkeypatch.setattr(kw, "_kv_resume", manifest)

    payload = settings_route.OpenAIAutoSwitchPayload(enabled = True, auto_unload_keep_kv = False)
    resp = settings_route.update_openai_auto_switch(payload, "tester")
    assert resp.auto_unload_keep_kv is False
    assert kw._kv_resume is None
    assert not state_file.exists()


def test_keep_kv_only_update_leaves_env_idle_ttl_active(monkeypatch):
    # A keep-KV-only update must not materialize the env TTL as a stored value.
    import routes.settings as settings_route
    import storage.studio_db as db

    store = {}
    monkeypatch.setattr(db, "upsert_app_settings", lambda m: store.update(m))
    monkeypatch.setattr(settings, "_cached_setting", lambda k, d = None: store.get(k, d))
    monkeypatch.setenv(settings.MODEL_IDLE_TTL_ENV_VAR, "600")

    assert settings_route.OpenAIAutoSwitchPayload(enabled = False).auto_unload_idle_seconds is None
    enabled, idle, keep_kv = settings.set_openai_auto_switch(False, None, False)
    assert settings.AUTO_UNLOAD_IDLE_SETTING_KEY not in store  # idle untouched
    assert settings.get_auto_unload_idle_seconds() == 600  # env TTL still active
    assert (enabled, idle, keep_kv) == (False, 600, False)


def test_load_impl_notes_loaded_with_backend_off_loop():
    import inspect
    src = inspect.getsource(inference_route._load_model_impl)
    assert "to_thread(note_model_loaded, llama_backend)" in src


def test_restore_matches_gguf_realpath_across_naming(tmp_path):
    from core.inference import llama_keepwarm as kw

    blob = tmp_path / "blob.gguf"
    blob.write_bytes(b"gguf")
    link = tmp_path / "snapshot.gguf"
    try:
        link.symlink_to(blob)
    except OSError:
        pytest.skip("symlinks unsupported on this host")

    backend = _FakeBackend("/hf/snapshots/d7f5", hf_variant = None)
    backend._gguf_path = str(link)  # reload resolved the symlink spelling
    backend._slot_save_binary = ("/bin/llama-server", 111)
    restored = []
    backend.restore_slots_for_resume = lambda manifest: restored.append(manifest)
    state_file, manifest = _seed_kv_manifest(
        tmp_path, identity = ("unsloth/A-GGUF", None, "unsloth/A-GGUF"), gguf = str(blob)
    )

    kw.restore_kv_resume(backend, manifest)
    assert len(restored) == 1  # names differ, file identical: restore ran
    assert not state_file.exists()


def test_setter_rejects_idle_below_floor(monkeypatch):
    import storage.studio_db as db

    writes = []
    monkeypatch.setattr(db, "upsert_app_settings", lambda m: writes.append(dict(m)))
    settings._cache.clear()

    with pytest.raises(ValueError, match = "at least 60"):
        settings.set_openai_auto_switch(True, 30)
    assert writes == []  # rejected before any persist
    # 0 (off) and >= 60 pass through unchanged.
    assert settings.set_openai_auto_switch(True, 0)[1] == 0
    assert settings.set_openai_auto_switch(True, 60)[1] == 60
    assert settings.set_openai_auto_switch(True, 3600)[1] == 3600


def test_put_route_rejects_idle_below_floor():
    import routes.settings as settings_route
    from fastapi import HTTPException

    payload = settings_route.OpenAIAutoSwitchPayload(enabled = True, auto_unload_idle_seconds = 30)
    with pytest.raises(HTTPException) as excinfo:
        settings_route.update_openai_auto_switch(payload, "tester")
    assert excinfo.value.status_code == 400


def test_stored_legacy_idle_below_floor_is_clamped(monkeypatch):
    # Values persisted before the floor existed are raised to it on read, for
    # both the effective TTL and the value the settings UI displays.
    store = {settings.AUTO_UNLOAD_IDLE_SETTING_KEY: 5}
    monkeypatch.setattr(settings, "_cached_setting", lambda k, d = None: store.get(k, d))
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    assert settings.get_auto_unload_idle_seconds() == 60
    assert settings.get_stored_auto_unload_idle_seconds() == 60
    store[settings.AUTO_UNLOAD_IDLE_SETTING_KEY] = 90
    assert settings.get_auto_unload_idle_seconds() == 90


def test_env_idle_below_floor_is_clamped(monkeypatch):
    monkeypatch.setattr(settings, "_cached_setting", lambda k, d = None: d)
    monkeypatch.setenv(settings.MODEL_IDLE_TTL_ENV_VAR, "5")
    assert settings.get_auto_unload_idle_seconds() == 60
    monkeypatch.setenv(settings.MODEL_IDLE_TTL_ENV_VAR, "0")
    assert settings.get_auto_unload_idle_seconds() == 0
    monkeypatch.setenv(settings.MODEL_IDLE_TTL_ENV_VAR, "600")
    assert settings.get_auto_unload_idle_seconds() == 600
    monkeypatch.delenv(settings.MODEL_IDLE_TTL_ENV_VAR)
    assert settings.get_auto_unload_idle_seconds() == 0
