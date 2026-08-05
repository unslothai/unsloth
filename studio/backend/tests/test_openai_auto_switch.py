# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in OpenAI /v1 model auto-switch: resolver, hook, settings coercion.

Backend and load route are mocked (no GPU/llama-server).
"""

import asyncio
import json
import os
import threading

import pytest
from fastapi import HTTPException

import routes.inference as inference_route
from models.inference import LoadRequest
from core.inference import local_model_resolver as resolver
from utils import openai_auto_switch_settings as settings


@pytest.fixture(autouse = True)
def _clean_resolver_index():
    """Drop the scan cache around every test.

    The /v1 admission hook warms the index in the background, so a test exercising it
    can publish its fixture's scan and, inside the TTL, hand it to the next test.
    """
    resolver.invalidate_index()
    yield
    resolver.invalidate_index()


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
        # Mirror _load_model_impl: a load advertises its own id until auto-switch overwrites it.
        self.backend._openai_advertised_id = None
        from core.inference import llama_keepwarm as kw

        kw.note_model_loaded(self.backend)
        return None


def _wire(monkeypatch, *, enabled, resolves_to, backend, recorder):
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: enabled)
    monkeypatch.setattr(resolver, "resolve_local_gguf", lambda _m, **_kw: resolves_to)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    # Auto-switch calls _load_model_impl directly: it already holds the lifecycle gate.
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
    # Off means no load, but A must not answer as B either: say why instead.
    with pytest.raises(HTTPException) as excinfo:
        _run_hook("unsloth/B-GGUF")
    assert excinfo.value.status_code == 404
    assert "Switch model by request" in str(excinfo.value.detail)
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
    # The hook must run before either dispatcher so streaming requests switch too.
    # Asserted on source (order-independent on the shared inference module).
    import inspect

    src = inspect.getsource(inference_route.openai_responses)
    assert "_maybe_auto_switch_model" in src
    hook_at = src.index("_maybe_auto_switch_model")
    assert hook_at < src.index("_responses_stream")
    assert hook_at < src.index("_responses_non_streaming")


def test_embeddings_endpoint_wires_auto_switch_before_loaded_check():
    # /v1/embeddings must auto-switch before the loaded-state gate. Asserted on source.
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
    # A helper wedged between @router.post and its handler silently rebinds the route
    # and drops its auth dependency (this hit /messages/count_tokens). Lock the path ->
    # (handler, auth) mapping at the route level to catch any decorator/handler split.
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

    # HF-cache snapshot with a quant subdir (nested layout the old shallow glob missed).
    repo = tmp_path / "models--org--repo"
    (repo / "snapshots" / "abc" / "BF16").mkdir(parents = True)
    (repo / "snapshots" / "abc" / "BF16" / "model-BF16.gguf").write_text("x")
    e2 = resolver._local_gguf_entry("org/repo", SimpleNamespace(path = str(repo)))
    assert e2 is not None and e2.variants


def test_local_gguf_entry_rejects_standalone_companions(tmp_path, monkeypatch):
    # Codex P2: the standalone-.gguf scan pass emits a bare mmproj projector (mmproj is
    # only filtered inside directory scans). A projector is not servable, so the resolver
    # must reject it, else /v1/models advertises it and a switch could load it over the weights.
    from types import SimpleNamespace

    proj = tmp_path / "mmproj-F16.gguf"
    proj.write_text("x")
    assert resolver._local_gguf_entry("p", SimpleNamespace(path = str(proj))) is None
    assert resolver.info_has_local_gguf(SimpleNamespace(id = str(proj), path = str(proj))) is False
    root = tmp_path / "MTP"
    root.mkdir()
    main = root / "Qwen3.6-27B-MTP-Q6_K.gguf"
    terminal = root / "gemma-4-12b-it-Q8_0-MTP.gguf"
    prefixed = root / "mtp-gemma-4-12b-it.gguf"
    for file in (main, terminal, prefixed):
        file.write_text("x")
    monkeypatch.setattr("storage.studio_db.list_scan_folders", lambda: [{"path": str(root)}])
    assert resolver._local_gguf_entry("main", SimpleNamespace(path = str(main))) is not None
    assert resolver._local_gguf_entry("terminal", SimpleNamespace(path = str(terminal))) is None
    assert resolver._local_gguf_entry("prefixed", SimpleNamespace(path = str(prefixed))) is None


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
    # Best-effort: any internal failure falls through to None so the request still
    # serves the loaded model instead of 500-ing (the hook has no guard of its own).
    def boom():
        raise RuntimeError("scan blew up")

    monkeypatch.setattr(resolver, "_build_index", boom)
    resolver._scan = (0.0, {})
    assert resolver.resolve_local_gguf("unsloth/B-GGUF") is None


def test_resolver_nonstring_model_is_failsafe():
    # Raw endpoints pass body.get("model") through, so a non-string must not raise on .strip().
    assert resolver.resolve_local_gguf(123) is None
    assert resolver.resolve_local_gguf({"a": 1}) is None
    assert resolver.resolve_local_gguf(None) is None


def test_describe_local_miss_separates_missing_repo_from_missing_quant(monkeypatch):
    # Two different misses: the repo isn't downloaded, or only that quant is absent.
    monkeypatch.setattr(
        resolver,
        "_build_index",
        lambda: {"unsloth/b-gguf": _entry("unsloth/B-GGUF", "UD-Q5_K_XL", "Q4_K_M")},
    )
    resolver._scan = (0.0, {})
    assert resolver.describe_local_miss("unsloth/B-GGUF:Q8_0") == (
        resolver.MISS_VARIANT_NOT_FOUND,
        ("UD-Q5_K_XL", "Q4_K_M"),
    )
    # Split the same way resolve_local_gguf does, so the two never disagree.
    assert resolver.describe_local_miss("unsloth/b-gguf:q8_0")[0] == (
        resolver.MISS_VARIANT_NOT_FOUND
    )
    # Unknown repo, and a bare id with no ":VARIANT" to blame.
    assert resolver.describe_local_miss("totally/unknown:Q8_0") == (
        resolver.MISS_MODEL_NOT_FOUND,
        (),
    )
    assert resolver.describe_local_miss("unsloth/B-GGUF") == (resolver.MISS_MODEL_NOT_FOUND, ())


def test_describe_local_miss_is_failsafe(monkeypatch):
    # Runs inside an error path, so a broken scan must degrade, not turn a 4xx into a 500.
    def boom():
        raise RuntimeError("scan blew up")

    monkeypatch.setattr(resolver, "_build_index", boom)
    resolver._scan = (0.0, {})
    assert resolver.describe_local_miss("unsloth/B-GGUF:Q8_0") == (
        resolver.MISS_MODEL_NOT_FOUND,
        (),
    )
    assert resolver.describe_local_miss(123) == (resolver.MISS_MODEL_NOT_FOUND, ())
    assert resolver.describe_local_miss("") == (resolver.MISS_MODEL_NOT_FOUND, ())


def test_resolver_exact_id_with_colon_wins(monkeypatch):
    # A local id containing a colon (e.g. a Windows path) matches exactly, not split at it.
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
    # A load stamps activity, so the next poll must not unload a freshly loaded model.
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
    # Headline path: TTL elapsed and nothing in flight -> free the GGUF once and stash
    # its identity so a later alias request can reload that variant.
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
        # Wall clock, not an iteration count: Windows rounds a 10 ms sleep to its ~15.6 ms tick.
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
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
    monkeypatch.setattr(
        settings_route, "set_openai_auto_switch", lambda *a: (False, 300, True, False)
    )
    monkeypatch.setattr(settings_route, "get_auto_unload_idle_seconds", lambda: 0)

    payload = settings_route.OpenAIAutoSwitchPayload(enabled = False)
    resp = settings_route.update_openai_auto_switch(payload, "tester")
    assert resp.idle_unload_active is False and resp.auto_unload_keep_kv is True
    assert kw._kv_resume is None and not saved.exists()


def test_audio_generate_is_tracked_as_inference_path():
    # Direct GGUF TTS uses the llama backend and can outlive the TTL, so it must be tracked.
    from core.inference.llama_keepwarm import _is_inference_path

    assert _is_inference_path("/api/inference/audio/generate") is True
    assert _is_inference_path("/v1/chat/completions") is True
    assert _is_inference_path("/api/inference/models/list") is False


def test_idle_loop_does_not_unload_while_request_inflight(monkeypatch):
    # An in-flight request protects the model even when idle by wall-clock past the TTL.
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
    # A configured model loads with its saved launch flags, not defaults.
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
    # Only llama_extra_args configured: applied; max_seq_length stays default.
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
    """In-memory dict backing the override read + atomic-merge write."""
    import storage.studio_db as db

    store = {}

    def _merge_entry(
        key,
        entry_key,
        entry_value,
        *,
        fill_absent_fields = False,
    ):
        current = dict(store.get(key) or {})
        if fill_absent_fields:
            # Fill only: every stored value wins, nothing is deleted.
            if not entry_value:
                return current
            stored = current.get(entry_key)
            if isinstance(stored, dict):
                current[entry_key] = {**entry_value, **stored}
            else:
                current[entry_key] = entry_value
        elif entry_value:
            current[entry_key] = entry_value
        else:
            current.pop(entry_key, None)
        store[key] = current
        return current

    monkeypatch.setattr(db, "upsert_app_setting_map_entry", _merge_entry)
    monkeypatch.setattr(db, "get_app_setting", lambda k, default = None: store.get(k, default))
    settings._cache.clear()
    return store


@pytest.fixture
def override_store(monkeypatch):
    """The in-memory override store, for a test that needs nothing else mocked."""
    _mock_override_store(monkeypatch)


def _put(model_id, **fields):
    """One override PUT through the route, spelled the way the UI sends it."""
    import routes.settings as settings_route
    return settings_route.update_openai_auto_switch_override(
        settings_route.ModelOverridePayload(model_id = model_id, **fields),
        "tester",
    )


def test_model_override_roundtrip(monkeypatch):
    _mock_override_store(monkeypatch)

    settings.set_model_override(
        "unsloth/B-GGUF", llama_extra_args = ["--n-gpu-layers", "20"], max_seq_length = 4096
    )
    assert settings.get_model_override("unsloth/B-GGUF") == {
        "llama_extra_args": ["--n-gpu-layers", "20"],
        "max_seq_length": 4096,
    }
    # An empty override removes the entry rather than storing {}.
    settings.set_model_override("unsloth/B-GGUF", llama_extra_args = [], max_seq_length = None)
    assert settings.get_model_override("unsloth/B-GGUF") == {}
    assert settings.get_model_overrides() == {}


def test_override_route_rejects_managed_flag_and_removes(monkeypatch):
    import routes.settings as settings_route
    from fastapi import HTTPException

    _mock_override_store(monkeypatch)

    # A managed/denylisted llama-server flag rejects with 400, not 500.
    bad = settings_route.ModelOverridePayload(
        model_id = "unsloth/B-GGUF", llama_extra_args = ["--port", "1234"]
    )
    with pytest.raises(HTTPException) as excinfo:
        settings_route.update_openai_auto_switch_override(bad, "tester")
    assert excinfo.value.status_code == 400

    # Store a valid override, then an empty payload removes it through the route.
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
    # 0 is invalid and the setter drops falsy values, so the payload must reject it at the boundary.
    import pydantic
    import routes.settings as settings_route

    with pytest.raises(pydantic.ValidationError):
        settings_route.ModelOverridePayload(model_id = "x", max_seq_length = 0)
    assert settings_route.ModelOverridePayload(model_id = "x", max_seq_length = 1).max_seq_length == 1


def test_update_openai_auto_switch_writes_both_keys_in_one_transaction(monkeypatch):
    # The PUT must persist enabled + idle in one upsert, never leaving one key stale.
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
    # Codex P2: UNSLOTH_MODEL_IDLE_TTL drives idle-unload while the toggle is off, so
    # the settings response must report idle_unload_active (env-active, not "needs enable").
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
    # The resolver lowercases its index, so a case-differing retrieve must still hit
    # (200), not 404. Guards the .lower() compare in openai_retrieve_model (the loaded
    # fast-path is exact, the catalog loop is lenient; catalog is main's #6519).
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
    # Models hidden from Unsloth's pickers (validation probe, RAG weights) must never
    # become auto-switch targets.
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
    assert "org/normal-gguf" in index  # keys normalized to lowercase
    assert "ggml-org/models" not in index
    assert "unsloth/bge-small-en-v1.5-gguf" not in index
    assert str(local_default_embedder).lower() not in index
    # And the hidden probe cannot be auto-switched to by name.
    resolver._scan = (0.0, {})
    assert resolver.resolve_local_gguf("ggml-org/models") is None


def test_idle_disabled_when_auto_switch_off(monkeypatch):
    # Off means unchanged: a stored idle TTL reports 0 while auto-switch is off, so
    # nothing can unload the model.
    store = {settings.AUTO_UNLOAD_IDLE_SETTING_KEY: 60}
    monkeypatch.setattr(settings, "_cached_setting", lambda k, d = None: store.get(k, d))
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: False)
    assert settings.get_auto_unload_idle_seconds() == 0
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    assert settings.get_auto_unload_idle_seconds() == 60


def test_count_tokens_is_tracked_as_inference_path():
    # count_tokens uses the loaded tokenizer, so it must be a tracked in-flight path.
    from core.inference.llama_keepwarm import _is_inference_path

    assert _is_inference_path("/v1/messages/count_tokens") is True
    assert _is_inference_path("/api/inference/messages/count_tokens") is True
    assert _is_inference_path("/api/inference/chat/count_tokens") is True
    assert _is_inference_path("/v1/messages") is True


# ── review follow-ups: bare-id reuse, responses order, in-flight tracking ──


def test_bare_id_tolerates_any_loaded_variant(monkeypatch):
    # Repo loaded as Q4_K_M; a BARE request for the same repo (resolver picks Q8_0)
    # must NOT reload a different quant.
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
    # An explicit :VARIANT still honors the quant (reloads to Q8_0).
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
    # A 400 on empty input must not have triggered a load, so the hook runs after validation.
    import inspect

    src = inspect.getsource(inference_route.openai_responses)
    assert "No input provided" in src
    assert src.index("No input provided") < src.index("_maybe_auto_switch_model")


def test_responses_system_only_rejected_before_switch(monkeypatch):
    # Codex P2: instructions-only input normalises to a lone system message and passes
    # the empty-input check, so it must 400 before the switch, never evicting the model.
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
    # In-flight is counted whenever auto-switch is on (even idle TTL 0), so enabling
    # idle mid-stream cannot unload an in-flight request.
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
    # OFF + nothing loaded + unparseable body still 503s (pre-feature), not 500 from the body read.
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
    # A non-string model must be treated as absent (never raising in membership checks),
    # even when an idle-unload stash exists.
    from core.inference import llama_keepwarm as kw

    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = True, resolves_to = None, backend = backend, recorder = rec)
    monkeypatch.setattr(kw, "_last_unloaded_model", ("unsloth/A-GGUF", None))
    asyncio.run(inference_route._maybe_auto_switch_model(123, object(), "tester"))
    assert rec.calls == []  # no load, no TypeError


def test_anthropic_validates_max_tokens_before_auto_switch():
    # A missing max_tokens must 400 before the hook, never loading a model. Asserted on source order.
    import inspect

    src = inspect.getsource(inference_route.anthropic_messages)
    assert "_maybe_auto_switch_model" in src
    assert src.index("max_tokens: field required") < src.index("_maybe_auto_switch_model")


def test_alias_reloads_model_freed_by_idle_unload_with_quant(monkeypatch):
    # After idle-unload, an unknown/alias name (resolves to None) reloads the freed
    # model with its exact quant instead of 503-ing.
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
    # Reload only fires on an empty backend; with a model loaded, an unknown name
    # falls through without resurrecting the stash.
    from core.inference import llama_keepwarm as kw

    backend = _FakeBackend("unsloth/B-GGUF")
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = True, resolves_to = None, backend = backend, recorder = rec)
    monkeypatch.setattr(kw, "_last_unloaded_model", ("unsloth/A-GGUF", None))
    _run_hook("gpt-4o-mini")
    assert rec.calls == []


def test_idle_loop_does_not_unload_while_request_pending(monkeypatch):
    # A pending request (waiting on the unload gate, not yet started) blocks idle unload.
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
    # A stream started while OFF is still counted, so enabling idle-unload mid-stream can't unload it.
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
    # _build_index must scan the same roots the picker lists, else a UI-shown model is
    # silently served as the loaded one. Verify each root is consulted.
    from pathlib import Path
    import routes.models as models_route
    from utils import paths as upaths
    from utils import hf_cache_settings
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
        lambda d, **_: scanned.append(("hf", str(Path(d).resolve()))) or [],
    )
    monkeypatch.setattr(
        models_route,
        "_scan_lmstudio_dir",
        lambda d: scanned.append(("lm", str(Path(d).resolve()))) or [],
    )
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: tmp_path / "active")
    monkeypatch.setattr(models_route, "_is_hidden_model", lambda *a, **k: False)
    monkeypatch.setattr(
        hf_cache_settings,
        "known_hf_hub_caches",
        lambda: [tmp_path / "active", tmp_path / "previous"],
    )
    monkeypatch.setattr(upaths, "legacy_hf_cache_dir", lambda: tmp_path / "legacy")
    monkeypatch.setattr(upaths, "hf_default_cache_dir", lambda: tmp_path / "default")
    monkeypatch.setattr(upaths, "lmstudio_model_dirs", lambda: [tmp_path / "lmstudio"])
    monkeypatch.setattr(
        studio_db, "list_scan_folders", lambda: [{"path": str(tmp_path / "custom")}]
    )
    for sub in ("active", "previous", "legacy", "default", "lmstudio", "custom"):
        (tmp_path / sub).mkdir()

    resolver._build_index()

    hf = {p for k, p in scanned if k == "hf"}
    lm = {p for k, p in scanned if k == "lm"}
    assert str((tmp_path / "legacy").resolve()) in hf
    assert str((tmp_path / "default").resolve()) in hf
    assert str((tmp_path / "previous").resolve()) in hf
    assert str((tmp_path / "custom").resolve()) in hf
    assert str((tmp_path / "lmstudio").resolve()) in lm


# ── gemini round: list-body 400, non-POST not tracked ──


def _json_body_request(payload):
    class _Req:
        async def json(self):
            return payload

    return _Req()


def test_completions_list_body_is_400_not_500(monkeypatch):
    # A valid JSON non-dict body (a list) on a loaded backend is a clean 400, not a 500.
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
    # CORS preflight (OPTIONS) on an inference path is not tracked as in-flight.
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
    # Same repo, different quant is a fresh model: the idle timer resets, so the new
    # variant isn't unloaded before one TTL of its own.
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
        assert unloads == []  # timer reset by the variant change
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
    # An HF-cache repo resolves to its on-disk snapshot dir (local branch, no download);
    # loader_id stays the repo id.
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


def _revision_pair(root, complete: bool):
    """Two revisions of one cache repo; the newer one is optionally half-downloaded."""
    snaps = root / "models--org--Repo" / "snapshots"
    old, new = snaps / "rev-old", snaps / "rev-new"
    for path in (old, new):
        path.mkdir(parents = True)
    (old / "model-Q8_0.gguf").write_bytes(b"GGUF stub")
    name = "model-Q4_K_M.gguf" if complete else "model-Q4_K_M-00001-of-00003.gguf"
    (new / name).write_bytes(b"GGUF stub")
    return old, new


def test_sibling_revision_resolves_to_its_own_weights(tmp_path):
    # /v1/models advertises only the snapshot dir name, so a durable pin holds one
    # revision hash. A newer snapshot must not strand it, and the old revision must
    # resolve to ITS OWN directory rather than be redirected onto the newest.
    old, new = _revision_pair(tmp_path, complete = True)

    found = dict(resolver._sibling_revision_entries(str(new), "org/Repo"))

    assert "rev-old" in found
    assert found["rev-old"].load_path == str(old)


def test_incomplete_sibling_revision_is_not_indexed(tmp_path):
    # A half-downloaded revision cannot load, so naming it must not resolve to it.
    old, _new = _revision_pair(tmp_path, complete = False)
    # Point the scan at the complete one; the partial sibling is the candidate here.
    found = dict(resolver._sibling_revision_entries(str(old), "org/Repo"))

    assert "rev-new" not in found


def test_sibling_revisions_ignore_a_scan_folder_named_snapshots(tmp_path):
    # A user scan folder called "snapshots" holds unrelated models, not revisions of
    # one repo; treating them as revisions would silently serve model-a as model-b.
    snaps = tmp_path / "snapshots"
    for name in ("model-a", "model-b"):
        (snaps / name).mkdir(parents = True)
        (snaps / name / "model-Q4_K_M.gguf").write_bytes(b"GGUF stub")

    found = dict(resolver._sibling_revision_entries(str(snaps / "model-a"), "model-a"))

    assert found == {}


def test_sibling_revisions_skip_plain_repo_ids():
    assert dict(resolver._sibling_revision_entries("org/Repo-GGUF", "org/Repo-GGUF")) == {}


def test_already_loaded_by_repo_id_is_not_reswapped(monkeypatch):
    # A normal load has model_identifier == repo id while the resolver returns the
    # concrete path. A request for that repo counts as already serving (no reload, no
    # 409) even with another inference active.
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
    # After a load-by-path the backend advertises the repo id (override key), not the
    # path, so /v1/models and the idle stash stay name-based.
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
    # Codex P2: a model loaded by path and requested via an alias resolving to that path
    # is already serving (no reload). Unless the alias is recorded as the advertised id
    # on the already-serving return, /v1/models reports the path basename and lists the
    # alias as loaded:false.
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
    # _llama_public_model_id (prefers _openai_advertised_id), not raw model_identifier.
    # After a switch to a cached HF GGUF the identifier is the snapshot path, so the raw
    # form would stream a basename while everything else reports the repo id.
    import inspect

    src = inspect.getsource(inference_route._responses_stream)
    assert "_clean_model = _llama_public_model_id(llama_backend" in src
    assert 'public_model_id(getattr(llama_backend, "model_identifier"' not in src


def test_concurrent_same_target_requests_load_once(monkeypatch):
    # Two concurrent requests for the same unloaded model load once, not 409 each other.
    # The second is already registered while the first runs the hook, _inflight counts both.
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
    # The stash carries (load_path, quant, advertised_id). An alias reload looks up the
    # override by the advertised repo id, not the path, so saved launch flags survive.
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
    # The manual /load route must wrap the load in inference_lifecycle_gate so idle-unload
    # can't fire mid-load. Asserted on the gated coroutine: the route only adds the padding.
    import inspect

    src = inspect.getsource(inference_route.load_model_gated)
    assert "inference_lifecycle_gate" in src
    assert "_load_model_impl" in src


def test_model_replacements_recheck_sidecar_swap_before_either_backend_is_unloaded():
    # Both replacement directions drain, then recheck whether a sidecar install reserved the
    # gate meanwhile. That recheck is the last thing that can reject the load, so the
    # destructive cancel must follow it. Exact-model reuse exits earlier and never waits.
    import inspect

    src = inspect.getsource(inference_route._load_model_impl)
    already_loaded = src.index('status = "already_loaded"')
    standard_branch = src.index("# ── Standard path")

    gguf_wait = src.index("await _wait_for_model_switch_idle", src.index("if config.is_gguf:"))
    gguf_sidecar_check = src.index("_raise_if_sidecar_swap_in_progress()", gguf_wait)
    gguf_cancel = src.index("on_reload_confirmed(cancel = True)", gguf_wait)
    unload_unsloth = src.index("unsloth_backend.unload_model", gguf_wait)

    standard_wait = src.index("await _wait_for_model_switch_idle", standard_branch)
    standard_sidecar_check = src.index("_raise_if_sidecar_swap_in_progress()", standard_wait)
    standard_cancel = src.index("on_reload_confirmed(cancel = True)", standard_wait)
    # No parens: both teardowns are asyncio.to_thread args (on-loop a 160s one would
    # block /load's tunnel padding).
    unload_gguf = src.index("llama_backend.unload_model", standard_wait)

    assert already_loaded < gguf_wait < gguf_sidecar_check < gguf_cancel < unload_unsloth
    assert standard_branch < standard_wait < standard_sidecar_check
    assert standard_sidecar_check < standard_cancel < unload_gguf


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
    # Default-off parity: unloaded backend + auto-switch off 503s before the max_tokens 400.
    from fastapi import HTTPException

    backend = _FakeBackend(None)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: False)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.anthropic_messages(_anthropic_payload(), object(), "tester"))
    assert exc.value.status_code == 503


def test_anthropic_400_when_auto_switch_on_and_max_tokens_missing(monkeypatch):
    # With auto-switch on, shape validation runs first: a missing max_tokens 400s before any load.
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
    # The middleware counts a concurrent same-model request as in-flight before it resolves
    # and registers a target waiter. Treat it as active until its target is known, then
    # recognize it as another queued switch request.
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


def test_untrack_preview_request_balances_both_counters():
    # A /p/ request untracking itself must drop from BOTH _inflight and the preview
    # subset, or the busy guard sees phantom foreign traffic.
    from core.inference import llama_keepwarm as kw

    kw._inflight = 1
    kw._preview_inflight = 1
    scope = {"type": "http", "path": "/p/demorun/v1/chat/completions"}
    kw.untrack_current_request(scope)
    assert kw._inflight == 0
    assert kw._preview_inflight == 0
    assert scope.get(kw._UNTRACKED_SCOPE_KEY) is True
    kw.untrack_current_request(scope)  # idempotent
    assert kw._inflight == 0 and kw._preview_inflight == 0


def test_preview_pending_counter_balances_and_flags_studio_waiters():
    # A queued preview is not foreign to a preview swap, but a queued Studio request is;
    # the subset tracks _pending at the same sites.
    from core.inference import llama_keepwarm as kw

    kw._pending = 0
    kw._preview_pending = 0
    kw._note_pending(is_preview = True)
    assert kw._pending == 1 and kw._preview_pending == 1
    assert kw.other_non_preview_pending_count() == 0  # queued preview is not foreign
    kw._note_pending(is_preview = False)
    assert kw._pending == 2 and kw._preview_pending == 1
    assert kw.other_non_preview_pending_count() == 1  # queued Studio request is foreign
    kw._note_start(is_preview = True)  # preview moves pending -> inflight
    assert kw._pending == 1 and kw._preview_pending == 0
    kw._note_unpending(is_preview = False)  # Studio request leaves the queue
    assert kw._pending == 0 and kw._preview_pending == 0
    kw._inflight = 0
    kw._preview_inflight = 0


def test_already_serving_clears_preview_marker(monkeypatch):
    # A non-preview auto-switch adopting a preview-owned model (already serving, no
    # reload) must claim it for Studio, else a later preview swaps it out from under
    # the active OpenAI caller.
    path = "/cache/models--org--Repo-GGUF/snapshots/abc"
    backend = _FakeBackend(path, hf_variant = "Q4_K_M")
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = (path, "Q4_K_M", "org/Repo-GGUF"),
        backend = backend,
        recorder = rec,
    )
    inference_route._set_preview_resident(path)
    assert inference_route._is_preview_resident(path)
    _run_hook("org/Repo-GGUF:Q4_K_M")
    assert rec.calls == []  # already serving -> no reload
    assert not inference_route._is_preview_resident(path)  # claimed for Studio
    inference_route._set_preview_resident(None)  # cleanup


def test_should_validate_before_switch_fires_on_preview_owned_slot(monkeypatch):
    # With both features off no load runs, but the switch helper still claims a preview-
    # owned slot, so the pre-load validation must run: the predicate is True whenever the
    # slot is preview-owned.
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: False)
    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 0)
    inference_route._set_preview_resident(None)
    assert inference_route._should_validate_before_switch() is False
    inference_route._set_preview_resident("/outputs/run/ckpt")
    assert inference_route._should_validate_before_switch() is True
    inference_route._set_preview_resident(None)  # cleanup


def test_chat_system_only_preserves_preview_marker_when_both_off(monkeypatch):
    # Both features off: a rejected system-only chat still reaches the slot claim, so the
    # pre-load validation must run (slot preview-owned) and 400 before the claim converts
    # the preview model to Studio-owned and strands the next preview.
    from fastapi import HTTPException
    from models.inference import ChatCompletionRequest

    path = "/outputs/run/ckpt-a"
    backend = _FakeBackend(path)
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = False, resolves_to = None, backend = backend, recorder = rec)
    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 0)
    inference_route._set_preview_resident(path)
    assert inference_route._is_preview_resident(path)
    payload = ChatCompletionRequest(model = path, messages = [{"role": "system", "content": "sys"}])
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))
    assert exc.value.status_code == 400
    assert rec.calls == []  # never switched
    assert inference_route._is_preview_resident(path)  # claim never ran, preview kept
    inference_route._set_preview_resident(None)  # cleanup


def test_completions_non_object_body_preserves_preview_marker(monkeypatch):
    # A non-object body ([]) is rejected but reaches the slot claim via
    # _auto_switch_from_request_body (model=None). Reject before the claim so a preview-
    # owned model isn't converted to Studio-owned for a request that never runs.
    from fastapi import HTTPException

    path = "/outputs/run/ckpt-a"
    backend = _FakeBackend(path)
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = False, resolves_to = None, backend = backend, recorder = rec)
    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 0)
    inference_route._set_preview_resident(path)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_completions(_json_body_request([]), "tester"))
    assert exc.value.status_code == 400
    assert rec.calls == []  # never switched
    assert inference_route._is_preview_resident(path)  # claim never ran, preview kept
    inference_route._set_preview_resident(None)  # cleanup


def test_embeddings_non_object_body_preserves_preview_marker(monkeypatch):
    # Like completions: a non-object embeddings body ([]) is rejected before the slot
    # claim so it can't strand a preview-owned model.
    from fastapi import HTTPException

    path = "/outputs/run/ckpt-a"
    backend = _FakeBackend(path)
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = False, resolves_to = None, backend = backend, recorder = rec)
    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 0)
    inference_route._set_preview_resident(path)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_embeddings(_json_body_request([]), "tester"))
    assert exc.value.status_code == 400
    assert rec.calls == []  # never switched
    assert inference_route._is_preview_resident(path)  # claim never ran, preview kept
    inference_route._set_preview_resident(None)  # cleanup


def test_completions_missing_prompt_preserves_preview_marker_when_both_off(monkeypatch):
    # Both-off gap on /v1/completions: a no-prompt body reaches the slot claim via
    # _auto_switch_from_request_body, so the missing-prompt 400 must fire first when the
    # slot is preview-owned, or it strands the preview.
    from fastapi import HTTPException

    path = "/outputs/run/ckpt-a"
    backend = _FakeBackend(path)
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = False, resolves_to = None, backend = backend, recorder = rec)
    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 0)
    inference_route._set_preview_resident(path)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            inference_route.openai_completions(_json_body_request({"model": path}), "tester")
        )
    assert exc.value.status_code == 400
    assert rec.calls == []  # never switched
    assert inference_route._is_preview_resident(path)  # claim never ran, preview kept
    inference_route._set_preview_resident(None)  # cleanup


def test_manual_unload_interrupts_even_while_inference_active(monkeypatch):
    # A manual /unload is deliberate: it tears down immediately even with a request in
    # flight (only the idle loop defers). No 409.
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
    # The advertised repo id wins.
    assert inference_route._llama_public_model_id(backend) == "org/Repo-GGUF"
    backend._openai_advertised_id = None
    # No advertised id: the identifier is cleaned to a public id, never the raw .gguf path.
    cleaned = inference_route._llama_public_model_id(backend)
    assert cleaned and "/cache/" not in cleaned and not cleaned.endswith(".gguf")
    # An already-clean repo id passes through.
    backend.model_identifier = "org/Repo-GGUF"
    assert inference_route._llama_public_model_id(backend) == "org/Repo-GGUF"
    backend.model_identifier = None
    assert inference_route._llama_public_model_id(backend, "req") == "req"


def test_chat_validates_non_system_message_before_auto_switch():
    # A system-only chat is rejected before the hook, never swapping the model. Source order.
    import inspect
    src = inspect.getsource(inference_route.openai_chat_completions)
    assert src.index("At least one non-system message is required.") < src.index(
        "_maybe_auto_switch_model"
    )


def test_chat_untracks_external_provider_before_proxy():
    # The external-provider branch untracks before proxying so its stream can't block a local auto-switch.
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
    # API-key caller: training 409s while a request streams, so it can't free VRAM by
    # unloading the chat model out from under the stream.
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
    # UI (session auth) caller: the API guard is skipped, so training proceeds past it
    # even with inference active (hits the normal already-active path).
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
    # With nothing stored, the env var enables idle-unload even while auto-switch is off
    # (headless/ops default), and the UI reader reflects it.
    monkeypatch.setattr(settings, "_cached_setting", lambda k, d = None: d)  # nothing stored
    monkeypatch.setenv("UNSLOTH_MODEL_IDLE_TTL", "600")
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: False)
    assert settings.get_auto_unload_idle_seconds() == 600
    assert settings.get_stored_auto_unload_idle_seconds() == 600


def test_stored_idle_value_overrides_env_and_stays_gated(monkeypatch):
    # An explicit stored value wins over the env default and stays gated on the toggle.
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
    # C3: a standalone UNSLOTH_MODEL_IDLE_TTL (auto-switch OFF) freed the model on idle;
    # the next request restores exactly what was freed, though the resolver never runs.
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
    # A is restored, but the request named B, so it is told so rather than served A.
    with pytest.raises(HTTPException) as excinfo:
        _run_hook("org/B-GGUF")
    assert excinfo.value.status_code == 404
    # Resolver skipped (auto-switch off), so only the stash reload runs: the freed A
    # is restored, not the resolves_to target B.
    assert len(rec.calls) == 1
    assert rec.calls[0].model_path == "/cache/snap/A"
    assert rec.calls[0].gguf_variant == "Q4_K_M"


def test_no_stash_reload_when_idle_off_and_auto_switch_off(monkeypatch):
    # C3 guard: with both features off the hook is a no-op and must not resurrect a
    # stashed model (that path only serves the idle feature).
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
    # An Unsloth/Transformers model loaded after idle-unload leaves the GGUF slot empty
    # but is live; an unknown /v1 name must NOT resurrect the stale GGUF stash (that
    # reload would tear the active Unsloth model down).
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
    # No alias: strip the path to a public id so a host path is never advertised.
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
    # C1 end-to-end: a scanner reporting the path as the id must not advertise the host
    # path in /v1/models, yet the model stays resolvable by that path.
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
    # Still resolvable by its on-disk path (an indexed alias).
    resolver._scan = (0.0, {})
    assert resolver.resolve_local_gguf(str(gguf)) is not None


def test_build_index_survives_a_failing_scanner(tmp_path, monkeypatch):
    # gemini: one bad scanner (e.g. a permission error) drops only that source, not the
    # whole index.
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
    # Codex: HF-cache GGUF snapshots leave model_format unset, so /v1/models decides
    # GGUF-ness from the files: a standalone .gguf is servable, a safetensors-only dir is not.
    from types import SimpleNamespace

    gguf = tmp_path / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"x" * 32)
    assert resolver.info_has_local_gguf(SimpleNamespace(id = str(gguf), path = str(gguf))) is True

    st = tmp_path / "safetensors_model"
    st.mkdir()
    (st / "model.safetensors").write_bytes(b"x" * 32)
    assert resolver.info_has_local_gguf(SimpleNamespace(id = str(st), path = str(st))) is False


def test_info_has_local_gguf_excludes_ollama_links(tmp_path):
    # Codex P2: Ollama entries come from a scanner _build_index skips, so their ids never
    # resolve; the catalog must not report them as servable.
    from types import SimpleNamespace

    links = tmp_path / ".studio_links"
    links.mkdir()
    ollama_gguf = links / "model-Q4_K_M.gguf"
    ollama_gguf.write_bytes(b"x" * 32)
    assert (
        resolver.info_has_local_gguf(SimpleNamespace(id = "ollama/foo:latest", path = str(ollama_gguf)))
        is False
    )
    # The same GGUF outside an ollama-link dir is servable.
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
    # C2: an embeddings request with no input 400s before the hook, never swapping the model.
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
    # Codex P2: only the non-streaming GGUF path returns multiple choices, so stream=true
    # + n>1 is invalid everywhere. Both fields are known pre-switch, so 400 before the switch.
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
    # Codex P2: the cache is stamped AFTER _build_index, else a scan slower than the TTL
    # stores an already-expired cache and rebuilds every request.
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
    # Codex P2: the middleware runs before auth, so a 401 decrements in-flight without
    # stamping activity, else unauthenticated probes keep the model warm and block idle-unload.
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
    """Standalone-idle reload setup: feature off, idle TTL on, an idle-freed model
    stashed, nothing loaded, no in-flight requests."""
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
    # #1: /v1/completions lacked a prompt pre-check, so a malformed request naming another
    # GGUF loaded it before failing. Now it 400s first.
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
    # #4: the chat pre-load guard only checked auto-switch, so a standalone idle TTL could
    # reload a system-only chat before the 400. Now it 400s first.
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
    # #5: same gap on /v1/embeddings; the missing-input 400 must fire under a standalone
    # idle TTL too, not only when auto-switch is on.
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
    # standalone idle TTL never restored the freed model. The early 503 now defers to
    # any automatic-load trigger, so the reload hook runs.
    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = False, resolves_to = None, backend = backend, recorder = rec)
    _stash(monkeypatch)
    # Proceeds past the hook to real generation (no llama-server), so tolerate the
    # downstream failure; the reload having run is the assertion.
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
    # #6: a raw body omitting `model` passed None, skipping the idle-stash reload and
    # 503-ing. A non-empty sentinel now lets the reload run while still resolving as unknown.
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
    # #2: /audio/generate is keep-warm-tracked but had no reload hook, so an idle-freed
    # audio GGUF stayed unloaded. The hook now restores it.
    from models.inference import ChatCompletionRequest

    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = False, resolves_to = None, backend = backend, recorder = rec)
    _stash(monkeypatch)
    payload = ChatCompletionRequest(model = "x", messages = [{"role": "user", "content": "say hi"}])
    # Falls through to the non-audio backend path after the reload; tolerate the failure,
    # the reload having run is the assertion.
    try:
        asyncio.run(inference_route.generate_audio(payload, object(), "tester"))
    except Exception:
        pass
    assert len(rec.calls) == 1
    assert rec.calls[0].model_path == "/cache/snap/A"


def test_audio_generate_does_not_reload_on_invalid_request(monkeypatch):
    # The audio reload hook runs after message validation, so an empty request never reloads.
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
    # #7: the preview route delegates to the chat handler; a caller-supplied model must
    # not switch away from the pinned checkpoint. The scope opt-out flag makes the hook a no-op.
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
    # #8: long preview streams share the backend, so the middleware must count them and
    # the idle loop can't unload mid-response.
    from core.inference.llama_keepwarm import _is_inference_path

    assert _is_inference_path("/p/my-run/v1/chat/completions") is True
    assert _is_inference_path("/p/my-run/ckpt-100/v1/chat/completions") is True
    assert _is_inference_path("/p/my-run/v1/models") is False


def test_untrack_does_not_reset_idle_timer():
    # #9: untrack stamped _last_active, keeping the GGUF warm on external traffic. It must
    # decrement in-flight without restamping.
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
    # The start stamp was removed so a later-untracked external request can't reset the
    # timer at start; in-flight count still protects it.
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


def test_preview_inflight_updates_alongside_general_inflight():
    # Both counters update under the same lock at the same call site, so a preview's
    # inflight count can't race other_inference_request_count's decrement the way a
    # route-level counter (torn down on body-iterator exhaustion) could.
    from core.inference import llama_keepwarm as kw

    kw._inflight = 0
    kw._preview_inflight = 0
    kw._note_start(is_preview = True)
    kw._note_start(is_preview = False)
    try:
        assert kw._inflight == 2
        assert kw._preview_inflight == 1
        other_busy = kw.other_inference_request_count(
            current_request_counted = True, include_pending = False
        ) - kw.other_preview_inflight_count(current_request_counted = True)
        assert other_busy == 1  # only the non-preview request reads as foreign traffic
    finally:
        kw._note_end(is_preview = True)
        kw._note_end(is_preview = False)
    assert kw._inflight == 0
    assert kw._preview_inflight == 0


# ── codex review (merge round): reload-only sentinel, Anthropic tool validation ──


def test_omitted_model_does_not_resolve_to_a_named_gguf(monkeypatch):
    # Codex P2: a raw body omitting `model` must never run the resolver, so a GGUF named
    # "default" can't be switched to. The resolver here would switch to B if it ran.
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
    # The reload-only sentinel still restores an idle-freed model (round-9); it just
    # never runs the resolver.
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
    # Codex P2: a malformed client tool (no input_schema, no server-tool type) 400s
    # before the hook, never evicting the model.
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
    # Source order: tool-shape validation precedes the hook, for both /messages and
    # /messages/count_tokens (shared helper).
    import inspect
    for fn in (inference_route.anthropic_messages, inference_route.anthropic_count_tokens):
        src = inspect.getsource(fn)
        assert src.index("_validate_anthropic_client_tools") < src.index("_maybe_auto_switch_model")


def test_anthropic_mixed_tools_rejected_before_switch(monkeypatch):
    # Codex P2: mixing an Anthropic server tool (type) with a client tool (input_schema)
    # is unsupported and 400s before the switch, never evicting the model.
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
    # Codex P2: an omitted `model` (pydantic fills "default") is reload-only; an explicit
    # model -- including a literal "default" -- is honored.
    from models.inference import ChatCompletionRequest

    omitted = ChatCompletionRequest(messages = [_chat_msg()])
    assert inference_route._switch_model_for_payload(omitted) == inference_route._RELOAD_ONLY_MODEL
    explicit_default = ChatCompletionRequest(model = "default", messages = [_chat_msg()])
    assert inference_route._switch_model_for_payload(explicit_default) == "default"
    explicit = ChatCompletionRequest(model = "org/B-GGUF", messages = [_chat_msg()])
    assert inference_route._switch_model_for_payload(explicit) == "org/B-GGUF"


def test_omitted_schema_model_skips_resolver(monkeypatch):
    # End to end: a schema request omitting `model` doesn't run the resolver (a GGUF named
    # "default" is never swapped to); an explicit model still switches.
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
    # Codex P2: a malformed function tool (no name) 400s before the hook, never switching
    # or evicting the model.
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
    # A well-formed function tool and a built-in tool pass the pre-switch check. Stub the
    # hook so the test stops right after validation.
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
    # Source order: tool validation precedes the switch hook.
    import inspect
    src = inspect.getsource(inference_route.openai_responses)
    assert src.index("each function tool must have a 'name'") < src.index(
        "_maybe_auto_switch_model"
    )


def test_responses_forcing_tool_choice_without_name_rejected_before_switch(monkeypatch):
    # Codex P2: a forcing-function tool_choice with no name ({"type": "function"}) 400s
    # before the switch, so the streaming path can't forward a bad choice and evict the model.
    from fastapi import HTTPException
    from models.inference import ResponsesRequest

    async def _boom(*a, **k):
        raise AssertionError("must not switch on an invalid tool_choice")

    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _boom)
    payload = ResponsesRequest(model = "org/B-GGUF", input = "hi", tool_choice = {"type": "function"})
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_responses(payload, object(), "tester"))
    assert exc.value.status_code == 400
    # A named forcing choice is accepted (reaches the mocked-to-raise switch).
    ok = ResponsesRequest(
        model = "org/B-GGUF", input = "hi", tool_choice = {"type": "function", "name": "f"}
    )
    with pytest.raises(AssertionError):
        asyncio.run(inference_route.openai_responses(ok, object(), "tester"))


# ── codex review (round 3): process-wide swap gate across event loops ──


def test_swap_acquires_process_gate_before_load():
    # The process-wide gate is acquired before the load and always released, so a
    # cross-loop swap can't reach _load_model_impl unguarded.
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
    # Codex P2: confirm_tool_calls=true + stream=false + local tools is invalid; it 400s
    # before the switch hook, never evicting the model.
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
    # bypass_permissions suppresses the confirm gate, so the pre-check doesn't fire and
    # the request reaches the switch hook (stubbed to a sentinel).
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
    # Codex P2: a chat request with audio_base64 guards the target before the switch
    # (audio rides the same companion mmproj as vision), so a text-only target can't
    # evict the working audio model. Assert the handler flags require_vision so the
    # hook's multimodal probe runs.
    class _Reached(Exception):
        pass

    captured = {}

    async def _capture(
        model,
        request,
        subject,
        *,
        require_vision = False,
        claim_resident = True,
    ):
        captured["require_vision"] = require_vision
        captured["claim_resident"] = claim_resident
        raise _Reached()

    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _capture)
    payload = _chat_request(model = "org/B-GGUF", audio_base64 = "AAAA")
    with pytest.raises(_Reached):
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))
    assert captured["require_vision"] is True
    # Chat defers the resident claim to the middleware (on a 2xx), so post-switch
    # capability rejections can't strand a preview-owned model.
    assert captured["claim_resident"] is False


def test_completions_rejects_object_prompt_before_switch(monkeypatch):
    # Codex P2: an object prompt {"prompt": {}} is a client error (only string/array
    # valid), so it 400s before the switch, never loading the named GGUF and evicting.
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
    # Codex P2: an object input {"input": {}} is a client error (only string/array valid);
    # reject before the switch, like completions.
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
    # Codex P2: the audio size cap is a cheap target-independent check, so an oversized
    # upload 413s before the switch, not after loading a GGUF.
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
    # Codex P2: mcp_enabled opens the tool loop on its own, so confirm+no-stream+mcp is the
    # same invalid shape as confirm+no-stream+tools and 400s before the switch. The old
    # guard only checked explicit tool fields and missed it.
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
    # Codex P2: an image request naming a text-only GGUF 400s before the swap, so the
    # resident vision model isn't evicted for a rejected request.
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
    # The reload-stash restores the model the request was already using; the modality
    # check applies only to an explicit resolver target, not a restore.
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
    # 404 because the restored A is not the requested B, whose quant makes it a real reference.
    with pytest.raises(HTTPException):
        asyncio.run(
            inference_route._maybe_auto_switch_model(
                "org/B-GGUF:UD-Q6_K_XL", object(), "t", require_vision = True
            )
        )
    assert len(rec.calls) == 1
    assert rec.calls[0].model_path == "/cache/snap/A"  # restored despite require_vision


def test_chat_validates_confirm_and_modality_before_switch():
    # Source order: confirm-shape rejection precedes the hook, and the hook rejects a
    # non-vision target before the load.
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
    # P2: the modality guard must fire on /v1/responses and /v1/messages too, so an image
    # request can't evict a vision model for a text-only target. Source: each hook derives
    # require_vision from the request's images.
    import inspect

    responses_src = inspect.getsource(inference_route.openai_responses)
    assert "require_vision = _messages_have_image(" in responses_src
    anthropic_src = inspect.getsource(inference_route.anthropic_messages)
    assert "require_vision = _anthropic_request_has_image(" in anthropic_src
    # count_tokens shares the /messages translation, so it needs the same guard: an image
    # count must not evict a vision model for a text-only target.
    count_src = inspect.getsource(inference_route.anthropic_count_tokens)
    assert "require_vision = _anthropic_request_has_image(" in count_src


def test_responses_count_tokens_messages_defer_slot_claim():
    # Codex P2: openai_responses (streaming 400s in _responses_stream), count_tokens
    # (tokenize-only) and anthropic_messages (the image normalizer can 400 after the
    # switch) must NOT claim the resident model in the switch hook, or a non-generating
    # request strands a preview-owned slot as Studio-owned and 503s a later preview.
    # They pass claim_resident=False and rely on the middleware's claim-on-2xx instead.
    import inspect
    for fn in (
        inference_route.openai_responses,
        inference_route.anthropic_count_tokens,
        inference_route.anthropic_messages,
    ):
        assert "claim_resident = False" in inspect.getsource(fn), fn.__name__


# ── codex review (round 5): count_tokens tools, tool_choice, process-wide gate ──


def test_count_tokens_rejects_malformed_tool_before_switch(monkeypatch):
    # Codex P2: count_tokens rejects a malformed tool before the switch, like /messages,
    # so a count request can't evict the model.
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
    # Codex P2: an image count_tokens naming a text-only GGUF carries the same
    # require_vision guard as /messages, so it can't evict a loaded vision model.
    class _Reached(Exception):
        pass

    captured = {}

    async def _capture(
        model,
        request,
        subject,
        *,
        require_vision = False,
        claim_resident = True,
    ):
        captured["require_vision"] = require_vision
        captured["claim_resident"] = claim_resident
        raise _Reached()

    monkeypatch.setattr(inference_route, "_anthropic_request_has_image", lambda p: True)
    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _capture)
    payload = _anthropic_payload_with_tools(None)  # no tools -> tool validation passes
    with pytest.raises(_Reached):
        asyncio.run(inference_route.anthropic_count_tokens(payload, object(), "tester"))
    assert captured["require_vision"] is True
    # Codex P2: count_tokens only tokenizes (no generation), so it must NOT adopt the
    # resident model for Studio -- clearing the marker here would 503 a later preview
    # even though Studio never generated.
    assert captured["claim_resident"] is False


def test_count_tokens_switch_marks_new_model_preview_owned(monkeypatch):
    # Codex P2: a count_tokens auto-switch can load a new GGUF via _load_model_impl, which
    # clears the preview marker. Counting never generates, so the switched-in model must be
    # marked preview-owned, else it blocks later previews for other checkpoints.
    from types import SimpleNamespace
    from fastapi import HTTPException

    inference_route._set_preview_resident(None)
    # _loaded_slot_ident() returns A before the switch, B after.
    slots = iter(["/outputs/run/A"])
    monkeypatch.setattr(
        inference_route, "_loaded_slot_ident", lambda: next(slots, "/outputs/run/B")
    )

    async def _noop_switch(*a, **k):
        return None

    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _noop_switch)
    # Backend not loaded, so the route 503s right AFTER the compensation.
    monkeypatch.setattr(
        inference_route, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False)
    )
    payload = _anthropic_payload_with_tools(None)
    with pytest.raises(HTTPException):
        asyncio.run(inference_route.anthropic_count_tokens(payload, object(), "tester"))
    # B (switched-in) is marked preview-owned, not left Studio-owned.
    assert inference_route._is_preview_resident("/outputs/run/B")
    inference_route._set_preview_resident(None)  # cleanup


# ── /chat/count_tokens: what the recount prices ───────────────────


def _count_tokens_backend(
    monkeypatch,
    loaded_id = "org/A-GGUF",
    count = 10,
    *,
    supports_tools = False,
    reasoning_style = "enable_thinking",
):
    """A loaded GGUF backend wired into the count endpoint, as ``(switched, counted)``: auto-switch
    attempts, and the messages/system/tools/template kwargs the route hands to the tokenizer."""
    from core.inference.llama_cpp import LlamaCppBackend

    backend = _FakeBackend(loaded_id)
    backend.supports_tools = supports_tools
    # The real kwargs builder, so this pins the route against the completion path's mapping.
    backend._supports_reasoning = True
    backend._reasoning_always_on = False
    backend._reasoning_style = reasoning_style
    backend._reasoning_effort_levels = ["high", "max"]
    backend._supports_preserve_thinking = True
    backend._architecture = None
    backend._request_reasoning_kwargs = LlamaCppBackend._request_reasoning_kwargs.__get__(
        backend, type(backend)
    )
    switched: list = []
    counted: dict = {}

    def _count(
        messages,
        system,
        tools,
        strict = False,
        chat_template_kwargs = None,
        should_abort = None,
    ):
        counted.update(
            messages = messages,
            system = system,
            tools = tools,
            strict = strict,
            chat_template_kwargs = chat_template_kwargs,
        )
        return count

    async def _switch(*args, **kwargs):
        switched.append(True)
        return None

    backend.count_chat_tokens = _count
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _switch)
    return switched, counted


def _count_request(
    messages,
    model = "org/A-GGUF",
    **fields,
):
    """A /chat/count_tokens payload built from plain message dicts."""
    from models.inference import ChatCountTokensRequest, ChatMessage
    return ChatCountTokensRequest(
        model = model,
        messages = [ChatMessage(**message) for message in messages],
        **fields,
    )


def _counted_body(payload):
    """Run the count endpoint and return its decoded JSON body."""
    response = asyncio.run(inference_route.chat_count_tokens(payload, "tester"))
    return json.loads(response.body)


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("hello", id = "plain_string"),
        # Control for the image guard below: a text-only parts array is unaffected.
        pytest.param([{"type": "text", "text": "hello"}], id = "text_parts"),
    ],
)
def test_chat_count_tokens_prices_the_loaded_model_without_switching(monkeypatch, content):
    # The recount has no abort signal, so a stale payload naming A must not drag the backend back.
    switched, _counted = _count_tokens_backend(monkeypatch, "org/B-GGUF", count = 42)
    payload = _count_request([{"role": "user", "content": content}])
    # The reply names B, the tokenizer that produced the total, not the A asked for.
    assert _counted_body(payload) == {"input_tokens": 42, "model": "org/B-GGUF"}
    assert switched == []


def test_chat_count_tokens_forwards_enabled_tools(monkeypatch):
    # Schemas and the nudge are a large share of the prompt: price the completion's own selection.
    _switched, counted = _count_tokens_backend(monkeypatch, count = 99, supports_tools = True)
    gate = {}

    async def _select(payload, *, tools_on, mcp_allowed):
        gate.update(tools_on = tools_on, mcp_allowed = mcp_allowed)
        return [{"type": "function", "function": {"name": "web_search"}}]

    monkeypatch.setattr(inference_route, "_select_request_tools", _select)
    payload = _count_request(
        [{"role": "user", "content": "hello"}],
        enable_tools = True,
        enabled_tools = ["web_search"],
    )
    assert _counted_body(payload) == {"input_tokens": 99, "model": "org/A-GGUF"}
    assert gate.get("tools_on") is True
    assert [t.get("function", {}).get("name") for t in counted.get("tools") or []] == ["web_search"]
    assert any(
        message.get("role") == "system" and "web_search" in str(message.get("content", ""))
        for message in counted.get("messages") or []
    )


# A leaked tool call in a replayed turn, plus an example naming a NOT-enabled tool the gate keeps.
_LEAKED_TOOL_HISTORY = [
    {"role": "user", "content": "weather?"},
    {
        "role": "assistant",
        "content": 'sunny <tool_call>{"name": "web_search", "arguments": {"q": "weather"}}'
        "</tool_call> and call it as offline_tool[ARGS]{}",
    },
    {"role": "user", "content": "and tomorrow?"},
]


@pytest.mark.parametrize(
    ("fields", "expect_markup"),
    [
        # Auto-Heal on (default): the tool path strips before rendering, so raw markup reads high.
        pytest.param({}, False, id = "auto_heal_default_on"),
        pytest.param({"auto_heal_tool_calls": True}, False, id = "auto_heal_on"),
        # Off leaves the markup in the real prompt, so the count has to keep it as well.
        pytest.param({"auto_heal_tool_calls": False}, True, id = "auto_heal_off"),
    ],
)
def test_chat_count_tokens_strips_replayed_tool_markup(monkeypatch, fields, expect_markup):
    """The GGUF tool path strips stale tool-call XML out of replayed assistant turns before
    rendering, so a count that keeps it prices text the completion removes."""
    _switched, counted = _count_tokens_backend(monkeypatch, count = 99, supports_tools = True)

    async def _select(_payload, *, tools_on, mcp_allowed):
        return [{"type": "function", "function": {"name": "web_search"}}]

    monkeypatch.setattr(inference_route, "_select_request_tools", _select)
    payload = _count_request(
        _LEAKED_TOOL_HISTORY,
        enable_tools = True,
        enabled_tools = ["web_search"],
        **fields,
    )
    assert _counted_body(payload)["input_tokens"] == 99
    assistant = [m for m in counted["messages"] if m.get("role") == "assistant"]
    assert len(assistant) == 1
    content = str(assistant[0].get("content", ""))
    assert (
        "<tool_call>" in content
    ) is expect_markup, "the count must render the same replayed history the completion does"
    assert (
        "offline_tool[ARGS]" in content
    ), "an inactive tool name is prose in the real prompt, so the count keeps it too"


_PASSTHROUGH_CATALOG = [
    {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}
]
_PASSTHROUGH_TOOL_HISTORY = [
    {"role": "user", "content": "weather?"},
    {
        "role": "assistant",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
]
_PASSTHROUGH_PLAIN = [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize(
    ("cli_policy", "messages", "fields", "priced_tools"),
    [
        # The reported shape: `unsloth run --enable-tools` sets the process policy without asking
        # for the tool loop, so tool history still goes to llama-server verbatim, bare.
        pytest.param(
            True,
            _PASSTHROUGH_TOOL_HISTORY,
            {},
            None,
            id = "cli_policy_does_not_price_a_passthrough_prompt",
        ),
        # Negative control: with no tool history the same policy takes the ordinary GGUF loop.
        pytest.param(
            True,
            _PASSTHROUGH_PLAIN,
            {},
            ["web_search"],
            id = "cli_policy_prices_an_ordinary_chat",
        ),
        # The passthrough is the one route that forwards the caller's own catalog.
        pytest.param(
            None,
            _PASSTHROUGH_PLAIN,
            {"tools": _PASSTHROUGH_CATALOG},
            ["get_weather"],
            id = "client_catalog_is_priced_verbatim",
        ),
        # tool_choice "none" withdraws the catalog, leaving the request on the ordinary path.
        pytest.param(
            None,
            _PASSTHROUGH_PLAIN,
            {"tools": _PASSTHROUGH_CATALOG, "tool_choice": "none"},
            None,
            id = "withdrawn_catalog_is_not_priced",
        ),
        # ... unless tool history needs those schemas to replay, keeping it on the passthrough.
        pytest.param(
            None,
            _PASSTHROUGH_TOOL_HISTORY,
            {"tools": _PASSTHROUGH_CATALOG, "tool_choice": "none"},
            ["get_weather"],
            id = "withdrawn_catalog_with_tool_history_is_priced",
        ),
        # It withdraws the process policy's own catalog too, as _client_disabled_tool_calls does.
        pytest.param(
            True,
            _PASSTHROUGH_PLAIN,
            {"tool_choice": "none"},
            None,
            id = "withdrawn_catalog_beats_the_cli_policy",
        ),
    ],
)
def test_chat_count_tokens_prices_the_route_the_completion_takes(
    monkeypatch, cli_policy, messages, fields, priced_tools
):
    """The count must describe the request the completion actually sends (#7453).

    Applying the process tool policy without first asking which route the request takes prices a
    built-in catalog plus the action nudge, while the completion forwards verbatim and sends neither.
    """
    import state.tool_policy as _tp

    _switched, counted = _count_tokens_backend(monkeypatch, count = 99, supports_tools = True)

    async def _select(payload, *, tools_on, mcp_allowed):
        return [{"type": "function", "function": {"name": "web_search"}}]

    monkeypatch.setattr(inference_route, "_select_request_tools", _select)
    monkeypatch.setattr(_tp, "get_tool_policy", lambda: cli_policy)

    assert _counted_body(_count_request(messages, **fields))["input_tokens"] == 99
    assert [(tool.get("function") or {}).get("name") for tool in counted.get("tools") or []] == (
        priced_tools or []
    )
    # The nudge rides with the built-in selection, so it must follow the same verdict.
    nudged = any(
        message.get("role") == "system" and "web_search" in str(message.get("content", ""))
        for message in counted.get("messages") or []
    )
    assert nudged is (priced_tools == ["web_search"])


def test_chat_count_tokens_keeps_adjacent_user_turns_on_the_passthrough(monkeypatch):
    """Coalescing is an ordinary-GGUF-path step, so it has to follow the routing.

    ``_openai_messages_for_passthrough`` drops the empty assistant sentinel but keeps the two user
    turns around it (a stopped response's shape), so merging prices a prompt that route never sends.
    """
    _switched, counted = _count_tokens_backend(monkeypatch, count = 99, supports_tools = True)
    sentinel_thread = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "second"},
    ]

    _counted_body(_count_request(sentinel_thread, tools = _PASSTHROUGH_CATALOG))
    assert [message.get("content") for message in counted.get("messages") or []] == [
        "first",
        "second",
    ]

    # Negative control: off the passthrough the same thread merges, so no two user turns in a row.
    _counted_body(_count_request(sentinel_thread))
    assert [message.get("content") for message in counted.get("messages") or []] == [
        "first\n\nsecond"
    ]


def _in_flight_generation():
    """One registered generation, as the completion path registers it."""
    from state import active_generations
    return active_generations.ActiveGeneration(threading.Event(), thread_id = "t1")


def test_chat_count_tokens_refuses_while_a_generation_is_in_flight(monkeypatch):
    # The whole point of the endpoint's cost budget: a count must never share llama-server with a
    # decode. The frontend gate only covers our own tab, so the refusal has to live here too.
    switched, counted = _count_tokens_backend(monkeypatch, count = 1234)
    # Reached only after the tool selection and message rewriting, so it doubles as proof that the
    # refusal happens on entry rather than after the handler has already done that work.
    reached: list = []
    real = inference_route._llama_status_checkpoint_id
    monkeypatch.setattr(
        inference_route,
        "_llama_status_checkpoint_id",
        lambda backend: (reached.append(1), real(backend))[1],
    )
    payload = _count_request([{"role": "user", "content": "hello"}])
    with _in_flight_generation():
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(inference_route.chat_count_tokens(payload, "tester"))
    assert excinfo.value.status_code == 503
    assert "generation" in str(excinfo.value.detail).lower()
    assert reached == [], "the handler must decline before doing any of the count's work"
    assert counted == {}, "the tokenizer must not be reached"
    assert switched == [], "and neither must the auto-switch hook"


def test_chat_count_tokens_counts_again_once_the_generation_ends(monkeypatch):
    # Control: the refusal keys on a live generation, not on the request, and it does not latch.
    _switched, counted = _count_tokens_backend(monkeypatch, count = 1234)
    payload = _count_request([{"role": "user", "content": "hello"}])
    with _in_flight_generation():
        with pytest.raises(HTTPException):
            asyncio.run(inference_route.chat_count_tokens(payload, "tester"))
    body = _counted_body(payload)
    assert body["input_tokens"] == 1234
    assert counted != {}, "the tokenizer must be reached once nothing is decoding"


def test_chat_count_tokens_refuses_a_generation_that_starts_mid_count(monkeypatch):
    # Everything between the entry guard and the tokenizer awaits, so a run can begin in the gap.
    # _llama_status_checkpoint_id is the last call before the second guard: start a generation
    # from inside it and the count must abandon rather than proceed with what it already built.
    _switched, counted = _count_tokens_backend(monkeypatch, count = 1234)
    started: list = []
    real = inference_route._llama_status_checkpoint_id

    def _start_a_run(backend):
        if not started:
            handle = _in_flight_generation()
            handle.__enter__()
            started.append(handle)
        return real(backend)

    monkeypatch.setattr(inference_route, "_llama_status_checkpoint_id", _start_a_run)
    payload = _count_request([{"role": "user", "content": "hello"}])
    try:
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(inference_route.chat_count_tokens(payload, "tester"))
    finally:
        for handle in started:
            handle.__exit__(None, None, None)
    assert started, "the hook must have fired, or the test proves nothing"
    assert excinfo.value.status_code == 503
    assert "generation" in str(excinfo.value.detail).lower()
    assert counted == {}, "the tokenizer must not be reached"


def _enabled_mcp_server(
    tmp_path,
    monkeypatch,
    *,
    cached = None,
    cooloff = False,
):
    """One enabled MCP server, with its discovery cache in a known state.

    Both cache dicts are module globals shared across the whole test session, so they are
    replaced rather than mutated: a leftover entry would make an "undiscovered" case look
    discovered and quietly pass.
    """
    from core.inference import mcp_client
    from core.inference import tools as tools_mod
    from storage import mcp_servers_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", False)
    monkeypatch.setattr(tools_mod, "stdio_mcp_enabled", lambda: True)
    monkeypatch.setattr(mcp_client, "_tool_cache", {})
    monkeypatch.setattr(mcp_client, "_probe_cooloff_until", {})
    mcp_servers_db.create_server(
        id = "s1", display_name = "S", url = "http://mcp.test/sse", is_enabled = True
    )
    if cached is not None:
        mcp_client.cache_tools("s1", cached)
    if cooloff:
        mcp_client.record_probe_failure("s1")

    # Any probe at all is a failure of the whole design: a count must not reach the network.
    async def _no_probes(**_kwargs):
        raise AssertionError("a count must never probe an MCP server")

    monkeypatch.setattr(tools_mod, "list_tools_async", _no_probes)


MCP_TOOL_PAYLOAD = [{"name": "lookup", "description": "d", "inputSchema": {"type": "object"}}]


def test_cached_mcp_tools_reads_the_cache_without_probing(tmp_path, monkeypatch):
    from core.inference.tools import cached_mcp_tools

    _enabled_mcp_server(tmp_path, monkeypatch, cached = MCP_TOOL_PAYLOAD)
    specs, complete = cached_mcp_tools()
    assert complete is True
    assert [spec["function"]["name"] for spec in specs] == ["mcp__s1__lookup"]


def test_cached_mcp_tools_reports_an_undiscovered_server_as_incomplete(tmp_path, monkeypatch):
    from core.inference.tools import cached_mcp_tools

    _enabled_mcp_server(tmp_path, monkeypatch)
    specs, complete = cached_mcp_tools()
    assert specs == []
    assert complete is False, (
        "a completion would probe this server and render its schemas, so a count that skips "
        "them is short, not exact"
    )


def test_cached_mcp_tools_counts_a_cooloff_server_as_complete(tmp_path, monkeypatch):
    # The completion renders nothing for a cool-off server either, so skipping it is exact rather
    # than short. Declining here would blank the bar over an agreement.
    from core.inference.tools import cached_mcp_tools

    _enabled_mcp_server(tmp_path, monkeypatch, cooloff = True)
    specs, complete = cached_mcp_tools()
    assert specs == []
    assert complete is True


def test_chat_count_tokens_declines_an_undiscovered_mcp_server(tmp_path, monkeypatch):
    # Undercounting is the dangerous direction: it tells the user room exists that the next
    # request will not find. Discovery is not an option on this path, so decline instead.
    _switched, counted = _count_tokens_backend(monkeypatch, count = 1234, supports_tools = True)
    _enabled_mcp_server(tmp_path, monkeypatch)
    payload = _count_request([{"role": "user", "content": "hello"}], mcp_enabled = True)
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(inference_route.chat_count_tokens(payload, "tester"))
    assert excinfo.value.status_code == 503
    assert "mcp" in str(excinfo.value.detail).lower()
    assert counted == {}, "the tokenizer must not be reached with a short tool list"


def test_chat_count_tokens_prices_cached_mcp_schemas(tmp_path, monkeypatch):
    # MCP alone turns tools on for the completion, so the count has to render them even with the
    # built-in tools off, or the bar is short by a whole catalog.
    _switched, counted = _count_tokens_backend(monkeypatch, count = 1234, supports_tools = True)
    _enabled_mcp_server(tmp_path, monkeypatch, cached = MCP_TOOL_PAYLOAD)
    payload = _count_request(
        [{"role": "user", "content": "hello"}], mcp_enabled = True, enabled_tools = []
    )
    body = _counted_body(payload)
    assert body["input_tokens"] == 1234
    names = [tool["function"]["name"] for tool in (counted.get("tools") or [])]
    assert (
        "mcp__s1__lookup" in names
    ), "a cached MCP schema is in the completion's prompt, so it must be in the count"


def test_chat_count_tokens_ignores_an_mcp_server_the_request_did_not_enable(tmp_path, monkeypatch):
    # Control: the decline keys on the request asking for MCP, not on a server merely existing.
    # tool_choice "none" would NOT be a control here: _explicit_studio_tool_loop_requested treats
    # mcp_enabled as an explicit ask, so the completion still renders the catalog.
    _switched, counted = _count_tokens_backend(monkeypatch, count = 1234, supports_tools = True)
    _enabled_mcp_server(tmp_path, monkeypatch)
    body = _counted_body(_count_request([{"role": "user", "content": "hello"}]))
    assert body["input_tokens"] == 1234
    assert counted != {}, "the tokenizer must still be reached"


def test_a_count_admitted_while_idle_stands_down_if_a_run_starts(monkeypatch):
    """Admission and the work are separate steps. A run that registers in between cannot be
    prevented without a lock in front of generation startup, so the count abandons at the
    checkpoint between /apply-template and /tokenize instead of spending the second trip."""
    _switched, counted = _count_tokens_backend(monkeypatch, count = 1234)
    payload = _count_request([{"role": "user", "content": "hello"}])
    started: list = []

    def _count(
        messages,
        system,
        tools,
        strict = False,
        chat_template_kwargs = None,
        should_abort = None,
    ):
        # Stand in for /apply-template returning: the run lands, then the checkpoint is polled.
        handle = _in_flight_generation()
        handle.__enter__()
        started.append(handle)
        assert should_abort is not None, "the route must give the tokenizer a way to stand down"
        if should_abort():
            from core.inference.llama_cpp import CountAborted
            raise CountAborted()
        counted.update(messages = messages)
        return 1234

    from core.inference.llama_cpp import LlamaCppBackend

    backend = inference_route.get_llama_cpp_backend()
    monkeypatch.setattr(backend, "count_chat_tokens", _count)
    assert LlamaCppBackend is not None
    try:
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(inference_route.chat_count_tokens(payload, "tester"))
    finally:
        for handle in started:
            handle.__exit__(None, None, None)
    assert started, "the hook must have fired, or the test proves nothing"
    assert excinfo.value.status_code == 503
    assert "generation" in str(excinfo.value.detail).lower()
    assert counted == {}, "the second round trip must not happen"


def test_a_count_that_stays_idle_is_not_aborted(monkeypatch):
    # Control: the checkpoint fires on a live run, not on every count.
    _switched, counted = _count_tokens_backend(monkeypatch, count = 1234)
    seen: list = []

    def _count(
        messages,
        system,
        tools,
        strict = False,
        chat_template_kwargs = None,
        should_abort = None,
    ):
        seen.append(should_abort() if should_abort else None)
        counted.update(messages = messages)
        return 1234

    backend = inference_route.get_llama_cpp_backend()
    monkeypatch.setattr(backend, "count_chat_tokens", _count)
    body = _counted_body(_count_request([{"role": "user", "content": "hello"}]))
    assert body["input_tokens"] == 1234
    assert seen == [False], "an idle server must report nothing to stand down for"


@pytest.mark.parametrize(
    ("abort", "expect_tokenize"),
    [(True, False), (False, True)],
    ids = ["run_started", "still_idle"],
)
def test_count_chat_tokens_stands_down_before_tokenizing(monkeypatch, abort, expect_tokenize):
    """The abort has to escape the template except-block. Swallowed, it would set
    apply_template_failed and the text fallback would tokenize anyway, which is the work
    being declined. The control shows the poll alone does not stop an idle count."""
    from core.inference import llama_cpp as llama_cpp_mod
    from core.inference.llama_cpp import CountAborted, LlamaCppBackend

    posted: list = []

    class _FakeResponse:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    class _FakeClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def post(
            self,
            url,
            json = None,
        ):
            posted.append(url)
            if url.endswith("/apply-template"):
                return _FakeResponse({"prompt": "user hi"})
            return _FakeResponse({"tokens": [1, 2]})

    class _CountBackend(LlamaCppBackend):
        is_loaded = True
        base_url = "http://127.0.0.1:1"
        _auth_headers = None

        def __init__(self):
            pass

    monkeypatch.setattr(llama_cpp_mod.httpx, "Client", _FakeClient)
    call = lambda: _CountBackend().count_chat_tokens(
        [{"role": "user", "content": "hi"}],
        strict = True,
        should_abort = lambda: abort,
    )
    if abort:
        with pytest.raises(CountAborted):
            call()
    else:
        assert call() == 2
    assert any(u.endswith("/tokenize") for u in posted) is expect_tokenize


def test_chat_count_tokens_refuses_image_messages(monkeypatch):
    # Images become a short /apply-template marker: refuse rather than undercount, before the switch.
    switched, counted = _count_tokens_backend(monkeypatch, count = 1234)
    payload = _count_request(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                    },
                ],
            }
        ]
    )
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(inference_route.chat_count_tokens(payload, "tester"))
    assert excinfo.value.status_code == 503
    assert counted == {}, "the tokenizer must not be reached"
    assert switched == [], "and neither must the auto-switch hook"


def test_chat_count_tokens_refuses_audio_messages(monkeypatch):
    # extra = "allow", so without this guard audio_base64 is accepted, dropped and undercounted.
    switched, counted = _count_tokens_backend(monkeypatch, count = 1234)
    payload = _count_request(
        [{"role": "user", "content": "what did I just say"}],
        audio_base64 = "UklGRiQAAABXQVZF",
    )
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(inference_route.chat_count_tokens(payload, "tester"))
    assert excinfo.value.status_code == 503
    assert "audio" in str(excinfo.value.detail).lower()
    assert counted == {}, "the tokenizer must not be reached"
    assert switched == [], "and neither must the auto-switch hook"


def test_chat_count_tokens_still_counts_without_audio(monkeypatch):
    # Control: the refusal keys on the audio, not on the shape of the request.
    _switched, counted = _count_tokens_backend(monkeypatch, count = 1234)
    body = _counted_body(_count_request([{"role": "user", "content": "what did I just say"}]))
    assert body["input_tokens"] == 1234
    assert counted != {}, "the tokenizer must be reached"


# Shapes the recount sends for a thread with documents in scope. Only a PENDING turn is answered
# from these exact messages, and the tool loop opens it by splicing in what RAG retrieves.
_PENDING_USER_TURN = [{"role": "user", "content": "what does the contract say"}]
_PENDING_TOOL_TURN = [
    {"role": "user", "content": "what does the contract say"},
    {"role": "assistant", "content": "checking"},
    {"role": "tool", "content": "{}", "tool_call_id": "call_1"},
]
_SETTLED_TURN = [
    {"role": "user", "content": "what does the contract say"},
    {"role": "assistant", "content": "it renews yearly"},
]


@pytest.mark.parametrize(
    ("messages", "rag_scope", "expected_total"),
    [
        # Retrieval splices passages in front of this turn, so counting alone says it fits.
        pytest.param(_PENDING_USER_TURN, {"thread_id": "t1"}, None, id = "pending_user_turn"),
        # An interrupted loop's unanswered tool result is the same pending shape.
        pytest.param(_PENDING_TOOL_TURN, {"project_id": "p1"}, None, id = "pending_tool_turn"),
        # Ends on an assistant turn: retrieval has no user message yet, so nothing is omitted.
        pytest.param(_SETTLED_TURN, {"thread_id": "t1"}, 4242, id = "settled_turn_still_counts"),
        # No documents in scope means no injection to miss, pending turn or not.
        pytest.param(_PENDING_USER_TURN, None, 4242, id = "no_rag_scope_still_counts"),
    ],
)
def test_chat_count_tokens_declines_a_pending_turn_that_would_retrieve(
    monkeypatch, messages, rag_scope, expected_total
):
    """A recount that omits RAG injection under-reports, the one direction the context bar must
    never be wrong in. Decline as the image case does and leave the usage the bar already had."""
    _switched, counted = _count_tokens_backend(monkeypatch, count = 4242)
    payload = _count_request(
        messages,
        **({"rag_scope": rag_scope} if rag_scope else {}),
    )
    try:
        total = _counted_body(payload).get("input_tokens")
    except HTTPException as exc:
        if exc.status_code != 503:
            raise
        total = None

    assert total == expected_total, (
        "a pending turn whose generation would retrieve documents must be declined, "
        "not priced without them"
    )
    if expected_total is None:
        assert counted == {}, "the tokenizer must not be reached for a declined count"


def test_chat_count_tokens_declines_when_the_model_changes_mid_count(monkeypatch):
    """A load landing while the tokenizer runs leaves a total attributable to neither model, and
    the caller's checkpoint guard never moved, so either identity would have it trust the number."""
    _switched, counted = _count_tokens_backend(monkeypatch, "org/A-GGUF", count = 555)
    backend = inference_route.get_llama_cpp_backend()
    inner = backend.count_chat_tokens

    def _count_then_swap(*args, **kwargs):
        result = inner(*args, **kwargs)
        # Another tab finishes loading B while this count is in the worker thread.
        backend.model_identifier = "org/B-GGUF"
        return result

    backend.count_chat_tokens = _count_then_swap
    payload = _count_request([{"role": "user", "content": "hello"}])
    try:
        total = _counted_body(payload).get("input_tokens")
    except HTTPException as exc:
        if exc.status_code != 503:
            raise
        total = None

    assert (
        total is None
    ), "a total counted across a model change must not be published as either model's"
    assert counted.get("messages"), "the tokenizer still ran; only its result is dropped"


def test_chat_count_tokens_collapses_system_turns(monkeypatch):
    # The completion path joins every system/developer turn into one; the count renders that.
    _switched, counted = _count_tokens_backend(monkeypatch, count = 13)
    payload = _count_request(
        [
            {"role": "system", "content": "Runtime rules."},
            {"role": "system", "content": "Studio prompt."},
            {"role": "user", "content": "hello"},
        ]
    )
    asyncio.run(inference_route.chat_count_tokens(payload, "tester"))
    messages = counted.get("messages") or []
    systems = [m for m in messages if m.get("role") in ("system", "developer")]
    assert len(systems) == 1, messages
    assert "Runtime rules." in systems[0].get("content", "")
    assert "Studio prompt." in systems[0].get("content", "")


@pytest.mark.parametrize(
    ("reasoning_style", "fields", "expected"),
    [
        # Qwen3-style gate: with Thinking off the completion sends enable_thinking=false and the
        # template prefills an empty <think> block; without it, the LOADED mode renders.
        pytest.param(
            "enable_thinking",
            {"enable_thinking": False},
            {"enable_thinking": False},
            id = "thinking_turned_off",
        ),
        # gpt-oss-style: the effort level is rendered into the system turn.
        pytest.param(
            "reasoning_effort",
            {"reasoning_effort": "low"},
            {"reasoning_effort": "low"},
            id = "effort_level",
        ),
        # preserve_thinking decides whether past <think> blocks stay: a short count or the history.
        pytest.param(
            "enable_thinking",
            {"enable_thinking": True, "preserve_thinking": True},
            {"enable_thinking": True, "preserve_thinking": True},
            id = "preserve_thinking",
        ),
        # Nothing selected: send nothing, so llama-server keeps its load-time defaults.
        pytest.param("enable_thinking", {}, None, id = "template_default"),
    ],
)
def test_chat_count_tokens_renders_the_requested_reasoning_mode(
    monkeypatch, reasoning_style, fields, expected
):
    # llama-server layers request kwargs over the load-time ones: omitting them prices LOAD mode.
    _switched, counted = _count_tokens_backend(
        monkeypatch, count = 7, reasoning_style = reasoning_style
    )
    payload = _count_request([{"role": "user", "content": "hello"}], **fields)
    assert _counted_body(payload) == {"input_tokens": 7, "model": "org/A-GGUF"}
    assert counted.get("chat_template_kwargs") == expected


@pytest.mark.parametrize(
    ("template_kwargs", "expected_tokens"),
    [
        # Thinking off: the template prefills an empty <think></think> pair the completion pays for.
        pytest.param({"enable_thinking": False}, 5, id = "thinking_off"),
        pytest.param(None, 3, id = "template_default"),
    ],
)
def test_count_chat_tokens_renders_with_the_requested_template_kwargs(
    monkeypatch, template_kwargs, expected_tokens
):
    """The kwargs have to reach llama-server itself: /apply-template runs the same parser
    as /v1/chat/completions, so the rendered prompt only moves when they are in the body."""
    from core.inference import llama_cpp as llama_cpp_mod
    from core.inference.llama_cpp import LlamaCppBackend

    class _FakeResponse:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    class _FakeClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def post(
            self,
            url,
            json = None,
        ):
            body = json or {}
            if not url.endswith("/apply-template"):
                return _FakeResponse({"tokens": str(body.get("content", "")).split()})
            # A Qwen3-style template: thinking off prefills an empty reasoning block.
            kwargs = body.get("chat_template_kwargs") or {}
            prefill = "" if kwargs.get("enable_thinking", True) else " <think> </think>"
            return _FakeResponse({"prompt": "user hi assistant" + prefill})

    class _CountBackend(LlamaCppBackend):
        is_loaded = True
        base_url = "http://127.0.0.1:1"
        _auth_headers = None

        def __init__(self):
            pass

    monkeypatch.setattr(llama_cpp_mod.httpx, "Client", _FakeClient)
    assert (
        _CountBackend().count_chat_tokens(
            [{"role": "user", "content": "hi"}],
            strict = True,
            chat_template_kwargs = template_kwargs,
        )
        == expected_tokens
    )


@pytest.mark.parametrize(
    "failure",
    [
        # A minja template that cannot render this history, or a build without the endpoint.
        pytest.param("status", id = "apply_template_rejects"),
        # The template call times out while the tokenizer answers.
        pytest.param("raise", id = "apply_template_unreachable"),
    ],
)
def test_strict_count_refuses_a_text_only_template_fallback(monkeypatch, failure):
    """/apply-template failing on a TEXT-ONLY prompt used to fall through to concatenating message
    text, dropping every role marker, special token and tool schema (~30% of a six-turn two-tool
    prompt). Strict callers publish what they get, so it must be an error, not an estimate."""
    from core.inference import llama_cpp as llama_cpp_mod
    from core.inference.llama_cpp import LlamaCppBackend

    class _FakeResponse:
        def __init__(
            self,
            payload,
            status_code = 200,
        ):
            self._payload = payload
            self.status_code = status_code

        def json(self):
            return self._payload

    class _FakeClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def post(
            self,
            url,
            json = None,
        ):
            body = json or {}
            if url.endswith("/apply-template"):
                if failure == "raise":
                    raise RuntimeError("timed out")
                return _FakeResponse({"error": "template error"}, status_code = 500)
            return _FakeResponse({"tokens": str(body.get("content", "")).split()})

    class _CountBackend(LlamaCppBackend):
        is_loaded = True
        base_url = "http://127.0.0.1:1"
        _auth_headers = None

        def __init__(self):
            pass

    monkeypatch.setattr(llama_cpp_mod.httpx, "Client", _FakeClient)
    messages = [{"role": "user", "content": "hi there"}]
    tools = [{"type": "function", "function": {"name": "web_search"}}]
    with pytest.raises(RuntimeError):
        _CountBackend().count_chat_tokens(messages, None, tools, strict = True)
    # Non-strict callers keep the best-effort approximation they have always had.
    assert _CountBackend().count_chat_tokens(messages, None, tools) > 0


def test_an_empty_chat_sends_the_empty_list_unchanged(monkeypatch):
    """A fresh New Chat has no messages and, by default, no system prompt. The count must
    forward that empty list as-is rather than inventing a turn to make the template happy.

    Templates that index ``messages[0]`` look like they must reject an empty list, and under
    python jinja2 they do. llama-server renders through minja, where that yields undefined
    instead of raising, so the real engine returns the bare preamble. Checked against the
    shipped templates for Llama-3.2-1B-Instruct, Qwen3-8B, Phi-4, gemma-3-270m-it and
    mistral-7b-instruct-v0.3 driven through llama-server with --jinja: all five render.
    Injecting a placeholder system turn would add a system block to the count for Qwen3
    (+30 chars) and Phi-4 (+38), overcounting the empty chat the bar exists to show."""
    from core.inference import llama_cpp as llama_cpp_mod
    from core.inference.llama_cpp import LlamaCppBackend

    seen = {}

    class _FakeResponse:
        def __init__(
            self,
            payload,
            status_code = 200,
        ):
            self._payload = payload
            self.status_code = status_code

        def json(self):
            return self._payload

    class _FakeClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def post(
            self,
            url,
            json = None,
        ):
            body = json or {}
            if url.endswith("/apply-template"):
                seen["messages"] = body.get("messages")
                # What minja does: no messages renders the preamble, it does not raise.
                return _FakeResponse({"prompt": "<start_of_turn>model\n"})
            return _FakeResponse({"tokens": str(body.get("content", "")).split()})

    class _CountBackend(LlamaCppBackend):
        is_loaded = True
        base_url = "http://127.0.0.1:1"
        _auth_headers = None

        def __init__(self):
            pass

    monkeypatch.setattr(llama_cpp_mod.httpx, "Client", _FakeClient)
    count = _CountBackend().count_chat_tokens([], None, None, strict = True)
    assert seen["messages"] == [], "the count must not invent a turn the caller never sent"
    assert count > 0, "a fresh chat still prices the template preamble"


def test_a_count_never_spawns_mcp_servers():
    """get_enabled_mcp_tools starts stdio MCP server processes, writes cache and cooloff state,
    and blocks for a whole probe timeout against a server that is down. A background recount
    must not do host work the user's completion never asked for, so the count path pins
    mcp_allowed False rather than deriving it from payload.mcp_enabled."""
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "routes" / "inference.py"
    tree = ast.parse(src.read_text(encoding = "utf-8"))
    handler = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "chat_count_tokens"
    )

    # The only assignment to _mcp_allowed in the handler must be the constant False.
    assigned = [
        node.value
        for node in ast.walk(handler)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_mcp_allowed" for t in node.targets)
    ]
    assert assigned, "the count handler no longer pins _mcp_allowed; this test is stale"
    assert all(
        isinstance(v, ast.Constant) and v.value is False for v in assigned
    ), "a count must never enable MCP discovery"

    called = {
        node.func.id
        for node in ast.walk(handler)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "get_enabled_mcp_tools" not in called


def test_audio_generate_is_reload_only(monkeypatch):
    # Codex P2: /audio/generate must not switch to a client-named GGUF. A GGUF's audio-input
    # capability isn't a cheap pre-load probe (mmproj can't tell an audio projector from a
    # vision one), so resolving the client model could evict the working audio model for a
    # target that then fails the audio check. Only the idle-stash restore runs (reload-only sentinel).
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
        claim_resident = True,
    ):
        captured["model"] = model
        captured["claim_resident"] = claim_resident
        raise _Reached()

    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _capture)
    payload = ChatCompletionRequest(
        model = "org/B-GGUF", messages = [{"role": "user", "content": "say hi"}]
    )
    with pytest.raises(_Reached):
        asyncio.run(inference_route.generate_audio(payload, object(), "tester"))
    assert captured["model"] == inference_route._RELOAD_ONLY_MODEL
    # Audio defers the resident claim until after its modality checks.
    assert captured["claim_resident"] is False


def test_audio_generate_text_only_model_preserves_preview_marker(monkeypatch):
    # A non-preview /audio/generate against a text-only resident model 400s at the audio
    # check. The claim is deferred until after that check, so a preview-owned checkpoint
    # isn't stranded as Studio-owned by a request that never generated.
    from types import SimpleNamespace
    from fastapi import HTTPException
    from models.inference import ChatCompletionRequest

    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _noop)
    monkeypatch.setattr(
        inference_route, "get_llama_cpp_backend", lambda: SimpleNamespace(is_loaded = False)
    )
    infer = SimpleNamespace(
        active_model_name = "/outputs/run/ckpt-a",
        models = {"/outputs/run/ckpt-a": {"is_audio": False}},
    )
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: infer)
    inference_route._set_preview_resident("/outputs/run/ckpt-a")
    payload = ChatCompletionRequest(model = "x", messages = [{"role": "user", "content": "hi"}])
    fake_req = SimpleNamespace(scope = {"path": "/api/inference/audio/generate"})
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.generate_audio(payload, fake_req, "tester"))
    assert exc.value.status_code == 400
    assert inference_route._is_preview_resident("/outputs/run/ckpt-a")  # claim deferred
    inference_route._set_preview_resident(None)  # cleanup


def test_completions_no_gguf_loaded_preserves_preview_marker(monkeypatch):
    # A non-preview /v1/completions with a preview-owned non-GGUF (transformers/LoRA)
    # resident 503s "No GGUF model loaded". The claim is deferred to after that check,
    # so the failed request can't strand the preview as Studio-owned.
    from fastapi import HTTPException

    backend = _FakeBackend(None)  # llama backend not loaded (a non-GGUF model is resident)
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = False, resolves_to = None, backend = backend, recorder = rec)
    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 0)
    inference_route._set_preview_resident("/outputs/run/lora-a")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            inference_route.openai_completions(_json_body_request({"prompt": "hi"}), "tester")
        )
    assert exc.value.status_code == 503
    assert rec.calls == []  # never switched
    assert inference_route._is_preview_resident("/outputs/run/lora-a")  # claim deferred
    inference_route._set_preview_resident(None)  # cleanup


def test_note_model_unloaded_clears_reload_stash(monkeypatch):
    # Codex P2: a deliberate unload drops the idle reload stash so the next /v1 request
    # can't resurrect the just-unloaded model.
    import core.inference.llama_keepwarm as kw

    kw._set_last_unloaded(("org/A-GGUF", "Q4_K_M"))
    assert kw.get_last_unloaded_model() == ("org/A-GGUF", "Q4_K_M")
    kw.note_model_unloaded()
    assert kw.get_last_unloaded_model() is None


def test_unload_route_clears_reload_stash(monkeypatch):
    # The /unload route clears the stash on both the GGUF and non-GGUF branches.
    # Asserted on the impl: the route only adds the padded response.
    import inspect
    src = inspect.getsource(inference_route._unload_model_impl)
    assert src.count("note_model_unloaded()") >= 2


def test_non_gguf_load_clears_reload_stash():
    # A non-GGUF load clears the stash like the GGUF branch, so it never lingers until
    # the idle poll (or forever, idle-unload off).
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
    # A well-formed forcing object passes the pre-check and reaches the hook.
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
    # Codex P2: the lifecycle gate is process-wide so a swap on one loop blocks inference
    # on another. Two loops must never hold it at once.
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
    # Codex P2: a per-loop asyncio lock can't serialize two swaps on different event loops;
    # the process-wide gate does, so the two slow loads never overlap on the model slot.
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
        await asyncio.sleep(0.1)  # widen the window so an unguarded race overlaps
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
    # A waiter cancelled while waiting for the gate (client disconnect mid-swap) must not
    # leak it: after the holder releases, a fresh acquire still succeeds. A to_thread(acquire)
    # approach would leak (its worker keeps acquiring after cancel, so the gate is never released).
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
    # The "no model loaded" errors point at the auto-switch toggle (so a listed-but-
    # unloaded model is self-explanatory) only when it's off. With it on, the name simply
    # didn't resolve, so no hint.
    base = "No GGUF model loaded. Load a GGUF model first."

    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: False)
    off = inference_route._no_model_loaded_detail(base)
    assert off.startswith(base)
    assert "Model auto-switch" in off and "Settings > API" in off

    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    assert inference_route._no_model_loaded_detail(base) == base


def _run_responses_stream_no_model(
    monkeypatch,
    *,
    enabled,
    active_model_name,
    resolves_to = None,
):
    # Drive _responses_stream's GGUF-not-loaded guard (llama unloaded, inference backend
    # maybe holding a non-GGUF model). Returns (status, detail).
    from fastapi import HTTPException
    from models.inference import ResponsesRequest, ChatMessage

    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: enabled)
    monkeypatch.setattr(resolver, "resolve_local_gguf", lambda name: resolves_to)
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
    return exc.value.status_code, exc.value.detail


def test_responses_stream_hint_matches_toggle_regardless_of_active_model(monkeypatch):
    # The hint attaches whenever the toggle is off, whatever is active -- including while a
    # non-GGUF model is active, since auto-switch evicts it to load a resolved GGUF. With the
    # toggle on the name resolved to nothing local, so 404 rather than 400.
    off_status, hinted = _run_responses_stream_no_model(
        monkeypatch, enabled = False, active_model_name = None
    )
    assert off_status == 400
    assert "Model auto-switch" in hinted

    on_status, on = _run_responses_stream_no_model(
        monkeypatch, enabled = True, active_model_name = None
    )
    assert on_status == 404
    assert "Model auto-switch" not in on
    assert "unsloth/Qwen3.5-4B-GGUF" in on

    non_gguf_status, non_gguf_loaded = _run_responses_stream_no_model(
        monkeypatch, enabled = False, active_model_name = "unsloth/Llama-3.2-1B-Instruct"
    )
    assert non_gguf_status == 400
    assert "Model auto-switch" in non_gguf_loaded


def _wire_unloaded_chat(
    monkeypatch,
    *,
    enabled,
    catalog = ("org/A-GGUF", "org/B-GGUF"),
):
    # Nothing loaded, so a chat request hits "no model loaded". Pin the catalog for determinism.
    async def _catalog():
        return [{"id": mid} for mid in catalog]

    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: enabled)
    monkeypatch.setattr(resolver, "resolve_local_gguf", lambda _m, **_kw: None)
    monkeypatch.setattr(
        resolver, "describe_local_miss", lambda _m: (resolver.MISS_MODEL_NOT_FOUND, ())
    )
    monkeypatch.setattr(inference_route, "_openai_catalog_objects", _catalog)
    monkeypatch.setattr(
        inference_route, "get_llama_cpp_backend", lambda: _FakeBackend(loaded_id = None)
    )
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: type("_B", (), {"active_model_name": None, "models": {}})(),
    )


def _chat_error(payload):
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))
    return exc.value.status_code, exc.value.detail


def test_chat_names_undownloaded_model_404s_with_available_ids(monkeypatch):
    # The reported bug: the model is not here, so the switch did nothing and /inference/load
    # cannot fix it. Name it and list what can serve.
    _wire_unloaded_chat(monkeypatch, enabled = True)
    status, detail = _chat_error(_chat_request(model = "unsloth/gemma-4-E4B-it-GGUF:UD-Q5_K_XL"))
    assert status == 404
    assert "unsloth/gemma-4-E4B-it-GGUF:UD-Q5_K_XL" in detail
    assert "org/A-GGUF, org/B-GGUF" in detail
    assert "GET /v1/models" in detail
    assert "POST /inference/load" not in detail


def test_chat_undownloaded_model_with_empty_catalog(monkeypatch):
    # Nothing downloaded: an empty list would read as a bug, so say so plainly.
    _wire_unloaded_chat(monkeypatch, enabled = True, catalog = ())
    status, detail = _chat_error(_chat_request(model = "org/nope-GGUF"))
    assert status == 404
    assert "no models are downloaded yet" in detail


def test_chat_wrong_quant_lists_the_local_quants(monkeypatch):
    # Repo downloaded, only the quant missing: sibling quants, not the catalog.
    _wire_unloaded_chat(monkeypatch, enabled = True)
    monkeypatch.setattr(
        resolver,
        "describe_local_miss",
        lambda _m: (resolver.MISS_VARIANT_NOT_FOUND, ("Q4_K_M", "Q8_0")),
    )
    status, detail = _chat_error(_chat_request(model = "org/A-GGUF:UD-Q5_K_XL"))
    assert status == 404
    assert "'org/A-GGUF' is downloaded, but the quant 'UD-Q5_K_XL' is not" in detail
    assert "Q4_K_M, Q8_0" in detail


def test_chat_error_unchanged_when_auto_switch_off(monkeypatch):
    # Toggle off: nothing resolved, so keep the pre-existing status and text, hint included.
    _wire_unloaded_chat(monkeypatch, enabled = False)
    status, detail = _chat_error(_chat_request(model = "org/nope-GGUF"))
    assert status == 400
    assert detail.startswith("No model loaded. Call POST /inference/load first.")
    assert "Model auto-switch" in detail


def test_chat_error_unchanged_when_no_model_named(monkeypatch):
    # An omitted model means "serve whatever is loaded", so there is no name to report.
    _wire_unloaded_chat(monkeypatch, enabled = True)
    status, detail = _chat_error(_chat_request())
    assert status == 400
    assert detail == "No model loaded. Call POST /inference/load first."


def test_chat_not_downloaded_error_survives_a_broken_catalog_scan(monkeypatch):
    # Layered onto an already-failing path, so a broken scan must not make it a 500.
    async def _boom():
        raise RuntimeError("catalog scan blew up")

    _wire_unloaded_chat(monkeypatch, enabled = True)
    monkeypatch.setattr(inference_route, "_openai_catalog_objects", _boom)
    status, detail = _chat_error(_chat_request(model = "org/nope-GGUF"))
    assert status == 400
    assert detail.startswith("No model loaded. Call POST /inference/load first.")


def test_chat_available_id_list_is_capped(monkeypatch):
    # A machine with 40 GGUFs must not print all 40 into a terminal error.
    _wire_unloaded_chat(
        monkeypatch, enabled = True, catalog = tuple(f"org/m{i:02d}-GGUF" for i in range(20))
    )
    status, detail = _chat_error(_chat_request(model = "org/nope-GGUF"))
    assert status == 404
    assert "and 12 more" in detail
    assert "org/m08-GGUF" not in detail


def test_anthropic_undownloaded_model_uses_the_anthropic_envelope(monkeypatch):
    # Shared with /v1/messages, so the 404 must not leak an OpenAI-shaped body.
    from fastapi import HTTPException

    async def _noop_switch(*a, **k):
        return None

    _wire_unloaded_chat(monkeypatch, enabled = True)
    monkeypatch.setattr(inference_route, "_automatic_model_load_may_run", lambda: True)
    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _noop_switch)

    request = type("_R", (), {"url": type("_U", (), {"path": "/v1/messages"})()})()
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.anthropic_messages(_anthropic_payload(64), request, "tester"))
    assert exc.value.status_code == 404
    body = exc.value.detail
    assert body["type"] == "error"
    assert body["error"]["type"] == "not_found_error"
    assert "claude-x" in body["error"]["message"]


def test_chat_undownloaded_model_uses_the_openai_envelope(monkeypatch):
    # The OpenAI surface carries param/code so SDK clients can branch on it.
    from fastapi import HTTPException

    _wire_unloaded_chat(monkeypatch, enabled = True)
    request = type("_R", (), {"url": type("_U", (), {"path": "/v1/chat/completions"})()})()
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            inference_route.openai_chat_completions(
                _chat_request(model = "org/nope-GGUF"), request, "tester"
            )
        )
    assert exc.value.status_code == 404
    err = exc.value.detail["error"]
    assert err["type"] == "not_found_error"
    assert err["code"] == "model_not_found"
    assert err["param"] == "model"


def test_gguf_only_paths_keep_the_generic_error_for_the_resident_non_gguf_model(monkeypatch):
    # resolve_local_gguf misses a resident Transformers model the catalog does list, so
    # "not downloaded" would contradict itself.
    resident = "unsloth/Qwen3.5-4B-GGUF"  # the id _run_responses_stream_no_model asks for

    async def _catalog():
        return [{"id": resident}]

    monkeypatch.setattr(inference_route, "_openai_catalog_objects", _catalog)
    status, detail = _run_responses_stream_no_model(
        monkeypatch, enabled = True, active_model_name = resident
    )
    assert status == 400
    assert "requires a GGUF model" in detail
    assert "not downloaded" not in detail


def test_completions_keeps_the_generic_error_for_the_resident_non_gguf_model(monkeypatch):
    # Same contradiction on the raw-body surface, via _auto_switch_from_request_body.
    from fastapi import HTTPException

    resident = "unsloth/Llama-3.2-1B-Instruct"
    _wire_unloaded_chat(monkeypatch, enabled = True, catalog = (resident,))
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: type("_B", (), {"active_model_name": resident, "models": {}})(),
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            inference_route.openai_completions(
                _json_body_request({"model": resident, "prompt": "hi"}), "tester"
            )
        )
    assert exc.value.status_code == 503
    assert exc.value.detail.startswith("No GGUF model loaded.")
    assert "not downloaded" not in exc.value.detail


def test_responses_stream_keeps_generic_error_when_target_is_local(monkeypatch):
    # Resolves locally yet nothing is loaded: the switch failed, so keep the generic 400.
    status, detail = _run_responses_stream_no_model(
        monkeypatch,
        enabled = True,
        active_model_name = None,
        resolves_to = ("/p/A", "Q4_K_M", "unsloth/Qwen3.5-4B-GGUF"),
    )
    assert status == 400
    assert "not downloaded" not in detail


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
    enabled, idle, keep_kv, auto_dl = settings.set_openai_auto_switch(False, None, False)
    assert settings.AUTO_UNLOAD_IDLE_SETTING_KEY not in store  # idle untouched
    assert settings.OPENAI_AUTO_DOWNLOAD_SETTING_KEY not in store  # nor auto-download
    assert settings.get_auto_unload_idle_seconds() == 600  # env TTL still active
    assert (enabled, idle, keep_kv, auto_dl) == (False, 600, False, False)


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


# Per-model launch config: normalization, LoadRequest mapping, key resolution.


def test_normalize_model_override_drops_unusable_fields_and_keeps_the_rest():
    # A stale field must not cost the user the whole config, so bad values are
    # dropped one by one rather than rejecting the payload.
    entry = settings.normalize_model_override(
        {
            "max_seq_length": 8192,
            "kv_cache_dtype": "not_a_dtype",
            "speculative_type": "mtp",
            "spec_draft_n_max": 999,  # out of range for the MTP draft count
            "gpu_memory_mode": "auto",  # only "manual" is a real override
            "gpu_layers": -1,  # -1 is Auto, which is already the default
            "n_cpu_moe": 0,
            "gpu_ids": [1, 1, 0, "2", -5],
            "tensor_parallel": False,
            "llama_extra_args": [],
        }
    )
    assert entry == {"max_seq_length": 8192, "speculative_type": "mtp", "gpu_ids": [1, 0, 2]}


def test_normalize_model_override_rejects_oversized_chat_template():
    small = settings.normalize_model_override({"chat_template_override": "{{ bos }}"})
    assert small["chat_template_override"] == "{{ bos }}"
    # The limit is bytes, not characters, so a multi-byte template just under the
    # character limit can still be over.
    huge = "é" * settings.MAX_CHAT_TEMPLATE_OVERRIDE_BYTES
    assert "chat_template_override" not in settings.normalize_model_override(
        {"chat_template_override": huge}
    )


def test_spec_draft_n_max_only_stored_for_mtp_modes():
    mtp = settings.normalize_model_override({"speculative_type": "mtp", "spec_draft_n_max": 4})
    assert mtp["spec_draft_n_max"] == 4
    # A non-MTP mode ignores the draft count, so storing it shows an edit that
    # never takes effect.
    ngram = settings.normalize_model_override({"speculative_type": "ngram", "spec_draft_n_max": 4})
    assert "spec_draft_n_max" not in ngram


def test_resolve_fit_max_seq_length_hands_sizing_to_fit_under_manual_auto_layers():
    # Manual GPU memory with Auto layers hands the context to llama.cpp --fit, so
    # the load sends the context pin (or 0), not the stored max seq length.
    override = {"gpu_memory_mode": "manual", "max_seq_length": 8192}
    assert settings.resolve_fit_max_seq_length(override, is_gguf = True) == 0
    assert (
        settings.resolve_fit_max_seq_length(
            {**override, "custom_context_length": 4096}, is_gguf = True
        )
        == 4096
    )
    # Pinning the layer count takes --fit back out of the picture.
    assert settings.resolve_fit_max_seq_length({**override, "gpu_layers": 20}, is_gguf = True) == 8192
    # Not a GGUF, so none of this applies.
    assert settings.resolve_fit_max_seq_length(override, is_gguf = False) == 8192


def test_model_override_load_kwargs_gates_gpu_placement_on_gguf():
    override = {
        "max_seq_length": 4096,
        "kv_cache_dtype": "q8_0",
        "tensor_parallel": True,
        "gpu_memory_mode": "manual",
        "gpu_layers": 20,
        "n_cpu_moe": 3,
        "gpu_ids": [0, 1],
    }
    gguf = settings.model_override_load_kwargs(override, is_gguf = True)
    assert gguf["cache_type_kv"] == "q8_0"
    assert gguf["tensor_parallel"] is True
    assert gguf["gpu_layers"] == 20
    assert gguf["gpu_ids"] == [0, 1]

    # A safetensors model loads through HF auto-placement, so a GGUF GPU pin would
    # silently change where the weights land.
    safetensors = settings.model_override_load_kwargs(override, is_gguf = False)
    assert safetensors["max_seq_length"] == 4096
    assert "gpu_layers" not in safetensors
    assert "gpu_ids" not in safetensors
    assert "n_cpu_moe" not in safetensors
    assert "gpu_memory_mode" not in safetensors

    # Every key must be a real LoadRequest field, or the load raises TypeError when
    # the user's request arrives.
    LoadRequest(model_path = "unsloth/B-GGUF", **gguf)


def test_a_carried_ctx_flag_cannot_outrank_a_freshly_saved_context(monkeypatch):
    # The settings page has no control for pass-through flags, so a save carries over the
    # ones already stored while writing the field the user just edited, leaving one entry
    # holding both. Sent together they reach llama-server with the extras appended last,
    # and its parser takes the final -c, so the stale flag would pin the old context.
    _mock_override_store(monkeypatch)
    _put("unsloth/B-GGUF:Q4_K_M", llama_extra_args = ["--ctx-size", "8192", "--top-k", "40"])
    saved = _put("unsloth/B-GGUF:Q4_K_M", max_seq_length = 32768)
    entry = saved.overrides["unsloth/B-GGUF:Q4_K_M"]
    assert entry["max_seq_length"] == 32768
    assert entry["llama_extra_args"] == ["--ctx-size", "8192", "--top-k", "40"]

    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("unsloth/B-GGUF", "Q4_K_M", "unsloth/B-GGUF"),
        backend = backend,
        recorder = rec,
    )

    _run_hook("unsloth/B-GGUF")
    request = rec.calls[0]
    assert request.max_seq_length == 32768
    # The shadowing flag goes; the sampling flag beside it is nobody's first-class field.
    assert request.llama_extra_args == ["--top-k", "40"]


def test_load_kwargs_strip_only_the_shadow_groups_the_override_supplies():
    # Same rule the /load route applies to inherited extras: one group per first-class
    # field actually being sent, so a flag with nothing to shadow still passes through.
    override = {
        "llama_extra_args": [
            "-c",
            "8192",
            "--cache-type-k",
            "f16",
            "--spec-type",
            "ngram",
            "--jinja",
            "--split-mode",
            "row",
            "--top-p",
            "0.9",
        ],
        "max_seq_length": 32768,
        "kv_cache_dtype": "q8_0",
        "speculative_type": "mtp",
        "chat_template_override": "{{ bos_token }}",
        "tensor_parallel": True,
    }
    stripped = settings.model_override_load_kwargs(override, is_gguf = True)
    assert stripped["llama_extra_args"] == ["--top-p", "0.9"]

    # Nothing is supplied to shadow them, so every flag survives.
    kept = settings.model_override_load_kwargs(
        {"llama_extra_args": override["llama_extra_args"]}, is_gguf = True
    )
    assert kept["llama_extra_args"] == override["llama_extra_args"]

    # And one field strips one group.
    ctx_only = settings.model_override_load_kwargs(
        {"llama_extra_args": override["llama_extra_args"], "max_seq_length": 32768},
        is_gguf = True,
    )
    assert ctx_only["llama_extra_args"] == override["llama_extra_args"][2:]


def test_saved_parallel_slots_reach_an_api_load(monkeypatch):
    # Parallel decode slots are a per-model setting the picker sends on every GGUF load.
    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("unsloth/B-GGUF", "Q4_K_M", "unsloth/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    monkeypatch.setattr(settings, "get_model_override", lambda mid: {"n_parallel": 8})

    _run_hook("unsloth/B-GGUF")
    assert rec.calls[0].n_parallel == 8


def test_parallel_slots_are_stored_and_gated_on_gguf():
    override = settings.normalize_model_override({"n_parallel": 8})
    assert override == {"n_parallel": 8}
    # Blank, out of range and non-integer all mean "follow the server-wide default".
    for bad in (None, 0, -1, settings.PARALLEL_SLOTS_MAX + 1, "many", True):
        assert "n_parallel" not in settings.normalize_model_override({"n_parallel": bad})

    gguf = settings.model_override_load_kwargs(override, is_gguf = True)
    assert gguf["n_parallel"] == 8
    # A safetensors load has no llama-server slots, exactly as the picker gates it.
    assert "n_parallel" not in settings.model_override_load_kwargs(override, is_gguf = False)
    LoadRequest(model_path = "unsloth/B-GGUF", **gguf)


def test_override_route_persists_parallel_slots(override_store):
    # The picker's mirror must carry the field, or a slot-count-only change saves as empty.
    resp = _put("unsloth/B-GGUF:Q4_K_M", n_parallel = 8)
    assert resp.overrides["unsloth/B-GGUF:Q4_K_M"] == {"n_parallel": 8}


def test_eviction_cleanup_clears_mirrored_fields_but_keeps_launch_flags(override_store):
    # Evicting a local entry for storage budget is not a forget, so cleanup sends remove=false.
    settings.set_model_override(
        "unsloth/B-GGUF:Q4_K_M",
        llama_extra_args = ["--flash-attn"],
        custom_context_length = 32768,
        kv_cache_dtype = "q8_0",
    )
    resp = _put("unsloth/B-GGUF:Q4_K_M", remove = False)
    assert resp.overrides["unsloth/B-GGUF:Q4_K_M"] == {"llama_extra_args": ["--flash-attn"]}

    # Nothing server-owned left, so the row goes rather than lingering empty.
    settings.set_model_override("unsloth/C-GGUF:Q4_K_M", custom_context_length = 32768)
    gone = _put("unsloth/C-GGUF:Q4_K_M", remove = False)
    assert "unsloth/C-GGUF:Q4_K_M" not in gone.overrides


def test_auto_switch_prefers_variant_qualified_override(monkeypatch):
    # Settings are per quant; the bare repo id is only the fallback.
    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("unsloth/B-GGUF", "Q4_K_M", "unsloth/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    stored = {
        "unsloth/B-GGUF": {"max_seq_length": 1024},
        "unsloth/B-GGUF:Q4_K_M": {"max_seq_length": 8192, "gpu_layers": 20},
    }
    monkeypatch.setattr(settings, "get_model_override", lambda mid: stored.get(mid, {}))

    _run_hook("unsloth/B-GGUF")
    req = rec.calls[0]
    assert req.max_seq_length == 8192
    assert req.gpu_layers == 20


def test_auto_switch_falls_back_to_bare_repo_override(monkeypatch):
    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("unsloth/B-GGUF", "Q4_K_M", "unsloth/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    stored = {"unsloth/B-GGUF": {"max_seq_length": 1024}}
    monkeypatch.setattr(settings, "get_model_override", lambda mid: stored.get(mid, {}))

    _run_hook("unsloth/B-GGUF")
    assert rec.calls[0].max_seq_length == 1024


def test_override_route_preserves_launch_flags_across_a_settings_only_update(override_store):
    # The settings page has no control for llama_extra_args, so it omits the field.
    _put("unsloth/B-GGUF", llama_extra_args = ["--flash-attn"])
    resp = _put("unsloth/B-GGUF", max_seq_length = 4096)
    entry = resp.overrides["unsloth/B-GGUF"]
    assert entry["llama_extra_args"] == ["--flash-attn"]
    assert entry["max_seq_length"] == 4096

    # An explicit empty list is the UI's "forget", and with no fields left the entry goes.
    gone = _put("unsloth/B-GGUF", llama_extra_args = [])
    assert "unsloth/B-GGUF" not in gone.overrides


def test_override_found_under_a_concrete_path_with_variant(monkeypatch):
    # A local folder resolves to repo id + path; settings saved against the path must be found.
    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("/models/local/Qwen3-8B-Q4_K_M.gguf", "Q4_K_M", "unsloth/Qwen3-8B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    stored = {"/models/local/Qwen3-8B-Q4_K_M.gguf:Q4_K_M": {"max_seq_length": 8192}}
    monkeypatch.setattr(settings, "get_model_override", lambda mid: stored.get(mid, {}))

    _run_hook("unsloth/Qwen3-8B-GGUF")
    assert rec.calls[0].max_seq_length == 8192


def test_path_qualified_override_beats_repo_qualified(monkeypatch):
    # Most specific first: the settings page keys a local row by the path being loaded,
    # while the repo id is only the advertised alias. The row the user edited wins.
    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("/models/local/x.gguf", "Q4_K_M", "unsloth/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    stored = {
        "unsloth/B-GGUF:Q4_K_M": {"max_seq_length": 8192},
        "/models/local/x.gguf:Q4_K_M": {"max_seq_length": 1024},
    }
    monkeypatch.setattr(settings, "get_model_override", lambda mid: stored.get(mid, {}))

    _run_hook("unsloth/B-GGUF")
    assert rec.calls[0].max_seq_length == 1024


def test_first_quant_save_keeps_legacy_bare_repo_launch_flags(override_store):
    # Flags predating per-quant settings live under the bare repo id.
    settings.set_model_override("unsloth/B-GGUF", llama_extra_args = ["--flash-attn"])

    resp = _put("unsloth/B-GGUF:Q4_K_M", max_seq_length = 4096)
    entry = resp.overrides["unsloth/B-GGUF:Q4_K_M"]
    assert entry["max_seq_length"] == 4096
    assert entry["llama_extra_args"] == ["--flash-attn"]


def test_bare_repo_carry_over_does_not_split_a_windows_path(override_store):
    # The colon in "C:\models\x.gguf" is not a variant separator: splitting looks up "C".
    settings.set_model_override("C", llama_extra_args = ["--flash-attn"])

    resp = _put(r"C:\models\x.gguf", max_seq_length = 4096)
    assert "llama_extra_args" not in resp.overrides[r"C:\models\x.gguf"]


def test_windows_path_with_quant_still_carries_over(override_store):
    settings.set_model_override(r"C:\models\x.gguf", llama_extra_args = ["--flash-attn"])

    resp = _put(r"C:\models\x.gguf:Q4_K_M", max_seq_length = 4096)
    assert resp.overrides[r"C:\models\x.gguf:Q4_K_M"]["llama_extra_args"] == ["--flash-attn"]


# The snapshot dir a repo cached outside the active HF cache loads from, which is what an
# older release keyed that row by and what the loader still reads before the repo id.
_LEGACY_SNAPSHOT = "/home/u/.cache/hub-alt/models--unsloth--B-GGUF/snapshots/2f1c9ab"


def test_a_repo_save_retires_the_legacy_snapshot_path_entry(monkeypatch):
    """The two spellings of one cached repo cannot both be stored, or the older wins.

    The one-time backfill mirrors the pre-upgrade path-qualified key to the server, the
    Settings page then keys the same row by its repo id, and the loader reads the load
    path first: without retiring the leftover, every API load applies the settings the
    user just replaced.
    """
    _mock_override_store(monkeypatch)
    settings.set_model_override(f"{_LEGACY_SNAPSHOT}:Q4_K_M", max_seq_length = 4096)

    resp = _put("unsloth/B-GGUF:Q4_K_M", max_seq_length = 32768)
    assert resp.overrides["unsloth/B-GGUF:Q4_K_M"]["max_seq_length"] == 32768
    assert f"{_LEGACY_SNAPSHOT}:Q4_K_M" not in resp.overrides

    # End to end: the load the request triggers carries the saved value, not the retired one.
    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = (_LEGACY_SNAPSHOT, "Q4_K_M", "unsloth/B-GGUF"),
        backend = backend,
        recorder = rec,
    )

    _run_hook("unsloth/B-GGUF:Q4_K_M")
    assert rec.calls[0].max_seq_length == 32768


def test_the_retired_snapshot_path_entry_hands_over_its_launch_flags(override_store):
    # The page can neither show nor restore llama_extra_args, so retiring the entry has to
    # carry them, exactly as the bare repo id does on a first per-quant save.
    settings.set_model_override(
        f"{_LEGACY_SNAPSHOT}:Q4_K_M",
        llama_extra_args = ["--flash-attn"],
        max_seq_length = 4096,
    )

    resp = _put("unsloth/B-GGUF:Q4_K_M", max_seq_length = 32768)
    entry = resp.overrides["unsloth/B-GGUF:Q4_K_M"]
    assert entry["llama_extra_args"] == ["--flash-attn"]
    assert entry["max_seq_length"] == 32768
    assert f"{_LEGACY_SNAPSHOT}:Q4_K_M" not in resp.overrides


def test_a_standalone_gguf_save_keeps_its_filename_label_launch_flags(override_store):
    # An early build keyed a loose .gguf by the quant label from its filename, and the bare
    # path the picker writes today is read before that key (see
    # test_the_filename_label_key_no_longer_shadows_the_bare_path), so writing the bare entry
    # without the legacy flags puts them out of reach of a page that cannot show or restore
    # them. The forget path already consults the same derived key.
    path = "/models/Qwen3-8B-Q4_K_M.gguf"
    settings.set_model_override(
        f"{path}:q4_k_m",
        llama_extra_args = ["--flash-attn"],
        max_seq_length = 4096,
    )

    resp = _put(path, max_seq_length = 32768)
    entry = resp.overrides[path]
    assert entry["llama_extra_args"] == ["--flash-attn"]
    assert entry["max_seq_length"] == 32768


def test_a_snapshot_path_save_retires_the_repo_id_entry(override_store):
    # The same rule the other way round: a row the picker still keys by its path (an inactive
    # cache reached as a local row) must not be shadowed, so the last spelling saved survives.
    settings.set_model_override("unsloth/B-GGUF:Q4_K_M", max_seq_length = 32768)

    resp = _put(f"{_LEGACY_SNAPSHOT}:Q4_K_M", max_seq_length = 4096)
    assert resp.overrides[f"{_LEGACY_SNAPSHOT}:Q4_K_M"]["max_seq_length"] == 4096
    assert "unsloth/B-GGUF:Q4_K_M" not in resp.overrides


def test_forgetting_a_cached_repo_also_clears_its_snapshot_path_entry(override_store):
    # Clearing only the repo id would leave the path entry applying what was forgotten.
    settings.set_model_override(f"{_LEGACY_SNAPSHOT}:Q4_K_M", max_seq_length = 4096)
    settings.set_model_override("unsloth/B-GGUF:Q4_K_M", max_seq_length = 32768)

    resp = _put("unsloth/B-GGUF:Q4_K_M", remove = True, llama_extra_args = [])
    assert resp.overrides == {}


def test_retiring_a_spelling_leaves_every_other_entry_alone(override_store):
    # Only the same repo and the same quant fold together: a ./models path is keyed by its path
    # alone, another quant has its own settings, and a bare entry backs every quant.
    settings.set_model_override("/models/local/x.gguf:Q4_K_M", max_seq_length = 1024)
    settings.set_model_override(f"{_LEGACY_SNAPSHOT}:Q8_0", max_seq_length = 2048)
    settings.set_model_override(_LEGACY_SNAPSHOT, max_seq_length = 4096)

    resp = _put("unsloth/B-GGUF:Q4_K_M", max_seq_length = 32768)
    assert resp.overrides["/models/local/x.gguf:Q4_K_M"]["max_seq_length"] == 1024
    assert resp.overrides[f"{_LEGACY_SNAPSHOT}:Q8_0"]["max_seq_length"] == 2048
    assert resp.overrides[_LEGACY_SNAPSHOT]["max_seq_length"] == 4096


def test_the_one_time_fill_retires_nothing(override_store):
    # fill_absent_fields only adds what is missing; the migration mirroring both spellings
    # of one row must not have its own first write deleted by its second.
    settings.set_model_override(f"{_LEGACY_SNAPSHOT}:Q4_K_M", max_seq_length = 4096)

    resp = _put("unsloth/B-GGUF:Q4_K_M", max_seq_length = 32768, fill_absent_fields = True)
    assert resp.overrides[f"{_LEGACY_SNAPSHOT}:Q4_K_M"]["max_seq_length"] == 4096
    assert resp.overrides["unsloth/B-GGUF:Q4_K_M"]["max_seq_length"] == 32768


def test_a_fill_never_creates_a_snapshot_path_key_over_a_repo_id_entry(override_store):
    # A fill only adds, so it cannot retire the other spelling the way a save does. Creating
    # the snapshot path key would leave two entries for one quant, and the loader reads the
    # load path before the advertised repo id, so an upgraded browser's pre-upgrade copy would
    # shadow the newer server config on every API load. Fill into the entry that is already
    # there instead: nothing outranks it and the fields it lacks still arrive.
    settings.set_model_override("unsloth/B-GGUF:Q4_K_M", max_seq_length = 32768)

    resp = _put(
        f"{_LEGACY_SNAPSHOT}:Q4_K_M",
        max_seq_length = 2048,
        kv_cache_dtype = "q8_0",
        fill_absent_fields = True,
    )
    assert f"{_LEGACY_SNAPSHOT}:Q4_K_M" not in resp.overrides
    entry = resp.overrides["unsloth/B-GGUF:Q4_K_M"]
    assert entry["max_seq_length"] == 32768
    assert entry["kv_cache_dtype"] == "q8_0"


def test_stale_gpu_ids_are_dropped_not_fatal(monkeypatch):
    # A two-GPU pin on a one-GPU box used to 400 the load; one dead field degrades to defaults.
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
        lambda mid: {"gpu_ids": [0, 1], "max_seq_length": 4096},
    )

    async def _unusable(ids):
        return False

    monkeypatch.setattr(inference_route, "_override_gpu_ids_still_resolve", _unusable)

    _run_hook("unsloth/B-GGUF")
    req = rec.calls[0]
    assert not req.gpu_ids
    # The rest of the config still applies.
    assert req.max_seq_length == 4096


def test_usable_gpu_ids_are_kept(monkeypatch):
    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("unsloth/B-GGUF", "Q4_K_M", "unsloth/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    monkeypatch.setattr(settings, "get_model_override", lambda mid: {"gpu_ids": [0, 1]})

    async def _usable(ids):
        return True

    monkeypatch.setattr(inference_route, "_override_gpu_ids_still_resolve", _usable)

    _run_hook("unsloth/B-GGUF")
    assert rec.calls[0].gpu_ids == [0, 1]


def test_override_gpu_ids_probe_never_raises(monkeypatch):
    # On the load path, so a hardware error must read as "unusable", not a 500.
    import utils.hardware.hardware as hw

    def boom(*args, **kwargs):
        raise RuntimeError("driver exploded")

    monkeypatch.setattr(hw, "resolve_requested_gpu_ids", boom)
    assert asyncio.run(inference_route._override_gpu_ids_still_resolve([0])) is False


def test_vulkan_ordinal_absent_from_the_probe_is_unusable(monkeypatch):
    # resolve_requested_gpu_ids only rejects malformed ordinals; presence needs the ggml probe.
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda: True))
    monkeypatch.setattr(
        LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: "/bin/llama-server")
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_get_gpu_memory", staticmethod(lambda binary: [(0, 8192)])
    )
    assert asyncio.run(inference_route._override_gpu_ids_still_resolve([0])) is True
    assert asyncio.run(inference_route._override_gpu_ids_still_resolve([7])) is False
    assert asyncio.run(inference_route._override_gpu_ids_still_resolve([0, 1])) is False


def test_vulkan_probe_without_a_binary_does_not_block_the_load(monkeypatch):
    # Nothing to probe with, and refusing would drop a valid pin on every load.
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda: True))
    monkeypatch.setattr(LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: None))
    assert asyncio.run(inference_route._override_gpu_ids_still_resolve([0])) is True


def test_default_save_preserves_flags_instead_of_removing(override_store):
    # All-default values send no fields, shape-identical to a removal; guessing wrong wipes flags.
    settings.set_model_override("unsloth/B-GGUF", llama_extra_args = ["--flash-attn"])

    resp = _put("unsloth/B-GGUF", remove = False)
    assert resp.overrides["unsloth/B-GGUF"]["llama_extra_args"] == ["--flash-attn"]


def test_explicit_remove_still_clears_everything(override_store):
    settings.set_model_override(
        "unsloth/B-GGUF", llama_extra_args = ["--flash-attn"], max_seq_length = 4096
    )
    resp = _put("unsloth/B-GGUF", remove = True, llama_extra_args = [])
    assert "unsloth/B-GGUF" not in resp.overrides


def test_bare_payload_without_remove_flag_still_removes(override_store):
    # The original contract, kept for any caller that predates the flag.
    settings.set_model_override("unsloth/B-GGUF", max_seq_length = 4096)
    resp = _put("unsloth/B-GGUF")
    assert "unsloth/B-GGUF" not in resp.overrides


def test_remove_false_with_real_fields_saves_normally(override_store):
    resp = _put("unsloth/B-GGUF", remove = False, max_seq_length = 8192)
    assert resp.overrides["unsloth/B-GGUF"]["max_seq_length"] == 8192


def test_override_lookup_falls_back_to_case_insensitive(override_store):
    # The browser lowercases ids while the resolver asks for the repo's real casing.
    settings.set_model_override("unsloth/qwen3-8b-gguf:q4_k_m", max_seq_length = 8192)
    got = settings.get_model_override("unsloth/Qwen3-8B-GGUF:Q4_K_M")
    assert got["max_seq_length"] == 8192


def test_exact_override_match_beats_a_case_variant(override_store):
    settings.set_model_override("/models/foo.gguf", max_seq_length = 1024)
    settings.set_model_override("/models/Foo.gguf", max_seq_length = 8192)
    assert settings.get_model_override("/models/Foo.gguf")["max_seq_length"] == 8192
    assert settings.get_model_override("/models/foo.gguf")["max_seq_length"] == 1024


def test_ambiguous_case_fallback_matches_nothing(override_store):
    # Two POSIX paths differing only in case are two files; guessing applies the wrong settings.
    settings.set_model_override("/models/foo.gguf", max_seq_length = 1024)
    settings.set_model_override("/models/FOO.gguf", max_seq_length = 8192)
    assert settings.get_model_override("/models/Foo.gguf") == {}


def test_request_used_api_key_distinguishes_key_from_session():
    from auth.authentication import API_KEY_PREFIX

    class _Req:
        def __init__(self, header):
            self.headers = {"authorization": header} if header else {}

    assert inference_route._request_used_api_key(_Req(f"Bearer {API_KEY_PREFIX}abc")) is True
    assert inference_route._request_used_api_key(_Req(f"bearer {API_KEY_PREFIX}abc")) is True
    assert inference_route._request_used_api_key(_Req("Bearer eyJhbGciOiJIUzI1NiJ9.x")) is False
    assert inference_route._request_used_api_key(_Req("")) is False
    assert inference_route._request_used_api_key(_Req(None)) is False
    # Hot path: a malformed request object must read as "not an API key" rather than raise.
    assert inference_route._request_used_api_key(object()) is False
    # A stand-in whose headers answer with anything at all, which is how the load
    # routes are driven in tests: a monitor label must never take a load down.
    from unittest.mock import MagicMock

    assert inference_route._request_used_api_key(MagicMock()) is False


def test_case_fallback_never_applies_to_a_posix_path(override_store):
    # Two casings are two models on Linux, so a near miss loads defaults, not the other's pin.
    settings.set_model_override("/models/foo.gguf", max_seq_length = 8192, gpu_ids = [1])
    assert settings.get_model_override("/models/Foo.gguf") == {}
    assert settings.get_model_override("/models/foo.gguf")["max_seq_length"] == 8192


def test_case_fallback_does_apply_to_a_windows_path(override_store):
    # NTFS is case-insensitive: these name one file, and the browser folds drive paths.
    settings.set_model_override(r"c:\models\foo.gguf", max_seq_length = 8192)
    assert settings.get_model_override(r"C:\models\FOO.gguf")["max_seq_length"] == 8192
    assert settings.get_model_override("C:/Models/Foo.gguf")["max_seq_length"] == 8192


def test_case_fallback_applies_to_unc_and_wsl_drive_paths(override_store):
    settings.set_model_override(r"\\server\share\foo.gguf", max_seq_length = 4096)
    settings.set_model_override("/mnt/c/models/bar.gguf", max_seq_length = 2048)
    assert settings.get_model_override(r"\\Server\Share\FOO.gguf")["max_seq_length"] == 4096
    assert settings.get_model_override("/mnt/C/Models/Bar.gguf")["max_seq_length"] == 2048


def test_a_plain_posix_path_under_mnt_stays_case_sensitive(override_store):
    # Only /mnt/<letter> is a WSL drive mount; /mnt/data is an ordinary case-sensitive mount.
    settings.set_model_override("/mnt/data/models/foo.gguf", max_seq_length = 8192)
    assert settings.get_model_override("/mnt/data/models/Foo.gguf") == {}


def test_an_ambiguous_windows_case_fallback_still_matches_nothing(override_store):
    # Two keys folding to one has no single answer, so the load takes defaults.
    settings.set_model_override(r"c:\models\foo.gguf", max_seq_length = 1024)
    settings.set_model_override("C:/models/FOO.gguf", max_seq_length = 8192)
    assert settings.get_model_override(r"C:\Models\Foo.gguf") == {}


def test_case_fallback_still_covers_repo_ids(override_store):
    # The migration case this fallback exists for.
    settings.set_model_override("unsloth/qwen3-8b-gguf:q4_k_m", max_seq_length = 8192)
    assert settings.get_model_override("unsloth/Qwen3-8B-GGUF:Q4_K_M")["max_seq_length"] == 8192


def test_explicit_remove_is_not_blocked_by_stale_invalid_flags(override_store):
    # remove is the operation discriminator: a rejected flag must not turn a forget into a 400.
    settings.set_model_override("unsloth/B-GGUF", max_seq_length = 4096)
    resp = _put("unsloth/B-GGUF", remove = True, llama_extra_args = ["--port", "1234"])
    assert "unsloth/B-GGUF" not in resp.overrides


def test_explicit_remove_wins_over_config_fields_in_the_same_payload(override_store):
    # remove is the operation discriminator: a stale field beside it must not make it an update.
    settings.set_model_override("unsloth/B-GGUF", max_seq_length = 4096)
    resp = _put("unsloth/B-GGUF", remove = True, max_seq_length = 8192, tensor_parallel = True)
    assert "unsloth/B-GGUF" not in resp.overrides


def test_posix_colon_in_a_path_is_not_treated_as_a_quant(override_store):
    # "/models/foo:bar.gguf" is one filename: splitting grafts /models/foo's flags onto it.
    settings.set_model_override("/models/foo", llama_extra_args = ["--flash-attn"])
    resp = _put("/models/foo:bar.gguf", max_seq_length = 4096)
    assert "llama_extra_args" not in resp.overrides["/models/foo:bar.gguf"]


def test_unknown_quant_label_on_a_gguf_still_carries_flags_over(override_store):
    # A .gguf with no quant token is labelled by its stem, so the UI saves ":custom".
    settings.set_model_override("/models/custom.gguf", llama_extra_args = ["--flash-attn"])
    resp = _put("/models/custom.gguf:custom", max_seq_length = 4096)
    assert resp.overrides["/models/custom.gguf:custom"]["llama_extra_args"] == ["--flash-attn"]


def test_bpw_qualified_variants_still_carry_flags_over(override_store):
    # model_config.py keeps a bits-per-weight modifier on the label, and that form reaches keys.
    settings.set_model_override("unsloth/Repo-GGUF", llama_extra_args = ["--flash-attn"])
    resp = _put("unsloth/Repo-GGUF:IQ4_XS-3.53bpw", max_seq_length = 4096)
    assert resp.overrides["unsloth/Repo-GGUF:IQ4_XS-3.53bpw"]["llama_extra_args"] == [
        "--flash-attn"
    ]


def test_a_posix_path_variant_folds_while_the_path_does_not(override_store):
    # The browser lowercases the quant but keeps POSIX path casing.
    settings.set_model_override("/models/Foo:q4_k_m", max_seq_length = 8192)
    assert settings.get_model_override("/models/Foo:Q4_K_M")["max_seq_length"] == 8192
    assert settings.get_model_override("/models/foo:Q4_K_M") == {}


def test_an_unknown_gguf_label_is_reachable_in_either_casing(override_store):
    # The stem-derived label is lowercased in storage while the scanner keeps filename casing.
    settings.set_model_override("/models/CustomModel.gguf:custommodel", max_seq_length = 8192)
    got = settings.get_model_override("/models/CustomModel.gguf:CustomModel")
    assert got["max_seq_length"] == 8192
    # The path itself is still case-sensitive on POSIX.
    assert settings.get_model_override("/models/custommodel.gguf:CustomModel") == {}


def test_a_posix_colon_filename_is_not_folded_as_a_variant(override_store):
    # "/models/foo:Bar.gguf" is one filename: folding its tail reaches a different file.
    settings.set_model_override("/models/foo:bar.gguf", max_seq_length = 8192)
    assert settings.get_model_override("/models/foo:Bar.gguf") == {}


def test_a_suffix_the_scanner_would_not_derive_carries_nothing_over(override_store):
    # Only the scanner's exact label is accepted, so a stray colon suffix reaches nothing.
    settings.set_model_override("/models/custom.gguf", llama_extra_args = ["--flash-attn"])
    resp = _put("/models/custom.gguf:something-else", max_seq_length = 4096)
    assert "llama_extra_args" not in resp.overrides["/models/custom.gguf:something-else"]


def test_unknown_quant_label_carries_over_for_a_windows_path(override_store):
    # Written on Windows, read back where a backslash is an ordinary filename character.
    settings.set_model_override(r"C:\models\custom.gguf", llama_extra_args = ["--flash-attn"])
    resp = _put(r"C:\models\custom.gguf:custom", max_seq_length = 4096)
    assert resp.overrides[r"C:\models\custom.gguf:custom"]["llama_extra_args"] == ["--flash-attn"]


def test_real_quant_suffix_on_a_path_still_carries_flags_over(override_store):
    settings.set_model_override("/models/x.gguf", llama_extra_args = ["--flash-attn"])
    resp = _put("/models/x.gguf:Q4_K_M", max_seq_length = 4096)
    assert resp.overrides["/models/x.gguf:Q4_K_M"]["llama_extra_args"] == ["--flash-attn"]


def test_load_retries_without_gpu_ids_when_the_loader_rejects_the_pin(monkeypatch):
    # The pre-flight check can't mirror every loader rule, and a stale pin must not block a load.
    from fastapi import HTTPException

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
        settings, "get_model_override", lambda mid: {"gpu_ids": [0], "max_seq_length": 4096}
    )

    async def _usable(ids):
        return True

    monkeypatch.setattr(inference_route, "_override_gpu_ids_still_resolve", _usable)

    calls = {"n": 0}

    async def _load(request, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise HTTPException(
                status_code = 400,
                detail = "GPU selection (gpu_ids) is not supported for a DiffusionGemma GGUF",
            )
        return await rec(request, *args, **kwargs)

    monkeypatch.setattr(inference_route, "_load_model_impl", _load)

    _run_hook("unsloth/B-GGUF")
    assert calls["n"] == 2
    served = rec.calls[-1]
    assert not served.gpu_ids
    assert served.max_seq_length == 4096


def test_a_non_gpu_load_failure_is_not_retried(monkeypatch):
    from fastapi import HTTPException

    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("unsloth/B-GGUF", "Q4_K_M", "unsloth/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    monkeypatch.setattr(settings, "get_model_override", lambda mid: {"gpu_ids": [0]})

    async def _usable(ids):
        return True

    monkeypatch.setattr(inference_route, "_override_gpu_ids_still_resolve", _usable)

    calls = {"n": 0}

    async def _load(request, *args, **kwargs):
        calls["n"] += 1
        raise HTTPException(status_code = 400, detail = "Corrupt GGUF header")

    monkeypatch.setattr(inference_route, "_load_model_impl", _load)

    with pytest.raises(HTTPException):
        _run_hook("unsloth/B-GGUF")
    assert calls["n"] == 1


def test_removal_clears_the_entry_a_load_would_actually_resolve(override_store):
    # A forget can carry a different casing; removing only the literal key leaves a live entry.
    settings.set_model_override("unsloth/B-GGUF:Q4_K_M", max_seq_length = 8192)
    assert settings.get_model_override("unsloth/b-gguf:q4_k_m")["max_seq_length"] == 8192

    _put("unsloth/b-gguf:q4_k_m", remove = True)
    assert settings.get_model_overrides() == {}
    assert settings.get_model_override("unsloth/B-GGUF:Q4_K_M") == {}


def test_save_updates_the_existing_case_variant_instead_of_forking_it(override_store):
    # The backfill stores lowercase keys while a later UI save carries the catalog's casing.
    settings.set_model_override("unsloth/b-gguf:q4_k_m", max_seq_length = 8192)
    _put("unsloth/B-GGUF:Q4_K_M", max_seq_length = 4096)
    assert list(settings.get_model_overrides()) == ["unsloth/b-gguf:q4_k_m"]
    assert settings.get_model_override("Unsloth/B-GGUF:Q4_K_M")["max_seq_length"] == 4096


def test_removal_of_a_path_still_only_touches_the_exact_key(override_store):
    settings.set_model_override("/models/foo.gguf", max_seq_length = 8192)
    _put("/models/Foo.gguf", remove = True)
    # A different file must survive its neighbour being forgotten.
    assert settings.get_model_override("/models/foo.gguf")["max_seq_length"] == 8192


def test_forget_clears_the_filename_derived_key_a_load_still_reads(override_store):
    # The picker once keyed a standalone .gguf by its filename quant label; backfill carries it.
    settings.set_model_override(
        "/models/Qwen3-8B-Q4_K_M.gguf:q4_k_m",
        max_seq_length = 8192,
    )
    _put("/models/Qwen3-8B-Q4_K_M.gguf", remove = True)
    assert settings.get_model_override("/models/Qwen3-8B-Q4_K_M.gguf:Q4_K_M") == {}
    assert settings.get_model_overrides() == {}


def test_forget_clears_the_bare_repo_entry_the_quant_inherited_from(override_store):
    # A save under repo:QUANT copies the flags off a legacy bare entry and leaves it in place,
    # and the loader falls back to it, so clearing only the qualified key hands them back.
    settings.set_model_override(
        "unsloth/B-GGUF",
        llama_extra_args = ["--flash-attn"],
        max_seq_length = 8192,
    )
    _put("unsloth/B-GGUF:Q4_K_M", max_seq_length = 4096)
    _put("unsloth/B-GGUF:Q4_K_M", remove = True)
    assert settings.get_model_override("unsloth/B-GGUF") == {}
    assert settings.get_model_overrides() == {}


def test_forget_keeps_a_bare_entry_another_quant_still_has_settings_under(override_store):
    # The bare entry backs every quant with no entry of its own, so forgetting Q4 must not
    # strip it while Q8 is still there: this forget is not the last word on the model.
    settings.set_model_override("unsloth/B-GGUF", max_seq_length = 8192)
    settings.set_model_override("unsloth/B-GGUF:Q8_0", max_seq_length = 2048)
    _put("unsloth/B-GGUF:Q4_K_M", max_seq_length = 4096)
    _put("unsloth/B-GGUF:Q4_K_M", remove = True)
    assert settings.get_model_override("unsloth/B-GGUF") == {"max_seq_length": 8192}
    assert settings.get_model_override("unsloth/B-GGUF:Q8_0") == {"max_seq_length": 2048}


def test_forget_clears_every_spelling_of_one_model(override_store):
    # Two spellings of one repo can coexist; clearing only the named one makes the survivor
    # the sole fold match, so the next load reapplies what was just forgotten.
    settings.set_model_override("unsloth/B-GGUF:Q4_K_M", max_seq_length = 8192)
    settings.set_model_override("unsloth/b-gguf:q4_k_m", max_seq_length = 8192)
    _put("unsloth/B-GGUF:Q4_K_M", remove = True)
    assert settings.get_model_overrides() == {}
    assert settings.get_model_override("unsloth/B-GGUF:Q4_K_M") == {}


def test_forget_of_one_windows_spelling_clears_the_other(override_store):
    # Windows paths fold, so two spellings are one file and both have to go.
    settings.set_model_override(r"C:\Models\x.gguf", max_seq_length = 8192)
    settings.set_model_override("c:/models/X.gguf", max_seq_length = 4096)
    _put(r"C:\Models\x.gguf", remove = True)
    assert settings.get_model_overrides() == {}


def test_forget_of_a_posix_path_still_spares_its_case_sibling(override_store):
    # POSIX paths do not fold: two casings are two files, so a forget spares the sibling.
    settings.set_model_override("/models/foo.gguf", max_seq_length = 8192)
    settings.set_model_override("/models/Foo.gguf", max_seq_length = 4096)
    _put("/models/foo.gguf", remove = True)
    assert list(settings.get_model_overrides()) == ["/models/Foo.gguf"]


def test_forget_leaves_another_file_own_derived_key_alone(override_store):
    # The derived key uses the forgotten file's own path, so a quant-sharing neighbour survives.
    settings.set_model_override("/models/Other-Q4_K_M.gguf:q4_k_m", max_seq_length = 4096)
    _put("/models/Qwen3-8B-Q4_K_M.gguf", remove = True)
    assert settings.get_model_override("/models/Other-Q4_K_M.gguf:Q4_K_M")["max_seq_length"] == 4096


def test_forget_of_a_repo_quant_key_derives_nothing(override_store):
    # Only a bare .gguf path derives a label.
    settings.set_model_override("unsloth/b-gguf:q4_k_m", max_seq_length = 8192)
    _put("unsloth/B-GGUF", remove = True)
    assert settings.get_model_override("unsloth/b-gguf:q4_k_m")["max_seq_length"] == 8192


def test_a_tag_that_names_no_quant_resolves_to_the_repo(monkeypatch):
    # A downloaded but unloaded GGUF asked for as org/model:latest missed the resolver,
    # so the switch could not load it (404ing on a quant that was never a quant with
    # auto-download on, refusing with it off). A real quant that is not on disk must
    # still miss, or a swap would serve the wrong weights under the right name.
    from core.inference.local_model_resolver import _LocalGgufEntry

    import time

    entry = _LocalGgufEntry("org/model", "/srv/models/org--model", ("Q4_K_M",))
    # Fresh stamp so _index serves this instead of rescanning over it.
    monkeypatch.setattr(resolver, "_scan", (time.monotonic(), {"org/model": entry}))
    for tag in ("org/model:latest", "org/model:8b", "org/model"):
        assert resolver.resolve_local_gguf(tag) == (
            "/srv/models/org--model",
            "Q4_K_M",
            "org/model",
        )
    assert resolver.resolve_local_gguf("org/model:Q8_0") is None
    assert resolver.resolve_local_gguf("org/model:Q4_K_M") == (
        "/srv/models/org--model",
        "Q4_K_M",
        "org/model",
    )


def test_any_finished_download_drops_the_resolver_cache(monkeypatch):
    # Only the API auto-download watcher invalidated, so a GGUF fetched in the Hub UI
    # stayed absent to the cache-only request path and the resident model answered.
    # Every worker exits through here.
    import logging

    from hub.services import download_lifecycle

    class _Proc:
        stderr = None

        def wait(self):
            return 0

    class _Registry:
        def cancel_requested(self, key):
            return False

        def drop_process(self, key, proc):
            return True

        def get_job_metadata(self, key):
            return None

        def set_job(self, key, state):
            self.state = state

    resolver._scan = (1234.0, {"already-here": "entry"})
    assert (
        download_lifecycle.finalize_worker_exit(
            _Registry(),
            "org/model:Q4_K_M",
            _Proc(),
            hf_token = None,
            label = "org/model",
            log_prefix = "[test]",
            logger = logging.getLogger(__name__),
            repo_type = "model",
            repo_id = "org/model",
        )
        == "complete"
    )
    stamp, entries = resolver._scan
    assert stamp == 0.0, "a finished download left the scan looking fresh"
    # Evidence for models already indexed has to survive, or a bare request for one
    # of them during the rebuild is answered by whatever is resident.
    assert entries == {"already-here": "entry"}


def test_invalidating_keeps_the_entries_it_already_had(monkeypatch):
    # The request path reads this cache without scanning, so emptying it leaves no
    # evidence until the rebuild lands. Only a completed download invalidates, and
    # that only adds, so the entries stay true.
    import time

    entry = resolver._LocalGgufEntry("org/old", "/srv/models/org--old", ("Q4_K_M",))
    monkeypatch.setattr(resolver, "_scan", (time.monotonic(), {"org/old": entry}))
    resolver.invalidate_index()
    assert resolver._scan[0] == 0.0
    assert resolver.resolve_local_gguf("org/old", allow_scan = False) == (
        "/srv/models/org--old",
        "Q4_K_M",
        "org/old",
    )


def test_a_bare_local_id_takes_the_quant_a_plain_load_would(monkeypatch, tmp_path):
    # list_local_gguf_variants orders by descending size, so the head is the biggest
    # quant. Resolving a bare id to that could evict a working model and then OOM on an
    # F16 next to a fitting Q4, and /v1/models advertised the same head for pinning.
    from core.inference.local_model_resolver import _local_gguf_entry

    for name, size in (("model-F16.gguf", 900), ("model-Q4_K_M.gguf", 100)):
        (tmp_path / name).write_bytes(b"\0" * size)
    entry = _local_gguf_entry("org/model", type("I", (), {"path": str(tmp_path)})())
    assert entry is not None
    assert set(entry.variants) == {"F16", "Q4_K_M"}
    assert entry.variants[0] == "Q4_K_M", "a bare id would have resolved to F16"


def test_local_and_remote_agree_on_the_preferred_quant():
    # A bare id must mean the same quant whichever side answered it.
    from core.inference.openai_auto_download import _match_variant, preferred_quant

    labels = ("F16", "Q8_0", "UD-Q4_K_XL", "Q4_K_M")
    assert preferred_quant(labels) == _match_variant(None, dict.fromkeys(labels, 1))
    assert preferred_quant(labels) not in ("F16",)


def test_a_just_downloaded_model_is_evidence_before_the_scan_indexes_it(monkeypatch):
    # The retained index covers what was known, but nothing covers the model that just
    # landed until the next scan: a bare request for it was answered by the resident one.
    import logging

    from hub.services import download_lifecycle

    class _Proc:
        stderr = None

        def wait(self):
            return 0

    class _Registry:
        def cancel_requested(self, key):
            return False

        def drop_process(self, key, proc):
            return True

        def get_job_metadata(self, key):
            return None

        def set_job(self, key, state):
            pass

    assert not resolver.recently_downloaded("org/fresh")
    download_lifecycle.finalize_worker_exit(
        _Registry(),
        "org/fresh:Q4_K_M",
        _Proc(),
        hf_token = None,
        label = "org/fresh",
        log_prefix = "[test]",
        logger = logging.getLogger(__name__),
        repo_type = "model",
        repo_id = "org/fresh",
    )
    assert resolver.recently_downloaded("org/fresh"), "no evidence for the new model"
    assert resolver.recently_downloaded("ORG/Fresh"), "evidence must be case-insensitive"
    assert not resolver.recently_downloaded("org/other")

    # The scan that indexes it supersedes the note.
    monkeypatch.setattr(resolver, "_build_index", dict)
    resolver._index()
    assert not resolver.recently_downloaded("org/fresh")


def test_a_finished_dataset_is_not_recorded_as_a_local_model(monkeypatch):
    # finalize_worker_exit is shared with dataset downloads. Noting one as a local model
    # would refuse a bare /v1 request naming that id instead of letting a foreign id
    # fall through, and would kick off a multi-directory scan for nothing.
    import logging
    import time

    from hub.services import download_lifecycle

    class _Proc:
        stderr = None

        def wait(self):
            return 0

    class _Registry:
        def cancel_requested(self, key):
            return False

        def drop_process(self, key, proc):
            return True

        def get_job_metadata(self, key):
            return None

        def set_job(self, key, state):
            pass

    stamp = time.monotonic()
    monkeypatch.setattr(resolver, "_scan", (stamp, {"kept": "entry"}))
    download_lifecycle.finalize_worker_exit(
        _Registry(),
        "org/corpus",
        _Proc(),
        hf_token = None,
        label = "org/corpus",
        log_prefix = "[test]",
        logger = logging.getLogger(__name__),
        repo_type = "dataset",
        repo_id = "org/corpus",
    )
    assert not resolver.recently_downloaded("org/corpus")
    assert resolver._scan == (stamp, {"kept": "entry"}), "a dataset invalidated the index"


def test_two_local_paths_differing_only_in_case_are_not_the_same_model(monkeypatch):
    # _loaded_satisfies lowercased the request and every backend identifier, so on a
    # case-sensitive filesystem /srv/models/foo.gguf read as satisfied by a resident
    # /srv/models/Foo.gguf. A repo alias must still stay case-insensitive.
    import os

    loaded = _FakeBackend(loaded_id = "/srv/models/Foo.gguf")
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: loaded)
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: type("B", (), {"active_model_name": None})(),
    )
    assert inference_route._loaded_satisfies("/srv/models/Foo.gguf") is True
    same = os.path.normcase("A") == os.path.normcase("a")
    assert inference_route._loaded_satisfies("/srv/models/foo.gguf") is same

    alias = _FakeBackend(loaded_id = "unsloth/Qwen3-4B-GGUF")
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: alias)
    assert inference_route._loaded_satisfies("unsloth/qwen3-4b-gguf") is True


def test_abs_path_ids_are_recognised_in_either_platform_spelling():
    """Path() follows the running OS, so a Windows host read "/home/me/x.gguf" as
    relative and a POSIX host read "C:\\models\\x.gguf" the same way, and either
    then reached /v1/models as a published host path. Ids outlive the machine
    that wrote them (settings sync, WSL, a copied config), and the model-override
    identity already folds both spellings."""
    from types import SimpleNamespace

    for spelling in ("/home/me/models/x.gguf", "C:\\models\\x.gguf", "//host/share/x.gguf"):
        assert resolver._is_abs_path_id(spelling) is True, spelling
        # An alias wins, and with none the path is stripped to a public id.
        assert (
            resolver._advertised_loader_id(
                SimpleNamespace(id = spelling, model_id = "org/X-GGUF", display_name = "X")
            )
            == "org/X-GGUF"
        )
        advertised = resolver._advertised_loader_id(
            SimpleNamespace(id = spelling, model_id = None, display_name = None)
        )
        assert advertised is not None
        assert "/" not in advertised and "\\" not in advertised, advertised

    # A repo id has no leading separator, drive or UNC prefix under either reading.
    for repo_id in ("org/Repo-GGUF", "Repo", "org/Repo-GGUF:Q4_K_M"):
        assert resolver._is_abs_path_id(repo_id) is False, repo_id


def test_fill_absent_fields_put_never_replaces_a_newer_server_value(override_store):
    """The one-time localStorage backfill reads the override map once and then
    writes each model in turn, so a save by another tab during that pass was
    overwritten by this browser's older copy. fill_absent_fields writes only what
    the entry lacks, so every value already on the server wins."""
    import routes.settings as settings_route

    # The other tab's save lands first.
    newer = settings_route.ModelOverridePayload(model_id = "unsloth/B-GGUF", max_seq_length = 8192)
    settings_route.update_openai_auto_switch_override(newer, "tester")

    # The backfill's write, carrying this browser's older localStorage value.
    backfill = settings_route.ModelOverridePayload(
        model_id = "unsloth/B-GGUF", max_seq_length = 2048, fill_absent_fields = True
    )
    resp = settings_route.update_openai_auto_switch_override(backfill, "tester")
    assert resp.overrides["unsloth/B-GGUF"]["max_seq_length"] == 8192

    # With nothing stored it still creates, or the migration would never run.
    fresh = settings_route.ModelOverridePayload(
        model_id = "unsloth/C-GGUF", max_seq_length = 2048, fill_absent_fields = True
    )
    resp2 = settings_route.update_openai_auto_switch_override(fresh, "tester")
    assert resp2.overrides["unsloth/C-GGUF"]["max_seq_length"] == 2048


def test_fill_absent_fields_carries_the_browser_only_settings_into_a_legacy_entry(monkeypatch):
    """Codex P1: the override map shipped before the browser mirror did, storing only
    llama_extra_args and max_seq_length. An upgraded install holds such an entry while
    localStorage holds the context, KV cache, speculative and GPU settings, and an
    entry-level skip would strand exactly what the migration exists to carry."""
    import routes.settings as settings_route

    store = _mock_override_store(monkeypatch)

    legacy = settings_route.ModelOverridePayload(
        model_id = "unsloth/B-GGUF:Q4_K_M",
        llama_extra_args = ["--flash-attn"],
        max_seq_length = 8192,
    )
    settings_route.update_openai_auto_switch_override(legacy, "tester")

    backfill = settings_route.ModelOverridePayload(
        model_id = "unsloth/B-GGUF:Q4_K_M",
        # A field the server already has, plus the ones only the browser holds.
        max_seq_length = 2048,
        custom_context_length = 32768,
        kv_cache_dtype = "q8_0",
        speculative_type = "ngram",
        gpu_ids = [0, 1],
        fill_absent_fields = True,
    )
    resp = settings_route.update_openai_auto_switch_override(backfill, "tester")
    entry = resp.overrides["unsloth/B-GGUF:Q4_K_M"]
    # The server's own values survive untouched.
    assert entry["max_seq_length"] == 8192
    assert entry["llama_extra_args"] == ["--flash-attn"]
    # The browser-only settings are now there, so an API load applies them.
    assert entry["custom_context_length"] == 32768
    assert entry["kv_cache_dtype"] == "q8_0"
    assert entry["speculative_type"] == "ngram"
    assert entry["gpu_ids"] == [0, 1]
    # One entry, not two: the fill resolves onto the key a load reads.
    assert list(store[settings.MODEL_OVERRIDES_SETTING_KEY]) == ["unsloth/B-GGUF:Q4_K_M"]

    # An ordinary save is still a replacement, or an edit could never clear a field.
    edit = settings_route.ModelOverridePayload(
        model_id = "unsloth/B-GGUF:Q4_K_M", max_seq_length = 4096
    )
    resp2 = settings_route.update_openai_auto_switch_override(edit, "tester")
    assert resp2.overrides["unsloth/B-GGUF:Q4_K_M"]["max_seq_length"] == 4096
    assert "kv_cache_dtype" not in resp2.overrides["unsloth/B-GGUF:Q4_K_M"]


def test_fill_absent_fields_matches_a_legacy_casing_and_never_deletes(override_store):
    """The stored key can carry the casing an older install typed, and it must not
    be duplicated or emptied by a fill for the folded spelling."""
    import routes.settings as settings_route

    stored = settings_route.ModelOverridePayload(
        model_id = "Unsloth/B-GGUF:Q4_K_M", max_seq_length = 8192
    )
    settings_route.update_openai_auto_switch_override(stored, "tester")

    folded = settings_route.ModelOverridePayload(
        model_id = "unsloth/b-gguf:q4_k_m", max_seq_length = 2048, fill_absent_fields = True
    )
    resp = settings_route.update_openai_auto_switch_override(folded, "tester")
    assert list(resp.overrides) == ["Unsloth/B-GGUF:Q4_K_M"]
    assert resp.overrides["Unsloth/B-GGUF:Q4_K_M"]["max_seq_length"] == 8192

    # An all-default fill is a no-op, not the "empty payload means forget" path.
    empty = settings_route.ModelOverridePayload(
        model_id = "Unsloth/B-GGUF:Q4_K_M", fill_absent_fields = True
    )
    resp2 = settings_route.update_openai_auto_switch_override(empty, "tester")
    assert resp2.overrides["Unsloth/B-GGUF:Q4_K_M"]["max_seq_length"] == 8192

    # A fill that is also a delete has no meaning.
    with pytest.raises(HTTPException) as excinfo:
        _put("Unsloth/B-GGUF:Q4_K_M", remove = True, fill_absent_fields = True)
    assert excinfo.value.status_code == 400


def test_fill_absent_fields_does_not_break_the_empty_payload_removal(override_store):
    """fill_absent_fields is a write mode, not a saved field: leaving it in the dumped
    payload would make every request look non-empty and silently retire the legacy
    "a payload carrying only model_id forgets this model" contract."""
    import routes.settings as settings_route

    stored = settings_route.ModelOverridePayload(model_id = "unsloth/B-GGUF", max_seq_length = 4096)
    settings_route.update_openai_auto_switch_override(stored, "tester")
    empty = settings_route.ModelOverridePayload(model_id = "unsloth/B-GGUF")
    resp = settings_route.update_openai_auto_switch_override(empty, "tester")
    assert "unsloth/B-GGUF" not in resp.overrides


def test_map_entry_fill_reads_and_writes_in_one_transaction(tmp_path, monkeypatch):
    """The real store, not the in-memory stand-in: the read has to share the write's
    transaction, or a concurrent writer still slips between them."""
    import storage.studio_db as db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(db, "_schema_ready", False)

    key = "test_map_entry_create"
    assert db.upsert_app_setting_map_entry(key, "a", {"v": 1}) == {"a": {"v": 1}}
    # Present: the stored value stays, and a field it lacks is added.
    assert db.upsert_app_setting_map_entry(key, "a", {"v": 2}, fill_absent_fields = True) == {
        "a": {"v": 1}
    }
    assert db.get_app_setting(key) == {"a": {"v": 1}}
    assert db.upsert_app_setting_map_entry(key, "a", {"v": 2, "w": 7}, fill_absent_fields = True) == {
        "a": {"v": 1, "w": 7}
    }
    assert db.get_app_setting(key) == {"a": {"v": 1, "w": 7}}
    # Absent: created.
    assert db.upsert_app_setting_map_entry(key, "b", {"v": 3}, fill_absent_fields = True) == {
        "a": {"v": 1, "w": 7},
        "b": {"v": 3},
    }
    # A fill never deletes, even with nothing to store.
    assert db.upsert_app_setting_map_entry(key, "a", None, fill_absent_fields = True) == {
        "a": {"v": 1, "w": 7},
        "b": {"v": 3},
    }
    # The ordinary write still replaces and still removes.
    db.upsert_app_setting_map_entry(key, "a", {"v": 9})
    db.upsert_app_setting_map_entry(key, "b", None)
    assert db.get_app_setting(key) == {"a": {"v": 9}}


def test_gpu_ids_dedupe_is_not_a_scan_of_the_list_being_built():
    """gpu_ids arrives from an authenticated client and normalize_model_override
    de-duplicates it. Testing membership against the growing list walks up to
    MAX_GPU_ID entries per element; a set keeps the pass linear. Order, bounds and
    the bool rejection all have to survive the change."""
    import time

    from utils.openai_auto_switch_settings import MAX_GPU_ID, normalize_model_override

    assert normalize_model_override({"gpu_ids": [3, 1, 3, 0, 1, 2]})["gpu_ids"] == [3, 1, 0, 2]
    # bool is an int subclass; [True, False] must not pin GPUs 1 and 0.
    assert normalize_model_override({"gpu_ids": [True, False]}) == {}
    assert normalize_model_override({"gpu_ids": [MAX_GPU_ID + 1, -1, 2]})["gpu_ids"] == [2]

    ids = [index % (MAX_GPU_ID + 1) for index in range(200_000)]
    started = time.perf_counter()
    normalized = normalize_model_override({"gpu_ids": ids})
    elapsed = time.perf_counter() - started
    assert len(normalized["gpu_ids"]) == MAX_GPU_ID + 1
    # The scan version took ~1s for this input on a dev box; a linear pass is ~50ms.
    assert elapsed < 0.5, elapsed


def test_gpu_ids_payload_is_bounded():
    """A list longer than the number of ids the normalizer can store adds nothing
    but work, so it is rejected at the boundary. A real device list is tiny."""
    import pydantic

    import routes.settings as settings_route
    from utils.openai_auto_switch_settings import MAX_GPU_ID

    assert settings_route.MAX_GPU_IDS == MAX_GPU_ID + 1
    at_limit = settings_route.ModelOverridePayload(
        model_id = "x", gpu_ids = list(range(settings_route.MAX_GPU_IDS))
    )
    assert len(at_limit.gpu_ids) == settings_route.MAX_GPU_IDS
    with pytest.raises(pydantic.ValidationError):
        settings_route.ModelOverridePayload(
            model_id = "x", gpu_ids = [0] * (settings_route.MAX_GPU_IDS + 1)
        )
    # The ordinary case is untouched.
    assert settings_route.ModelOverridePayload(model_id = "x", gpu_ids = [0, 1]).gpu_ids == [0, 1]


# ── codex round: the concrete key beats the advertised alias ──────────


def _switch_with_overrides(monkeypatch, resolves_to, stored, requested):
    """Run the auto-switch hook against a real override map and return the load."""
    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = True, resolves_to = resolves_to, backend = backend, recorder = rec)
    _mock_override_store(monkeypatch)
    for key, max_seq_length in stored.items():
        settings.set_model_override(key, max_seq_length = max_seq_length)
    _run_hook(requested)
    assert len(rec.calls) == 1
    return rec.calls[0]


def test_a_loose_gguf_prefers_its_path_keyed_settings_over_the_alias(monkeypatch):
    """Codex: the settings UI keys a standalone .gguf by its bare path, while
    override_id is the filename stem /v1/models advertises and an overrides PUT can
    be written against. Reading the alias first let it shadow the saved settings for
    good, so an API load kept applying the old flags."""
    path = "/srv/models/Qwen3-8B-Q4_K_M.gguf"
    alias = "Qwen3-8B-Q4_K_M"
    req = _switch_with_overrides(
        monkeypatch,
        resolves_to = (path, None, alias),
        stored = {alias: 2048, path: 32768},
        requested = alias,
    )
    assert req.max_seq_length == 32768

    # The alias is still read when it is the only key.
    req2 = _switch_with_overrides(
        monkeypatch,
        resolves_to = (path, None, alias),
        stored = {alias: 2048},
        requested = alias,
    )
    assert req2.max_seq_length == 2048


def test_the_filename_label_key_no_longer_shadows_the_bare_path(monkeypatch):
    """An early build of this feature keyed a standalone .gguf by the quant label
    derived from its filename. Those entries stay readable, but the bare path the
    picker writes today comes first."""
    path = "/srv/models/Qwen3-8B-Q4_K_M.gguf"
    alias = "Qwen3-8B-Q4_K_M"
    req = _switch_with_overrides(
        monkeypatch,
        resolves_to = (path, None, alias),
        stored = {f"{path}:Q4_K_M": 2048, path: 32768},
        requested = alias,
    )
    assert req.max_seq_length == 32768

    req2 = _switch_with_overrides(
        monkeypatch,
        resolves_to = (path, None, alias),
        stored = {f"{path}:Q4_K_M": 2048},
        requested = alias,
    )
    assert req2.max_seq_length == 2048


def test_a_variant_qualified_path_key_beats_the_same_quant_under_the_alias(monkeypatch):
    """An LM Studio dir or a non-active HF cache is configured against its path, so
    a same-quant entry under the repo id (another copy of the same repo, or a
    hand-written PUT) must not win over the row the user actually edited."""
    path = "/srv/lmstudio/publisher/Qwen3-8B-GGUF"
    repo = "publisher/Qwen3-8B-GGUF"
    req = _switch_with_overrides(
        monkeypatch,
        resolves_to = (path, "Q4_K_M", repo),
        stored = {f"{repo}:Q4_K_M": 2048, f"{path}:Q4_K_M": 32768},
        requested = f"{repo}:Q4_K_M",
    )
    assert req.max_seq_length == 32768


def test_a_cached_repo_still_resolves_by_its_repo_id(monkeypatch):
    """The Hub keys a cached repo row by its repo id, which is the advertised id,
    and no path entry exists for it, so it still resolves on the second try."""
    snapshot = "/mnt/old-cache/models--unsloth--Qwen3-8B-GGUF/snapshots/abc123"
    repo = "unsloth/Qwen3-8B-GGUF"
    req = _switch_with_overrides(
        monkeypatch,
        resolves_to = (snapshot, "Q4_K_M", repo),
        stored = {f"{repo}:Q4_K_M": 32768},
        requested = f"{repo}:Q4_K_M",
    )
    assert req.max_seq_length == 32768

    # A bare entry under the repo id keeps working too.
    req2 = _switch_with_overrides(
        monkeypatch,
        resolves_to = (snapshot, "Q4_K_M", repo),
        stored = {repo: 16384},
        requested = f"{repo}:Q4_K_M",
    )
    assert req2.max_seq_length == 16384


def test_a_fill_does_not_replay_a_stored_flag_through_validation(monkeypatch):
    """The migration now writes for entries it used to skip, and an omitted
    llama_extra_args is normally carried over from the stored entry. Replaying a
    flag that has been denylisted since it was saved would 400 the one-time
    migration, which then retries on every start. A fill keeps the stored flags
    without sending them back."""
    import routes.settings as settings_route
    from core.inference import llama_server_args

    store = _mock_override_store(monkeypatch)
    settings.set_model_override("unsloth/B-GGUF:Q4_K_M", llama_extra_args = ["--flash-attn"])

    # The flag is refused from now on, as a later release's denylist would.
    real_validate = llama_server_args.validate_extra_args

    def _reject_flash_attn(args):
        if args and "--flash-attn" in args:
            raise ValueError("--flash-attn is managed by the server.")
        return real_validate(args)

    monkeypatch.setattr(llama_server_args, "validate_extra_args", _reject_flash_attn)

    fill = settings_route.ModelOverridePayload(
        model_id = "unsloth/B-GGUF:Q4_K_M", custom_context_length = 32768, fill_absent_fields = True
    )
    resp = settings_route.update_openai_auto_switch_override(fill, "tester")
    entry = resp.overrides["unsloth/B-GGUF:Q4_K_M"]
    assert entry["llama_extra_args"] == ["--flash-attn"]
    assert entry["custom_context_length"] == 32768
    assert list(store[settings.MODEL_OVERRIDES_SETTING_KEY]) == ["unsloth/B-GGUF:Q4_K_M"]

    # An ordinary save still validates what it is handed.
    with pytest.raises(HTTPException) as excinfo:
        _put("unsloth/C-GGUF", llama_extra_args = ["--flash-attn"])
    assert excinfo.value.status_code == 400


def test_override_payload_rejects_booleans_for_numeric_fields():
    """bool subclasses int and pydantic parses non-strictly, so `true` would
    arrive as 1: `max_seq_length: true` becomes a one-token context and
    `gpu_ids: [true]` pins GPU 1. _bounded_int rejects bools for exactly that
    reason, but never sees one, because coercion happens at the route boundary
    first. Reject them there so that guard is reachable through this path."""
    import pytest
    from pydantic import ValidationError
    from routes.settings import ModelOverridePayload

    for field, value in (
        ("max_seq_length", True),
        ("custom_context_length", True),
        ("spec_draft_n_max", True),
        ("n_parallel", True),
        ("gpu_layers", False),
        ("n_cpu_moe", True),
        ("gpu_ids", [True]),
        ("gpu_ids", [0, False, 2]),
    ):
        with pytest.raises(ValidationError):
            ModelOverridePayload(model_id = "unsloth/x-GGUF:Q4_K_M", **{field: value})

    # Only bools are rejected: real values, including real booleans, still validate.
    ok = ModelOverridePayload(
        model_id = "unsloth/x-GGUF:Q4_K_M",
        max_seq_length = 4096,
        gpu_layers = -1,
        n_cpu_moe = 0,
        gpu_ids = [0, 1],
        tensor_parallel = True,
        remove = True,
        fill_absent_fields = True,
    )
    assert ok.max_seq_length == 4096
    assert ok.gpu_ids == [0, 1]
    assert ok.gpu_layers == -1
    assert ok.tensor_parallel is True
    assert ok.remove is True
    assert ok.fill_absent_fields is True


# ── codex round: the alias retirement has to be atomic with the write ──


def test_two_spellings_of_one_cached_quant_do_not_delete_each_others_save(monkeypatch):
    """A save writes its target key, then reads the map back to retire the other spelling
    of the same cached repo, in a second transaction. This route is a plain `def`, so
    FastAPI runs it in a threadpool: two clients saving one quant, one by repo id and one
    by the snapshot path an upgraded install still holds, can both write before either
    cleanup runs and then retire each other's row. Both calls return 200 and nothing is
    stored. Whichever runs second must retire the first instead."""
    import threading
    import routes.settings as settings_route

    _mock_override_store(monkeypatch)
    repo = "unsloth/Qwen3-8B-GGUF:Q4_K_M"
    snapshot = "/mnt/old-cache/models--unsloth--Qwen3-8B-GGUF/snapshots/abc123:Q4_K_M"

    ready = threading.Barrier(2)
    written = threading.Barrier(2)
    write_lock = threading.Lock()
    wrote = set()
    real_set = settings_route.set_model_override

    def _set_then_sync(model_id, *args, **kwargs):
        # One write is one transaction in production (BEGIN IMMEDIATE), so keep the fake
        # store's read-modify-write atomic too: only the gap between writes is on trial.
        with write_lock:
            result = real_set(model_id, *args, **kwargs)
        ident = threading.get_ident()
        if ident not in wrote:
            wrote.add(ident)
            try:
                # Hold each save just after it stored its own key, so neither looks for
                # aliases until both are there. Broken instead when the two are serialized:
                # the second saver cannot reach this while the first is still inside.
                written.wait(timeout = 1.0)
            except threading.BrokenBarrierError:
                pass
        return result

    monkeypatch.setattr(settings_route, "set_model_override", _set_then_sync)

    failures = []

    def _save(model_id, max_seq_length):
        try:
            ready.wait(timeout = 10.0)
            _put(model_id, max_seq_length = max_seq_length)
        except BaseException as exc:  # noqa: BLE001 - re-reported below
            failures.append(exc)

    threads = [
        threading.Thread(target = _save, args = (repo, 8192)),
        threading.Thread(target = _save, args = (snapshot, 4096)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout = 30.0)
        assert not thread.is_alive()
    assert not failures, failures

    stored = settings.get_model_overrides()
    assert list(stored) in ([repo], [snapshot]), stored
    assert stored[list(stored)[0]]["max_seq_length"] in (4096, 8192)
