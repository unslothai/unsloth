# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in OpenAI /v1 model auto-switch: resolver, hook, and settings coercion.

No GPU or llama-server: the backend and the load route are mocked, mirroring
tests/test_gguf_completion_usage.py.
"""

import asyncio
import json
import os
import threading
import time
import types

import pytest
from fastapi import HTTPException

import routes.inference as inference_route
from models.inference import LoadRequest
from core.inference import local_model_resolver as resolver
from utils import openai_auto_switch_settings as settings
from core.inference import llama_cpp as llama_cpp_mod
from core.inference import llama_keepwarm as kw
from core.inference.llama_cpp import LlamaCppBackend
from core.inference.llama_keepwarm import _is_inference_path
from models.inference import ChatCompletionRequest
from models.inference import ResponsesRequest
from pathlib import Path
from storage import studio_db
from types import SimpleNamespace
from utils import hf_cache_settings
from utils.hardware import hardware as hw
import base64 as _b64
import inspect
import io
import routes.models as models_route
import routes.settings as settings_route
import storage.studio_db as db
import utils.paths as paths
from core.inference.tools import cached_mcp_tools
from hub.services import download_lifecycle
from models.inference import ChatMessage, ImageContentPart, ImageUrl
import logging
import routes.models as model_routes
import state.tool_policy as _tp


@pytest.fixture(autouse = True)
def _auto_switch_waiters_are_not_carried_between_tests(request, monkeypatch):
    """Restore ``_auto_switch_waiters`` around every test, and fail the test that dirties it.

    It is a module-level dict in routes.inference, so a test that registers a waiting request
    and does not unregister it leaves that entry behind for every later test in the same xdist
    worker. It matters beyond tidiness because ``_switch_waiter_count()`` sums every key rather
    than reading one, so a single stranded entry inflates the count for the whole worker and
    ``_wait_for_model_switch_idle`` sees waiters that do not exist.

    Two jobs, deliberately. Restoring keeps the next test starting from a known state. Raising
    names the test that left the residue instead of the unrelated one that trips over it later,
    which is the whole difficulty with this class of bug: the failure surfaces nowhere near its
    cause.

    The two halves have different proofs, and one of them has none. Removing the marker from a
    staging test makes that test fail, so the detection half is covered. Removing the restore
    changes nothing any test here can observe: the growth check is per-test, so a carried-over
    entry only harms files that run LATER in the same worker, and which files share a worker is
    decided by xdist at run time. A cleanliness assertion in a second file would pass vacuously
    whenever the two land in different processes, which is worse than no test at all, so the
    restore is kept as a defensive measure and is deliberately left unproven.

    ``monkeypatch`` is requested, and not because this fixture patches anything. It is what
    fixes the teardown ORDER. ``_wire()`` rebinds the registry with
    ``monkeypatch.setattr(inference_route, "_auto_switch_waiters", {})``, so the entry a test
    stages goes into a temporary dict, and whichever of the two fixtures tears down second sees
    the original one restored and nothing amiss. Depending on ``monkeypatch`` here makes this
    fixture set up after it and therefore tear down before it, so the read below lands on the
    dict the test actually wrote to. That ordering held incidentally without the dependency,
    which is exactly the reason to state it: a guard that works by accident stops working
    silently.

    Three tests stage a waiting request on purpose, with ``_note_switch_waiter(key, 1)`` and no
    matching -1, because that is the honest way to set the condition up. They carry
    ``@pytest.mark.stages_switch_waiter`` to say so, which is checked here rather than inferred
    from a name, so a new leak cannot arrive silently by resembling them.
    """
    before = dict(inference_route._auto_switch_waiters)
    try:
        yield
    finally:
        after = dict(inference_route._auto_switch_waiters)
        inference_route._auto_switch_waiters.clear()
        inference_route._auto_switch_waiters.update(before)
    # Only growth is flagged, since tests that reset the registry are cleaning up, not leaking.
    leaked = {key: count for key, count in after.items() if count > before.get(key, 0)}
    if leaked and request.node.get_closest_marker("stages_switch_waiter") is None:
        raise AssertionError(
            "this test left routes.inference._auto_switch_waiters dirty: "
            f"{leaked!r} (registry went {before!r} -> {after!r}). _switch_waiter_count() sums "
            "every key, so the entry inflates the waiter count for every later test in this "
            "xdist worker. Unregister it, or mark the test @pytest.mark.stages_switch_waiter "
            "if the residue is the point."
        )


async def _boom(*a, **k):
    raise _Reached()


def _reset_keepwarm():
    """Clear the keep-warm counters and mark the model long idle."""
    kw._inflight = 0
    kw._pending = 0
    kw._last_active = time.monotonic() - 3600
    kw._last_unloaded_model = None
    kw._kv_resume = None


def _vision_gguf_cache_repo(tmp_path):
    """An HF cache repo whose older snapshot holds the weights and newer one the companions."""
    repo = tmp_path / "models--org--Vision-GGUF"
    old = repo / "snapshots" / "weights-revision"
    old.mkdir(parents = True)
    (old / "vision-model-Q4_K_M.gguf").write_bytes(b"GGUF weights")
    newer = repo / "snapshots" / "companion-revision"
    newer.mkdir(parents = True)
    return repo, old, newer


class _Reached(Exception):
    pass


class _CountBackend(LlamaCppBackend):
    is_loaded = True
    base_url = "http://127.0.0.1:1"
    _auth_headers = None

    def __init__(self):
        pass


async def _usable(ids, index_kind = "physical"):
    return True


class _Proc:
    stderr = None

    def wait(self):
        return 0


async def receive():
    return {"type": "http.request", "body": b"", "more_body": False}


async def send(_m):
    pass


# Captured before the autouse fixture below pins it, so its own test can reach the real one.
_REAL_HOST_HAS_NON_GGUF_BACKEND = resolver._host_has_a_non_gguf_backend


@pytest.fixture(autouse = True)
def _host_serves_non_gguf(monkeypatch):
    """Pin the host-capability gates for the classifier tests.

    They are about the config rules, not about whether this machine happens to have
    torch or MLX installed, and an unpinned MLX verdict made the whole file pass or fail
    by platform. Each gate is covered by its own test below.
    """
    monkeypatch.setattr(resolver, "_host_has_a_non_gguf_backend", lambda: True)
    # the device itself, not the helper, so a test setting DEVICE for itself still wins.
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.CUDA, raising = False)


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
    _loaded_by_user_action = False

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
        cache_environment = None,
        anonymous_hf_access = False,
        speech_codec_path = None,
    ):
        await inference_route._wait_for_model_switch_idle(
            current_request_counted = current_request_counted
        )
        self.calls.append(request)
        self.cache_environment = cache_environment
        self.anonymous_hf_access = anonymous_hf_access
        self.speech_codec_path = speech_codec_path
        if self.fail:
            raise HTTPException(status_code = 503, detail = "load failed")
        self.backend.model_identifier = request.model_path
        self.backend.hf_variant = getattr(request, "gguf_variant", None)
        self.backend._gguf_path = request.model_path
        self.backend.is_loaded = True
        self.backend._openai_advertised_id = None
        self.backend._openai_gguf_companion_roots = tuple(request._gguf_companion_roots)
        self.backend._openai_gguf_companion_state = resolver.local_gguf_companion_state(
            tuple(request._gguf_companion_roots)
        )

        kw.note_model_loaded(self.backend)
        return None


# Reload checks use the configured idle TTL, not the effective one residency may zero.
def _wire(monkeypatch, *, enabled, resolves_to, backend, recorder):
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: enabled)
    monkeypatch.setattr(resolver, "resolve_local_gguf", lambda _m, **_kw: resolves_to)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    # Auto-switch already owns the lifecycle gate, so it calls _load_model_impl directly.
    monkeypatch.setattr(inference_route, "_load_model_impl", recorder)
    monkeypatch.setattr(inference_route, "_auto_switch_waiters", {})
    monkeypatch.setattr(inference_route, "_preflight_speech_codec_for_switch", lambda *_a: None)


def _wire_on(
    *args,
    enabled = True,
    **kwargs,
):
    """_wire with auto-switch enabled."""
    return _wire(*args, enabled = enabled, **kwargs)


def _wired(
    monkeypatch,
    backend,
    resolves_to,
    *,
    enabled = True,
    fail = False,
):
    """Load recorder + _wire around a backend: the setup nearly every test below repeats."""
    rec = _LoadRecorder(backend, fail = fail)
    _wire(monkeypatch, enabled = enabled, resolves_to = resolves_to, backend = backend, recorder = rec)
    return backend, rec


async def _noop_reject(*_args, **_kwargs):
    return None


def _run_hook(model = "some/model"):
    asyncio.run(inference_route._maybe_auto_switch_model(model, object(), "tester"))


def test_flag_off_never_loads(monkeypatch):
    backend, rec = _wired(
        monkeypatch,
        _FakeBackend("unsloth/A-GGUF"),
        ("unsloth/B-GGUF", None, "unsloth/B-GGUF"),
        enabled = False,
    )
    # Off means no load, but A must not answer as B either: say why instead.
    with pytest.raises(HTTPException) as excinfo:
        _run_hook("unsloth/B-GGUF")
    assert excinfo.value.status_code == 404
    assert "Switch model by request" in str(excinfo.value.detail)
    assert rec.calls == []


def test_unknown_model_falls_through(monkeypatch):
    backend, rec = _wired(monkeypatch, _FakeBackend("unsloth/A-GGUF"), None)
    _run_hook("gpt-4o-mini")
    assert rec.calls == []


def test_already_loaded_does_not_reload(monkeypatch):
    backend, rec = _wired(
        monkeypatch, _FakeBackend("unsloth/A-GGUF"), ("unsloth/a-gguf", None, "unsloth/a-gguf")
    )
    _run_hook("unsloth/A-GGUF")
    assert rec.calls == []


def test_known_unloaded_model_switches_once(monkeypatch):
    backend, rec = _wired(
        monkeypatch, _FakeBackend("unsloth/A-GGUF"), ("unsloth/B-GGUF", "Q4_K_M", "unsloth/B-GGUF")
    )
    _run_hook("unsloth/B-GGUF:Q4_K_M")
    assert len(rec.calls) == 1
    req = rec.calls[0]
    assert isinstance(req, LoadRequest)
    assert req.model_path == "unsloth/B-GGUF"
    assert req.gguf_variant == "Q4_K_M"
    assert backend.model_identifier == "unsloth/B-GGUF"


def test_a_late_cancel_keeps_the_completed_switch_metadata(monkeypatch):
    backend = _FakeBackend("unsloth/A-GGUF")
    cancel_event = threading.Event()
    calls = []

    async def _load(
        request,
        *_args,
        load_cancel_event = None,
        **_kwargs,
    ):
        assert load_cancel_event is cancel_event
        calls.append(request)
        backend.model_identifier = request.model_path
        backend.is_loaded = True
        cancel_event.set()

    _wire_on(
        monkeypatch,
        resolves_to = ("/models/B.gguf", "Q4_K_M", "unsloth/B-GGUF"),
        backend = backend,
        recorder = _load,
    )
    request = types.SimpleNamespace(
        state = types.SimpleNamespace(generation_cancel_event = cancel_event)
    )
    inference_route._set_preview_resident(None)
    try:
        with pytest.raises(HTTPException) as exc:
            asyncio.run(
                inference_route._maybe_auto_switch_model(
                    "unsloth/B-GGUF", request, "tester", claim_resident = False
                )
            )

        assert exc.value.status_code == 409
        assert len(calls) == 1
        assert backend._openai_advertised_id == "unsloth/B-GGUF"
        assert backend._loaded_by_user_action is False
        assert inference_route._is_preview_resident("/models/B.gguf")
    finally:
        inference_route._set_preview_resident(None)


def test_resident_model_skips_the_filesystem_resolver(monkeypatch):
    backend, rec = _wired(
        monkeypatch, _FakeBackend("unsloth/Muse-Glimmer-30B-GGUF", "UD-Q4_K_XL"), None
    )
    warmed = []
    monkeypatch.setattr(resolver, "warm_index_soon", lambda: warmed.append(1))

    def _unexpected_resolve(*_args, **_kwargs):
        raise AssertionError("the resident model must not touch the filesystem index")

    monkeypatch.setattr(resolver, "resolve_local_gguf", _unexpected_resolve)
    _run_hook("unsloth/Muse-Glimmer-30B-GGUF")
    assert rec.calls == []
    assert warmed == [1]


def test_auto_switch_reads_an_additions_only_snapshot_without_rebuilding_it(monkeypatch):
    real_resolve = resolver.resolve_local_gguf
    backend, rec = _wired(monkeypatch, _FakeBackend("unsloth/A-GGUF", "Q4_K_M"), None)
    calls = []
    warmed = []

    entry = resolver._LocalGgufEntry("unsloth/B-GGUF", "/models/unsloth/B-GGUF", ("Q4_K_M",))
    monkeypatch.setattr(resolver, "_scan", (time.monotonic(), {"unsloth/b-gguf": entry}))
    resolver.invalidate_index(additions_only = True)
    assert resolver._scan[0] < 0.0
    assert resolver.index_is_built() is False

    def _resolve(name, **kwargs):
        calls.append((name, kwargs))
        return real_resolve(name, **kwargs)

    monkeypatch.setattr(resolver, "warm_index_soon", lambda: warmed.append(1))
    monkeypatch.setattr(resolver, "resolve_local_gguf", _resolve)

    async def _accept_loaded_target(*_args, **_kwargs):
        assert backend.model_identifier == "/models/unsloth/B-GGUF"
        assert backend._openai_advertised_id == "unsloth/B-GGUF"

    monkeypatch.setattr(inference_route, "_reject_unservable_model", _accept_loaded_target)

    _run_hook("unsloth/B-GGUF:Q4_K_M")

    assert calls == []
    assert warmed == [1]
    assert len(rec.calls) == 1


def test_non_additive_invalidation_keeps_unservable_check_cold(monkeypatch):
    real_resolve = resolver.resolve_local_gguf
    backend, rec = _wired(
        monkeypatch, _FakeBackend("unsloth/A-GGUF", "Q4_K_M"), None, enabled = False
    )
    monkeypatch.setattr(resolver, "resolve_local_gguf", real_resolve)
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: SimpleNamespace(active_model_name = None),
    )

    old = resolver._LocalGgufEntry("unsloth/A-GGUF", "/models/unsloth/A-GGUF", ("Q4_K_M",))
    added = resolver._LocalGgufEntry("unsloth/B-GGUF", "/models/unsloth/B-GGUF", ("Q4_K_M",))
    resolver._scan = (time.monotonic(), {"unsloth/a-gguf": old})
    resolver.invalidate_index()
    assert resolver.index_is_built() is False
    monkeypatch.setattr(
        resolver,
        "_build_index",
        lambda: {"unsloth/a-gguf": old, "unsloth/b-gguf": added},
    )

    with pytest.raises(HTTPException) as excinfo:
        _run_hook("unsloth/B-GGUF")

    assert excinfo.value.status_code == 404
    assert "Switch model by request" in str(excinfo.value.detail)
    assert rec.calls == []


def test_an_expired_positive_hit_refreshes_before_switching(monkeypatch):
    real_resolve = resolver.resolve_local_gguf
    backend, rec = _wired(monkeypatch, _FakeBackend("unsloth/A-GGUF", "Q4_K_M"), None)
    removed = resolver._LocalGgufEntry(
        "unsloth/B-GGUF", "/removed-root/unsloth/B-GGUF", ("Q4_K_M",)
    )
    monkeypatch.setattr(resolver, "_scan", (1.0, {"unsloth/b-gguf": removed}))
    scans = []
    calls = []
    monkeypatch.setattr(resolver, "_build_index", lambda: scans.append(1) or {})

    def _resolve(name, **kwargs):
        calls.append((name, kwargs))
        return real_resolve(name, **kwargs)

    monkeypatch.setattr(resolver, "resolve_local_gguf", _resolve)

    # A hub-style id is a concrete reference, so a name proven absent is refused, not served.
    with pytest.raises(HTTPException) as excinfo:
        _run_hook("unsloth/B-GGUF")

    assert excinfo.value.status_code == 404
    assert calls == [
        ("unsloth/B-GGUF", {"include_companion_scope": True}),
        ("unsloth/B-GGUF", {"allow_scan": False}),
        ("unsloth/A-GGUF:Q4_K_M", {"allow_scan": False}),
    ]
    assert scans == [1]
    assert rec.calls == []
    assert backend.model_identifier == "unsloth/A-GGUF"


def test_a_stale_miss_refreshes_before_the_resident_model_can_answer(monkeypatch):
    real_resolve = resolver.resolve_local_gguf
    backend, rec = _wired(monkeypatch, _FakeBackend("unsloth/A-GGUF", "Q4_K_M"), None)
    old = resolver._LocalGgufEntry("unsloth/A-GGUF", "/models/unsloth/A-GGUF", ("Q4_K_M",))
    added = resolver._LocalGgufEntry("unsloth/B-GGUF", "/models/unsloth/B-GGUF", ("Q4_K_M",))
    monkeypatch.setattr(resolver, "_scan", (1.0, {"unsloth/a-gguf": old}))
    scans = []
    calls = []
    monkeypatch.setattr(
        resolver,
        "_build_index",
        lambda: scans.append(1) or {"unsloth/a-gguf": old, "unsloth/b-gguf": added},
    )

    def _resolve(name, **kwargs):
        calls.append((name, kwargs))
        return real_resolve(name, **kwargs)

    monkeypatch.setattr(resolver, "resolve_local_gguf", _resolve)

    async def _accept_loaded_target(*_args, **_kwargs):
        assert backend.model_identifier == "/models/unsloth/B-GGUF"
        assert backend._openai_advertised_id == "unsloth/B-GGUF"

    monkeypatch.setattr(inference_route, "_reject_unservable_model", _accept_loaded_target)

    _run_hook("unsloth/B-GGUF")

    assert calls == [("unsloth/B-GGUF", {"include_companion_scope": True})]
    assert scans == [1]
    assert len(rec.calls) == 1


def test_concurrent_same_target_loads_once(monkeypatch):
    backend, rec = _wired(
        monkeypatch, _FakeBackend(None), ("unsloth/B-GGUF", None, "unsloth/B-GGUF")
    )

    async def _race():
        await asyncio.gather(
            inference_route._maybe_auto_switch_model("unsloth/B-GGUF", object(), "t"),
            inference_route._maybe_auto_switch_model("unsloth/B-GGUF", object(), "t"),
        )

    asyncio.run(_race())
    assert len(rec.calls) == 1


def test_load_failure_propagates(monkeypatch):
    backend, rec = _wired(
        monkeypatch,
        _FakeBackend("unsloth/A-GGUF"),
        ("unsloth/B-GGUF", None, "unsloth/B-GGUF"),
        fail = True,
    )
    with pytest.raises(HTTPException):
        _run_hook("unsloth/B-GGUF")


def test_same_repo_different_variant_switches(monkeypatch):
    # Q4_K_M loaded, Q8_0 requested: a different quant must trigger a reload.
    backend, rec = _wired(
        monkeypatch,
        _FakeBackend("unsloth/B-GGUF", hf_variant = "Q4_K_M"),
        ("unsloth/B-GGUF", "Q8_0", "unsloth/B-GGUF"),
    )
    _run_hook("unsloth/B-GGUF:Q8_0")
    assert len(rec.calls) == 1
    assert rec.calls[0].gguf_variant == "Q8_0"


def test_same_repo_same_variant_does_not_reload(monkeypatch):
    backend = _FakeBackend("unsloth/B-GGUF", hf_variant = "Q4_K_M")
    rec = _LoadRecorder(backend)
    _wire_on(
        monkeypatch,
        resolves_to = ("unsloth/B-GGUF", "q4_k_m", "unsloth/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    _run_hook("unsloth/B-GGUF:Q4_K_M")
    assert rec.calls == []


def test_responses_endpoint_wires_auto_switch_before_streaming_dispatch():
    # streaming bypasses chat, so its switch must run before dispatch.

    src = inspect.getsource(inference_route.openai_responses)
    assert "_maybe_auto_switch_model" in src
    hook_at = src.index("_maybe_auto_switch_model")
    assert src.index("if payload.stream:") < hook_at
    assert hook_at < src.index("_responses_stream")


def test_nonstreaming_responses_passes_subject_to_chat(monkeypatch):
    seen = {}

    async def _capture(_payload, _request, current_subject):
        seen["current_subject"] = current_subject
        raise RuntimeError("stop after dispatch")

    monkeypatch.setattr(inference_route, "openai_chat_completions", _capture)
    request = types.SimpleNamespace(
        state = types.SimpleNamespace(skip_api_monitor = True),
        url = types.SimpleNamespace(path = "/v1/responses"),
        method = "POST",
    )
    with pytest.raises(RuntimeError, match = "stop after dispatch"):
        asyncio.run(
            inference_route.openai_responses(
                ResponsesRequest(model = "org/B-GGUF", input = "hi"),
                request,
                "tester",
            )
        )

    assert seen == {"current_subject": "tester"}


def test_embeddings_endpoint_wires_auto_switch_before_loaded_check():
    # /v1/embeddings carries a model, so it must auto-switch before the loaded-state gate.

    src = inspect.getsource(inference_route.openai_embeddings)
    assert "_auto_switch_from_request_body" in src
    assert src.index("_auto_switch_from_request_body") < src.index("is_loaded")


def test_count_tokens_endpoint_wires_auto_switch_before_loaded_check():
    # The Anthropic token-count endpoint must count with the requested model.

    src = inspect.getsource(inference_route.anthropic_count_tokens)
    assert "_maybe_auto_switch_model" in src
    assert src.index("_maybe_auto_switch_model") < src.index("is_loaded")


def test_openai_compat_routes_bound_to_handlers_with_auth():
    # A helper inserted under @router.post silently rebinds the route and drops auth; lock the mapping.
    expected = {
        ("POST", "/chat/completions"): "openai_chat_completions",
        ("POST", "/completions"): "openai_completions",
        ("POST", "/embeddings"): "openai_embeddings",
        ("POST", "/responses"): "openai_responses",
        ("POST", "/messages"): "anthropic_messages",
        ("POST", "/messages/count_tokens"): "anthropic_count_tokens",
        ("POST", "/audio/generate"): "generate_audio",
        ("GET", "/models"): "openai_list_models",
        ("GET", "/models/"): "openai_list_models",
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


def test_local_gguf_entry_filters_non_gguf_and_recurses(tmp_path):
    # Transformers/safetensors folder: not a GGUF, must be rejected.
    tf = tmp_path / "tf-model"
    tf.mkdir()
    (tf / "config.json").write_text("{}")
    (tf / "model.safetensors").write_text("x")
    assert resolver._local_gguf_entry("tf", SimpleNamespace(path = str(tf))) is None

    bare = tmp_path / "x.gguf"
    bare.write_text("x")
    e = resolver._local_gguf_entry("x", SimpleNamespace(path = str(bare)))
    assert e is not None and e.variants == ()

    # Nested quant subdirs in HF-cache snapshots must still be detected.
    repo = tmp_path / "models--org--repo"
    (repo / "snapshots" / "abc" / "BF16").mkdir(parents = True)
    (repo / "snapshots" / "abc" / "BF16" / "model-BF16.gguf").write_text("x")
    e2 = resolver._local_gguf_entry("org/repo", SimpleNamespace(path = str(repo), source = "hf_cache"))
    assert e2 is not None and e2.variants


def test_local_gguf_entry_labels_a_quantless_filename_as_the_picker_does(tmp_path):
    # With no quant token, the picker and saved settings key this variant by the file stem.

    from utils.openai_auto_switch_settings import override_lookup_candidates

    repo = tmp_path / "models--org--KAT-GGUF"
    snapshot = repo / "snapshots" / "abc"
    snapshot.mkdir(parents = True)
    (snapshot / "KAT-Coder-V2.5-Dev-APEX-dynamic-v2.gguf").write_text("x")

    entry = resolver._local_gguf_entry("org/KAT-GGUF", SimpleNamespace(path = str(repo)))
    assert entry is not None
    assert entry.variants == ("KAT-Coder-V2.5-Dev-APEX-dynamic-v2",)
    assert "org/KAT-GGUF:KAT-Coder-V2.5-Dev-APEX-dynamic-v2" in override_lookup_candidates(
        entry.load_path, entry.loader_id, entry.variants[0]
    )


def test_local_gguf_entry_skips_a_quant_whose_blob_is_gone(tmp_path):
    evicted = tmp_path / "models--org--evicted" / "snapshots" / "abc"
    evicted.mkdir(parents = True)
    (evicted / "m-Q8_0.gguf").write_text("x")
    twins = tmp_path / "models--org--twins" / "snapshots" / "abc"
    twins.mkdir(parents = True)
    (twins / "z-BF16.gguf").write_text("x")
    try:
        (evicted / "m-Q4_K_M.gguf").symlink_to(tmp_path / "evicted-blob")
        (twins / "a-BF16.gguf").symlink_to(tmp_path / "evicted-blob")
    except (NotImplementedError, OSError):
        pytest.skip("symlinks unavailable (Windows without developer mode)")

    entry = resolver._local_gguf_entry(
        "org/evicted", SimpleNamespace(path = str(evicted.parent.parent))
    )
    assert entry is not None and entry.variants == ("Q8_0",)
    assert resolver._resolve_from_index("org/evicted", {"org/evicted": entry})[1] == "Q8_0"

    twin_entry = resolver._local_gguf_entry(
        "org/twins", SimpleNamespace(path = str(twins.parent.parent))
    )
    assert twin_entry is not None and twin_entry.variants == ("BF16",)


def _one_gguf_repo(tmp_path, name, filenames):
    """An HF-cache repo dir holding *filenames*, and the entry the index builds for it."""
    snapshot = tmp_path / f"models--org--{name}" / "snapshots" / "abc"
    snapshot.mkdir(parents = True)
    for index, filename in enumerate(filenames):
        (snapshot / filename).write_bytes(b"\0" * (4096 + index * 100))
    return resolver._local_gguf_entry(
        f"org/{name}", SimpleNamespace(path = str(snapshot.parent.parent))
    )


def test_a_quant_id_v1_models_used_to_publish_still_resolves(tmp_path):
    # A client pinned to "<repo>:BF16" must keep resolving after the label spelling changed.
    entry = _one_gguf_repo(tmp_path, "float-then-k", ["DeepSeek-R1-BF16-Q4_K_M.gguf"])
    assert entry is not None
    assert entry.variants == ("Q4_K_M",)
    assert entry.aliases == (("bf16", "Q4_K_M"),)

    index = {"org/float-then-k": entry}
    for requested in ("org/float-then-k:BF16", "org/float-then-k:bf16"):
        resolved = resolver._resolve_from_index(requested, index)
        assert resolved is not None, requested
        # the CURRENT label, so the override lookup reads the key the UI wrote
        assert resolved[1] == "Q4_K_M"
    assert resolver._resolve_from_index("org/float-then-k:Q8_0", index) is None


def test_a_legacy_quant_id_never_shadows_a_quant_that_is_really_there(tmp_path):
    # The real BF16 file keeps the name, so no alias is offered for the renamed one.
    entry = _one_gguf_repo(tmp_path, "collide", ["a-BF16-Q4_K_M.gguf", "b-BF16.gguf"])
    assert entry is not None
    assert set(entry.variants) == {"Q4_K_M", "BF16"}
    assert entry.aliases == ()

    index = {"org/collide": entry}
    assert resolver._resolve_from_index("org/collide:BF16", index)[1] == "BF16"
    assert resolver._resolve_from_index("org/collide:Q4_K_M", index)[1] == "Q4_K_M"


def test_an_ambiguous_legacy_quant_id_resolves_to_nothing(tmp_path):
    entry = _one_gguf_repo(tmp_path, "ambiguous", ["x-BF16-Q4_K_M.gguf", "y-BF16-Q6_K.gguf"])
    assert entry is not None
    assert set(entry.variants) == {"Q4_K_M", "Q6_K"}
    assert entry.aliases == ()
    assert resolver._resolve_from_index("org/ambiguous:BF16", {"org/ambiguous": entry}) is None


def test_a_quantless_pin_keeps_its_own_weights_instead_of_quietly_getting_the_quant(tmp_path):
    # Quantless labels alias too, else an old pin quietly falls through to a different quant.
    entry = _one_gguf_repo(
        tmp_path, "quantless-pin", ["Meta-Llama-3-8B.gguf", "Meta-Llama-3-8B-Q4_K_M.gguf"]
    )
    assert entry is not None
    assert entry.aliases == (("8b", "Meta-Llama-3-8B"),)
    resolved = resolver._resolve_from_index("org/quantless-pin:8b", {"org/quantless-pin": entry})
    assert resolved is not None and resolved[1] == "Meta-Llama-3-8B"
    # the bare id never reaches an alias, so preferred_quant still decides it
    bare = resolver._resolve_from_index("org/quantless-pin", {"org/quantless-pin": entry})
    assert bare[1] == entry.variants[0] == "Q4_K_M"


def test_a_bundle_repos_mirror_is_not_emptied_by_the_h3_filter(tmp_path):
    for name in ("MiniMax-H3-GGUF-mirror", "minimax-h3-gguf-i1", "MiniMax-H3-GGUF-BF16"):
        snapshot = tmp_path / f"models--unsloth--{name}" / "snapshots" / "abc"
        snapshot.mkdir(parents = True)
        (snapshot / "m-Q4_K_M.gguf").write_bytes(b"\0" * 4096)
        (snapshot / "m-Q8_0.gguf").write_bytes(b"\0" * 8192)
        entry = resolver._local_gguf_entry(
            f"unsloth/{name}", SimpleNamespace(path = str(snapshot.parent.parent))
        )
        assert entry is not None, f"{name} disappeared from the index"
        assert set(entry.variants) == {"Q4_K_M", "Q8_0"}, name


def test_a_broken_alias_helper_costs_the_aliases_and_not_the_model(tmp_path, monkeypatch):
    # _local_gguf_entry answers None on any raise, so an escaping shim would drop the repo.
    monkeypatch.setattr(
        resolver,
        "_legacy_variant_aliases",
        lambda variants: (_ for _ in ()).throw(RuntimeError("renamed private helper")),
    )
    entry = _one_gguf_repo(tmp_path, "shim-broke", ["DeepSeek-R1-BF16-Q4_K_M.gguf"])
    assert entry is None, "a raising helper must not be what deletes the entry"

    monkeypatch.undo()
    from utils.models import model_config

    def renamed_away(*_args, **_kwargs):
        raise AttributeError("_qualified_variant_name")

    monkeypatch.setattr(model_config, "_qualified_variant_name", renamed_away)
    entry = _one_gguf_repo(tmp_path, "shim-renamed", ["DeepSeek-R1-BF16-Q4_K_M.gguf"])
    assert entry is not None and entry.variants == ("Q4_K_M",)
    assert entry.aliases == ()


def test_a_repo_the_two_listers_agree_on_carries_no_aliases(tmp_path):
    # Aliases only exist when needed, and an id naming no local quant still misses.
    entry = _one_gguf_repo(tmp_path, "ordinary", ["m-Q4_K_M.gguf", "m-Q8_0.gguf"])
    assert entry is not None
    assert entry.aliases == ()
    assert resolver._resolve_from_index("org/ordinary:BF16", {"org/ordinary": entry}) is None


def test_local_gguf_entry_rejects_standalone_companions(tmp_path, monkeypatch):
    # A standalone mmproj projector is not servable, so the resolver must never return it.

    proj = tmp_path / "mmproj-F16.gguf"
    proj.write_text("x")
    assert resolver._local_gguf_entry("p", SimpleNamespace(path = str(proj))) is None
    assert resolver.local_servable_model(SimpleNamespace(id = str(proj), path = str(proj))) is None
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
    return resolver._LocalGgufEntry(loader_id, loader_id, tuple(variants))


def test_resolver_matches_and_splits_variant(monkeypatch):
    monkeypatch.setattr(
        resolver,
        "_build_index",
        lambda: {"unsloth/b-gguf": _entry("unsloth/B-GGUF", "UD-Q5_K_XL", "Q4_K_M")},
    )
    resolver._scan = (0.0, {})
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
    # Best-effort: any failure returns None; the hook has no guard of its own, so it lives here.
    def boom():
        raise RuntimeError("scan blew up")

    monkeypatch.setattr(resolver, "_build_index", boom)
    resolver._scan = (0.0, {})
    assert resolver.resolve_local_gguf("unsloth/B-GGUF") is None


def test_resolver_nonstring_model_is_failsafe():
    # Raw-body endpoints pass the model through, so a non-string must not crash on .strip().
    assert resolver.resolve_local_gguf(123) is None
    assert resolver.resolve_local_gguf({"a": 1}) is None
    assert resolver.resolve_local_gguf(None) is None


def test_describe_local_miss_separates_missing_repo_from_missing_quant(monkeypatch):
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
    # A local id containing a colon, like a Windows path, must match exactly, not be split.
    win = r"C:\models\foo.gguf"
    monkeypatch.setattr(resolver, "_build_index", lambda: {win.lower(): _entry(win)})
    resolver._scan = (0.0, {})
    assert resolver.resolve_local_gguf(win) == (win, None, win)


def test_setting_coercion():
    assert settings._coerce_bool("on") is True
    assert settings._coerce_bool("off") is False
    assert settings._coerce_bool("garbage") is None
    assert settings._coerce_int("5") == 5
    assert settings._coerce_int(-3) == 0
    assert settings._coerce_int("nope") is None


def test_idle_loop_does_not_unload_freshly_loaded_model(monkeypatch):
    # The load transition stamps activity, so a long-idle server does not unload a fresh model.

    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 1)
    monkeypatch.setattr(settings, "idle_unload_is_configured", lambda: 1 > 0)
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
    # With nothing in flight past the TTL, the GGUF is freed once and its identity stashed.

    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 0.005)
    monkeypatch.setattr(settings, "idle_unload_is_configured", lambda: 0.005 > 0)
    kw._inflight = 0
    kw._pending = 0
    kw._last_active = time.monotonic() - 3600
    kw._last_unloaded_model = None

    unloads = []
    backend = _FakeBackend("unsloth/Idle-GGUF", hf_variant = "Q4_K_M")

    def _unload():
        unloads.append(1)
        backend.is_loaded = False

    backend.unload_model = _unload
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)

    async def _drive():
        task = asyncio.create_task(kw.idle_unload_loop(poll_seconds = 0.02))
        # Use a generous wall-clock deadline, since a fixed short sleep flakes on loaded runners.
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            await asyncio.sleep(0.01)
            if unloads:
                break
        # Keep polling after the first unload so a loop that frees repeatedly is still caught.
        await asyncio.sleep(0.15)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_drive())
    assert unloads == [1]
    stash = kw.get_last_unloaded_model()
    assert stash is not None and stash[0] == "unsloth/Idle-GGUF" and stash[1] == "Q4_K_M"


def test_a_request_landing_during_the_pin_read_is_not_unloaded_out_from_under(monkeypatch):
    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 0.005)
    monkeypatch.setattr(settings, "idle_unload_is_configured", lambda: True)
    monkeypatch.setattr(kw, "_inflight", 0)
    monkeypatch.setattr(kw, "_pending", 0)
    monkeypatch.setattr(kw, "_last_active", time.monotonic() - 3600)
    monkeypatch.setattr(kw, "_last_unloaded_model", None)

    inside_the_unload_block = {"flag": False}
    landed = []

    def _keep_kv_marks_the_block():
        inside_the_unload_block["flag"] = True
        return False

    def _api_only_while_a_request_lands():
        if inside_the_unload_block["flag"]:
            inside_the_unload_block["flag"] = False
            kw._note_pending()
            landed.append(1)
        return False

    monkeypatch.setattr(settings, "get_auto_unload_keep_kv", _keep_kv_marks_the_block)
    monkeypatch.setattr(settings, "get_auto_unload_api_only", _api_only_while_a_request_lands)

    unloads = []
    backend = _FakeBackend("unsloth/Idle-GGUF", hf_variant = "Q4_K_M")
    backend.unload_model = lambda: unloads.append(1)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)

    async def _drive():
        task = asyncio.create_task(kw.idle_unload_loop(poll_seconds = 0.01))
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            await asyncio.sleep(0.01)
            if landed or unloads:
                break
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_drive())
    assert landed, "the tick never reached the guard's pin read, so the window was never hit"
    assert unloads == [], "the loop freed the model out from under a request on the gate"


def test_idle_loop_deletes_saved_kv_when_unload_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 0.005)
    monkeypatch.setattr(settings, "idle_unload_is_configured", lambda: 0.005 > 0)
    monkeypatch.setattr(settings, "get_auto_unload_keep_kv", lambda: True)
    _reset_keepwarm()

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

    saved = tmp_path / "resume-abc-slot0.bin"
    saved.write_bytes(b"kv")
    kw._kv_resume = {
        "identity": ("m", None, "m"),
        "dir": str(tmp_path),
        "slots": [{"id": 0, "filename": saved.name}],
    }
    monkeypatch.setattr(
        settings_route,
        "set_openai_auto_switch",
        lambda *a: (False, 300, True, False, False, 0, False),
    )
    monkeypatch.setattr(settings_route, "get_auto_unload_idle_seconds", lambda: 0)

    payload = settings_route.OpenAIAutoSwitchPayload(enabled = False)
    resp = settings_route.update_openai_auto_switch(payload, "tester")
    assert resp.idle_unload_active is False and resp.auto_unload_keep_kv is True
    assert kw._kv_resume is None and not saved.exists()


def test_residency_does_not_purge_saved_kv(monkeypatch, tmp_path):
    # Residency zeroes only the effective TTL; idle unload is still on, so the saved KV stays.

    saved = tmp_path / "resume-abc-slot0.bin"
    saved.write_bytes(b"kv")
    manifest = {
        "identity": ("m", None, "m"),
        "dir": str(tmp_path),
        "slots": [{"id": 0, "filename": saved.name}],
    }
    kw._kv_resume = manifest
    monkeypatch.setattr(
        settings_route,
        "set_openai_auto_switch",
        lambda *a: (True, 300, True, False, False, 0, False),
    )
    monkeypatch.setattr(settings_route, "get_auto_unload_idle_seconds", lambda: 0)
    monkeypatch.setattr(settings_route, "idle_unload_is_configured", lambda: True)

    payload = settings_route.OpenAIAutoSwitchPayload(enabled = True)
    resp = settings_route.update_openai_auto_switch(payload, "tester")
    try:
        assert resp.idle_unload_active is False
        assert kw._kv_resume is manifest and saved.exists()
    finally:
        kw._kv_resume = None


def test_idle_unload_is_configured_ignores_the_residency_veto(monkeypatch):
    # The reader the purge uses must report the user's setting, not the veto.
    import utils.model_memory_settings as mm
    import utils.openai_auto_switch_settings as settings

    monkeypatch.setattr(settings, "_stored_idle_seconds", lambda: 300)
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    monkeypatch.setattr(mm, "get_keep_resident", lambda: True)

    assert settings.get_auto_unload_idle_seconds() == 0
    assert settings.idle_unload_is_configured() is True

    monkeypatch.setattr(settings, "_stored_idle_seconds", lambda: 0)
    assert settings.idle_unload_is_configured() is False


def test_audio_generate_is_tracked_as_inference_path():
    # Direct GGUF TTS uses the llama backend, so keep-warm must count it as in-flight.

    assert _is_inference_path("/api/inference/audio/generate") is True
    assert _is_inference_path("/v1/chat/completions") is True
    assert _is_inference_path("/api/inference/models/list") is False


def test_idle_loop_does_not_unload_while_request_inflight(monkeypatch):
    # An in-flight request protects the model from unload even past the idle TTL.

    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 0.01)
    monkeypatch.setattr(settings, "idle_unload_is_configured", lambda: 0.01 > 0)
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


def test_auto_switch_applies_model_override(monkeypatch):
    backend, rec = _wired(
        monkeypatch, _FakeBackend(None), ("unsloth/B-GGUF", "Q4_K_M", "unsloth/B-GGUF")
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
    backend, rec = _wired(
        monkeypatch, _FakeBackend(None), ("unsloth/B-GGUF", "Q4_K_M", "unsloth/B-GGUF")
    )
    monkeypatch.setattr(
        settings, "get_model_override", lambda model_id: {"llama_extra_args": ["--flash-attn"]}
    )

    _run_hook("unsloth/B-GGUF")
    req = rec.calls[0]
    assert req.llama_extra_args == ["--flash-attn"]
    assert req.max_seq_length == 0


def test_auto_switch_applies_reasoning_budget_override(monkeypatch):
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
        lambda model_id: {
            "reasoning_budget": 2048,
            "reasoning_budget_message": "Conclude now",
        },
    )

    _run_hook("unsloth/B-GGUF")
    req = rec.calls[0]
    assert req.reasoning_budget == 2048
    assert req.reasoning_budget_message == "Conclude now"


def _mock_override_store(monkeypatch):
    """Back the override read + atomic-merge write with an in-memory dict."""
    store = {}

    def _merge_entry(
        key,
        entry_key,
        entry_value,
        *,
        fill_absent_fields = False,
        coupled_fields = (),
    ):
        current = dict(store.get(key) or {})
        if fill_absent_fields:
            if not entry_value:
                return current
            stored = current.get(entry_key)
            if isinstance(stored, dict):
                incoming = entry_value
                for group in coupled_fields:
                    if any(field in stored for field in group):
                        incoming = {k: v for k, v in incoming.items() if k not in group}
                current[entry_key] = {**incoming, **stored}
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
    return settings_route.update_openai_auto_switch_override(
        settings_route.ModelOverridePayload(model_id = model_id, **fields),
        "tester",
    )


def test_an_older_clients_cache_width_survives_the_override_route(monkeypatch):
    _mock_override_store(monkeypatch)
    _put("org/m", mlx_kv_bits = 8)
    assert settings.get_model_overrides()["org/m"]["mlx_kv_quant"] == "8"


def test_model_override_roundtrip(monkeypatch):
    _mock_override_store(monkeypatch)

    settings.set_model_override(
        "unsloth/B-GGUF", llama_extra_args = ["--n-gpu-layers", "20"], max_seq_length = 4096
    )
    assert settings.get_model_override("unsloth/B-GGUF") == {
        "llama_extra_args": ["--n-gpu-layers", "20"],
        "max_seq_length": 4096,
    }
    settings.set_model_override("unsloth/B-GGUF", llama_extra_args = [], max_seq_length = None)
    assert settings.get_model_override("unsloth/B-GGUF") == {}
    assert settings.get_model_overrides() == {}


def test_reasoning_budget_override_route_roundtrip(monkeypatch):
    _mock_override_store(monkeypatch)

    response = _put(
        "unsloth/B-GGUF",
        reasoning_budget = 2048,
        reasoning_budget_message = "Conclude now",
    )

    assert response.overrides["unsloth/B-GGUF"] == {
        "reasoning_budget": 2048,
        "reasoning_budget_message": "Conclude now",
    }
    assert settings.model_override_load_kwargs(
        response.overrides["unsloth/B-GGUF"], is_gguf = True
    ) == {
        "reasoning_budget": 2048,
        "reasoning_budget_message": "Conclude now",
    }


def test_reasoning_resets_strip_only_matching_carried_flags(monkeypatch):
    _mock_override_store(monkeypatch)
    extras = [
        "--reasoning-budget",
        "2048",
        "--reasoning-budget-message",
        "Conclude now",
        "--top-k",
        "40",
    ]
    for suffix in ("budget", "message", "both", "fill"):
        _put(f"unsloth/B-GGUF:{suffix}", llama_extra_args = extras)

    budget = _put("unsloth/B-GGUF:budget", reasoning_budget = -1).overrides["unsloth/B-GGUF:budget"]
    assert budget["llama_extra_args"] == [
        "--reasoning-budget-message",
        "Conclude now",
        "--top-k",
        "40",
    ]
    assert budget["reasoning_budget"] == -1

    message = _put("unsloth/B-GGUF:message", reasoning_budget_message = "").overrides[
        "unsloth/B-GGUF:message"
    ]
    assert message["llama_extra_args"] == ["--reasoning-budget", "2048", "--top-k", "40"]
    assert message["reasoning_budget_message"] == ""

    both = _put(
        "unsloth/B-GGUF:both",
        reasoning_budget = -1,
        reasoning_budget_message = "",
    ).overrides["unsloth/B-GGUF:both"]
    assert both["llama_extra_args"] == ["--top-k", "40"]
    assert settings.model_override_load_kwargs(both, is_gguf = True) == {
        "llama_extra_args": ["--top-k", "40"],
        "reasoning_budget": -1,
        "reasoning_budget_message": "",
    }

    filled = _put(
        "unsloth/B-GGUF:fill",
        reasoning_budget = -1,
        reasoning_budget_message = "",
        fill_absent_fields = True,
    ).overrides["unsloth/B-GGUF:fill"]
    assert filled["llama_extra_args"] == extras
    assert "reasoning_budget" not in filled
    assert "reasoning_budget_message" not in filled


def test_reasoning_reset_tombstone_blocks_bare_and_legacy_fallbacks(monkeypatch):
    _mock_override_store(monkeypatch)

    _put("unsloth/B-GGUF", llama_extra_args = ["--reasoning-budget", "2048"])
    qualified = _put("unsloth/B-GGUF:Q4_K_M", reasoning_budget = -1).overrides
    assert qualified["unsloth/B-GGUF:Q4_K_M"] == {"reasoning_budget": -1}
    assert qualified["unsloth/B-GGUF"]["llama_extra_args"] == ["--reasoning-budget", "2048"]
    assert settings.model_override_load_kwargs(
        settings.get_model_override("unsloth/B-GGUF:Q4_K_M"), is_gguf = True
    ) == {"reasoning_budget": -1}

    path = "/tmp/model-Q4_K_M.gguf"
    _put(f"{path}:Q4_K_M", llama_extra_args = ["--reasoning-budget-message", "Stop"])
    standalone = _put(path, reasoning_budget_message = "").overrides
    assert standalone[path] == {"reasoning_budget_message": ""}
    assert settings.model_override_load_kwargs(settings.get_model_override(path), is_gguf = True) == {
        "reasoning_budget_message": ""
    }


@pytest.mark.parametrize("message", ["😀" * 2_049, "bad\0message"])
def test_reasoning_budget_override_rejects_unsafe_argv(message):
    import pydantic
    import routes.settings as settings_route

    with pytest.raises(pydantic.ValidationError):
        settings_route.ModelOverridePayload(
            model_id = "unsloth/B-GGUF", reasoning_budget_message = message
        )
    assert settings.normalize_model_override({"reasoning_budget_message": message}) == {}

    from routes.chat_history import ChatPresetLoadConfig

    with pytest.raises(pydantic.ValidationError):
        ChatPresetLoadConfig(reasoningBudgetMessage = message)


@pytest.mark.parametrize("message", ["😀" * 2_049, "bad\0message"])
def test_unsafe_passthrough_is_rejected_before_backend_lookup(monkeypatch, message):
    from fastapi import HTTPException

    monkeypatch.setattr(
        inference_route.api_monitor, "record_lifecycle", lambda **kwargs: "load-event"
    )
    monkeypatch.setattr(inference_route.api_monitor, "fail_open", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        inference_route,
        "get_llama_cpp_backend",
        lambda: pytest.fail("backend lookup happened before argument rejection"),
    )
    request = LoadRequest(
        model_path = "unsloth/B-GGUF",
        llama_extra_args = ["--reasoning-budget-message", message],
    )

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(inference_route._load_model_impl(request, object(), "tester"))

    assert excinfo.value.status_code == 400


def test_override_route_rejects_managed_flag_and_removes(monkeypatch):
    _mock_override_store(monkeypatch)

    bad = settings_route.ModelOverridePayload(
        model_id = "unsloth/B-GGUF", llama_extra_args = ["--port", "1234"]
    )
    with pytest.raises(HTTPException) as excinfo:
        settings_route.update_openai_auto_switch_override(bad, "tester")
    assert excinfo.value.status_code == 400

    ok = settings_route.ModelOverridePayload(
        model_id = "unsloth/B-GGUF", llama_extra_args = ["--flash-attn"], max_seq_length = 4096
    )
    resp = settings_route.update_openai_auto_switch_override(ok, "tester")
    assert resp.overrides["unsloth/B-GGUF"]["max_seq_length"] == 4096
    assert "llama_extra_args" in resp.overrides["unsloth/B-GGUF"]

    empty = settings_route.ModelOverridePayload(model_id = "unsloth/B-GGUF")
    resp2 = settings_route.update_openai_auto_switch_override(empty, "tester")
    assert "unsloth/B-GGUF" not in resp2.overrides


def test_override_route_stores_llama_server_tuning(override_store):
    """Load mode, draft KV dtype, checkpoints and cache RAM survive the route.

    The picker has a control for each and mirrors all four, and
    ``model_override_load_kwargs`` already applies them off a stored row, so a
    payload that drops them leaves the setting reaching a picker load and nothing
    else -- and the panel, which hydrates from this row, reads it back as unset.
    """
    _put(
        "unsloth/B-GGUF",
        load_mode = "mmap+mlock",
        speculative_type = "dspark",
        spec_draft_cache_type = "q8_0",
        ctx_checkpoints = 0,
        cache_ram = -1,
    )
    stored = settings.get_model_override("unsloth/B-GGUF")
    assert stored["load_mode"] == "mmap+mlock"
    assert stored["spec_draft_cache_type"] == "q8_0"
    assert stored["ctx_checkpoints"] == 0
    assert stored["cache_ram"] == -1
    kwargs = settings.model_override_load_kwargs(stored, is_gguf = True)
    assert kwargs["load_mode"] == "mmap+mlock"
    assert kwargs["spec_draft_cache_type"] == "q8_0"
    assert kwargs["ctx_checkpoints"] == 0
    assert kwargs["cache_ram"] == -1


def test_override_route_keeps_a_row_holding_only_tuning(override_store):
    """One of the four on its own is a saved field, not an empty payload.

    ``is_removal`` counts what the payload carries, so a field it did not declare
    would read as "nothing set" and delete the model's row instead of writing it.
    """
    _put("unsloth/B-GGUF", cache_ram = 0)
    assert settings.get_model_override("unsloth/B-GGUF") == {"cache_ram": 0}


def test_model_override_rejects_zero_max_seq_length():
    # The setter drops falsy values, so 0 must be rejected at the boundary, not silently ignored.
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        settings_route.ModelOverridePayload(model_id = "x", max_seq_length = 0)
    assert settings_route.ModelOverridePayload(model_id = "x", max_seq_length = 1).max_seq_length == 1


def test_update_openai_auto_switch_writes_both_keys_in_one_transaction(monkeypatch):
    # Enabled and idle persist in one upsert so a write cannot leave one key stale.
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
    assert len(calls) == 1
    written = calls[0]
    assert written.get(OPENAI_AUTO_SWITCH_SETTING_KEY) is True
    assert written.get(AUTO_UNLOAD_IDLE_SETTING_KEY) == 120


def test_settings_report_idle_unload_active_when_env_backed(monkeypatch):
    # An env-driven idle TTL must report idle_unload_active so the UI shows it as active.

    monkeypatch.setattr(settings_route, "get_openai_auto_switch_enabled", lambda: False)
    monkeypatch.setattr(settings_route, "get_stored_auto_unload_idle_seconds", lambda: 600)
    monkeypatch.setattr(settings_route, "get_auto_unload_idle_seconds", lambda: 600)
    resp = settings_route.get_openai_auto_switch("tester")
    assert resp.enabled is False and resp.idle_unload_active is True
    monkeypatch.setattr(settings_route, "get_auto_unload_idle_seconds", lambda: 0)
    assert settings_route.get_openai_auto_switch("tester").idle_unload_active is False


def test_v1_models_retrieve_is_case_insensitive(monkeypatch):
    # The resolver lowercases its index, so a case-only mismatch on retrieve must still return 200.

    monkeypatch.setattr(inference_route, "_openai_model_objects", lambda: [])

    async def _catalog():
        return [
            {"id": "unsloth/A-GGUF", "object": "model", "created": 1, "owned_by": "local"},
            {"id": "unsloth/B-GGUF", "object": "model", "created": 1, "owned_by": "local"},
        ]

    monkeypatch.setattr(inference_route, "_openai_catalog_objects", _catalog)

    obj = asyncio.run(inference_route.openai_retrieve_model("unsloth/a-gguf", "tester"))
    assert obj["id"] == "unsloth/A-GGUF"
    with pytest.raises(HTTPException) as unknown:
        asyncio.run(inference_route.openai_retrieve_model("totally/unknown", "tester"))
    assert unknown.value.status_code == 404


def test_index_excludes_hidden_models(tmp_path, monkeypatch):
    # The validation probe and RAG embedding weights must never become auto-switch targets.

    normal = tmp_path / "normal-Q4_K_M.gguf"
    normal.write_bytes(b"x" * 32)
    probe = tmp_path / "stories260K.gguf"
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
    assert "org/normal-gguf" in index
    assert "ggml-org/models" not in index
    assert "unsloth/bge-small-en-v1.5-gguf" not in index
    assert str(local_default_embedder).lower() not in index
    resolver._scan = (0.0, {})
    assert resolver.resolve_local_gguf("ggml-org/models") is None


def test_idle_disabled_when_auto_switch_off(monkeypatch):
    # With auto-switch off the idle TTL reports 0, so nothing can unload the model.
    store = {settings.AUTO_UNLOAD_IDLE_SETTING_KEY: 60}
    monkeypatch.setattr(settings, "_cached_setting", lambda k, d = None: store.get(k, d))
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: False)
    assert settings.get_auto_unload_idle_seconds() == 0
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    assert settings.get_auto_unload_idle_seconds() == 60


def test_count_tokens_is_tracked_as_inference_path():
    # count_tokens uses the loaded tokenizer, so it must be tracked as in-flight.

    assert _is_inference_path("/v1/messages/count_tokens") is True
    assert _is_inference_path("/api/inference/messages/count_tokens") is True
    assert _is_inference_path("/api/inference/chat/count_tokens") is True
    assert _is_inference_path("/v1/messages") is True


def test_bare_id_tolerates_any_loaded_variant(monkeypatch):
    # A bare request for the loaded repo must not reload a different (larger) quant.
    backend, rec = _wired(
        monkeypatch,
        _FakeBackend("unsloth/B-GGUF", hf_variant = "Q4_K_M"),
        ("unsloth/B-GGUF", "Q8_0", "unsloth/B-GGUF"),
    )
    _run_hook("unsloth/B-GGUF")
    assert rec.calls == []
    _, rec2 = _wired(monkeypatch, backend, ("unsloth/B-GGUF", "Q8_0", "unsloth/B-GGUF"))
    _run_hook("unsloth/B-GGUF:Q8_0")
    assert len(rec2.calls) == 1


def test_responses_hook_runs_after_input_validation():
    # The auto-switch hook must run after input validation so an empty request loads nothing.

    src = inspect.getsource(inference_route.openai_responses)
    assert "No input provided" in src
    assert src.index("No input provided") < src.index("_maybe_auto_switch_model")


def test_responses_system_only_rejected_before_switch(monkeypatch):
    # Instructions-only input passes the empty check, so it must 400 before the switch.

    async def _boom(*a, **k):
        raise AssertionError("must not switch a system-only Responses request")

    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _boom)
    payload = ResponsesRequest(model = "org/B-GGUF", instructions = "be helpful", input = "")
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_responses(payload, object(), "tester"))
    assert exc.value.status_code == 400


def test_keepwarm_tracks_inflight_when_enabled_even_if_idle_zero(monkeypatch):
    # In-flight is counted whenever auto-switch is on, so enabling idle mid-stream is safe.

    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    kw._inflight = 0
    seen = {}

    async def app(scope, receive, send):
        seen["inflight"] = kw._inflight
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})

    async def drive():
        scope = {"type": "http", "path": "/v1/chat/completions", "method": "POST", "headers": []}
        await kw.LlamaKeepWarmMiddleware(app)(scope, receive, send)

    asyncio.run(drive())
    assert seen["inflight"] == 1
    assert kw._inflight == 0


def _bad_body_request():
    import json as _json
    class _BadReq:
        async def json(self):
            raise _json.JSONDecodeError("expecting value", "", 0)

    return _BadReq()


def test_completions_malformed_body_503_not_500_when_unloaded(monkeypatch):
    # Off, unloaded and unparseable must still 503 as before, not 500 from the early body read.

    backend, rec = _wired(monkeypatch, _FakeBackend(None), None, enabled = False)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_completions(_bad_body_request(), "tester"))
    assert exc.value.status_code == 503


def test_embeddings_malformed_body_503_not_500_when_unloaded(monkeypatch):
    backend, rec = _wired(monkeypatch, _FakeBackend(None), None, enabled = False)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_embeddings(_bad_body_request(), "tester"))
    assert exc.value.status_code == 503


def test_non_string_model_falls_through_without_error(monkeypatch):
    # A non-string model is treated as absent and must not raise, even with an idle stash.

    backend, rec = _wired(monkeypatch, _FakeBackend(None), None)
    monkeypatch.setattr(kw, "_last_unloaded_model", ("unsloth/A-GGUF", None))
    asyncio.run(inference_route._maybe_auto_switch_model(123, object(), "tester"))
    assert rec.calls == []


def test_anthropic_validates_max_tokens_before_auto_switch():
    # A missing max_tokens must 400 before the hook so an invalid request never loads a model.

    src = inspect.getsource(inference_route.anthropic_messages)
    assert "_maybe_auto_switch_model" in src
    assert src.index("max_tokens: field required") < src.index("_maybe_auto_switch_model")


def test_alias_reloads_model_freed_by_idle_unload_with_quant(monkeypatch):
    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = True, resolves_to = None, backend = backend, recorder = rec)
    monkeypatch.setattr(kw, "_inflight", 0)
    monkeypatch.setattr(kw, "_last_unloaded_model", ("unsloth/A-GGUF", "Q4_K_M"))
    _run_hook("gpt-4o-mini")
    assert len(rec.calls) == 1
    assert rec.calls[0].model_path == "unsloth/A-GGUF"
    assert rec.calls[0].gguf_variant == "Q4_K_M"


def test_alias_does_not_reload_when_model_already_loaded(monkeypatch):
    backend, rec = _wired(monkeypatch, _FakeBackend("unsloth/B-GGUF"), None)
    monkeypatch.setattr(kw, "_last_unloaded_model", ("unsloth/A-GGUF", None))
    _run_hook("gpt-4o-mini")
    assert rec.calls == []


def test_idle_loop_does_not_unload_while_request_pending(monkeypatch):
    # A pending request waiting on the unload gate must still block idle unload.

    monkeypatch.setattr(kw, "_inflight", 0)
    monkeypatch.setattr(kw, "_pending", 0)
    monkeypatch.setattr(kw, "_last_active", 0.0)
    kw._note_pending()
    try:
        assert kw._is_idle(1.0) is False
    finally:
        kw._note_unpending()
    assert kw._is_idle(1.0) is True


def test_keepwarm_tracks_inflight_even_when_auto_switch_off(monkeypatch):
    # A stream started while the feature is off must be counted so enabling it cannot unload it.

    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: False)
    monkeypatch.setattr(kw, "_inflight", 0)
    seen = {}

    async def app(scope, receive, send):
        seen["inflight"] = kw._inflight
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})

    async def drive():
        scope = {"type": "http", "path": "/v1/chat/completions", "method": "POST", "headers": []}
        await kw.LlamaKeepWarmMiddleware(app)(scope, receive, send)

    asyncio.run(drive())
    assert seen["inflight"] == 1
    assert kw._inflight == 0


def test_build_index_covers_legacy_default_lmstudio_and_custom_roots(monkeypatch, tmp_path):
    # _build_index must scan every root the picker lists, or shown models get the loaded one.
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
    for sub in ("active", "previous", "legacy", "default", "lmstudio", "custom", "custom/hub"):
        (tmp_path / sub).mkdir()

    resolver._build_index()

    hf = {p for k, p in scanned if k == "hf"}
    lm = {p for k, p in scanned if k == "lm"}
    assert str((tmp_path / "legacy").resolve()) in hf
    assert str((tmp_path / "default").resolve()) in hf
    assert str((tmp_path / "previous").resolve()) in hf
    assert str((tmp_path / "custom").resolve()) in hf
    assert str((tmp_path / "custom" / "hub").resolve()) in hf
    assert str((tmp_path / "lmstudio").resolve()) in lm


def test_build_index_covers_hermes_downloads(monkeypatch, tmp_path):
    # Hermes requests its staged GGUF by name, so the name must resolve to that exact file.
    from pathlib import Path
    import routes.models as models_route
    from hub.services.models import hermes as hermes_scan
    from utils import paths as upaths
    from utils import hf_cache_settings
    import storage.studio_db as studio_db

    monkeypatch.setattr(models_route, "_scan_models_dir", lambda d, limit = None: [])
    monkeypatch.setattr(models_route, "_scan_hf_cache", lambda d, **_: [])
    monkeypatch.setattr(models_route, "_scan_lmstudio_dir", lambda d: [])
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: tmp_path / "active")
    monkeypatch.setattr(models_route, "_is_hidden_model", lambda *a, **k: False)
    monkeypatch.setattr(hf_cache_settings, "known_hf_hub_caches", lambda: [])
    monkeypatch.setattr(upaths, "legacy_hf_cache_dir", lambda: tmp_path / "legacy")
    monkeypatch.setattr(upaths, "hf_default_cache_dir", lambda: tmp_path / "default")
    monkeypatch.setattr(upaths, "lmstudio_model_dirs", lambda: [])
    monkeypatch.setattr(studio_db, "list_scan_folders", lambda: [])

    hermes_models = tmp_path / ".hermes" / "models"
    hermes_models.mkdir(parents = True)
    weight = hermes_models / "Qwen3.8-27B-UD-Q4_K_M.gguf"
    weight.write_bytes(b"\x00" * 32)
    monkeypatch.setattr(upaths, "hermes_model_dirs", lambda: [hermes_models])

    scanned = []
    real_scan = hermes_scan.scan_hermes_dir
    monkeypatch.setattr(
        hermes_scan,
        "scan_hermes_dir",
        lambda d, **kw: scanned.append(str(Path(d).resolve())) or real_scan(d, **kw),
    )

    index = resolver._build_index()

    assert scanned == [str(hermes_models.resolve())]
    # The id Hermes configures as model.default is the stem; it must land on the staged file.
    entry = index.get("qwen3.8-27b-ud-q4_k_m")
    assert entry is not None
    assert entry.is_gguf
    assert Path(entry.load_path).resolve() == weight.resolve()


def _json_body_request(payload):
    class _Req:
        async def json(self):
            return payload

    return _Req()


def test_completions_list_body_is_400_not_500(monkeypatch):
    backend = _FakeBackend("unsloth/A-GGUF")
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
    backend, rec = _wired(monkeypatch, _FakeBackend("unsloth/A-GGUF"), None, enabled = False)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_embeddings(_json_body_request([]), "tester"))
    assert exc.value.status_code == 400


def test_middleware_ignores_non_post(monkeypatch):
    # CORS preflight (OPTIONS) on an inference path must not be tracked as in-flight.

    monkeypatch.setattr(kw, "_inflight", 0)
    seen = {}

    async def app(scope, receive, send):
        seen["inflight"] = kw._inflight
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def drive():
        scope = {"type": "http", "path": "/v1/chat/completions", "method": "OPTIONS", "headers": []}
        await kw.LlamaKeepWarmMiddleware(app)(scope, receive, send)

    asyncio.run(drive())
    assert seen["inflight"] == 0
    assert kw._inflight == 0


def test_auto_switch_waits_for_another_inference_to_finish(monkeypatch):
    backend, rec = _wired(
        monkeypatch, _FakeBackend("org/A-GGUF", hf_variant = "Q4_K_M"), ("/p/B", "Q8_0", "org/B-GGUF")
    )
    monkeypatch.setattr(kw, "_inflight", 2)
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


def test_auto_switch_swaps_when_only_caller_is_active(monkeypatch):
    # Only the caller is in flight: nothing else to protect, so the swap proceeds.

    backend, rec = _wired(monkeypatch, _FakeBackend("org/A-GGUF"), ("/p/B", None, "org/B-GGUF"))
    monkeypatch.setattr(kw, "_inflight", 1)
    monkeypatch.setattr(kw, "_pending", 0)
    _run_hook("org/B-GGUF")
    assert len(rec.calls) == 1
    assert rec.calls[0].model_path == "/p/B"


def test_idle_loop_resets_timer_for_same_repo_different_variant(monkeypatch):
    # A different quant of the same repo resets the idle timer, so it gets a full TTL of its own.

    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 0.05)
    monkeypatch.setattr(settings, "idle_unload_is_configured", lambda: 0.05 > 0)
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
        # Write under the loop's gate so the reset cannot race; acquire off-loop (`with` deadlocks).
        await asyncio.to_thread(kw._lifecycle_lock.acquire)
        try:
            kw._last_active = time.monotonic() - 60
            backend.hf_variant = "Q8_0"
        finally:
            kw._lifecycle_lock.release()
        await asyncio.sleep(0.03)
        assert unloads == []
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_drive())


def test_generate_stream_is_tracked_as_inference_path():
    assert _is_inference_path("/api/inference/generate/stream") is True
    assert _is_inference_path("/api/inference/audio/generate") is True
    assert _is_inference_path("/v1/responses") is True


def test_successful_manual_load_clears_last_unloaded_stash():
    kw._set_last_unloaded(("org/A-GGUF", "Q4_K_M"))
    assert kw.get_last_unloaded_model() == ("org/A-GGUF", "Q4_K_M")
    kw.note_model_loaded()
    assert kw.get_last_unloaded_model() is None


def test_hf_cache_entry_loads_from_local_snapshot_path(tmp_path):
    repo = tmp_path / "models--org--Repo"
    snap = repo / "snapshots" / "abc123"
    snap.mkdir(parents = True)
    (snap / "model-Q4_K_M.gguf").write_bytes(b"GGUF stub")

    entry = resolver._local_gguf_entry(
        "org/Repo", SimpleNamespace(id = "org/Repo", path = str(repo), source = "hf_cache")
    )
    assert entry is not None
    assert entry.loader_id == "org/Repo"
    assert "snapshots" in entry.load_path
    assert entry.load_path != "org/Repo"
    assert entry.variants


def test_hf_cache_entry_skips_newer_companion_only_snapshot(tmp_path):
    """A companion fetched at a newer revision must not hide older complete weights."""
    repo = tmp_path / "models--unsloth--Qwen3.8-Flash-Next-GGUF"
    old = repo / "snapshots" / "weights-revision"
    quant_dir = old / "UD-Q4_K_XL"
    quant_dir.mkdir(parents = True)
    for shard in range(1, 5):
        (quant_dir / f"Qwen3.8-Flash-Next-UD-Q4_K_XL-{shard:05d}-of-00004.gguf").write_bytes(
            b"GGUF stub"
        )

    newer = repo / "snapshots" / "companion-revision" / "MTP"
    newer.mkdir(parents = True)
    (newer / "mtp-Qwen3.8-Flash-Next-Q8_0.gguf").write_bytes(b"GGUF companion")
    os.utime(old, (1_000, 1_000))
    os.utime(newer.parent, (2_000, 2_000))

    # GGUF discovery needs a stronger rule than latest revision, since this one has no weights.
    assert Path(resolver._resolve_load_dir(repo)) == newer.parent.resolve()
    info = SimpleNamespace(id = "unsloth/Qwen3.8-Flash-Next-GGUF", path = str(repo), source = "hf_cache")
    entry = resolver._local_gguf_entry(
        "unsloth/Qwen3.8-Flash-Next-GGUF",
        info,
    )

    assert entry is not None
    assert Path(entry.load_path) == old
    assert entry.variants == ("UD-Q4_K_XL",)
    assert resolver.local_servable_model(info) == (True, ("UD-Q4_K_XL",))


@pytest.mark.parametrize("repo_level", [True, False])
@pytest.mark.parametrize(
    "filename, legacy, canonical",
    [
        ("DeepSeek-R1-BF16-Q4_K_M.gguf", "BF16", "Q4_K_M"),
        ("Meta-Llama-3-8B.gguf", "8B", "Meta-Llama-3-8B"),
    ],
)
def test_snapshot_selection_preserves_legacy_alias_companion_scope(
    tmp_path, repo_level, filename, legacy, canonical
):
    repo = tmp_path / "models--org--alias-scope"
    old = repo / "snapshots" / "weights"
    newer = repo / "snapshots" / "companions"
    old.mkdir(parents = True)
    newer.mkdir()
    (old / filename).write_bytes(b"GGUF weights")
    (newer / "mmproj-F16.gguf").write_bytes(b"GGUF companion")
    os.utime(old, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))
    info = SimpleNamespace(
        path = str(repo if repo_level else old),
        source = "hf_cache" if repo_level else "local",
    )
    entry = resolver._local_gguf_entry("org/alias-scope", info, exact_snapshot = not repo_level)
    assert entry is not None
    assert entry.load_path == str(old)
    assert entry.variants == (canonical,)
    index = {"org/alias-scope": entry}
    for label in (legacy, canonical):
        assert resolver._resolve_from_index(
            f"org/alias-scope:{label}", index, include_companion_scope = True
        ) == (str(old), canonical, "org/alias-scope", repo_level)
    assert resolver.local_gguf_companion_roots(
        entry.load_path, repo_level = entry.repo_level_companions
    ) == ((str(old), str(newer)) if repo_level else ())
    assert (
        resolver._local_gguf_entry(
            "org/alias-scope", SimpleNamespace(path = str(newer)), exact_snapshot = True
        )
        is None
    )


def test_hf_cache_entry_keeps_newer_companions_for_auto_switch(tmp_path, monkeypatch):
    """Selected weights stay pinned while request and load probes see companions."""
    from utils.models import model_config as model_config_module
    from utils.models.drafters import dflash as dflash_module

    ModelConfig = model_config_module.ModelConfig
    detect_mmproj_file = model_config_module.detect_mmproj_file

    repo, old, newer = _vision_gguf_cache_repo(tmp_path)
    mmproj = newer / "mmproj-vision-model-F16.gguf"
    mmproj.write_bytes(b"GGUF companion")
    mtp = newer / "mtp-vision-model-Q4_0.gguf"
    mtp.write_bytes(b"GGUF drafter")
    dspark = newer / "dspark-vision-model-Q8_0.gguf"
    dspark.write_bytes(b"GGUF drafter")
    dflash = newer / "dflash-kquant.gguf"
    dflash.write_bytes(b"GGUF drafter")
    os.utime(old, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))

    entry = resolver._local_gguf_entry(
        "org/Vision-GGUF",
        SimpleNamespace(id = "org/Vision-GGUF", path = str(repo), source = "hf_cache"),
    )

    assert entry is not None
    assert Path(entry.load_path) == old
    assert resolver.local_gguf_companion_roots(entry.load_path) == ()
    assert resolver._resolve_from_index(
        "Vision-GGUF",
        {"vision-gguf": entry},
        include_companion_scope = True,
    ) == (entry.load_path, entry.variants[0], "org/Vision-GGUF", True)
    roots = resolver.local_gguf_companion_roots(entry.load_path, repo_level = True)
    assert tuple(map(Path, roots)) == (old, newer)
    assert detect_mmproj_file(str(old / "vision-model-Q4_K_M.gguf"), search_root = str(newer)) is None
    monkeypatch.setattr(
        dflash_module,
        "read_gguf_architecture",
        lambda path: "dflash" if Path(path).name == dflash.name else None,
    )
    assert inference_route._target_accepts_request_input(
        entry.load_path,
        True,
        True,
        False,
        entry.variants[0],
        True,
        False,
        roots,
    )
    config = ModelConfig.from_identifier(
        entry.load_path,
        gguf_variant = entry.variants[0],
        gguf_companion_roots = roots,
    )
    assert config is not None
    assert config.gguf_mmproj_file == str(mmproj.resolve())
    assert config.gguf_mtp_file == str(mtp.resolve())
    assert config.gguf_dspark_file == str(dspark.resolve())
    assert config.gguf_dflash_file == str(dflash.resolve())


def test_hf_cache_entry_skips_unreadable_sibling_for_mmproj(tmp_path, monkeypatch):
    from utils.models import model_config as model_config_module

    repo, old, newer = _vision_gguf_cache_repo(tmp_path)
    (newer / "mmproj-vision-model-F16.gguf").write_bytes(b"GGUF companion")
    unreadable = repo / "snapshots" / "unreadable-revision"
    unreadable.mkdir(parents = True)
    os.utime(old, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))
    os.utime(unreadable, (3_000, 3_000))

    entry = resolver._local_gguf_entry(
        "org/Vision-GGUF",
        SimpleNamespace(id = "org/Vision-GGUF", path = str(repo), source = "hf_cache"),
    )
    assert entry is not None
    roots = resolver.local_gguf_companion_roots(entry.load_path, repo_level = True)
    assert tuple(map(Path, roots)) == (old, unreadable, newer)

    original_iter = model_config_module._iter_gguf_files

    def _raise_for_unreadable(directory, recursive = False):
        if Path(directory).resolve() == unreadable.resolve():
            raise PermissionError("simulated unreadable snapshot")
        return original_iter(directory, recursive)

    monkeypatch.setattr(model_config_module, "_iter_gguf_files", _raise_for_unreadable)
    assert model_config_module.is_vision_model(
        entry.load_path,
        gguf_variant = entry.variants[0],
        gguf_companion_roots = roots,
    )


def test_image_preflight_uses_the_projector_selected_for_load(tmp_path, monkeypatch):
    from utils.models import model_config as config_module

    selected = tmp_path / "selected"
    sibling = tmp_path / "sibling"
    selected.mkdir()
    sibling.mkdir()
    (selected / "model-Q4_K_M.gguf").write_bytes(b"GGUF weights")
    audio = selected / "mmproj-model-F16.gguf"
    image = sibling / "mmproj-model-F16.gguf"
    audio.write_bytes(b"GGUF audio")
    image.write_bytes(b"GGUF image")
    roots = (str(selected), str(sibling))
    monkeypatch.setattr(
        config_module, "mmproj_accepts_image", lambda path: str(path) == str(image.resolve())
    )
    config = config_module.ModelConfig.from_identifier(
        str(selected), gguf_variant = "Q4_K_M", gguf_companion_roots = roots
    )
    assert config.gguf_mmproj_file == str(audio.resolve())
    assert not config_module.is_vision_model(
        str(selected),
        gguf_variant = "Q4_K_M",
        gguf_companion_roots = roots,
        require_image = True,
    )


def test_companion_roots_skip_only_the_unreadable_sibling(tmp_path, monkeypatch):
    repo = tmp_path / "models--org--Vision-GGUF"
    selected = repo / "snapshots" / "weights-revision"
    readable = repo / "snapshots" / "companion-revision"
    unreadable = repo / "snapshots" / "unreadable-revision"
    for path in (selected, readable, unreadable):
        path.mkdir(parents = True)

    original_is_dir = Path.is_dir

    def _is_dir(path):
        if path == unreadable:
            raise PermissionError("simulated unreadable snapshot")
        return original_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", _is_dir)

    assert tuple(
        map(Path, resolver.local_gguf_companion_roots(str(selected), repo_level = True))
    ) == (selected, readable)


def test_snapshot_selector_skips_only_the_unreadable_child(tmp_path, monkeypatch):
    from hub.utils.gguf import select_gguf_cache_snapshot_for_repo_dir

    repo = tmp_path / "models--org--Vision-GGUF"
    readable = repo / "snapshots" / "weights-revision"
    unreadable = repo / "snapshots" / "unreadable-revision"
    readable.mkdir(parents = True)
    unreadable.mkdir()
    (readable / "vision-model-Q4_K_M.gguf").write_bytes(b"GGUF weights")

    original_is_dir = Path.is_dir

    def _is_dir(path):
        if path == unreadable:
            raise PermissionError("simulated unreadable snapshot")
        return original_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", _is_dir)

    selected = select_gguf_cache_snapshot_for_repo_dir(repo)
    assert selected is not None
    assert selected[3] == readable


def test_path_companion_roots_widen_only_the_snapshot_the_repo_would_hand_out(tmp_path):
    """#10599: loading by path widens to the sibling revisions of the SAME repo dir,
    and only when the path is the one a repo-level selection resolves to."""
    repo, old, newer = _vision_gguf_cache_repo(tmp_path)
    (newer / "mmproj-vision-model-F16.gguf").write_bytes(b"GGUF companion")
    os.utime(old, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))

    assert tuple(map(Path, resolver.local_path_gguf_companion_roots(str(old)))) == (old, newer)
    # A revision the selector would not hand out is pinned, so it keeps its own root only.
    assert resolver.local_path_gguf_companion_roots(str(newer)) == ()

    pinned = repo / "snapshots" / "newer-weights-revision"
    pinned.mkdir(parents = True)
    (pinned / "vision-model-Q4_K_M.gguf").write_bytes(b"GGUF weights")
    os.utime(pinned, (3_000, 3_000))
    assert resolver.local_path_gguf_companion_roots(str(old)) == ()
    assert tuple(map(Path, resolver.local_path_gguf_companion_roots(str(pinned)))) == (
        pinned,
        newer,
        old,
    )


@pytest.mark.parametrize("kind", ["plain_dir", "repo_dir", "missing", "file", "repo_id"])
def test_path_companion_roots_refuse_anything_outside_an_hf_cache_snapshot(tmp_path, kind):
    """The widening reaches sibling revisions of one ``models--`` dir and nothing else."""
    repo, old, _newer = _vision_gguf_cache_repo(tmp_path)
    candidates = {
        "plain_dir": tmp_path / "loose-model-dir",
        "repo_dir": repo,
        "missing": old.parent / "absent-revision",
        "file": old / "vision-model-Q4_K_M.gguf",
        "repo_id": Path("org/Vision-GGUF"),
    }
    target = candidates[kind]
    if kind == "plain_dir":
        target.mkdir()
        (target / "vision-model-Q4_K_M.gguf").write_bytes(b"GGUF weights")
    assert resolver.local_path_gguf_companion_roots(str(target)) == ()


def test_disjoint_companion_roots_preserve_selected_snapshot_ancestor_walk(tmp_path):
    """A selected snapshot still walks intermediate parents before sibling revisions."""
    from utils.models.model_config import detect_mmproj_file

    snapshot = tmp_path / "models--org--Vision-GGUF" / "snapshots" / "weights-revision"
    checkpoint = snapshot / "checkpoint"
    quant = checkpoint / "Q4_K_M"
    quant.mkdir(parents = True)
    weights = quant / "vision-model-Q4_K_M.gguf"
    weights.write_bytes(b"GGUF weights")
    mmproj = checkpoint / "mmproj-vision-model-F16.gguf"
    mmproj.write_bytes(b"GGUF companion")

    assert detect_mmproj_file(
        str(weights),
        search_root = str(snapshot),
        allow_disjoint_search_root = True,
    ) == str(mmproj.resolve())


def test_auto_switch_carries_hf_cache_companion_roots_into_load(tmp_path, monkeypatch):
    repo, old, newer = _vision_gguf_cache_repo(tmp_path)
    (newer / "mmproj-vision-model-F16.gguf").write_bytes(b"GGUF companion")
    os.utime(old, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))

    backend, recorder = _wired(
        monkeypatch,
        _FakeBackend("org/Other-GGUF", "Q4_K_M"),
        (str(old), "Q4_K_M", "org/Vision-GGUF"),
    )

    asyncio.run(
        inference_route._maybe_auto_switch_model(
            "org/Vision-GGUF",
            object(),
            "tester",
            require_vision = True,
        )
    )

    assert len(recorder.calls) == 1
    assert tuple(map(Path, recorder.calls[0]._gguf_companion_roots)) == (old, newer)
    assert "_gguf_companion_roots" not in recorder.calls[0].model_dump()
    assert (
        LoadRequest(
            model_path = str(old),
            _gguf_companion_roots = (str(tmp_path / "untrusted"),),
        )._gguf_companion_roots
        == ()
    )


def test_auto_switch_display_alias_keeps_repo_level_companion_scope(tmp_path, monkeypatch):
    repo, old, newer = _vision_gguf_cache_repo(tmp_path)
    (newer / "mmproj-vision-model-F16.gguf").write_bytes(b"GGUF companion")

    backend, recorder = _wired(
        monkeypatch,
        _FakeBackend("org/Other-GGUF", "Q4_K_M"),
        (str(old), "Q4_K_M", "org/Vision-GGUF", True),
    )

    asyncio.run(
        inference_route._maybe_auto_switch_model(
            "Vision-GGUF",
            object(),
            "tester",
        )
    )

    assert len(recorder.calls) == 1
    assert tuple(map(Path, recorder.calls[0]._gguf_companion_roots)) == (old, newer)


@pytest.mark.parametrize("advertised", [False, True])
def test_repo_level_request_reloads_resident_snapshot_without_companion_roots(
    tmp_path, monkeypatch, advertised
):
    repo, old, newer = _vision_gguf_cache_repo(tmp_path)
    (newer / "mmproj-vision-model-F16.gguf").write_bytes(b"GGUF companion")

    backend = _FakeBackend(str(old), "Q4_K_M")
    if advertised:
        backend._openai_advertised_id = "org/Vision-GGUF"
        backend._openai_gguf_companion_roots = (str(old),)
    _, recorder = _wired(monkeypatch, backend, (str(old), "Q4_K_M", "org/Vision-GGUF", True))

    asyncio.run(
        inference_route._maybe_auto_switch_model(
            "org/Vision-GGUF",
            object(),
            "tester",
        )
    )

    assert len(recorder.calls) == 1
    assert tuple(map(Path, recorder.calls[0]._gguf_companion_roots)) == (old, newer)
    assert tuple(map(Path, backend._openai_gguf_companion_roots)) == (old, newer)


@pytest.mark.parametrize("precreated", [False, True])
def test_resident_repo_reloads_when_companion_finishes_in_existing_snapshot(
    tmp_path, monkeypatch, precreated
):
    repo = tmp_path / "models--org--Vision-GGUF"
    selected = repo / "snapshots" / "weights"
    companion = repo / "snapshots" / "companion"
    selected.mkdir(parents = True)
    companion.mkdir()
    if precreated:
        (companion / "mmproj-vision-model-F16.gguf").write_bytes(b"")
    (selected / "vision-model-Q4_K_M.gguf").write_bytes(b"GGUF weights")
    backend, recorder = _wired(
        monkeypatch,
        _FakeBackend("org/Other-GGUF", "Q4_K_M"),
        (str(selected), "Q4_K_M", "org/Vision-GGUF", True),
    )

    async def run():
        await inference_route._maybe_auto_switch_model("org/Vision-GGUF", object(), "tester")
        assert len(recorder.calls) == 1
        await inference_route._maybe_auto_switch_model("org/Vision-GGUF", object(), "tester")
        assert len(recorder.calls) == 1
        (companion / "mmproj-vision-model-F16.gguf").write_bytes(b"GGUF companion")
        await inference_route._maybe_auto_switch_model("org/Vision-GGUF", object(), "tester")
        assert len(recorder.calls) == 2

    asyncio.run(run())


def test_auto_switch_exact_revision_does_not_widen_companion_roots(tmp_path, monkeypatch):
    repo, old, newer = _vision_gguf_cache_repo(tmp_path)
    (newer / "mmproj-other-model-F16.gguf").write_bytes(b"GGUF companion")

    backend, recorder = _wired(
        monkeypatch,
        _FakeBackend("org/Other-GGUF", "Q4_K_M"),
        (str(old), "Q4_K_M", "org/Vision-GGUF"),
    )

    asyncio.run(
        inference_route._maybe_auto_switch_model(
            old.name,
            object(),
            "tester",
        )
    )

    assert len(recorder.calls) == 1
    assert recorder.calls[0]._gguf_companion_roots == ()
    assert resolver.local_gguf_companion_roots(str(old), repo_level = False) == ()
    assert tuple(map(Path, resolver.local_gguf_companion_roots(str(old), repo_level = True))) == (
        old,
        newer,
    )


def test_idle_stash_reload_carries_hf_cache_companion_roots(tmp_path, monkeypatch):
    repo, old, newer = _vision_gguf_cache_repo(tmp_path)
    (newer / "mmproj-vision-model-F16.gguf").write_bytes(b"GGUF companion")
    os.utime(old, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))

    backend = _FakeBackend(None)
    recorder = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = True, resolves_to = None, backend = backend, recorder = recorder)
    monkeypatch.setattr(kw, "_inflight", 0)
    monkeypatch.setattr(
        kw,
        "_last_unloaded_model",
        (str(old), "Q4_K_M", "org/Vision-GGUF", (str(old), str(newer))),
    )

    asyncio.run(
        inference_route._maybe_auto_switch_model(
            inference_route._RELOAD_ONLY_MODEL,
            object(),
            "tester",
            require_vision = True,
        )
    )

    assert len(recorder.calls) == 1
    assert tuple(map(Path, recorder.calls[0]._gguf_companion_roots)) == (old, newer)


def test_loaded_identity_stashes_only_roots_used_by_the_load():
    backend = _FakeBackend("cache/snapshots/weights-revision", "Q4_K_M")
    backend._openai_advertised_id = "org/Vision-GGUF"
    backend._openai_gguf_companion_roots = ("weights-revision", "companion-revision")

    assert kw._loaded_identity(backend) == (
        "cache/snapshots/weights-revision",
        "Q4_K_M",
        "org/Vision-GGUF",
        ("weights-revision", "companion-revision"),
    )
    backend._openai_gguf_companion_roots = ()
    assert kw._loaded_identity(backend) == (
        "cache/snapshots/weights-revision",
        "Q4_K_M",
        "org/Vision-GGUF",
    )


def test_idle_stash_reload_of_manual_snapshot_does_not_add_sibling_roots(tmp_path, monkeypatch):
    repo, old, newer = _vision_gguf_cache_repo(tmp_path)
    (newer / "mmproj-other-model-F16.gguf").write_bytes(b"GGUF companion")

    backend = _FakeBackend(None)
    recorder = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = True, resolves_to = None, backend = backend, recorder = recorder)
    monkeypatch.setattr(kw, "_inflight", 0)
    monkeypatch.setattr(
        kw,
        "_last_unloaded_model",
        (str(old), "Q4_K_M", "org/Vision-GGUF"),
    )

    asyncio.run(
        inference_route._maybe_auto_switch_model(
            inference_route._RELOAD_ONLY_MODEL,
            object(),
            "tester",
        )
    )

    assert len(recorder.calls) == 1
    assert recorder.calls[0]._gguf_companion_roots == ()


def test_companion_root_scan_does_not_block_the_event_loop(tmp_path, monkeypatch):
    old = tmp_path / "models--org--Vision-GGUF" / "snapshots" / "weights-revision"
    old.mkdir(parents = True)
    (old / "vision-model-Q4_K_M.gguf").write_bytes(b"GGUF weights")

    backend, recorder = _wired(
        monkeypatch,
        _FakeBackend("org/Other-GGUF", "Q4_K_M"),
        (str(old), "Q4_K_M", "org/Vision-GGUF"),
    )
    entered = threading.Event()
    release = threading.Event()
    scan_thread: dict[str, int] = {}

    def _slow_companion_scan(_load_path, *, repo_level = False):
        assert repo_level is True
        scan_thread["ident"] = threading.get_ident()
        entered.set()
        release.wait(5.0)
        return ()

    monkeypatch.setattr(resolver, "local_gguf_companion_roots", _slow_companion_scan)

    async def _drive():
        # Captured inside the coroutine so it is the loop's own thread, not the caller's.
        loop_thread = threading.get_ident()
        task = asyncio.create_task(
            inference_route._maybe_auto_switch_model(
                "org/Vision-GGUF",
                object(),
                "tester",
            )
        )
        assert await asyncio.to_thread(entered.wait, 10.0), "the companion scan never started"
        release.set()
        await task
        return loop_thread

    loop_thread = asyncio.run(_drive())
    assert len(recorder.calls) == 1

    # Assert which thread ran the scan; elapsed time was a flaky proxy on busy runners.
    assert scan_thread.get("ident") is not None, "the companion scan never ran"
    assert scan_thread["ident"] != loop_thread, (
        "the companion scan ran on the event loop thread, so it blocks every other request "
        "for as long as it takes to walk the cache"
    )


def test_inactive_hf_cache_entry_skips_newer_companion_only_snapshot(tmp_path):
    """Inactive cache rows point at a snapshot but still select complete weights."""
    repo = tmp_path / "models--org--Repo"
    old = repo / "snapshots" / "weights-revision"
    old.mkdir(parents = True)
    (old / "model-Q4_K_M.gguf").write_bytes(b"GGUF stub")

    newer = repo / "snapshots" / "companion-revision" / "MTP"
    newer.mkdir(parents = True)
    (newer / "mtp-model-Q8_0.gguf").write_bytes(b"GGUF companion")
    os.utime(old, (1_000, 1_000))
    os.utime(newer.parent, (2_000, 2_000))

    [info] = models_route._scan_hf_cache(
        tmp_path,
        active_cache = False,
        classify_format = False,
    )
    assert Path(info.path) == newer.parent.resolve()

    entry = resolver._local_gguf_entry("org/Repo", info)

    assert entry is not None
    assert Path(entry.load_path) == old
    assert entry.variants == ("Q4_K_M",)


def test_hf_cache_entry_stays_within_the_scanned_case_variant(tmp_path):
    """A cache row must load from its exact repo directory, not a case-folded peer."""
    scanned_repo = tmp_path / "models--Org--Repo"
    scanned_snapshot = scanned_repo / "snapshots" / "scanned-revision"
    scanned_snapshot.mkdir(parents = True)
    (scanned_snapshot / "model-Q4_K_M.gguf").write_bytes(b"GGUF stub")

    other_repo = tmp_path / "models--org--repo"
    other_snapshot = other_repo / "snapshots" / "other-revision"
    other_snapshot.mkdir(parents = True)
    (other_snapshot / "model-Q8_0.gguf").write_bytes(b"GGUF stub")
    os.utime(scanned_snapshot, (1_000, 1_000))
    os.utime(other_snapshot, (2_000, 2_000))

    entry = resolver._local_gguf_entry(
        "Org/Repo",
        SimpleNamespace(id = "Org/Repo", path = str(scanned_repo), source = "hf_cache"),
    )

    assert entry is not None
    assert Path(entry.load_path) == scanned_snapshot
    assert entry.variants == ("Q4_K_M",)


def test_hf_cache_entry_excludes_torn_quant_from_selected_snapshot(tmp_path):
    """Only complete quants from the selected snapshot may be advertised as loadable."""
    repo = tmp_path / "models--org--Repo"
    snapshot = repo / "snapshots" / "revision"
    snapshot.mkdir(parents = True)
    (snapshot / "model-Q4_K_M.gguf").write_bytes(b"GGUF stub")
    (snapshot / "model-Q8_0-00001-of-00002.gguf").write_bytes(b"GGUF stub")

    entry = resolver._local_gguf_entry(
        "org/Repo",
        SimpleNamespace(id = "org/Repo", path = str(repo), source = "hf_cache"),
    )

    assert entry is not None
    assert entry.variants == ("Q4_K_M",)


def test_hf_cache_entry_keeps_partial_only_fallback(tmp_path):
    """A repo with no complete quant keeps the pre-existing fallback listing."""
    repo = tmp_path / "models--org--Repo"
    snapshot = repo / "snapshots" / "revision"
    snapshot.mkdir(parents = True)
    (snapshot / "model-Q4_K_M-00001-of-00002.gguf").write_bytes(b"GGUF stub")

    entry = resolver._local_gguf_entry(
        "org/Repo",
        SimpleNamespace(id = "org/Repo", path = str(repo), source = "hf_cache"),
    )

    assert entry is not None
    assert entry.variants == ("Q4_K_M",)


def test_selected_snapshot_preserves_local_variant_labels(tmp_path):
    """Selected snapshots use picker identities and still accept legacy API pins."""
    repo = tmp_path / "models--org--Repo"
    snapshot = repo / "snapshots" / "revision"
    snapshot.mkdir(parents = True)
    (snapshot / "model-small.gguf").write_bytes(b"small")
    (snapshot / "model-large.gguf").write_bytes(b"larger")

    entry = resolver._local_gguf_entry(
        "org/Repo",
        SimpleNamespace(id = "org/Repo", path = str(repo), source = "hf_cache"),
    )

    assert entry is not None
    assert set(entry.variants) == {"model-small", "model-large"}
    for legacy in ("small", "large"):
        assert resolver._resolve_from_index(
            f"org/Repo:{legacy}", {"org/repo": entry}, include_companion_scope = True
        ) == (str(snapshot), f"model-{legacy}", "org/Repo", True)


def test_model_dir_with_snapshots_subdir_keeps_root_gguf(tmp_path):
    """A regular model dir is not an HF cache just because snapshots/ exists."""
    nested = tmp_path / "snapshots" / "revision"
    nested.mkdir(parents = True)
    (nested / "mmproj-model-F16.gguf").write_bytes(b"GGUF companion")
    (tmp_path / "model-Q4_K_M.gguf").write_bytes(b"GGUF stub")

    entry = resolver._local_gguf_entry(
        "custom/model",
        SimpleNamespace(id = "custom/model", path = str(tmp_path), source = "models_dir"),
    )

    assert entry is not None
    assert entry.load_path == str(tmp_path)
    assert entry.variants == ("Q4_K_M",)


def test_hf_cache_entries_do_not_rescan_the_cache_root(monkeypatch, tmp_path):
    """Each scanned row already owns an exact repo directory under the cache root."""
    infos = []
    for index in range(3):
        repo = tmp_path / f"models--org--Repo-{index}"
        snapshot = repo / "snapshots" / "revision"
        snapshot.mkdir(parents = True)
        (snapshot / "model-Q4_K_M.gguf").write_bytes(b"GGUF stub")
        infos.append(SimpleNamespace(id = f"org/Repo-{index}", path = str(repo), source = "hf_cache"))

    original_iterdir = Path.iterdir
    root_scans = 0

    def counting_iterdir(path):
        nonlocal root_scans
        if path == tmp_path:
            root_scans += 1
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", counting_iterdir)
    assert all(resolver._local_gguf_entry(info.id, info) is not None for info in infos)
    assert root_scans == 0


def _revision_pair(root, complete: bool):
    """Two revisions of one cache repo; the newer one is optionally half-downloaded."""
    snaps = root / "models--org--Repo" / "snapshots"
    old, new = snaps / "rev-old", snaps / "rev-new"
    for path in (old, new):
        path.mkdir(parents = True)
    (old / "model-Q8_0.gguf").write_bytes(b"GGUF stub")
    name = "model-Q4_K_M.gguf" if complete else "model-Q4_K_M-00001-of-00003.gguf"
    (new / name).write_bytes(b"GGUF stub")
    os.utime(old, (1_000, 1_000))
    os.utime(new, (2_000, 2_000))
    return old, new


def test_sibling_revision_resolves_to_its_own_weights(tmp_path):
    # A pinned old revision must resolve to its own directory, not be redirected to the newest.
    old, new = _revision_pair(tmp_path, complete = True)

    found = dict(resolver._sibling_revision_entries(str(new), "org/Repo"))

    assert "rev-old" in found
    assert found["rev-old"].load_path == str(old)


def test_incomplete_sibling_revision_is_not_indexed(tmp_path):
    # A half-downloaded revision cannot load, so naming it must not resolve to it.
    old, _new = _revision_pair(tmp_path, complete = False)
    found = dict(resolver._sibling_revision_entries(str(old), "org/Repo"))

    assert "rev-new" not in found


def test_sibling_revisions_ignore_a_scan_folder_named_snapshots(tmp_path):
    # A user folder named "snapshots" holds unrelated models, not revisions of one repo.
    snaps = tmp_path / "snapshots"
    for name in ("model-a", "model-b"):
        (snaps / name).mkdir(parents = True)
        (snaps / name / "model-Q4_K_M.gguf").write_bytes(b"GGUF stub")

    found = dict(resolver._sibling_revision_entries(str(snaps / "model-a"), "model-a"))

    assert found == {}


def test_sibling_revisions_skip_plain_repo_ids():
    assert dict(resolver._sibling_revision_entries("org/Repo-GGUF", "org/Repo-GGUF")) == {}


def test_already_loaded_by_repo_id_is_not_reswapped(monkeypatch):
    # The resolver returns the load path, so a request by repo id must count as already serving.

    backend, rec = _wired(
        monkeypatch,
        _FakeBackend("org/Repo-GGUF", hf_variant = "Q4_K_M"),
        ("/cache/models--org--Repo-GGUF/snapshots/abc", "Q4_K_M", "org/Repo-GGUF"),
    )
    monkeypatch.setattr(kw, "_inflight", 2)
    monkeypatch.setattr(kw, "_pending", 0)
    _run_hook("org/Repo-GGUF:Q4_K_M")
    _run_hook("org/Repo-GGUF")
    assert rec.calls == []


def test_auto_switch_advertises_repo_id_after_load(monkeypatch):
    backend, rec = _wired(
        monkeypatch, _FakeBackend("org/A-GGUF"), ("/p/B-snapshot", "Q8_0", "org/B-GGUF")
    )
    _run_hook("org/B-GGUF:Q8_0")
    assert rec.calls[0].model_path == "/p/B-snapshot"
    assert backend._openai_advertised_id == "org/B-GGUF"


def test_already_serving_by_path_records_advertised_alias(monkeypatch):
    # An alias resolving to the loaded path must be recorded as the advertised id, not the basename.
    path = "/cache/models--org--Repo-GGUF/snapshots/abc"
    backend = _FakeBackend(path, hf_variant = "Q4_K_M")
    rec = _LoadRecorder(backend)
    _wire_on(
        monkeypatch,
        resolves_to = (path, "Q4_K_M", "org/Repo-GGUF"),
        backend = backend,
        recorder = rec,
    )
    assert backend._openai_advertised_id is None
    _run_hook("org/Repo-GGUF:Q4_K_M")
    assert rec.calls == []
    assert backend._openai_advertised_id == "org/Repo-GGUF"


def test_already_serving_requested_by_path_records_advertised_alias(monkeypatch):
    # The resident short circuit must still record the alias, or a loose .gguf shows its filename.
    path = "/models/lmstudio/TheBloke/weights-file-01.gguf"
    backend = _FakeBackend(path)
    rec = _LoadRecorder(backend)
    _wire_on(
        monkeypatch,
        resolves_to = (path, None, "Qwen3-4B-Instruct-GGUF"),
        backend = backend,
        recorder = rec,
    )
    _run_hook(path)
    assert rec.calls == []
    assert backend._openai_advertised_id == "Qwen3-4B-Instruct-GGUF"
    assert inference_route._llama_public_model_id(backend) == "Qwen3-4B-Instruct-GGUF"
    # Recorded, so the path now short circuits without the resolver.
    monkeypatch.setattr(
        resolver, "resolve_local_gguf", lambda _m, **_kw: pytest.fail("resolver re-entered")
    )
    _run_hook(path)


def test_streaming_responses_uses_advertised_id_helper():
    # Streamed responses must use _llama_public_model_id so they report the repo id, not a path.

    src = inspect.getsource(inference_route._responses_stream)
    assert "_clean_model = _llama_public_model_id(llama_backend" in src
    assert 'public_model_id(getattr(llama_backend, "model_identifier"' not in src


@pytest.mark.stages_switch_waiter
def test_concurrent_same_target_requests_load_once(monkeypatch):
    # Two concurrent requests for one unloaded model must load once, not 409 each other.

    backend, rec = _wired(monkeypatch, _FakeBackend("org/A-GGUF"), ("/p/B", "Q8_0", "org/B-GGUF"))
    monkeypatch.setattr(kw, "_inflight", 2)
    monkeypatch.setattr(kw, "_pending", 0)
    inference_route._note_switch_waiter(inference_route._switch_key("org/B-GGUF", "Q8_0"), 1)
    _run_hook("org/B-GGUF:Q8_0")
    assert len(rec.calls) == 1


@pytest.mark.stages_switch_waiter
def test_queued_different_target_does_not_deadlock_current_swap(monkeypatch):
    # A request queued for another target is not generating, so it must not block this swap.

    backend, rec = _wired(monkeypatch, _FakeBackend("org/A-GGUF"), ("/p/B", "Q8_0", "org/B-GGUF"))
    monkeypatch.setattr(kw, "_inflight", 2)
    monkeypatch.setattr(kw, "_pending", 0)
    inference_route._note_switch_waiter(inference_route._switch_key("org/C-GGUF", "Q4_K_M"), 1)
    _run_hook("org/B-GGUF:Q8_0")
    assert len(rec.calls) == 1


def test_v1_models_advertises_repo_id_not_load_path(monkeypatch):
    # /v1/models must report the advertised repo id, never the host load path.

    llama = _FakeBackend("/cache/models--org--Repo/snapshots/abc")
    llama._openai_advertised_id = "org/Repo-GGUF"
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: llama)
    monkeypatch.setattr(
        inference_route, "get_inference_backend", lambda: SimpleNamespace(active_model_name = None)
    )
    objects = inference_route._openai_model_objects()
    assert [o["id"] for o in objects] == ["org/Repo-GGUF"]


def test_idle_alias_reload_preserves_override_via_advertised_id(monkeypatch):
    # An alias reload looks up overrides by the advertised id so saved launch flags survive.

    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = True, resolves_to = None, backend = backend, recorder = rec)
    monkeypatch.setattr(kw, "_inflight", 0)
    monkeypatch.setattr(kw, "_last_unloaded_model", ("/cache/snap/A", "Q4_K_M", "org/A-GGUF"))
    overrides = {"org/A-GGUF": {"max_seq_length": 8192}}
    monkeypatch.setattr(settings, "get_model_override", lambda mid: overrides.get(mid, {}))
    _run_hook("gpt-4o-mini")
    assert rec.calls[0].model_path == "/cache/snap/A"
    assert rec.calls[0].gguf_variant == "Q4_K_M"
    assert rec.calls[0].max_seq_length == 8192


def test_load_route_holds_lifecycle_gate(monkeypatch):
    # The /load route must hold inference_lifecycle_gate so idle-unload cannot fire mid-load.

    src = inspect.getsource(inference_route.load_model_gated)
    assert "inference_lifecycle_gate" in src
    assert "_load_model_impl" in src


def test_model_replacements_recheck_sidecar_swap_before_either_backend_is_unloaded():
    # The sidecar-reservation recheck is the last rejection point, so the destructive cancel follows it.

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
    unload_gguf = src.index("_unload_llama_before_standard_load", standard_wait)

    assert already_loaded < gguf_wait < gguf_sidecar_check < gguf_cancel < unload_unsloth
    assert standard_branch < standard_wait < standard_sidecar_check
    assert standard_sidecar_check < standard_cancel < unload_gguf


def test_switch_waiter_deregisters_before_swap_gate_release():
    # A stale waiter after release lets another loop's swap drain early and unload a live request.

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
    backend = _FakeBackend(None)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: False)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.anthropic_messages(_anthropic_payload(), object(), "tester"))
    assert exc.value.status_code == 503


def test_anthropic_400_when_auto_switch_on_and_max_tokens_missing(monkeypatch):
    backend = _FakeBackend(None)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.anthropic_messages(_anthropic_payload(), object(), "tester"))
    assert exc.value.status_code == 400


def test_pending_same_target_request_does_not_block_swap(monkeypatch):
    # A pending same-target request in the middleware must not block the first one.

    backend, rec = _wired(monkeypatch, _FakeBackend("org/A-GGUF"), ("/p/B", "Q8_0", "org/B-GGUF"))
    monkeypatch.setattr(kw, "_inflight", 1)
    monkeypatch.setattr(kw, "_pending", 1)
    _run_hook("org/B-GGUF:Q8_0")
    assert len(rec.calls) == 1


@pytest.mark.stages_switch_waiter
def test_swap_waits_until_concurrent_request_finishes_resolving(monkeypatch):
    backend, rec = _wired(monkeypatch, _FakeBackend("org/A-GGUF"), ("/p/B", "Q8_0", "org/B-GGUF"))
    monkeypatch.setattr(kw, "_inflight", 2)
    monkeypatch.setattr(kw, "_pending", 0)
    # The twin is counted in-flight but has not yet joined the concrete target queue.

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
    kw._inflight = 2
    scope = {"type": "http"}
    kw.untrack_current_request(scope)
    assert kw._inflight == 1
    assert scope.get(kw._UNTRACKED_SCOPE_KEY) is True
    kw.untrack_current_request(scope)
    assert kw._inflight == 1
    kw._inflight = 0


def test_manual_unload_interrupts_even_while_inference_active(monkeypatch):
    from models.inference import UnloadRequest

    backend = _FakeBackend("org/A-GGUF")
    backend.is_active = True
    backend.unload_model = lambda: setattr(backend, "is_loaded", False)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(inference_route, "is_registered_native_path_label", lambda *a: False)
    monkeypatch.setattr(kw, "_inflight", 1)
    monkeypatch.setattr(kw, "_pending", 0)
    resp = asyncio.run(
        inference_route.unload_model(UnloadRequest(model_path = "org/A-GGUF"), "tester")
    )
    assert resp.status == "unloaded"
    assert not backend.is_loaded


def test_auto_switch_waits_when_unsloth_stream_active(monkeypatch):
    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire_on(
        monkeypatch,
        resolves_to = ("/p/B", "Q8_0", "org/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    monkeypatch.setattr(kw, "_inflight", 2)
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
    assert inference_route._llama_public_model_id(backend) == "org/Repo-GGUF"
    backend._openai_advertised_id = None
    # Without an advertised id the identifier is cleaned to a public id, never the raw path.
    cleaned = inference_route._llama_public_model_id(backend)
    assert cleaned and "/cache/" not in cleaned and not cleaned.endswith(".gguf")
    backend.model_identifier = "org/Repo-GGUF"
    assert inference_route._llama_public_model_id(backend) == "org/Repo-GGUF"
    backend.model_identifier = None
    assert inference_route._llama_public_model_id(backend, "req") == "req"


def test_chat_validates_non_system_message_before_auto_switch():
    # A system-only chat must be rejected before the hook so it never swaps the model.
    src = inspect.getsource(inference_route.produce_openai_chat_completions)
    assert src.index("At least one non-system message is required.") < src.index(
        "_maybe_auto_switch_model"
    )


def test_chat_untracks_external_provider_before_proxy():
    # External-provider requests untrack before proxying so they cannot block a local switch.
    src = inspect.getsource(inference_route.produce_openai_chat_completions)
    assert src.index("untrack_current_request") < src.index("_proxy_to_external_provider")


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
    # Training is refused while a request streams, so it cannot unload the model mid-stream.
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
    import routes.training as training_route

    monkeypatch.setattr(kw, "_inflight", 1)
    monkeypatch.setattr(kw, "_pending", 0)
    fake = SimpleNamespace(is_training_active = lambda: True, current_job_id = "job-1")
    monkeypatch.setattr(training_route, "get_training_backend", lambda: fake)
    resp = asyncio.run(
        training_route.start_training(_training_request(), current_subject = "t", via_api_key = False)
    )
    assert resp.status == "error" and "already" in (resp.error or "").lower()


def test_env_idle_ttl_standalone_when_no_stored_value(monkeypatch):
    monkeypatch.setattr(settings, "_cached_setting", lambda k, d = None: d)
    monkeypatch.setenv("UNSLOTH_MODEL_IDLE_TTL", "600")
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: False)
    assert settings.get_auto_unload_idle_seconds() == 600
    assert settings.get_stored_auto_unload_idle_seconds() == 600


def test_stored_idle_value_overrides_env_and_stays_gated(monkeypatch):
    store = {settings.AUTO_UNLOAD_IDLE_SETTING_KEY: 90}
    monkeypatch.setattr(settings, "_cached_setting", lambda k, d = None: store.get(k, d))
    monkeypatch.setenv("UNSLOTH_MODEL_IDLE_TTL", "600")
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    assert settings.get_auto_unload_idle_seconds() == 90
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: False)
    assert settings.get_auto_unload_idle_seconds() == 0


def test_env_idle_ttl_invalid_is_ignored(monkeypatch):
    monkeypatch.setattr(settings, "_cached_setting", lambda k, d = None: d)
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: False)
    monkeypatch.setenv("UNSLOTH_MODEL_IDLE_TTL", "not-a-number")
    assert settings.get_auto_unload_idle_seconds() == 0
    monkeypatch.delenv("UNSLOTH_MODEL_IDLE_TTL", raising = False)
    assert settings.get_auto_unload_idle_seconds() == 0


def test_env_idle_standalone_reloads_freed_model_with_auto_switch_off(monkeypatch):
    # An env idle TTL with auto-switch off must still restore exactly the freed model.

    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(
        monkeypatch,
        enabled = False,
        resolves_to = ("/p/B", "Q8_0", "org/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 600)
    monkeypatch.setattr(settings, "idle_unload_is_configured", lambda: 600 > 0)
    monkeypatch.setattr(kw, "_inflight", 0)
    monkeypatch.setattr(kw, "_last_unloaded_model", ("/cache/snap/A", "Q4_K_M", "org/A-GGUF"))
    # A is restored, but the request named B, so it is told so rather than served A.
    with pytest.raises(HTTPException) as excinfo:
        _run_hook("org/B-GGUF")
    assert excinfo.value.status_code == 404
    assert len(rec.calls) == 1
    assert rec.calls[0].model_path == "/cache/snap/A"
    assert rec.calls[0].gguf_variant == "Q4_K_M"


def test_no_stash_reload_when_idle_off_and_auto_switch_off(monkeypatch):
    # With both features off the hook is a no-op and must not resurrect a stashed model.

    backend, rec = _wired(monkeypatch, _FakeBackend(None), None, enabled = False)
    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 0)
    monkeypatch.setattr(settings, "idle_unload_is_configured", lambda: 0 > 0)
    monkeypatch.setattr(kw, "_inflight", 0)
    monkeypatch.setattr(kw, "_last_unloaded_model", ("/cache/snap/A", "Q4_K_M", "org/A-GGUF"))
    _run_hook("org/B-GGUF")
    assert rec.calls == []


def test_stash_reload_skipped_while_unsloth_model_active(monkeypatch):
    # A live Unsloth model after idle-unload must not be torn down to restore the GGUF stash.

    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = True, resolves_to = None, backend = backend, recorder = rec)
    monkeypatch.setattr(kw, "_inflight", 0)
    monkeypatch.setattr(kw, "_last_unloaded_model", ("/cache/snap/A", "Q4_K_M", "org/A-GGUF"))
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: SimpleNamespace(active_model_name = "unsloth/Qwen3-8B"),
    )
    _run_hook("gpt-4o-mini")
    assert rec.calls == []


def test_is_abs_path_id_distinguishes_path_from_repo_id():
    assert resolver._is_abs_path_id("/abs/path/model.gguf") is True
    assert resolver._is_abs_path_id("org/Repo-GGUF") is False
    assert resolver._is_abs_path_id("Repo") is False


def test_advertised_loader_id_prefers_alias_over_abs_path():
    f = resolver._advertised_loader_id
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
    assert (
        f(SimpleNamespace(id = "org/X-GGUF", model_id = "org/X-GGUF", display_name = "X")) == "org/X-GGUF"
    )


def test_index_advertises_alias_not_filesystem_path(tmp_path, monkeypatch):
    # A path-as-id scanner must not leak the host path in /v1/models, yet stay resolvable by it.

    gguf = tmp_path / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"x" * 32)
    info = SimpleNamespace(
        id = str(gguf),
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
    index = resolver._index()
    advertised = sorted({entry.loader_id for entry in index.values()})
    assert advertised == ["org/Repo-GGUF"]
    assert gguf.name.lower() not in index
    resolver._scan = (0.0, {})
    assert resolver.resolve_local_gguf(str(gguf)) is not None


def test_inactive_cache_revision_aliases_keep_exact_companion_scope(tmp_path, monkeypatch):
    repo = tmp_path / "models--org--Vision-GGUF"
    selected = repo / "snapshots" / "weights-revision"
    selected.mkdir(parents = True)
    (selected / "model-Q4_K_M.gguf").write_bytes(b"GGUF weights")
    (selected / "model-Q8_0-00001-of-00002.gguf").write_bytes(b"GGUF partial")
    companion = repo / "snapshots" / "companion-revision" / "MTP"
    companion.mkdir(parents = True)
    (companion / "mtp-model-Q8_0.gguf").write_bytes(b"GGUF companion")
    os.utime(selected, (1_000, 1_000))
    os.utime(companion.parent, (2_000, 2_000))

    info = SimpleNamespace(
        id = str(selected),
        path = str(selected),
        model_id = "org/Vision-GGUF",
        display_name = "Vision-GGUF",
        source = "hf_cache",
    )
    monkeypatch.setattr(models_route, "_scan_models_dir", lambda *a, **k: [info])
    monkeypatch.setattr(models_route, "_scan_hf_cache", lambda *a, **k: [])
    monkeypatch.setattr(models_route, "_scan_lmstudio_dir", lambda *a, **k: [])
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: tmp_path / "active")
    monkeypatch.setattr(models_route, "_is_hidden_model", lambda *a, **k: False)
    monkeypatch.setattr(hf_cache_settings, "known_hf_hub_caches", lambda: [])
    monkeypatch.setattr(paths, "legacy_hf_cache_dir", lambda: None)
    monkeypatch.setattr(paths, "hf_default_cache_dir", lambda: None)
    monkeypatch.setattr(paths, "lmstudio_model_dirs", lambda: [])
    monkeypatch.setattr(studio_db, "list_scan_folders", lambda: [])

    index = resolver._build_index()

    repo_entry = index["org/vision-gguf"]
    display_entry = index["vision-gguf"]
    path_entry = index[str(selected).lower()]
    revision_entry = index[selected.name.lower()]
    assert repo_entry.repo_level_companions is True
    assert display_entry.repo_level_companions is True
    assert path_entry.repo_level_companions is False
    assert revision_entry.repo_level_companions is False
    assert path_entry.variants == ("Q4_K_M",)
    assert revision_entry.variants == ("Q4_K_M",)
    assert {
        entry.load_path for entry in (repo_entry, display_entry, path_entry, revision_entry)
    } == {str(selected)}


def test_build_index_survives_a_failing_scanner(tmp_path, monkeypatch):
    # One failing scanner must drop only its own source, not abort the whole index.

    def _boom(*a, **k):
        raise OSError("permission denied")

    lm_info = SimpleNamespace(
        id = "org/Repo-GGUF", path = "/lm/Repo", model_id = "org/Repo-GGUF", display_name = "Repo"
    )
    monkeypatch.setattr(models_route, "_scan_models_dir", _boom)
    monkeypatch.setattr(models_route, "_scan_hf_cache", lambda *a, **k: [])
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(models_route, "_is_hidden_model", lambda *a, **k: False)
    monkeypatch.setattr(models_route, "_scan_lmstudio_dir", lambda *a, **k: [lm_info])
    monkeypatch.setattr(paths, "legacy_hf_cache_dir", lambda: None)
    monkeypatch.setattr(paths, "hf_default_cache_dir", lambda: None)
    monkeypatch.setattr(paths, "lmstudio_model_dirs", lambda: [tmp_path])
    monkeypatch.setattr(
        resolver,
        "_local_gguf_entry",
        lambda loader_id, info: resolver._LocalGgufEntry(loader_id, "/lm/Repo", ()),
    )
    resolver._scan = (0.0, {})
    index = resolver._build_index()
    assert any(e.loader_id == "org/Repo-GGUF" for e in index.values())


def test_build_index_groups_overlapping_custom_gguf_roots(tmp_path, monkeypatch):
    root = tmp_path / "custom"
    model = root / "publisher"
    model.mkdir(parents = True)
    (model / "model-Q4_K_M.gguf").write_bytes(b"x")
    quant_dir = model / "Q8_0"
    quant_dir.mkdir()
    (quant_dir / "model-Q8_0.gguf").write_bytes(b"xx")
    (quant_dir / "config.json").write_text("{}", encoding = "utf-8")
    scan_models_dir = models_route._scan_models_dir
    scan_lmstudio_dir = models_route._scan_lmstudio_dir

    monkeypatch.setattr(
        models_route,
        "_scan_models_dir",
        lambda path, **kwargs: scan_models_dir(path, **kwargs) if path in {root, quant_dir} else [],
    )
    monkeypatch.setattr(
        models_route,
        "_scan_lmstudio_dir",
        lambda path: scan_lmstudio_dir(path) if path in {root, quant_dir} else [],
    )
    monkeypatch.setattr(models_route, "_scan_hf_cache", lambda *a, **k: [])
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: tmp_path / "active")
    monkeypatch.setattr(models_route, "_is_hidden_model", lambda *a, **k: False)
    monkeypatch.setattr(hf_cache_settings, "known_hf_hub_caches", lambda: [])
    monkeypatch.setattr(paths, "legacy_hf_cache_dir", lambda: None)
    monkeypatch.setattr(paths, "hf_default_cache_dir", lambda: None)
    monkeypatch.setattr(paths, "lmstudio_model_dirs", lambda: [])
    monkeypatch.setattr(
        studio_db,
        "list_scan_folders",
        lambda: [{"path": str(root)}, {"path": str(quant_dir)}],
    )

    index = resolver._build_index()

    assert {entry.load_path for entry in index.values()} == {str(model)}
    assert {entry.variants for entry in index.values()} == {("Q4_K_M", "Q8_0")}


def test_non_gguf_entries_skip_gguf_sibling_revision_scans(tmp_path, monkeypatch):
    snapshot = tmp_path / "models--org--Repo" / "snapshots" / "rev-new"
    info = SimpleNamespace(
        id = str(snapshot),
        path = str(snapshot),
        model_id = "org/Repo",
        display_name = "Repo",
    )
    monkeypatch.setattr(models_route, "_scan_models_dir", lambda *a, **k: [info])
    monkeypatch.setattr(models_route, "_scan_hf_cache", lambda *a, **k: [])
    monkeypatch.setattr(models_route, "_scan_lmstudio_dir", lambda *a, **k: [])
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(models_route, "_is_hidden_model", lambda *a, **k: False)
    monkeypatch.setattr(hf_cache_settings, "known_hf_hub_caches", lambda: [])
    monkeypatch.setattr(paths, "legacy_hf_cache_dir", lambda: None)
    monkeypatch.setattr(paths, "hf_default_cache_dir", lambda: None)
    monkeypatch.setattr(paths, "lmstudio_model_dirs", lambda: [])
    monkeypatch.setattr(studio_db, "list_scan_folders", lambda: [])
    monkeypatch.setattr(
        resolver,
        "_local_servable_entry",
        lambda loader_id, _info: resolver._LocalGgufEntry(
            loader_id, str(snapshot), (), is_gguf = False
        ),
    )
    monkeypatch.setattr(
        resolver,
        "_sibling_revision_entries",
        lambda *_a: pytest.fail("non-GGUF discovery walked GGUF sibling revisions"),
    )

    index = resolver._build_index()

    assert any(not entry.is_gguf for entry in index.values())


def test_local_servable_model_reads_files_not_model_format(tmp_path):
    # HF-cache snapshots lack model_format, so detect from files; a model needs a root config.json.

    gguf = tmp_path / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"x" * 32)
    assert resolver.local_servable_model(SimpleNamespace(id = str(gguf), path = str(gguf))) == (
        True,
        (),
    )

    st = tmp_path / "safetensors_model"
    st.mkdir()
    (st / "model.safetensors").write_bytes(_safetensors_bytes())
    (st / "tokenizer.json").write_text("{}")
    (st / "tokenizer_config.json").write_text('{"chat_template": "{{ messages }}"}')
    assert resolver.local_servable_model(SimpleNamespace(id = str(st), path = str(st))) is None
    (st / "config.json").write_text(_CHAT_CONFIG)
    assert resolver.local_servable_model(SimpleNamespace(id = str(st), path = str(st))) == (
        False,
        (),
    )


def test_local_servable_model_excludes_ollama_links(tmp_path):
    # Ollama entries never resolve through _build_index, so they must not be reported as servable.

    links = tmp_path / ".studio_links"
    links.mkdir()
    ollama_gguf = links / "model-Q4_K_M.gguf"
    ollama_gguf.write_bytes(b"x" * 32)
    assert (
        resolver.local_servable_model(
            SimpleNamespace(id = "ollama/foo:latest", path = str(ollama_gguf))
        )
        is None
    )
    plain = tmp_path / "model-Q4_K_M.gguf"
    plain.write_bytes(b"x" * 32)
    assert resolver.local_servable_model(SimpleNamespace(id = str(plain), path = str(plain))) == (
        True,
        (),
    )


def test_embeddings_input_present_helper():
    f = inference_route._embeddings_input_present
    assert f({"input": "hi"}) is True
    assert f({"input": ["a", "b"]}) is True
    assert f({"input": [1, 2, 3]}) is True
    assert f({}) is False
    assert f({"input": ""}) is False
    assert f({"input": []}) is False


def test_embeddings_rejects_missing_input_before_switch(monkeypatch):
    # An embeddings request with no input must 400 before the hook so it never swaps the model.

    backend = _FakeBackend("org/A-GGUF")
    rec = _LoadRecorder(backend)
    _wire_on(
        monkeypatch,
        resolves_to = ("/p/B", "Q8_0", "org/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            inference_route.openai_embeddings(_json_body_request({"model": "org/B-GGUF"}), "tester")
        )
    assert exc.value.status_code == 400
    assert rec.calls == []


def test_retrieve_model_tolerates_non_string_id(monkeypatch):
    # A model with a non-string id is skipped rather than crashing the .lower() compare.

    async def _objs():
        return [{"id": 123, "object": "model"}, {"id": "org/B-GGUF", "object": "model"}]

    monkeypatch.setattr(inference_route, "_openai_model_objects", lambda: [])
    monkeypatch.setattr(inference_route, "_openai_catalog_objects", _objs)
    obj = asyncio.run(inference_route.openai_retrieve_model("org/B-GGUF", "tester"))
    assert obj["id"] == "org/B-GGUF"
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_retrieve_model("123", "tester"))
    assert exc.value.status_code == 404


def test_retrieve_model_resolves_raw_path_to_advertised_id(monkeypatch):
    # A legacy absolute .gguf path must map to the advertised repo id, or a loaded model 404s.

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
    # stream=true with n>1 is invalid everywhere, so it must 400 before loading another model.

    backend, rec = _wired(monkeypatch, _FakeBackend("org/A-GGUF"), ("/p/B", "Q8_0", "org/B-GGUF"))
    payload = _chat_request(model = "org/B-GGUF", stream = True, n = 2)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))
    assert exc.value.status_code == 400
    assert rec.calls == []


def test_resolver_cache_stamped_after_slow_build(monkeypatch):
    # Stamp the cache after _build_index, or a slow scan stores an already-expired cache.
    import core.inference.local_model_resolver as r

    clock = {"t": 1000.0}
    monkeypatch.setattr(r.time, "monotonic", lambda: clock["t"])
    calls = {"n": 0}

    def _slow_build():
        calls["n"] += 1
        clock["t"] += r._CACHE_TTL_S + 10.0
        return {}

    monkeypatch.setattr(r, "_build_index", _slow_build)
    r._scan = (0.0, {})
    r._index()
    r._index()
    assert calls["n"] == 1


def test_keepwarm_does_not_stamp_activity_on_401(monkeypatch):
    # Keep-warm runs before auth, so a 401 must not stamp activity and keep the model warm.
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
    assert kw._inflight == 0
    assert kw._last_active == 100.0
    asyncio.run(_run(200))
    assert kw._inflight == 0
    assert kw._last_active != 100.0


def _stash(monkeypatch, *, idle = 600):
    """Common setup for the standalone-idle reload paths: feature off, idle TTL on,
    an idle-freed model in the stash, nothing loaded, no in-flight requests."""
    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: idle)
    monkeypatch.setattr(settings, "idle_unload_is_configured", lambda: idle > 0)
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
    # /v1/completions must 400 a malformed prompt before loading a different GGUF.

    backend, rec = _wired(monkeypatch, _FakeBackend("org/A-GGUF"), ("/p/B", "Q8_0", "org/B-GGUF"))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            inference_route.openai_completions(
                _json_body_request({"model": "org/B-GGUF"}), "tester"
            )
        )
    assert exc.value.status_code == 400
    assert rec.calls == []


def test_chat_system_only_rejected_before_idle_reload(monkeypatch):
    backend, rec = _wired(monkeypatch, _FakeBackend(None), None, enabled = False)
    _stash(monkeypatch)
    payload = ChatCompletionRequest(model = "x", messages = [{"role": "system", "content": "sys"}])
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))
    assert exc.value.status_code == 400
    assert rec.calls == []


def test_embeddings_missing_input_rejected_before_idle_reload(monkeypatch):
    # /v1/embeddings must 400 missing input under a standalone idle TTL too.

    backend, rec = _wired(monkeypatch, _FakeBackend(None), None, enabled = False)
    _stash(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_embeddings(_json_body_request({"model": "x"}), "tester"))
    assert exc.value.status_code == 400
    assert rec.calls == []


def test_messages_does_not_503_before_reload_hook_when_idle_on(monkeypatch):
    # /v1/messages must defer its early 503 so a standalone idle TTL can restore the model.
    backend, rec = _wired(monkeypatch, _FakeBackend(None), None, enabled = False)
    _stash(monkeypatch)
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
    src = inspect.getsource(inference_route.anthropic_messages)
    assert "_automatic_model_load_may_run" in src


def test_raw_body_without_model_reloads_freed_model(monkeypatch):
    backend, rec = _wired(monkeypatch, _FakeBackend(None), None, enabled = False)
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
    backend, rec = _wired(monkeypatch, _FakeBackend(None), None, enabled = False)
    _stash(monkeypatch)
    payload = ChatCompletionRequest(model = "x", messages = [{"role": "user", "content": "say hi"}])
    try:
        asyncio.run(inference_route.generate_audio(payload, object(), "tester"))
    except Exception:
        pass
    assert len(rec.calls) == 1
    assert rec.calls[0].model_path == "/cache/snap/A"


def test_audio_generate_does_not_reload_on_invalid_request(monkeypatch):
    # The audio reload hook runs after validation so an empty request never reloads.

    backend, rec = _wired(monkeypatch, _FakeBackend(None), None, enabled = False)
    _stash(monkeypatch)
    payload = ChatCompletionRequest(model = "x", messages = [])
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.generate_audio(payload, object(), "tester"))
    assert exc.value.status_code == 400
    assert rec.calls == []


def test_preview_scope_disables_auto_switch(monkeypatch):
    # The preview route must stay on its pinned checkpoint, so the hook is opted out there.
    backend, rec = _wired(monkeypatch, _FakeBackend("org/A-GGUF"), ("/p/B", "Q8_0", "org/B-GGUF"))

    class _Req:
        def __init__(self):
            self.scope = {}

    req = _Req()
    inference_route.disable_openai_auto_switch_for_request(req.scope)
    asyncio.run(inference_route._maybe_auto_switch_model("org/B-GGUF", req, "tester"))
    assert rec.calls == []

    # Control: a fresh request without the flag would switch.
    req2 = _Req()
    asyncio.run(inference_route._maybe_auto_switch_model("org/B-GGUF", req2, "tester"))
    assert len(rec.calls) == 1


def test_preview_chat_is_tracked_as_inference_path():
    # Keep-warm counts long preview streams so the idle loop cannot unload mid-response.

    assert _is_inference_path("/p/my-run/v1/chat/completions") is True
    assert _is_inference_path("/p/my-run/ckpt-100/v1/chat/completions") is True
    assert _is_inference_path("/p/my-run/v1/models") is False


def test_untrack_does_not_reset_idle_timer():
    # Untracking external traffic must not restamp activity, or the local GGUF never idles.

    kw._inflight = 1
    kw._last_active = time.monotonic() - 3600
    before = kw._last_active
    scope = {"type": "http"}
    kw.untrack_current_request(scope)
    assert kw._inflight == 0
    assert kw._last_active == before
    kw._inflight = 0


def test_note_start_does_not_reset_idle_timer():
    kw._inflight = 0
    kw._pending = 0
    kw._last_active = time.monotonic() - 3600
    before = kw._last_active
    kw._note_start()
    try:
        assert kw._inflight == 1
        assert kw._last_active == before
        assert kw._is_idle(1.0) is False
    finally:
        kw._note_end()


def test_omitted_model_does_not_resolve_to_a_named_gguf(monkeypatch):
    # An omitted model must never run the resolver, so a GGUF named "default" is not switched to.
    backend = _FakeBackend("org/A-GGUF")
    rec = _LoadRecorder(backend)
    _wire_on(
        monkeypatch,
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
    assert rec.calls == []


def test_omitted_model_still_reloads_idle_freed_model(monkeypatch):
    # The reload-only sentinel still restores an idle-freed model without running the resolver.

    backend = _FakeBackend(None)
    rec = _LoadRecorder(backend)
    _wire(monkeypatch, enabled = False, resolves_to = None, backend = backend, recorder = rec)
    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 600)
    monkeypatch.setattr(settings, "idle_unload_is_configured", lambda: 600 > 0)
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
    # A malformed client tool must 400 before the hook so it never evicts the model.

    backend, rec = _wired(monkeypatch, _FakeBackend("org/A-GGUF"), ("/p/B", "Q8_0", "org/B-GGUF"))
    payload = _anthropic_payload_with_tools([{"name": "broken"}])
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.anthropic_messages(payload, object(), "tester"))
    assert exc.value.status_code == 400
    assert rec.calls == []


def test_anthropic_validates_tools_before_auto_switch():
    for fn in (inference_route.anthropic_messages, inference_route.anthropic_count_tokens):
        src = inspect.getsource(fn)
        assert src.index("_validate_anthropic_client_tools") < src.index("_maybe_auto_switch_model")


def test_anthropic_mixed_tools_rejected_before_switch(monkeypatch):
    # Mixing server and custom client tools is unsupported, so it must 400 before the switch.

    backend, rec = _wired(monkeypatch, _FakeBackend("org/A-GGUF"), ("/p/B", "Q8_0", "org/B-GGUF"))
    payload = _anthropic_payload_with_tools(
        [
            {"type": "web_search_20250305"},
            {"name": "my_func", "input_schema": {"type": "object"}},
        ]
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.anthropic_messages(payload, object(), "tester"))
    assert exc.value.status_code == 400
    assert rec.calls == []


def _chat_msg(text = "hi"):
    from models.inference import ChatMessage
    return ChatMessage(role = "user", content = text)


def _responses_payload(
    *,
    tools = None,
    set_model = True,
    stream = None,
):
    kwargs = dict(input = "hi")
    if set_model:
        kwargs["model"] = "org/B-GGUF"
    if tools is not None:
        kwargs["tools"] = tools
    if stream is not None:
        kwargs["stream"] = stream
    return ResponsesRequest(**kwargs)


def test_switch_model_for_payload_only_switches_when_explicit():
    # An omitted model is reload-only, while an explicit model, even "default", is honored.

    omitted = ChatCompletionRequest(messages = [_chat_msg()])
    assert inference_route._switch_model_for_payload(omitted) == inference_route._RELOAD_ONLY_MODEL
    explicit_default = ChatCompletionRequest(model = "default", messages = [_chat_msg()])
    assert inference_route._switch_model_for_payload(explicit_default) == "default"
    explicit = ChatCompletionRequest(model = "org/B-GGUF", messages = [_chat_msg()])
    assert inference_route._switch_model_for_payload(explicit) == "org/B-GGUF"


def test_omitted_schema_model_skips_resolver(monkeypatch):
    # An omitted model never runs the resolver, while an explicit model still switches.

    backend, rec = _wired(
        monkeypatch, _FakeBackend("org/A-GGUF"), ("org/B-GGUF", "Q8_0", "org/B-GGUF")
    )
    omitted = ChatCompletionRequest(messages = [_chat_msg()])
    asyncio.run(
        inference_route._maybe_auto_switch_model(
            inference_route._switch_model_for_payload(omitted), object(), "tester"
        )
    )
    assert rec.calls == []
    explicit = ChatCompletionRequest(model = "org/B-GGUF", messages = [_chat_msg()])
    asyncio.run(
        inference_route._maybe_auto_switch_model(
            inference_route._switch_model_for_payload(explicit), object(), "tester"
        )
    )
    assert len(rec.calls) == 1


def test_build_chat_request_propagates_omitted_model():
    # An omitted Responses model must not become "default", or the chat re-check would switch.
    omitted = _responses_payload(set_model = False)
    chat_req = inference_route._build_chat_request(omitted, [_chat_msg()], stream = False)
    assert "model" not in chat_req.model_fields_set
    explicit = _responses_payload(set_model = True)
    chat_req2 = inference_route._build_chat_request(explicit, [_chat_msg()], stream = False)
    assert "model" in chat_req2.model_fields_set


def test_responses_invalid_function_tool_rejected_before_switch(monkeypatch):
    # A function tool with no name must 400 before the hook so it never evicts the model.

    backend, rec = _wired(
        monkeypatch, _FakeBackend("org/A-GGUF"), ("org/B-GGUF", "Q8_0", "org/B-GGUF")
    )
    payload = _responses_payload(tools = [{"type": "function", "parameters": {}}])
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_responses(payload, object(), "tester"))
    assert exc.value.status_code == 400
    assert rec.calls == []


def test_responses_valid_and_builtin_tools_pass_validation(monkeypatch):
    # Valid function and built-in tools pass; the hook is stubbed to stop after validation.
    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _boom)
    payload = _responses_payload(
        tools = [{"type": "function", "name": "ok", "parameters": {}}, {"type": "web_search"}]
    )
    with pytest.raises(_Reached):
        asyncio.run(inference_route.openai_responses(payload, object(), "tester"))


def test_responses_validates_tools_before_auto_switch():
    src = inspect.getsource(inference_route.openai_responses)
    assert src.index("each function tool must have a 'name'") < src.index(
        "_maybe_auto_switch_model"
    )


def test_responses_forcing_tool_choice_without_name_rejected_before_switch(monkeypatch):
    # A function tool_choice with no name must 400 before the switch so it cannot evict the model.

    async def _boom(*a, **k):
        raise AssertionError("must not switch on an invalid tool_choice")

    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _boom)
    payload = ResponsesRequest(model = "org/B-GGUF", input = "hi", tool_choice = {"type": "function"})
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_responses(payload, object(), "tester"))
    assert exc.value.status_code == 400
    ok = ResponsesRequest(
        model = "org/B-GGUF", input = "hi", tool_choice = {"type": "function", "name": "f"}
    )
    with pytest.raises(AssertionError):
        asyncio.run(inference_route.openai_responses(ok, object(), "tester"))


def test_swap_acquires_process_gate_before_load():
    # The process-wide gate wraps the load and is always released, guarding cross-loop swaps.

    src = inspect.getsource(inference_route._maybe_auto_switch_model)
    assert src.index("_acquire_swap_gate") < src.index("_load_model_impl")
    assert "_auto_switch_process_lock.release()" in src


def _chat_request(**kw):
    from models.inference import ChatCompletionRequest, ChatMessage
    kw.setdefault("messages", [ChatMessage(role = "user", content = "hi")])
    return ChatCompletionRequest(**kw)


def _chat_request_b(
    *args,
    model = "org/B-GGUF",
    **kwargs,
):
    """_chat_request for the org/B-GGUF model these cases switch to."""
    return _chat_request(*args, model = model, **kwargs)


def test_chat_confirm_without_stream_rejected_before_switch(monkeypatch):
    # confirm_tool_calls without streaming plus local tools is invalid and must 400 before the switch.

    backend, rec = _wired(
        monkeypatch, _FakeBackend("org/A-GGUF"), ("org/B-GGUF", "Q8_0", "org/B-GGUF")
    )
    payload = _chat_request_b(enable_tools = True, confirm_tool_calls = True, stream = False)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))
    assert exc.value.status_code == 400
    assert rec.calls == []


def test_chat_confirm_with_bypass_permissions_reaches_hook(monkeypatch):
    # bypass_permissions disables the confirm gate, so the request must reach the switch hook.
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _boom)
    payload = _chat_request_b(
        enable_tools = True,
        confirm_tool_calls = True,
        stream = False,
        bypass_permissions = True,
    )
    with pytest.raises(_Reached):
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))


def test_chat_audio_input_guards_target_before_switch(monkeypatch):
    # Audio uses the mmproj too, so require the projector, but not a vision tower, before switching.

    captured = {}

    async def _capture(
        model,
        request,
        subject,
        *,
        require_vision = False,
        require_image = True,
        modality_label = "image or audio",
        claim_resident = True,
        require_audio_input = False,
        require_video = False,
        gguf_only = False,
        audio_preflight = None,
        image_preflight = None,
        tool_images_only = False,
    ):
        captured.update(
            require_vision = require_vision,
            require_image = require_image,
            modality_label = modality_label,
            claim_resident = claim_resident,
            require_audio_input = require_audio_input,
            audio_preflight_has_image = (
                audio_preflight.get("has_image") if audio_preflight is not None else None
            ),
            audio_preflight_has_video = (
                audio_preflight.get("has_video") if audio_preflight is not None else None
            ),
        )
        raise _Reached()

    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _capture)
    payload = _chat_request(model = "org/B-GGUF", audio_base64 = "AAAA")
    with pytest.raises(_Reached):
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))
    assert captured == {
        "require_vision": True,
        "require_image": False,
        "modality_label": "audio",
        "claim_resident": False,
        "require_audio_input": True,
        "audio_preflight_has_image": False,
        "audio_preflight_has_video": False,
    }

    img = ImageContentPart(type = "image_url", image_url = ImageUrl(url = "data:image/png;base64,AAAA"))
    payload = _chat_request_b(
        audio_base64 = "AAAA",
        messages = [ChatMessage(role = "user", content = [img])],
    )
    with pytest.raises(_Reached):
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))
    assert captured == {
        "require_vision": True,
        "require_image": True,
        "modality_label": "image or audio",
        "claim_resident": False,
        "require_audio_input": True,
        "audio_preflight_has_image": True,
        "audio_preflight_has_video": False,
    }

    # A clip beside the recording is refused after the load, so the switch must know first.
    payload = _chat_request(
        model = "org/B-GGUF", audio_base64 = "AAAA", video_base64 = "AAAAGGZ0eXBtcDQy"
    )
    with pytest.raises(_Reached):
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))
    assert captured["modality_label"] == "audio or video"
    assert captured["audio_preflight_has_image"] is False
    assert captured["audio_preflight_has_video"] is True

    # An image on an earlier turn is allowed: the non-GGUF audio route may refer back to it.
    payload = _chat_request(
        model = "org/B-GGUF",
        audio_base64 = "AAAA",
        messages = [
            ChatMessage(role = "user", content = [img]),
            ChatMessage(role = "assistant", content = "I can see it."),
            ChatMessage(role = "user", content = "Tell me more."),
        ],
    )
    with pytest.raises(_Reached):
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))
    assert captured["audio_preflight_has_image"] is False


def test_completions_rejects_object_prompt_before_switch(monkeypatch):
    # A non-string, non-array prompt must 400 before the switch so it cannot evict the model.

    backend, rec = _wired(monkeypatch, _FakeBackend("org/A-GGUF"), ("/p/B", "Q8_0", "org/B-GGUF"))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            inference_route.openai_completions(
                _json_body_request({"model": "org/B-GGUF", "prompt": {}}), "tester"
            )
        )
    assert exc.value.status_code == 400
    assert rec.calls == []


def _raise_reached(*_args, **_kwargs):
    raise _Reached()


_IGNORED_COMPLETIONS_PARAMS = [
    ({"echo": True}, "echo"),
    ({"suffix": " the end."}, "suffix"),
    ({"best_of": 3}, "best_of"),
    ({"best_of": 3, "n": 2}, "best_of"),
    ({"best_of": 2, "stream": True}, "best_of"),
]


@pytest.mark.parametrize("extra, param", _IGNORED_COMPLETIONS_PARAMS)
def test_completions_rejects_ignored_params_before_switch(monkeypatch, extra, param):
    backend, rec = _wired(monkeypatch, _FakeBackend("org/A-GGUF"), ("/p/B", "Q8_0", "org/B-GGUF"))
    body = {"model": "org/B-GGUF", "prompt": "hi", **extra}
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_completions(_json_body_request(body), "tester"))
    assert exc.value.status_code == 400
    assert exc.value.detail["error"]["code"] == "unsupported_parameter"
    assert exc.value.detail["error"]["param"] == param
    assert rec.calls == []


@pytest.mark.parametrize("extra, param", _IGNORED_COMPLETIONS_PARAMS)
def test_completions_rejects_ignored_params_without_switch(monkeypatch, extra, param):
    backend, rec = _wired(monkeypatch, _FakeBackend("org/A-GGUF"), None, enabled = False)
    monkeypatch.setattr(inference_route, "_fill_recommended_sampling_completions", _raise_reached)
    body = {"prompt": "hi", "max_tokens": 8, **extra}
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_completions(_json_body_request(body), "tester"))
    assert exc.value.status_code == 400
    assert exc.value.detail["error"]["param"] == param


@pytest.mark.parametrize(
    "extra",
    [
        {"echo": False, "suffix": "", "best_of": 1},
        {"echo": None, "suffix": None, "best_of": None},
        {"best_of": 2, "n": 2},
    ],
)
def test_completions_default_ignored_params_still_proxy(monkeypatch, extra):
    backend, rec = _wired(monkeypatch, _FakeBackend("org/A-GGUF"), None, enabled = False)
    monkeypatch.setattr(inference_route, "_fill_recommended_sampling_completions", _raise_reached)
    body = {"prompt": "hi", "max_tokens": 8, **extra}
    with pytest.raises(_Reached):
        asyncio.run(inference_route.openai_completions(_json_body_request(body), "tester"))


def test_embeddings_rejects_object_input_before_switch(monkeypatch):
    backend, rec = _wired(monkeypatch, _FakeBackend("org/A-GGUF"), ("/p/B", "Q8_0", "org/B-GGUF"))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            inference_route.openai_embeddings(
                _json_body_request({"model": "org/B-GGUF", "input": {}}), "tester"
            )
        )
    assert exc.value.status_code == 400
    assert rec.calls == []


def test_chat_oversized_audio_rejected_before_switch(monkeypatch):
    # The audio size cap is cheap and target-independent, so it must 413 before any load.

    backend, rec = _wired(monkeypatch, _FakeBackend("org/A-GGUF"), ("/p/B", "Q8_0", "org/B-GGUF"))
    big = "A" * (inference_route._MAX_AUDIO_B64_CHARS + 1)
    payload = _chat_request(model = "org/B-GGUF", audio_base64 = big)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))
    assert exc.value.status_code == 413
    assert rec.calls == []


def test_chat_confirm_without_stream_mcp_rejected_before_switch(monkeypatch):
    # mcp_enabled opens the tool loop alone, so confirm without streaming plus MCP must 400 too.

    monkeypatch.setattr(_tp, "get_tool_policy", lambda: None)
    backend, rec = _wired(
        monkeypatch, _FakeBackend("org/A-GGUF"), ("org/B-GGUF", "Q8_0", "org/B-GGUF")
    )
    payload = _chat_request_b(mcp_enabled = True, confirm_tool_calls = True, stream = False)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))
    assert exc.value.status_code == 400
    assert rec.calls == []


def test_require_vision_rejects_text_target_before_switch(monkeypatch):
    # An image request for a text-only GGUF must 400 before evicting the vision model.

    backend, rec = _wired(
        monkeypatch, _FakeBackend("org/A-GGUF"), ("/local/B.gguf", "Q8_0", "org/B-GGUF")
    )
    monkeypatch.setattr(inference_route, "_target_is_vision", lambda _p, _v = None, _i = True: False)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            inference_route._maybe_auto_switch_model(
                "org/B-GGUF", object(), "t", require_vision = True
            )
        )
    assert exc.value.status_code == 400
    assert rec.calls == []


def test_require_vision_allows_vision_target(monkeypatch):
    backend, rec = _wired(
        monkeypatch, _FakeBackend("org/A-GGUF"), ("/local/B.gguf", "Q8_0", "org/B-GGUF")
    )
    monkeypatch.setattr(inference_route, "_target_is_vision", lambda _p, _v = None, _i = True: True)
    asyncio.run(
        inference_route._maybe_auto_switch_model("org/B-GGUF", object(), "t", require_vision = True)
    )
    assert len(rec.calls) == 1


def test_an_audio_only_target_still_switches_for_an_audio_request(monkeypatch, tmp_path):
    # An audio-only projector snapshot must still be swapped in for an audio request.
    import struct

    key = "clip.has_audio_encoder"
    (tmp_path / "Voxtral-Mini-3B-Q4_K_M.gguf").write_bytes(b"\0" * 32)
    (tmp_path / "mmproj-F16.gguf").write_bytes(
        struct.pack("<IIQQ", 0x46554747, 3, 0, 1)
        + struct.pack("<Q", len(key))
        + key.encode()
        + struct.pack("<I", 7)
        + struct.pack("<?", True)
    )
    backend, rec = _wired(
        monkeypatch, _FakeBackend("org/A-GGUF"), (str(tmp_path), "Q4_K_M", "org/B-GGUF")
    )
    asyncio.run(
        inference_route._maybe_auto_switch_model(
            "org/B-GGUF", object(), "t", require_vision = True, require_image = False
        )
    )
    assert len(rec.calls) == 1


def test_require_vision_probes_the_quant_the_load_will_open(monkeypatch):
    # The probe must see the same directory and quant pair that the load uses.
    backend, rec = _wired(
        monkeypatch, _FakeBackend("org/A-GGUF"), ("/cache/snap", "UD-Q4_K_XL", "org/B-GGUF")
    )
    probed: list[tuple] = []
    monkeypatch.setattr(
        inference_route,
        "_target_is_vision",
        lambda path, variant = None, need_image = True: (
            probed.append((path, variant, need_image)) or True
        ),
    )
    asyncio.run(
        inference_route._maybe_auto_switch_model("org/B-GGUF", object(), "t", require_vision = True)
    )
    assert probed == [("/cache/snap", "UD-Q4_K_XL", True)]


def test_an_audio_request_is_not_refused_for_want_of_a_vision_tower(tmp_path):
    # Audio models have projectors with no vision tower, so gating on image support would 400 them.
    import struct

    key = "clip.has_audio_encoder"
    (tmp_path / "Voxtral-Mini-3B-Q4_K_M.gguf").write_bytes(b"\0" * 32)
    (tmp_path / "mmproj-F16.gguf").write_bytes(
        struct.pack("<IIQQ", 0x46554747, 3, 0, 1)
        + struct.pack("<Q", len(key))
        + key.encode()
        + struct.pack("<I", 7)
        + struct.pack("<?", True)
    )

    assert inference_route._target_is_vision(str(tmp_path), None, False) is True
    assert inference_route._target_is_vision(str(tmp_path), None, True) is False


def test_target_is_vision_reads_a_subdir_quants_projector(tmp_path):
    variant_dir = tmp_path / "UD-Q4_K_XL"
    variant_dir.mkdir()
    (variant_dir / "Qwen3-VL-235B-UD-Q4_K_XL.gguf").write_bytes(b"\0" * 32)
    (tmp_path / "mmproj-F32.gguf").write_bytes(b"\0" * 32)

    assert inference_route._target_is_vision(str(tmp_path), "UD-Q4_K_XL") is True


def test_require_vision_ignores_reload_stash(monkeypatch):
    backend, rec = _wired(monkeypatch, _FakeBackend(None), None, enabled = False)
    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 600)
    monkeypatch.setattr(settings, "idle_unload_is_configured", lambda: 600 > 0)
    monkeypatch.setattr(kw, "_inflight", 0)
    monkeypatch.setattr(kw, "_last_unloaded_model", ("/cache/snap/A", "Q4_K_M", "org/A-GGUF"))
    monkeypatch.setattr(inference_route, "_target_is_vision", lambda _p, _v = None, _i = True: False)
    # 404 because the restored A is not the requested B, whose quant makes it a real reference.
    with pytest.raises(HTTPException):
        asyncio.run(
            inference_route._maybe_auto_switch_model(
                "org/B-GGUF:UD-Q6_K_XL", object(), "t", require_vision = True
            )
        )
    assert len(rec.calls) == 1
    assert rec.calls[0].model_path == "/cache/snap/A"


def test_chat_validates_confirm_and_modality_before_switch():
    src = inspect.getsource(inference_route.produce_openai_chat_completions)
    assert src.index("confirm_tool_calls requires stream=true") < src.index(
        "_maybe_auto_switch_model"
    )
    assert "require_vision" in src
    hook = inspect.getsource(inference_route._maybe_auto_switch_model)
    assert hook.index("require_vision") < hook.index("_load_model_impl")
    assert "does not support the {modality_label} input" in hook


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
    from models.inference import AnthropicImageBlock

    f = inference_route._anthropic_request_has_image
    text = SimpleNamespace(messages = [SimpleNamespace(content = "hi")])
    assert f(text) is False
    text_block = SimpleNamespace(
        messages = [SimpleNamespace(content = [{"type": "text", "text": "hi"}])]
    )
    assert f(text_block) is False
    dict_img = SimpleNamespace(messages = [SimpleNamespace(content = [{"type": "image"}])])
    assert f(dict_img) is True
    typed_img = SimpleNamespace(
        messages = [
            SimpleNamespace(
                content = [
                    AnthropicImageBlock(
                        type = "image",
                        source = {"type": "base64", "media_type": "image/png", "data": "AAAA"},
                    )
                ]
            )
        ]
    )
    assert f(typed_img) is True


def test_responses_and_anthropic_wire_require_vision_from_images():
    # Responses and Messages hooks must derive require_vision from images like chat does.

    responses_src = inspect.getsource(inference_route.openai_responses)
    assert "_responses_has_image = _messages_have_image(" in responses_src
    assert "require_vision = _responses_has_image" in responses_src
    anthropic_src = inspect.getsource(inference_route.anthropic_messages)
    guard = "_anthropic_request_has_image(payload, tool_results = False)"
    assert f"_anthropic_top_level_image = {guard}" in anthropic_src
    assert "require_vision = _anthropic_top_level_image" in anthropic_src
    # count_tokens shares the /messages translation, so it needs the same vision guard.
    count_src = inspect.getsource(inference_route.anthropic_count_tokens)
    assert f"require_vision = {guard}" in count_src


def test_count_tokens_rejects_malformed_tool_before_switch(monkeypatch):
    # count_tokens must reject a malformed tool before the switch, like /messages.

    backend, rec = _wired(monkeypatch, _FakeBackend("org/A-GGUF"), ("/p/B", "Q8_0", "org/B-GGUF"))
    payload = _anthropic_payload_with_tools([{"name": "broken"}])
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.anthropic_count_tokens(payload, object(), "tester"))
    assert exc.value.status_code == 400
    assert rec.calls == []


def test_count_tokens_forwards_vision_guard_to_switch(monkeypatch):
    # An image count_tokens request needs the require_vision guard so it cannot evict a vision model.
    captured = {}

    async def _capture(
        model,
        request,
        subject,
        *,
        require_vision = False,
        claim_resident = True,
        require_audio_input = False,
        gguf_only = False,
    ):
        captured["require_vision"] = require_vision
        captured["claim_resident"] = claim_resident
        captured["gguf_only"] = gguf_only
        raise _Reached()

    monkeypatch.setattr(inference_route, "_anthropic_request_has_image", lambda p, **_: True)
    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _capture)
    payload = _anthropic_payload_with_tools(None)
    with pytest.raises(_Reached):
        asyncio.run(inference_route.anthropic_count_tokens(payload, object(), "tester"))
    assert captured["require_vision"] is True
    assert captured["claim_resident"] is False
    # llama.cpp serves this endpoint alone, so a non-GGUF swap must not be attempted.
    assert captured["gguf_only"] is True


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
    backend = _FakeBackend(loaded_id)
    backend.supports_tools = supports_tools
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
    monkeypatch.setattr(inference_route, "current_date_prompt_line", lambda **_kwargs: "")
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
        pytest.param([{"type": "text", "text": "hello"}], id = "text_parts"),
    ],
)
def test_chat_count_tokens_prices_the_loaded_model_without_switching(monkeypatch, content):
    # The recount has no abort signal, so a stale payload naming A must not drag the backend back.
    switched, _counted = _count_tokens_backend(monkeypatch, "org/B-GGUF", count = 42)
    payload = _count_request([{"role": "user", "content": content}])
    assert _counted_body(payload) == {"input_tokens": 42, "model": "org/B-GGUF"}
    assert switched == []


def test_chat_count_tokens_forwards_enabled_tools(monkeypatch):
    _switched, counted = _count_tokens_backend(monkeypatch, count = 99, supports_tools = True)
    gate = {}

    async def _select(
        payload,
        *,
        tools_on,
        mcp_allowed,
        supports_vision = False,
    ):
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

    async def _select(
        _payload,
        *,
        tools_on,
        mcp_allowed,
        supports_vision = False,
    ):
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
        pytest.param(
            True,
            _PASSTHROUGH_TOOL_HISTORY,
            {},
            None,
            id = "cli_policy_does_not_price_a_passthrough_prompt",
        ),
        pytest.param(
            True,
            _PASSTHROUGH_PLAIN,
            {},
            ["web_search"],
            id = "cli_policy_prices_an_ordinary_chat",
        ),
        pytest.param(
            None,
            _PASSTHROUGH_PLAIN,
            {"tools": _PASSTHROUGH_CATALOG},
            ["get_weather"],
            id = "client_catalog_is_priced_verbatim",
        ),
        pytest.param(
            None,
            _PASSTHROUGH_PLAIN,
            {"tools": _PASSTHROUGH_CATALOG, "tool_choice": "none"},
            None,
            id = "withdrawn_catalog_is_not_priced",
        ),
        pytest.param(
            None,
            _PASSTHROUGH_TOOL_HISTORY,
            {"tools": _PASSTHROUGH_CATALOG, "tool_choice": "none"},
            ["get_weather"],
            id = "withdrawn_catalog_with_tool_history_is_priced",
        ),
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
    _switched, counted = _count_tokens_backend(monkeypatch, count = 99, supports_tools = True)

    async def _select(
        payload,
        *,
        tools_on,
        mcp_allowed,
        supports_vision = False,
    ):
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

    _counted_body(_count_request(sentinel_thread))
    assert [message.get("content") for message in counted.get("messages") or []] == [
        "first\n\nsecond"
    ]


def test_chat_count_tokens_folds_a_stopped_studio_tool_thread(monkeypatch):
    """The counter skips its own coalesce on the passthrough, so only the fold helper keeps a
    Stop-sentinel thread alternating; without it the bar prices a prompt the completion 400s on.
    Unlike ``..._keeps_adjacent_user_turns_on_the_passthrough`` above, this thread IS folded.
    """
    _switched, counted = _count_tokens_backend(monkeypatch, count = 99)
    thread = [
        {"role": "user", "content": "what did we say about seeds?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search_conversation", "arguments": '{"query": "s"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "search_conversation",
            "content": "we said 3407",
        },
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "and now?"},
    ]

    _counted_body(
        _count_request(
            thread,
            studio_tool_history = True,
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": {"type": "object", "properties": {}}},
            },
        )
    )
    roles = [message.get("role") for message in counted.get("messages") or []]
    assert "tool" not in roles, roles
    assert not any(a == "user" and b == "user" for a, b in zip(roles, roles[1:])), roles


def test_chat_count_tokens_prices_the_current_date(monkeypatch):
    """The bar has to price what generation sends, and only the passthrough is sent undated."""
    _switched, counted = _count_tokens_backend(monkeypatch, count = 99, supports_tools = True)
    monkeypatch.setattr(
        inference_route,
        "current_date_prompt_line",
        lambda **_kwargs: "The current date is 2026-08-15.",
    )
    monkeypatch.setattr(inference_route, "_local_template_system_turn", lambda *_a: (True, ""))
    thread = [{"role": "user", "content": "hi"}]

    _counted_body(_count_request(thread))
    assert counted["messages"] == [
        {"role": "system", "content": "The current date is 2026-08-15."},
        {"role": "user", "content": "hi"},
    ]

    # The passthrough sends the request verbatim, so counting an unsent date would overcount.
    _counted_body(_count_request(thread, tools = _PASSTHROUGH_CATALOG))
    assert all(message.get("role") != "system" for message in counted["messages"])


def test_chat_count_tokens_dates_only_api_server_tool_prompts(monkeypatch):
    _switched, counted = _count_tokens_backend(monkeypatch, count = 99, supports_tools = True)
    monkeypatch.setattr(
        inference_route,
        "current_date_prompt_line",
        lambda **_kwargs: "The current date is 2026-08-15.",
    )

    async def _select(
        _payload,
        *,
        tools_on,
        mcp_allowed,
        supports_vision = False,
    ):
        return [{"type": "function", "function": {"name": "web_search"}}]

    monkeypatch.setattr(inference_route, "_select_request_tools", _select)
    monkeypatch.setattr(
        inference_route,
        "_request_is_internal_workflow",
        lambda _request: False,
    )
    request = types.SimpleNamespace(
        headers = {"authorization": "Bearer sk-unsloth-test"},
    )
    thread = [{"role": "user", "content": "hi"}]

    asyncio.run(
        inference_route.chat_count_tokens(
            _count_request(thread, enable_tools = True, enabled_tools = ["web_search"]),
            "tester",
            request,
        )
    )
    assert counted["messages"][0]["content"].startswith("The current date is 2026-08-15.\n\n")

    asyncio.run(inference_route.chat_count_tokens(_count_request(thread), "tester", request))
    assert all(message.get("role") != "system" for message in counted["messages"])


def _in_flight_generation():
    """One registered generation, as the completion path registers it."""
    from state import active_generations
    return active_generations.ActiveGeneration(threading.Event(), thread_id = "t1")


def test_chat_count_tokens_refuses_while_a_generation_is_in_flight(monkeypatch):
    # A count must never share llama-server with a decode, and the frontend gate is not enough.
    switched, counted = _count_tokens_backend(monkeypatch, count = 1234)
    # Reached after tool selection and rewriting, proving the refusal happens on entry.
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
    _switched, counted = _count_tokens_backend(monkeypatch, count = 1234)
    payload = _count_request([{"role": "user", "content": "hello"}])
    with _in_flight_generation():
        with pytest.raises(HTTPException):
            asyncio.run(inference_route.chat_count_tokens(payload, "tester"))
    body = _counted_body(payload)
    assert body["input_tokens"] == 1234
    assert counted != {}, "the tokenizer must be reached once nothing is decoding"


def test_chat_count_tokens_refuses_a_generation_that_starts_mid_count(monkeypatch):
    # A generation can start between the guards, so starting one here must make the count abandon.
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
    monkeypatch.setattr(mcp_servers_db, "_schema_ready", set())
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
    _enabled_mcp_server(tmp_path, monkeypatch, cached = MCP_TOOL_PAYLOAD)
    specs, complete = cached_mcp_tools()
    assert complete is True
    assert [spec["function"]["name"] for spec in specs] == ["mcp__s1__lookup"]


def test_cached_mcp_tools_reports_an_undiscovered_server_as_incomplete(tmp_path, monkeypatch):
    _enabled_mcp_server(tmp_path, monkeypatch)
    specs, complete = cached_mcp_tools()
    assert specs == []
    assert complete is False, (
        "a completion would probe this server and render its schemas, so a count that skips "
        "them is short, not exact"
    )


def test_cached_mcp_tools_counts_a_cooloff_server_as_complete(tmp_path, monkeypatch):
    # The completion renders nothing for a cool-off server, so skipping it here is exact.

    _enabled_mcp_server(tmp_path, monkeypatch, cooloff = True)
    specs, complete = cached_mcp_tools()
    assert specs == []
    assert complete is True


def test_chat_count_tokens_declines_an_undiscovered_mcp_server(tmp_path, monkeypatch):
    _switched, counted = _count_tokens_backend(monkeypatch, count = 1234, supports_tools = True)
    _enabled_mcp_server(tmp_path, monkeypatch)
    payload = _count_request([{"role": "user", "content": "hello"}], mcp_enabled = True)
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(inference_route.chat_count_tokens(payload, "tester"))
    assert excinfo.value.status_code == 503
    assert "mcp" in str(excinfo.value.detail).lower()
    assert counted == {}, "the tokenizer must not be reached with a short tool list"


def test_chat_count_tokens_prices_cached_mcp_schemas(tmp_path, monkeypatch):
    # MCP alone enables tools, so the count must render them even with built-in tools off.
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
    # The decline keys on the request asking for MCP, not on a server existing.
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
        handle = _in_flight_generation()
        handle.__enter__()
        started.append(handle)
        assert should_abort is not None, "the route must give the tokenizer a way to stand down"
        if should_abort():
            from core.inference.llama_cpp import CountAborted
            raise CountAborted()
        counted.update(messages = messages)
        return 1234

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
    _switched, counted = _count_tokens_backend(monkeypatch, count = 1234)
    body = _counted_body(_count_request([{"role": "user", "content": "what did I just say"}]))
    assert body["input_tokens"] == 1234
    assert counted != {}, "the tokenizer must be reached"


def test_chat_count_tokens_refuses_an_empty_prompt(monkeypatch):
    """#8882: an empty conversation renders the generation marker alone.

    unsloth/Phi-4-mini-instruct-GGUF Q4_K_M renders "<|assistant|>" for an empty message list, one
    token, and the header reported it as usage on a chat nobody had started.
    """
    switched, counted = _count_tokens_backend(monkeypatch, count = 1)
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(inference_route.chat_count_tokens(_count_request([]), "tester"))
    assert excinfo.value.status_code == 503
    assert "empty" in str(excinfo.value.detail).lower()
    assert counted == {}, "the tokenizer must not be reached"
    assert switched == []


def test_chat_count_tokens_counts_an_empty_chat_carrying_a_system_prompt(monkeypatch):
    # The system prompt is in every request the chat will send, so it occupies the window already.
    _switched, counted = _count_tokens_backend(monkeypatch, count = 7)
    body = _counted_body(_count_request([{"role": "system", "content": "You are helpful."}]))
    assert body["input_tokens"] == 7
    assert [message.get("role") for message in counted.get("messages") or []] == ["system"]


def test_chat_count_tokens_counts_an_empty_chat_the_cli_policy_fills(monkeypatch):
    """`--enable-tools` outranks the request's own `enable_tools: false`.

    The client cannot see that policy, so the emptiness verdict belongs here: the schemas and the
    action nudge it injects are real occupancy, and refusing them would blank a bar that has a
    number to show.
    """
    _switched, counted = _count_tokens_backend(monkeypatch, count = 850, supports_tools = True)

    async def _select(
        payload,
        *,
        tools_on,
        mcp_allowed,
        supports_vision = False,
    ):
        return [{"type": "function", "function": {"name": "web_search"}}]

    monkeypatch.setattr(inference_route, "_select_request_tools", _select)
    monkeypatch.setattr(_tp, "get_tool_policy", lambda: True)

    body = _counted_body(_count_request([], enable_tools = False))
    assert body["input_tokens"] == 850
    nudged = any(
        message.get("role") == "system" and "web_search" in str(message.get("content", ""))
        for message in counted.get("messages") or []
    )
    assert nudged is True


def test_chat_count_tokens_counts_a_passthrough_catalog_without_messages(monkeypatch):
    # /apply-template renders the caller's schemas, so the prompt is non-empty without messages.
    _switched, counted = _count_tokens_backend(monkeypatch, count = 640, supports_tools = True)
    catalog = [{"type": "function", "function": {"name": "lookup_order"}}]
    body = _counted_body(_count_request([], tools = catalog))
    assert body["input_tokens"] == 640
    assert [(tool.get("function") or {}).get("name") for tool in counted.get("tools") or []] == [
        "lookup_order"
    ]


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
        pytest.param(_PENDING_USER_TURN, {"thread_id": "t1"}, None, id = "pending_user_turn"),
        pytest.param(_PENDING_TOOL_TURN, {"project_id": "p1"}, None, id = "pending_tool_turn"),
        pytest.param(_SETTLED_TURN, {"thread_id": "t1"}, 4242, id = "settled_turn_still_counts"),
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
    _switched, counted = _count_tokens_backend(monkeypatch, count = 13)
    payload = _count_request(
        [
            {"role": "system", "content": "Runtime rules."},
            {"role": "system", "content": "Unsloth prompt."},
            {"role": "user", "content": "hello"},
        ]
    )
    asyncio.run(inference_route.chat_count_tokens(payload, "tester"))
    messages = counted.get("messages") or []
    systems = [m for m in messages if m.get("role") in ("system", "developer")]
    assert len(systems) == 1, messages
    assert "Runtime rules." in systems[0].get("content", "")
    assert "Unsloth prompt." in systems[0].get("content", "")


@pytest.mark.parametrize(
    ("reasoning_style", "fields", "expected"),
    [
        pytest.param(
            "enable_thinking",
            {"enable_thinking": False},
            {"enable_thinking": False},
            id = "thinking_turned_off",
        ),
        pytest.param(
            "reasoning_effort",
            {"reasoning_effort": "low"},
            {"reasoning_effort": "low"},
            id = "effort_level",
        ),
        pytest.param(
            "enable_thinking",
            {"enable_thinking": True, "preserve_thinking": True},
            {"enable_thinking": True, "preserve_thinking": True},
            id = "preserve_thinking",
        ),
        pytest.param(
            "reasoning_effort",
            {"chat_template_kwargs": {"reasoning_effort": "none"}},
            {"reasoning_effort": "none"},
            id = "nested_effort",
        ),
        pytest.param(
            "enable_thinking",
            {"chat_template_kwargs": {"preserve_thinking": True}},
            {"preserve_thinking": True},
            id = "nested_preserve_thinking",
        ),
        pytest.param("enable_thinking", {}, None, id = "template_default"),
    ],
)
def test_chat_count_tokens_renders_the_requested_reasoning_mode(
    monkeypatch, reasoning_style, fields, expected
):
    _switched, counted = _count_tokens_backend(
        monkeypatch, count = 7, reasoning_style = reasoning_style
    )
    payload = _count_request([{"role": "user", "content": "hello"}], **fields)
    assert _counted_body(payload) == {"input_tokens": 7, "model": "org/A-GGUF"}
    assert counted.get("chat_template_kwargs") == expected


@pytest.mark.parametrize(
    ("template_kwargs", "expected_tokens"),
    [
        pytest.param({"enable_thinking": False}, 5, id = "thinking_off"),
        pytest.param(None, 3, id = "template_default"),
    ],
)
def test_count_chat_tokens_renders_with_the_requested_template_kwargs(
    monkeypatch, template_kwargs, expected_tokens
):
    """The kwargs have to reach llama-server itself: /apply-template runs the same parser
    as /v1/chat/completions, so the rendered prompt only moves when they are in the body."""

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
            kwargs = body.get("chat_template_kwargs") or {}
            prefill = "" if kwargs.get("enable_thinking", True) else " <think> </think>"
            return _FakeResponse({"prompt": "user hi assistant" + prefill})

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
        pytest.param("status", id = "apply_template_rejects"),
        pytest.param("raise", id = "apply_template_unreachable"),
    ],
)
def test_strict_count_refuses_a_text_only_template_fallback(monkeypatch, failure):
    """/apply-template failing on a TEXT-ONLY prompt used to fall through to concatenating message
    text, dropping every role marker, special token and tool schema (~30% of a six-turn two-tool
    prompt). Strict callers publish what they get, so it must be an error, not an estimate."""

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

    monkeypatch.setattr(llama_cpp_mod.httpx, "Client", _FakeClient)
    messages = [{"role": "user", "content": "hi there"}]
    tools = [{"type": "function", "function": {"name": "web_search"}}]
    with pytest.raises(RuntimeError):
        _CountBackend().count_chat_tokens(messages, None, tools, strict = True)
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
    (+30 chars) and Phi-4 (+38), overcounting the empty chat the bar exists to show. Templates
    that raise on no messages (Qwen3.5+) are re-priced only after refusing; see below."""
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
                return _FakeResponse({"prompt": "<start_of_turn>model\n"})
            return _FakeResponse({"tokens": str(body.get("content", "")).split()})

    monkeypatch.setattr(llama_cpp_mod.httpx, "Client", _FakeClient)
    count = _CountBackend().count_chat_tokens([], None, None, strict = True)
    assert seen["messages"] == [], "the count must not invent a turn the caller never sent"
    assert count > 0, "a fresh chat still prices the template preamble"


class _RefusingEmptyRenderClient:
    """llama-server with a Qwen3.5+ template that raises on no messages; strips a trailing assistant."""

    sent = []
    down = False
    empty_status = 500

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
        if url.endswith(("/apply-template", "/input_tokens")):
            messages = body.get("messages")
            type(self).sent.append((url.rsplit("/", 1)[-1], messages))
            if type(self).down:
                raise RuntimeError("timed out")
            rendered = list(messages or [])
            if rendered and rendered[-1].get("role") == "assistant":
                rendered = rendered[:-1]
            if not rendered:
                status = type(self).empty_status
                return _CountResponse(
                    {"error": {"code": status, "message": "No messages provided."}},
                    status_code = status,
                )
            if url.endswith("/input_tokens"):
                return _CountResponse({"input_tokens": 7})
            return _CountResponse(
                {"prompt": "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n"}
            )
        return _CountResponse({"tokens": str(body.get("content", "")).split()})


class _CountResponse:
    def __init__(
        self,
        payload,
        status_code = 200,
    ):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


@pytest.fixture
def refusing_client(monkeypatch):
    _RefusingEmptyRenderClient.sent = []
    _RefusingEmptyRenderClient.down = False
    _RefusingEmptyRenderClient.empty_status = 500
    monkeypatch.setattr(llama_cpp_mod.httpx, "Client", _RefusingEmptyRenderClient)
    return _RefusingEmptyRenderClient


@pytest.mark.parametrize("prefer_native", [False, True])
@pytest.mark.parametrize(
    "messages",
    [[], [{"role": "assistant", "content": '{"name": "terminal"}'}]],
    ids = ["new_chat", "lone_pending_call"],
)
def test_a_template_refusing_an_empty_render_is_priced_behind_one_empty_user_turn(
    refusing_client, prefer_native, messages
):
    """#12327: an empty render the template refuses is priced behind one empty user turn, once per load."""
    backend = _CountBackend()
    count = backend.count_chat_tokens(
        messages, None, None, strict = True, prefer_native = prefer_native
    )
    assert count > 0, "a refused empty render must still be priced, not refused"
    assert refusing_client.sent[0][1] == messages, "what the caller sent is still tried first"
    assert refusing_client.sent[-1][1] == [{"role": "user", "content": ""}] + messages

    refusing_client.sent = []
    assert (
        backend.count_chat_tokens(messages, None, None, strict = True, prefer_native = prefer_native)
        == count
    )
    assert all(
        sent == [{"role": "user", "content": ""}] + messages for _, sent in refusing_client.sent
    ), "a known refusal must not be re-sent on every recount"


def test_a_conversation_the_template_renders_costs_one_request(refusing_client):
    """The chat path counts real conversations on every turn; the fallback must add nothing there."""
    backend = _CountBackend()
    backend._empty_chat_render_refused = True
    conversation = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "Read ./README.md"},
    ]
    backend.count_chat_tokens(conversation, None, None, strict = True)
    assert refusing_client.sent == [("apply-template", conversation)]


def test_an_unreachable_server_is_not_retried_or_taken_for_a_refusal(refusing_client):
    """A timeout says nothing about the template: no second round trip, and no remembered refusal."""
    refusing_client.down = True
    backend = _CountBackend()
    with pytest.raises(RuntimeError):
        backend.count_chat_tokens([], None, None, strict = True)
    assert len(refusing_client.sent) == 1
    assert backend._empty_chat_render_refused is False


def test_a_busy_server_is_not_taken_for_a_refusal(refusing_client):
    """llama-server answers 503 while loading or out of slots; only a 500 is a template refusal."""
    refusing_client.empty_status = 503
    backend = _CountBackend()
    with pytest.raises(RuntimeError):
        backend.count_chat_tokens([], None, None, strict = True)
    assert [m for _, m in refusing_client.sent] == [[]], "a busy server is not retried"
    assert backend._empty_chat_render_refused is False


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


def _capture_audio_switch(monkeypatch):
    captured = {}

    async def _capture(model, request, subject, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        raise _Reached()

    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _capture)
    return _Reached, captured


@pytest.mark.parametrize("model", [None, "", "org/B-GGUF"])
def test_audio_generate_model_selection(monkeypatch, model):
    reached, captured = _capture_audio_switch(monkeypatch)
    payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": "say hi"}],
        **({"model": model} if model is not None else {}),
    )
    with pytest.raises(reached):
        asyncio.run(inference_route.generate_audio(payload, object(), "tester"))
    assert captured["model"] == (model or inference_route._RELOAD_ONLY_MODEL)
    assert captured["claim_resident"] is False
    assert captured["require_speech"] is True


def test_note_model_unloaded_clears_reload_stash(monkeypatch):
    # A deliberate unload drops the idle stash so the next request cannot resurrect the model.
    import core.inference.llama_keepwarm as kw

    kw._set_last_unloaded(("org/A-GGUF", "Q4_K_M"))
    assert kw.get_last_unloaded_model() == ("org/A-GGUF", "Q4_K_M")
    kw.note_model_unloaded()
    assert kw.get_last_unloaded_model() is None


def test_unload_route_clears_reload_stash(monkeypatch):
    # /unload must clear the stash on both GGUF and non-GGUF branches.
    src = inspect.getsource(inference_route._unload_model_impl)
    assert src.count("note_model_unloaded()") >= 2


def test_non_gguf_load_clears_reload_stash():
    # A non-GGUF load clears the stash too, so it never lingers when idle-unload is off.

    src = inspect.getsource(inference_route._load_model_impl)
    assert src.count("note_model_loaded()") >= 1
    assert "to_thread(note_model_loaded, llama_backend)" in src


def test_chat_rejects_malformed_tool_choice_before_switch(monkeypatch):
    # A forcing object with no function name must 400 before the switch.

    backend, rec = _wired(
        monkeypatch, _FakeBackend("org/A-GGUF"), ("org/B-GGUF", "Q8_0", "org/B-GGUF")
    )
    payload = _chat_request(model = "org/B-GGUF", tool_choice = {"type": "function", "function": {}})
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))
    assert exc.value.status_code == 400
    assert rec.calls == []


def test_chat_valid_tool_choice_reaches_hook(monkeypatch):
    # A well-formed forcing object must pass the pre-check and reach the hook.
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _boom)
    payload = _chat_request_b(tool_choice = {"type": "function", "function": {"name": "ok"}})
    with pytest.raises(_Reached):
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))


def test_lifecycle_gate_serializes_across_loops():
    # The lifecycle gate is process-wide so two event loops never hold it at once.

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
    assert state["max"] == 1


def test_auto_switch_serializes_across_event_loops(monkeypatch):
    # Per-loop asyncio locks cannot serialize swaps across loops; the process-wide gate must.

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
        await asyncio.sleep(0.1)
        with slock:
            state["cur"] -= 1
            loaded.append(request.model_path)
        backend.model_identifier = request.model_path
        backend.is_loaded = True
        backend._openai_advertised_id = None

    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    monkeypatch.setattr(resolver, "resolve_local_gguf", lambda m, **_kw: (m, "Q8_0", m))
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(inference_route, "_load_model_impl", _slow_load)
    monkeypatch.setattr(inference_route, "_auto_switch_waiters", {})

    barrier = threading.Barrier(2)

    def _run(model):
        barrier.wait()
        asyncio.run(inference_route._maybe_auto_switch_model(model, object(), "t"))

    threads = [
        threading.Thread(target = _run, args = ("org/B-GGUF",)),
        threading.Thread(target = _run, args = ("org/C-GGUF",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert state["max"] == 1
    assert sorted(loaded) == ["org/B-GGUF", "org/C-GGUF"]


def test_acquire_swap_gate_is_cancellation_safe():
    # A cancelled waiter must not leak the gate; a to_thread acquire would keep it forever.
    async def main():
        await inference_route._acquire_swap_gate()
        try:

            async def waiter():
                await inference_route._acquire_swap_gate()

            t = asyncio.create_task(waiter())
            await asyncio.sleep(0.05)
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
    # The auto-switch hint appears only when it is off; with it on the name simply did not resolve.
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
    downloaded = (),
):
    # Nothing loaded, so a chat request hits "no model loaded". Pin the catalog for determinism.
    async def _catalog():
        return [{"id": mid} for mid in catalog]

    # _downloaded_model_ids reads the local catalog, so pin it or tests depend on host downloads.
    async def _local_catalog():
        return [
            type("_Row", (), {"model_id": mid, "id": mid, "partial": False})() for mid in downloaded
        ]

    monkeypatch.setattr(inference_route, "_cached_local_catalog", _local_catalog)
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: enabled)
    # The auto-download setting is memoized process-wide, so pin it off or 404s become downloads.
    monkeypatch.setattr(settings, "get_openai_auto_download_enabled", lambda: False)
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
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))
    return exc.value.status_code, exc.value.detail


def test_chat_mistyped_gguf_repo_404s_before_vision_guard(monkeypatch):
    # An image request with a mistyped GGUF id must 404, not 400 on the loaded model.

    backend, rec = _wired(monkeypatch, _FakeBackend("unsloth/text-only-GGUF", "UD-Q4_K_XL"), None)
    monkeypatch.setattr(
        "utils.openai_auto_switch_settings.get_openai_auto_download_enabled", lambda: False
    )
    monkeypatch.setattr(
        resolver,
        "describe_local_miss",
        lambda _m: (resolver.MISS_MODEL_NOT_FOUND, ()),
    )
    payload = _chat_request_b(model = "unsloth/typo-vision-GGUF", image_base64 = "aGVsbG8=")
    request = type("_R", (), {"url": type("_U", (), {"path": "/v1/chat/completions"})()})()
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_chat_completions(payload, request, "tester"))
    assert exc.value.status_code == 404
    assert exc.value.detail["error"]["code"] == "model_not_found"
    assert rec.calls == []


def test_chat_names_undownloaded_model_404s_with_available_ids(monkeypatch):
    # The model is absent, so the error names it and lists what can serve instead.
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


def _wire_withheld_chat(monkeypatch, *, objects, downloaded):
    async def _catalog():
        return list(objects)

    _wire_unloaded_chat(monkeypatch, enabled = True, downloaded = downloaded)
    monkeypatch.setattr(inference_route, "_openai_catalog_objects", _catalog)


def test_chat_withheld_model_is_not_offered_back_as_available(monkeypatch):
    # Whisper rows are withheld from chat, so they must not be suggested as alternatives.
    _wire_withheld_chat(
        monkeypatch,
        objects = [
            {"id": "org/A-GGUF"},
            {"id": "openai/whisper-large-v3", "task": "automatic-speech-recognition"},
        ],
        downloaded = ("org/A-GGUF", "openai/whisper-large-v3"),
    )
    status, detail = _chat_error(_chat_request(model = "openai/whisper-large-v3"))
    assert status == 404
    assert "cannot serve it here" in detail
    assert "Available models: org/A-GGUF." in detail
    assert detail.count("openai/whisper-large-v3") == 1


def test_chat_absent_model_with_only_task_rows_says_no_chat_model_is_here(monkeypatch):
    # Whisper is downloaded, so "no models are downloaded yet" would contradict GET /v1/models.
    _wire_withheld_chat(
        monkeypatch,
        objects = [{"id": "openai/whisper-large-v3", "task": "automatic-speech-recognition"}],
        downloaded = ("openai/whisper-large-v3",),
    )
    status, detail = _chat_error(_chat_request(model = "org/nope-GGUF"))
    assert status == 404
    assert "none of the downloaded models is a chat model" in detail
    assert "no models are downloaded yet" not in detail


def test_chat_withheld_model_with_no_chat_rows_offers_nothing(monkeypatch):
    _wire_withheld_chat(
        monkeypatch,
        objects = [{"id": "openai/whisper-large-v3", "task": "automatic-speech-recognition"}],
        downloaded = ("openai/whisper-large-v3",),
    )
    status, detail = _chat_error(_chat_request(model = "openai/whisper-large-v3"))
    assert status == 404
    assert "cannot serve it here" in detail
    assert "Available models" not in detail


def test_chat_error_unchanged_when_auto_switch_off(monkeypatch):
    _wire_unloaded_chat(monkeypatch, enabled = False)
    status, detail = _chat_error(_chat_request(model = "org/nope-GGUF"))
    assert status == 400
    assert detail.startswith("No model loaded. Call POST /inference/load first.")
    assert "Model auto-switch" in detail


def test_chat_error_unchanged_when_no_model_named(monkeypatch):
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
    # The catalog lists a resident Transformers model the resolver misses, so avoid "not downloaded".
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
    status, detail = _run_responses_stream_no_model(
        monkeypatch,
        enabled = True,
        active_model_name = None,
        resolves_to = ("/p/A", "Q4_K_M", "unsloth/Qwen3.5-4B-GGUF"),
    )
    assert status == 400
    assert "not downloaded" not in detail


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
    until = None,
    timeout = 10.0,
):
    """Pass `until` when the test asserts something the loop must DO: a loaded
    runner can otherwise be cancelled mid-sequence (save recorded, unload not),
    which is a flake, not a failure. Name the LAST state the test asserts: the
    loop signals most of these from inside a to_thread body and still has
    bookkeeping to run after it, so an earlier landmark cancels that away.
    The fixed window always runs afterwards, both as settle time for that
    bookkeeping and because most of these tests also assert the loop then
    stops, which needs a stretch of loop time to be worth anything."""
    import time as _time

    async def _drive():
        task = asyncio.create_task(kw.idle_unload_loop(poll_seconds = poll_seconds))
        if until is not None:
            deadline = _time.monotonic() + timeout
            while not until() and _time.monotonic() < deadline:
                await asyncio.sleep(poll_seconds / 4)
        await asyncio.sleep(run_for)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_drive())


def test_idle_unload_saves_slots_before_unload_and_stashes_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 0.005)
    monkeypatch.setattr(settings, "idle_unload_is_configured", lambda: 0.005 > 0)
    monkeypatch.setattr(settings, "get_auto_unload_keep_kv", lambda: True)
    _reset_keepwarm()

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

    _drive_idle_loop(kw, until = lambda: events == ["save", "unload"] and kw._kv_resume)
    # KV must be saved while the server is still alive, then exactly one unload.
    assert events == ["save", "unload"]
    assert kw.get_last_unloaded_model()[:2] == ("unsloth/Idle-GGUF", "Q4_K_M")
    resume = kw.take_kv_resume()
    assert resume is not None
    assert resume["identity"][:2] == ("unsloth/Idle-GGUF", "Q4_K_M")
    assert resume["slots"][0]["filename"] == "f.bin"


def test_idle_save_failure_still_unloads_plain(monkeypatch):
    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 0.005)
    monkeypatch.setattr(settings, "idle_unload_is_configured", lambda: 0.005 > 0)
    monkeypatch.setattr(settings, "get_auto_unload_keep_kv", lambda: True)
    _reset_keepwarm()

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

    _drive_idle_loop(kw, until = lambda: kw.get_last_unloaded_model() is not None)
    assert unloads == [1]
    assert kw.get_last_unloaded_model() is not None
    assert kw.take_kv_resume() is None


def test_keep_kv_setting_off_skips_save(monkeypatch):
    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 0.005)
    monkeypatch.setattr(settings, "idle_unload_is_configured", lambda: 0.005 > 0)
    monkeypatch.setattr(settings, "get_auto_unload_keep_kv", lambda: False)
    _reset_keepwarm()

    saves, unloads = [], []
    backend = _FakeBackend("unsloth/Idle-GGUF")

    def _unload():
        unloads.append(1)
        backend.is_loaded = False

    backend.save_slots_for_resume = lambda *a, **k: saves.append(1)
    backend.unload_model = _unload
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)

    _drive_idle_loop(kw, until = lambda: unloads)
    assert saves == []
    assert unloads == [1]
    assert kw.take_kv_resume() is None


def test_keep_kv_disabled_mid_save_discards_manifest(monkeypatch, tmp_path):
    keep = {"on": True}
    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 0.005)
    monkeypatch.setattr(settings, "idle_unload_is_configured", lambda: 0.005 > 0)
    monkeypatch.setattr(settings, "get_auto_unload_keep_kv", lambda: keep["on"])
    _reset_keepwarm()

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
        keep["on"] = False
        return manifest

    def _unload():
        unloads.append(1)
        backend.is_loaded = False

    backend.save_slots_for_resume = _save
    backend.unload_model = _unload
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)

    _drive_idle_loop(kw, until = lambda: unloads)
    assert unloads == [1]
    assert kw.take_kv_resume() is None
    assert not state_file.exists()


def test_idle_ttl_disabled_mid_save_skips_unload(monkeypatch, tmp_path):
    ttl = {"v": 0.005}
    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: ttl["v"])
    monkeypatch.setattr(settings, "idle_unload_is_configured", lambda: ttl["v"] > 0)
    monkeypatch.setattr(settings, "get_auto_unload_keep_kv", lambda: True)
    _reset_keepwarm()

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
        ttl["v"] = 0
        return manifest

    backend.save_slots_for_resume = _save
    backend.unload_model = lambda: unloads.append(1)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)

    _drive_idle_loop(kw, until = lambda: not state_file.exists())
    assert unloads == []
    assert kw.take_kv_resume() is None
    assert not state_file.exists()


def test_alias_reload_restores_slots_and_deletes_files(monkeypatch, tmp_path):
    backend = _FakeBackend(None)
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
    assert len(restored) == 1
    assert not state_file.exists()
    assert kw._kv_resume is None


def test_no_restore_when_different_model_loads(monkeypatch, tmp_path):
    backend = _FakeBackend(None)
    backend._slot_save_binary = ("/bin/llama-server", 111)
    restored = []
    backend.restore_slots_for_resume = lambda manifest: restored.append(manifest)
    rec = _LoadRecorder(backend)
    _wire_on(
        monkeypatch,
        resolves_to = ("unsloth/B-GGUF", None, "unsloth/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    monkeypatch.setattr(kw, "_inflight", 0)
    state_file, manifest = _seed_kv_manifest(tmp_path)
    monkeypatch.setattr(kw, "_kv_resume", manifest)

    _run_hook("unsloth/B-GGUF")
    assert len(rec.calls) == 1
    assert restored == []
    assert not state_file.exists()
    assert kw._kv_resume is None


def test_restore_skipped_when_binary_changed(monkeypatch, tmp_path):
    state_file, manifest = _seed_kv_manifest(tmp_path)
    backend = _FakeBackend("unsloth/A-GGUF", hf_variant = "Q4_K_M")
    backend._gguf_path = manifest["gguf"]
    backend._slot_save_binary = ("/bin/llama-server", 222)
    restored = []
    backend.restore_slots_for_resume = lambda manifest: restored.append(manifest)

    kw.restore_kv_resume(backend, manifest)
    assert restored == []
    assert not state_file.exists()


def test_restore_skipped_when_launch_config_changed(tmp_path):
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
    state_file, manifest = _seed_kv_manifest(tmp_path)
    with open(manifest["gguf"], "wb") as fh:
        fh.write(b"different weights")
    backend = _FakeBackend("unsloth/A-GGUF", hf_variant = "Q4_K_M")
    backend._gguf_path = manifest["gguf"]
    backend._slot_save_binary = ("/bin/llama-server", 111)
    restored = []
    backend.restore_slots_for_resume = lambda manifest: restored.append(manifest)

    kw.restore_kv_resume(backend, manifest)
    assert restored == []
    assert not state_file.exists()


def test_note_model_unloaded_purges_manifest_and_files(tmp_path):
    state_file, manifest = _seed_kv_manifest(tmp_path)
    kw._set_last_unloaded(("org/A-GGUF", "Q4_K_M"))
    kw._set_kv_resume(manifest)
    kw.note_model_unloaded()
    assert kw.get_last_unloaded_model() is None
    assert kw.take_kv_resume() is None
    assert not state_file.exists()


def test_note_model_loaded_purges_manifest_and_files(tmp_path):
    state_file, manifest = _seed_kv_manifest(tmp_path)
    kw._set_last_unloaded(("org/A-GGUF", "Q4_K_M"))
    kw._set_kv_resume(manifest)
    kw.note_model_loaded()
    assert kw.get_last_unloaded_model() is None
    assert kw.take_kv_resume() is None
    assert not state_file.exists()


def test_new_idle_save_purges_previous_manifest_files(tmp_path):
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
    assert not old_file.exists()
    assert new_file.exists()
    assert kw.take_kv_resume()["slots"][0]["filename"] == new_file.name


def test_sweep_slot_save_dir_removes_only_resume_files(monkeypatch, tmp_path):
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
    store = {}
    monkeypatch.setattr(db, "upsert_app_settings", lambda m: store.update(m))
    monkeypatch.setattr(settings, "_cached_setting", lambda k, d = None: store.get(k, d))

    assert settings.get_auto_unload_keep_kv() is True
    assert settings.set_openai_auto_switch(True, 60, False)[2] is False
    assert store[settings.AUTO_UNLOAD_KEEP_KV_SETTING_KEY] is False
    assert settings.get_auto_unload_keep_kv() is False
    assert settings.set_openai_auto_switch(True, 60, None)[2] is False
    assert store[settings.AUTO_UNLOAD_KEEP_KV_SETTING_KEY] is False
    with pytest.raises(ValueError, match = "true or false"):
        settings.set_openai_auto_switch(True, 60, "garbage")


def test_stale_stash_cleanup_waits_for_lifecycle_gate(monkeypatch, tmp_path):
    # The loop's stale-stash purge must wait on the gate a mid-reload holds.

    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 3600)
    monkeypatch.setattr(settings, "idle_unload_is_configured", lambda: 3600 > 0)
    kw._inflight = 0
    kw._pending = 0
    kw._last_active = time.monotonic()
    backend = _FakeBackend("unsloth/New-GGUF")
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    state_file, manifest = _seed_kv_manifest(tmp_path)
    kw._kv_resume = manifest
    kw._last_unloaded_model = ("unsloth/A-GGUF", "Q4_K_M")

    assert kw._lifecycle_lock.acquire(blocking = False)
    try:
        _drive_idle_loop(kw)
        assert kw._kv_resume is manifest
        assert state_file.exists()
    finally:
        kw._lifecycle_lock.release()
    _drive_idle_loop(kw, until = lambda: not state_file.exists())
    assert kw._kv_resume is None
    assert not state_file.exists()


def test_put_route_disabling_keep_kv_purges_saved_state(monkeypatch, tmp_path):
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

    store = {}
    monkeypatch.setattr(db, "upsert_app_settings", lambda m: store.update(m))
    monkeypatch.setattr(settings, "_cached_setting", lambda k, d = None: store.get(k, d))
    monkeypatch.setenv(settings.MODEL_IDLE_TTL_ENV_VAR, "600")

    assert settings_route.OpenAIAutoSwitchPayload(enabled = False).auto_unload_idle_seconds is None
    (
        enabled,
        idle,
        keep_kv,
        auto_dl,
        api_only,
        media_idle,
        media_switch,
    ) = settings.set_openai_auto_switch(False, None, False)
    assert settings.AUTO_UNLOAD_IDLE_SETTING_KEY not in store
    assert settings.OPENAI_AUTO_DOWNLOAD_SETTING_KEY not in store
    assert settings.MEDIA_AUTO_UNLOAD_IDLE_SETTING_KEY not in store
    assert settings.MEDIA_AUTO_SWITCH_SETTING_KEY not in store
    assert settings.get_auto_unload_idle_seconds() == 600
    assert (enabled, idle, keep_kv, auto_dl, api_only, media_idle, media_switch) == (
        False,
        600,
        False,
        False,
        False,
        0,
        False,
    )


def test_load_impl_notes_loaded_with_backend_off_loop():
    src = inspect.getsource(inference_route._load_model_impl)
    assert "to_thread(note_model_loaded, llama_backend)" in src


def test_restore_matches_gguf_realpath_across_naming(tmp_path):
    blob = tmp_path / "blob.gguf"
    blob.write_bytes(b"gguf")
    link = tmp_path / "snapshot.gguf"
    try:
        link.symlink_to(blob)
    except OSError:
        pytest.skip("symlinks unsupported on this host")

    backend = _FakeBackend("/hf/snapshots/d7f5", hf_variant = None)
    backend._gguf_path = str(link)
    backend._slot_save_binary = ("/bin/llama-server", 111)
    restored = []
    backend.restore_slots_for_resume = lambda manifest: restored.append(manifest)
    state_file, manifest = _seed_kv_manifest(
        tmp_path, identity = ("unsloth/A-GGUF", None, "unsloth/A-GGUF"), gguf = str(blob)
    )

    kw.restore_kv_resume(backend, manifest)
    assert len(restored) == 1
    assert not state_file.exists()


def test_setter_rejects_idle_below_floor(monkeypatch):
    writes = []
    monkeypatch.setattr(db, "upsert_app_settings", lambda m: writes.append(dict(m)))
    settings._cache.clear()

    with pytest.raises(ValueError, match = "at least 60"):
        settings.set_openai_auto_switch(True, 30)
    assert writes == []
    assert settings.set_openai_auto_switch(True, 0)[1] == 0
    assert settings.set_openai_auto_switch(True, 60)[1] == 60
    assert settings.set_openai_auto_switch(True, 3600)[1] == 3600


def test_media_idle_setting_roundtrip_and_default(monkeypatch):
    # The image/video TTL has its own key and starts off, so a chat TTL never evicts a pipeline.

    store = {}
    monkeypatch.setattr(db, "upsert_app_settings", lambda m: store.update(m))
    monkeypatch.setattr(settings, "_cached_setting", lambda k, d = None: store.get(k, d))
    monkeypatch.delenv(settings.MEDIA_IDLE_TTL_ENV_VAR, raising = False)

    assert settings.set_openai_auto_switch(True, 300)[5] == 0
    assert settings.MEDIA_AUTO_UNLOAD_IDLE_SETTING_KEY not in store
    assert settings.get_media_auto_unload_idle_seconds() == 0
    assert settings.get_auto_unload_idle_seconds() == 300

    assert settings.set_openai_auto_switch(True, 300, None, None, None, 600)[5] == 600
    assert store[settings.MEDIA_AUTO_UNLOAD_IDLE_SETTING_KEY] == 600
    assert settings.get_media_auto_unload_idle_seconds() == 600
    assert settings.get_stored_media_auto_unload_idle_seconds() == 600
    assert settings.set_openai_auto_switch(False, None)[5] == 600
    with pytest.raises(ValueError, match = "at least 60"):
        settings.set_openai_auto_switch(True, None, None, None, None, 30)
    assert settings.set_openai_auto_switch(True, None, None, None, None, 0)[5] == 0
    assert settings.get_media_auto_unload_idle_seconds() == 0


def test_settings_route_reports_the_media_idle_ttl(monkeypatch):
    monkeypatch.setattr(settings_route, "get_stored_media_auto_unload_idle_seconds", lambda: 600)
    monkeypatch.setattr(settings_route, "get_media_auto_unload_idle_seconds", lambda: 600)
    resp = settings_route.get_openai_auto_switch("tester")
    assert resp.media_auto_unload_idle_seconds == 600
    assert resp.media_idle_unload_active is True
    # On a veto the saved number stays but the flag drops, so the UI can show unload as paused.
    monkeypatch.setattr(settings_route, "get_media_auto_unload_idle_seconds", lambda: 0)
    resp = settings_route.get_openai_auto_switch("tester")
    assert resp.media_auto_unload_idle_seconds == 600
    assert resp.media_idle_unload_active is False


def test_put_route_rejects_media_idle_below_floor():
    payload = settings_route.OpenAIAutoSwitchPayload(
        enabled = True, media_auto_unload_idle_seconds = 30
    )
    with pytest.raises(HTTPException) as excinfo:
        settings_route.update_openai_auto_switch(payload, "tester")
    assert excinfo.value.status_code == 400


def test_put_route_rejects_idle_below_floor():
    payload = settings_route.OpenAIAutoSwitchPayload(enabled = True, auto_unload_idle_seconds = 30)
    with pytest.raises(HTTPException) as excinfo:
        settings_route.update_openai_auto_switch(payload, "tester")
    assert excinfo.value.status_code == 400


def test_stored_legacy_idle_below_floor_is_clamped(monkeypatch):
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


def test_normalize_model_override_drops_unusable_fields_and_keeps_the_rest():
    # Bad fields are dropped individually so one stale value does not reject the whole config.
    entry = settings.normalize_model_override(
        {
            "max_seq_length": 8192,
            "kv_cache_dtype": "not_a_dtype",
            "speculative_type": "mtp",
            "spec_draft_n_max": 999,
            "gpu_memory_mode": "auto",
            "gpu_layers": -1,
            "n_cpu_moe": 0,
            "gpu_ids": [1, 1, 0, "2", -5],
            "tensor_parallel": False,
            "llama_extra_args": [],
        }
    )
    assert entry == {"max_seq_length": 8192, "speculative_type": "mtp", "gpu_ids": [1, 0, 2]}


def test_gpu_index_kind_is_stored_only_when_it_is_not_the_legacy_default():
    # Absent means physical, so writing it back would churn every row for no information.
    physical = settings.normalize_model_override({"gpu_ids": [0], "gpu_index_kind": "physical"})
    assert physical == {"gpu_ids": [0]}
    assert settings.normalize_model_override({"gpu_ids": [0]}) == {"gpu_ids": [0]}
    vulkan = settings.normalize_model_override({"gpu_ids": [0], "gpu_index_kind": "vulkan"})
    assert vulkan == {"gpu_ids": [0], "gpu_index_kind": "vulkan"}
    assert settings.normalize_model_override({"gpu_index_kind": "vulkan"}) == {}


@pytest.mark.parametrize(
    "override, expected",
    [
        ({}, "physical"),
        ({"gpu_ids": [0]}, "physical"),
        ({"gpu_ids": [0], "gpu_index_kind": "vulkan"}, "vulkan"),
        ({"gpu_ids": [0], "gpu_index_kind": "metal"}, "physical"),
        ({"gpu_ids": [0], "gpu_index_kind": None}, "physical"),
    ],
)
def test_stored_gpu_index_kind_reads_absent_as_physical(override, expected):
    assert settings.stored_gpu_index_kind(override) == expected


def test_normalize_model_override_rejects_oversized_chat_template():
    small = settings.normalize_model_override({"chat_template_override": "{{ bos }}"})
    assert small["chat_template_override"] == "{{ bos }}"
    # The limit is in bytes, so a multi-byte template under the character limit can exceed it.
    huge = "é" * settings.MAX_CHAT_TEMPLATE_OVERRIDE_BYTES
    assert "chat_template_override" not in settings.normalize_model_override(
        {"chat_template_override": huge}
    )


def test_spec_draft_n_max_only_stored_for_mtp_modes():
    mtp = settings.normalize_model_override({"speculative_type": "mtp", "spec_draft_n_max": 4})
    assert mtp["spec_draft_n_max"] == 4
    # Non-MTP modes ignore the draft count, so storing it would show an edit with no effect.
    ngram = settings.normalize_model_override({"speculative_type": "ngram", "spec_draft_n_max": 4})
    assert "spec_draft_n_max" not in ngram


def test_resolve_fit_max_seq_length_hands_sizing_to_fit_under_manual_auto_layers():
    # Manual GPU memory with Auto layers lets llama.cpp --fit size context, so send the pin or 0.
    override = {"gpu_memory_mode": "manual", "max_seq_length": 8192}
    assert settings.resolve_fit_max_seq_length(override, is_gguf = True) == 0
    assert (
        settings.resolve_fit_max_seq_length(
            {**override, "custom_context_length": 4096}, is_gguf = True
        )
        == 4096
    )
    assert settings.resolve_fit_max_seq_length({**override, "gpu_layers": 20}, is_gguf = True) == 8192
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

    # Safetensors use HF auto-placement, so a GGUF GPU pin would silently move the weights.
    safetensors = settings.model_override_load_kwargs(override, is_gguf = False)
    assert safetensors["max_seq_length"] == 4096
    assert "gpu_layers" not in safetensors
    assert "gpu_ids" not in safetensors
    assert "n_cpu_moe" not in safetensors
    assert "gpu_memory_mode" not in safetensors

    # Every key must be a real LoadRequest field, or the load raises TypeError.
    LoadRequest(model_path = "unsloth/B-GGUF", **gguf)


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
@pytest.mark.parametrize("mode", ["tensor", "pipeline", "data"])
def test_optional_engine_override_preserves_precision_and_gpu_order(engine, mode):
    kwargs = settings.model_override_load_kwargs(
        {
            "engine": engine,
            "engine_precision": "int4",
            "engine_parallelism": mode,
            "gpu_ids": [1, 0],
        },
        is_gguf = False,
    )
    request = LoadRequest(model_path = "unsloth/Qwen2.5-0.5B-Instruct", **kwargs)
    assert request.engine == engine
    assert request.engine_precision == "int4"
    assert request.engine_parallelism == mode
    assert request.gpu_ids == [1, 0]
    assert request.load_in_4bit is False


def test_a_carried_ctx_flag_cannot_outrank_a_freshly_saved_context(monkeypatch):
    # llama-server takes the last -c, so stale pass-through flags must not shadow the edited field.
    _mock_override_store(monkeypatch)
    _put("unsloth/B-GGUF:Q4_K_M", llama_extra_args = ["--ctx-size", "8192", "--top-k", "40"])
    saved = _put("unsloth/B-GGUF:Q4_K_M", max_seq_length = 32768)
    entry = saved.overrides["unsloth/B-GGUF:Q4_K_M"]
    assert entry["max_seq_length"] == 32768
    assert entry["llama_extra_args"] == ["--ctx-size", "8192", "--top-k", "40"]

    backend, rec = _wired(
        monkeypatch, _FakeBackend(None), ("unsloth/B-GGUF", "Q4_K_M", "unsloth/B-GGUF")
    )

    _run_hook("unsloth/B-GGUF")
    request = rec.calls[0]
    assert request.max_seq_length == 32768
    assert request.llama_extra_args == ["--top-k", "40"]


def test_a_matching_explicit_ctx_flag_survives_auto_switch(monkeypatch):
    """A matching stored flag is the opt-in that bypasses the VRAM-fit ceiling."""
    _mock_override_store(monkeypatch)
    _put(
        "unsloth/B-GGUF:Q4_K_M",
        llama_extra_args = ["--ctx-size", "100352", "--spec-draft-n-max", "3"],
        custom_context_length = 100352,
        speculative_type = "mtp",
        spec_draft_n_max = 3,
    )

    backend, rec = _wired(
        monkeypatch, _FakeBackend(None), ("unsloth/B-GGUF", "Q4_K_M", "unsloth/B-GGUF")
    )

    _run_hook("unsloth/B-GGUF")
    request = rec.calls[0]
    assert request.max_seq_length == 100352
    # The context opt-in survives; the speculative flag is stripped as its fields always send.
    assert request.llama_extra_args == ["--ctx-size", "100352"]


def test_a_ctx_flag_saved_from_the_picker_reaches_an_api_load(monkeypatch):
    """#11511: the picker saves its slider context beside a typed -c. Its own load runs at the
    -c (llama.cpp takes the last one), so an API auto-switch of the same row must too, instead
    of stripping the flag as a stale shadow of the slider."""
    _mock_override_store(monkeypatch)
    saved = _put(
        "unsloth/B-GGUF:Q4_K_M",
        llama_extra_args = ["-c", "300000", "--rope-scaling", "yarn"],
        custom_context_length = 262144,
    )
    entry = saved.overrides["unsloth/B-GGUF:Q4_K_M"]
    assert entry["custom_context_length"] == 300000
    assert entry["llama_extra_args"] == ["-c", "300000", "--rope-scaling", "yarn"]

    backend, rec = _wired(
        monkeypatch, _FakeBackend(None), ("unsloth/B-GGUF", "Q4_K_M", "unsloth/B-GGUF")
    )
    _run_hook("unsloth/B-GGUF")
    request = rec.calls[0]
    assert request.max_seq_length == 300000
    assert request.llama_extra_args == ["-c", "300000", "--rope-scaling", "yarn"]


@pytest.mark.parametrize(
    "extra_args, fields, expected",
    [
        (
            ["--ctx-size", "8192", "-c", "65536"],
            {"max_seq_length": 4096, "custom_context_length": 4096},
            {"max_seq_length": 65536, "custom_context_length": 65536},
        ),
        (["-c", "0"], {"custom_context_length": 4096}, {"custom_context_length": 4096}),
        (["-c", "65536"], {"kv_cache_dtype": "q8_0"}, {}),
        (["--top-k", "40"], {"custom_context_length": 4096}, {"custom_context_length": 4096}),
        # Past the stored ceiling the slider value stays, so the flag is still checked on load.
        (["-c", "99999999"], {"custom_context_length": 4096}, {"custom_context_length": 4096}),
    ],
)
def test_a_saved_ctx_flag_sets_only_the_context_fields_sent(
    monkeypatch, extra_args, fields, expected
):
    _mock_override_store(monkeypatch)
    saved = _put("unsloth/B-GGUF:Q4_K_M", llama_extra_args = extra_args, **fields)
    entry = saved.overrides["unsloth/B-GGUF:Q4_K_M"]
    stored = {
        key: entry[key] for key in ("max_seq_length", "custom_context_length") if key in entry
    }
    assert stored == expected


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_context_save_preserves_managed_engine_settings(monkeypatch, engine):
    _mock_override_store(monkeypatch)
    model = "unsloth/model"
    _put(model, engine = engine, engine_parallelism = "pipeline", engine_precision = "fp8")
    saved = _put(model, llama_extra_args = ["-c", "65536"], custom_context_length = 4096)
    entry = saved.overrides[model]
    assert entry["custom_context_length"] == 65536
    assert entry["engine"] == engine
    assert entry["engine_parallelism"] == "pipeline"
    assert entry["engine_precision"] == "fp8"


def test_a_fill_keeps_the_sent_context_when_it_does_not_store_the_flag(monkeypatch):
    """A fill (the localStorage migration) keeps a stored row's flags, so a -c in its payload is
    not what any load will run with and must not rewrite the context."""
    _mock_override_store(monkeypatch)
    _put("unsloth/B-GGUF:Q4_K_M", llama_extra_args = ["--top-k", "7"])
    saved = _put(
        "unsloth/B-GGUF:Q4_K_M",
        llama_extra_args = ["-c", "65536"],
        custom_context_length = 4096,
        fill_absent_fields = True,
    )
    entry = saved.overrides["unsloth/B-GGUF:Q4_K_M"]
    assert entry["llama_extra_args"] == ["--top-k", "7"]
    assert entry["custom_context_length"] == 4096


@pytest.mark.parametrize(
    "stored_max_seq_length",
    ["100352", "not-a-number", 100352.0, True, [100352], {"v": 100352}],
)
def test_a_legacy_override_row_cannot_break_the_loader(stored_max_seq_length):
    """Rows are coerced on write but returned verbatim on read (get_model_overrides),
    so an entry written by an older build, by hand or through the API can hold any
    JSON type. Comparing the context against one must degrade, never raise: a raise
    here fails every auto-switch load of that model with a 500 the user cannot clear.
    A value that is not a plain positive int cannot be confirmed as matching, so the
    shadowing flag is stripped exactly as it was before the opt-in existed.
    """
    from utils.openai_auto_switch_settings import model_override_load_kwargs

    out = model_override_load_kwargs(
        {
            "llama_extra_args": ["--ctx-size", "100352", "--top-k", "40"],
            "max_seq_length": stored_max_seq_length,
        },
        is_gguf = True,
    )
    assert "--ctx-size" not in out["llama_extra_args"]
    assert "--top-k" in out["llama_extra_args"]


@pytest.mark.parametrize("stored_max_seq_length", ["", False, None, 0])
def test_a_falsy_legacy_context_leaves_the_flag_as_the_only_control(stored_max_seq_length):
    """These resolve to "no context field sent", so there is nothing to shadow and
    the pass-through flag stays the user's only way to set the knob -- the same
    answer this path gave before the opt-in existed.
    """
    from utils.openai_auto_switch_settings import model_override_load_kwargs

    out = model_override_load_kwargs(
        {
            "llama_extra_args": ["--ctx-size", "100352", "--top-k", "40"],
            "max_seq_length": stored_max_seq_length,
        },
        is_gguf = True,
    )
    assert out["llama_extra_args"] == ["--ctx-size", "100352", "--top-k", "40"]


def test_load_kwargs_strip_only_the_shadow_groups_the_override_supplies():
    # Strip one group per first-class field actually sent, so unshadowed flags pass through.
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

    kept = settings.model_override_load_kwargs(
        {"llama_extra_args": override["llama_extra_args"]}, is_gguf = True
    )
    assert kept["llama_extra_args"] == override["llama_extra_args"]

    ctx_only = settings.model_override_load_kwargs(
        {"llama_extra_args": override["llama_extra_args"], "max_seq_length": 32768},
        is_gguf = True,
    )
    assert ctx_only["llama_extra_args"] == override["llama_extra_args"][2:]


def test_saved_parallel_slots_reach_an_api_load(monkeypatch):
    backend, rec = _wired(
        monkeypatch, _FakeBackend(None), ("unsloth/B-GGUF", "Q4_K_M", "unsloth/B-GGUF")
    )
    monkeypatch.setattr(settings, "get_model_override", lambda mid: {"n_parallel": 8})

    _run_hook("unsloth/B-GGUF")
    assert rec.calls[0].n_parallel == 8


def test_parallel_slots_are_stored_and_reach_either_backend():
    override = settings.normalize_model_override({"n_parallel": 8})
    assert override == {"n_parallel": 8}
    for bad in (None, 0, -1, settings.PARALLEL_SLOTS_MAX + 1, "many", True):
        assert "n_parallel" not in settings.normalize_model_override({"n_parallel": bad})

    gguf = settings.model_override_load_kwargs(override, is_gguf = True)
    assert gguf["n_parallel"] == 8
    safetensors = settings.model_override_load_kwargs(override, is_gguf = False)
    assert safetensors["n_parallel"] == 8
    for flag in ("n_batch", "n_ubatch"):
        stored = settings.normalize_model_override({flag: 512})
        assert flag not in settings.model_override_load_kwargs(stored, is_gguf = False)
    LoadRequest(model_path = "unsloth/B-GGUF", **gguf)
    LoadRequest(model_path = "unsloth/B", **safetensors)


def test_override_route_persists_parallel_slots(override_store):
    # The picker's mirror must carry the field, or a slot-count-only change saves as empty.
    resp = _put("unsloth/B-GGUF:Q4_K_M", n_parallel = 8)
    assert resp.overrides["unsloth/B-GGUF:Q4_K_M"] == {"n_parallel": 8}


def test_eviction_cleanup_clears_mirrored_fields_but_keeps_launch_flags(override_store):
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
    backend, rec = _wired(
        monkeypatch, _FakeBackend(None), ("unsloth/B-GGUF", "Q4_K_M", "unsloth/B-GGUF")
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
    backend, rec = _wired(
        monkeypatch, _FakeBackend(None), ("unsloth/B-GGUF", "Q4_K_M", "unsloth/B-GGUF")
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

    gone = _put("unsloth/B-GGUF", llama_extra_args = [])
    assert "unsloth/B-GGUF" not in gone.overrides


def test_override_found_under_a_concrete_path_with_variant(monkeypatch):
    # A local folder resolves to repo id + path; settings saved against the path must be found.
    backend, rec = _wired(
        monkeypatch,
        _FakeBackend(None),
        ("/models/local/Qwen3-8B-Q4_K_M.gguf", "Q4_K_M", "unsloth/Qwen3-8B-GGUF"),
    )
    stored = {"/models/local/Qwen3-8B-Q4_K_M.gguf:Q4_K_M": {"max_seq_length": 8192}}
    monkeypatch.setattr(settings, "get_model_override", lambda mid: stored.get(mid, {}))

    _run_hook("unsloth/Qwen3-8B-GGUF")
    assert rec.calls[0].max_seq_length == 8192


def test_path_qualified_override_beats_repo_qualified(monkeypatch):
    backend, rec = _wired(
        monkeypatch, _FakeBackend(None), ("/models/local/x.gguf", "Q4_K_M", "unsloth/B-GGUF")
    )
    stored = {
        "unsloth/B-GGUF:Q4_K_M": {"max_seq_length": 8192},
        "/models/local/x.gguf:Q4_K_M": {"max_seq_length": 1024},
    }
    monkeypatch.setattr(settings, "get_model_override", lambda mid: stored.get(mid, {}))

    _run_hook("unsloth/B-GGUF")
    assert rec.calls[0].max_seq_length == 1024


def test_first_quant_save_keeps_legacy_bare_repo_launch_flags(override_store):
    settings.set_model_override("unsloth/B-GGUF", llama_extra_args = ["--flash-attn"])

    resp = _put("unsloth/B-GGUF:Q4_K_M", max_seq_length = 4096)
    entry = resp.overrides["unsloth/B-GGUF:Q4_K_M"]
    assert entry["max_seq_length"] == 4096
    assert entry["llama_extra_args"] == ["--flash-attn"]


def test_bare_repo_carry_over_does_not_split_a_windows_path(override_store):
    settings.set_model_override("C", llama_extra_args = ["--flash-attn"])

    resp = _put(r"C:\models\x.gguf", max_seq_length = 4096)
    assert "llama_extra_args" not in resp.overrides[r"C:\models\x.gguf"]


def test_windows_path_with_quant_still_carries_over(override_store):
    settings.set_model_override(r"C:\models\x.gguf", llama_extra_args = ["--flash-attn"])

    resp = _put(r"C:\models\x.gguf:Q4_K_M", max_seq_length = 4096)
    assert resp.overrides[r"C:\models\x.gguf:Q4_K_M"]["llama_extra_args"] == ["--flash-attn"]


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

    backend, rec = _wired(
        monkeypatch, _FakeBackend(None), (_LEGACY_SNAPSHOT, "Q4_K_M", "unsloth/B-GGUF")
    )

    _run_hook("unsloth/B-GGUF:Q4_K_M")
    assert rec.calls[0].max_seq_length == 32768


def test_the_retired_snapshot_path_entry_hands_over_its_launch_flags(override_store):
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
    # A row the picker keys by path must not be shadowed, so the last saved spelling survives.
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


def test_a_remove_reports_every_key_it_cleared(override_store):
    settings.set_model_override("unsloth/B-GGUF", max_seq_length = 2048)
    settings.set_model_override("unsloth/B-GGUF:Q4_K_M", max_seq_length = 32768)
    settings.set_model_override(f"{_LEGACY_SNAPSHOT}:Q4_K_M", max_seq_length = 4096)

    resp = _put("unsloth/B-GGUF:Q4_K_M", remove = True, llama_extra_args = [])
    assert resp.removed_keys == [
        "unsloth/B-GGUF:Q4_K_M",
        "unsloth/B-GGUF",
        f"{_LEGACY_SNAPSHOT}:Q4_K_M",
    ]
    assert resp.overrides == {}


def test_a_remove_reports_the_legacy_label_of_a_loose_gguf(override_store):
    settings.set_model_override("/models/Qwen3-4B-Q4_K_M.gguf:q4_k_m", max_seq_length = 4096)

    resp = _put("/models/Qwen3-4B-Q4_K_M.gguf", remove = True, llama_extra_args = [])
    assert resp.removed_keys == [
        "/models/Qwen3-4B-Q4_K_M.gguf",
        "/models/Qwen3-4B-Q4_K_M.gguf:q4_k_m",
    ]
    assert resp.overrides == {}


def test_a_remove_leaves_the_bare_row_for_a_remaining_quant_and_says_so(override_store):
    settings.set_model_override("unsloth/B-GGUF", max_seq_length = 2048)
    settings.set_model_override("unsloth/B-GGUF:Q4_K_M", max_seq_length = 32768)
    settings.set_model_override("unsloth/B-GGUF:Q8_0", max_seq_length = 16384)

    resp = _put("unsloth/B-GGUF:Q4_K_M", remove = True, llama_extra_args = [])
    assert resp.removed_keys == ["unsloth/B-GGUF:Q4_K_M"]
    assert set(resp.overrides) == {"unsloth/B-GGUF", "unsloth/B-GGUF:Q8_0"}


def test_a_save_reports_no_removed_keys(override_store):
    resp = _put("unsloth/B-GGUF:Q4_K_M", max_seq_length = 8192)
    assert resp.removed_keys == []


def test_a_fill_reports_no_removed_keys(override_store):
    settings.set_model_override(f"{_LEGACY_SNAPSHOT}:Q4_K_M", max_seq_length = 4096)

    resp = _put("unsloth/B-GGUF:Q4_K_M", max_seq_length = 32768, fill_absent_fields = True)
    assert resp.removed_keys == []


def test_retiring_a_spelling_leaves_every_other_entry_alone(override_store):
    settings.set_model_override("/models/local/x.gguf:Q4_K_M", max_seq_length = 1024)
    settings.set_model_override(f"{_LEGACY_SNAPSHOT}:Q8_0", max_seq_length = 2048)
    settings.set_model_override(_LEGACY_SNAPSHOT, max_seq_length = 4096)

    resp = _put("unsloth/B-GGUF:Q4_K_M", max_seq_length = 32768)
    assert resp.overrides["/models/local/x.gguf:Q4_K_M"]["max_seq_length"] == 1024
    assert resp.overrides[f"{_LEGACY_SNAPSHOT}:Q8_0"]["max_seq_length"] == 2048
    assert resp.overrides[_LEGACY_SNAPSHOT]["max_seq_length"] == 4096


def test_a_fill_never_labels_the_server_s_gpu_pin_with_this_browser_s_index_space(override_store):
    # The stored pin is physical while this offers Vulkan ordinals, so qualifier and ids travel together.
    settings.set_model_override("unsloth/B-GGUF:Q4_K_M", gpu_ids = [0, 1])

    resp = _put(
        "unsloth/B-GGUF:Q4_K_M",
        gpu_ids = [2],
        gpu_index_kind = "vulkan",
        max_seq_length = 32768,
        fill_absent_fields = True,
    )
    entry = resp.overrides["unsloth/B-GGUF:Q4_K_M"]
    assert entry["gpu_ids"] == [0, 1]
    assert "gpu_index_kind" not in entry
    assert settings.stored_gpu_index_kind(entry) == "physical"
    assert entry["max_seq_length"] == 32768


def test_the_one_time_fill_retires_nothing(override_store):
    # The migration mirrors both spellings, so its second write must not delete its first.
    settings.set_model_override(f"{_LEGACY_SNAPSHOT}:Q4_K_M", max_seq_length = 4096)

    resp = _put("unsloth/B-GGUF:Q4_K_M", max_seq_length = 32768, fill_absent_fields = True)
    assert resp.overrides[f"{_LEGACY_SNAPSHOT}:Q4_K_M"]["max_seq_length"] == 4096
    assert resp.overrides["unsloth/B-GGUF:Q4_K_M"]["max_seq_length"] == 32768


def test_a_fill_never_creates_a_snapshot_path_key_over_a_repo_id_entry(override_store):
    # Fill the existing entry, since a new path key would shadow newer server config on API loads.
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
    backend, rec = _wired(
        monkeypatch, _FakeBackend(None), ("unsloth/B-GGUF", "Q4_K_M", "unsloth/B-GGUF")
    )
    monkeypatch.setattr(
        settings,
        "get_model_override",
        lambda mid: {"gpu_ids": [0, 1], "max_seq_length": 4096},
    )

    async def _unusable(ids, index_kind = "physical"):
        return False

    monkeypatch.setattr(inference_route, "_override_gpu_ids_still_resolve", _unusable)

    _run_hook("unsloth/B-GGUF")
    req = rec.calls[0]
    assert not req.gpu_ids
    assert req.max_seq_length == 4096


def test_usable_gpu_ids_are_kept(monkeypatch):
    backend, rec = _wired(
        monkeypatch, _FakeBackend(None), ("unsloth/B-GGUF", "Q4_K_M", "unsloth/B-GGUF")
    )
    monkeypatch.setattr(settings, "get_model_override", lambda mid: {"gpu_ids": [0, 1]})

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
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda: True))
    monkeypatch.setattr(
        LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: "/bin/llama-server")
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_get_gpu_memory", staticmethod(lambda binary: [(0, 8192)])
    )
    assert asyncio.run(inference_route._override_gpu_ids_still_resolve([0], "vulkan")) is True
    assert asyncio.run(inference_route._override_gpu_ids_still_resolve([7], "vulkan")) is False
    assert asyncio.run(inference_route._override_gpu_ids_still_resolve([0, 1], "vulkan")) is False


def test_vulkan_probe_without_a_binary_does_not_block_the_load(monkeypatch):
    # Nothing to probe with, and refusing would drop a valid pin on every load.

    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda: True))
    monkeypatch.setattr(LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: None))
    assert asyncio.run(inference_route._override_gpu_ids_still_resolve([0], "vulkan")) is True


def _pin_resolves_on_a_host_with_device_0(monkeypatch):
    """Make device 0 exist, so the index-kind check is the only thing under test.

    Without this the whole helper falls to its `except: return False` on a CPU-only
    runner, which is how the mismatch cases below would pass for the wrong reason and
    how the matching case failed on CI while passing on a GPU box.
    """
    import utils.hardware as hardware_pkg
    from utils.hardware import DeviceType
    from utils.hardware import hardware as hardware_mod

    monkeypatch.setattr(hardware_pkg, "get_device", lambda: DeviceType.CUDA)
    monkeypatch.setattr(
        hardware_mod, "resolve_requested_gpu_ids", lambda ids, is_vulkan = False: list(ids)
    )


def test_a_pin_written_in_the_other_index_space_is_unusable(monkeypatch):
    # Ordinal 0 and physical device 0 both exist, so a stale pin passes every presence check.

    _pin_resolves_on_a_host_with_device_0(monkeypatch)
    monkeypatch.setattr(
        LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: "/bin/llama-server")
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_get_gpu_memory", staticmethod(lambda binary: [(0, 8192)])
    )
    for installed_is_vulkan, stored_kind in (
        (True, "physical"),
        (False, "vulkan"),
    ):
        monkeypatch.setattr(
            LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda: installed_is_vulkan)
        )
        assert (
            asyncio.run(inference_route._override_gpu_ids_still_resolve([0], stored_kind)) is False
        ), (installed_is_vulkan, stored_kind)


def test_a_rocm_pin_survives_while_the_backend_is_still_rocm(monkeypatch):
    _pin_resolves_on_a_host_with_device_0(monkeypatch)
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda: False))
    assert asyncio.run(inference_route._override_gpu_ids_still_resolve([0], "physical")) is True


def test_default_save_preserves_flags_instead_of_removing(override_store):
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
    settings.set_model_override("unsloth/B-GGUF", max_seq_length = 4096)
    resp = _put("unsloth/B-GGUF")
    assert "unsloth/B-GGUF" not in resp.overrides


def test_remove_false_with_real_fields_saves_normally(override_store):
    resp = _put("unsloth/B-GGUF", remove = False, max_seq_length = 8192)
    assert resp.overrides["unsloth/B-GGUF"]["max_seq_length"] == 8192


def test_override_lookup_falls_back_to_case_insensitive(override_store):
    settings.set_model_override("unsloth/qwen3-8b-gguf:q4_k_m", max_seq_length = 8192)
    got = settings.get_model_override("unsloth/Qwen3-8B-GGUF:Q4_K_M")
    assert got["max_seq_length"] == 8192


def test_exact_override_match_beats_a_case_variant(override_store):
    settings.set_model_override("/models/foo.gguf", max_seq_length = 1024)
    settings.set_model_override("/models/Foo.gguf", max_seq_length = 8192)
    assert settings.get_model_override("/models/Foo.gguf")["max_seq_length"] == 8192
    assert settings.get_model_override("/models/foo.gguf")["max_seq_length"] == 1024


def test_ambiguous_case_fallback_matches_nothing(override_store):
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
    # A monitor label must never take a load down, whatever the stand-in headers return.
    from unittest.mock import MagicMock

    assert inference_route._request_used_api_key(MagicMock()) is False


def test_case_fallback_never_applies_to_a_posix_path(override_store):
    # Two casings are two models on Linux, so a near miss loads defaults, not the other's pin.
    settings.set_model_override("/models/foo.gguf", max_seq_length = 8192, gpu_ids = [1])
    assert settings.get_model_override("/models/Foo.gguf") == {}
    assert settings.get_model_override("/models/foo.gguf")["max_seq_length"] == 8192


def test_case_fallback_does_apply_to_a_windows_path(override_store):
    settings.set_model_override(r"c:\models\foo.gguf", max_seq_length = 8192)
    assert settings.get_model_override(r"C:\models\FOO.gguf")["max_seq_length"] == 8192
    assert settings.get_model_override("C:/Models/Foo.gguf")["max_seq_length"] == 8192


def test_case_fallback_applies_to_unc_and_wsl_drive_paths(override_store):
    settings.set_model_override(r"\\server\share\foo.gguf", max_seq_length = 4096)
    settings.set_model_override("/mnt/c/models/bar.gguf", max_seq_length = 2048)
    assert settings.get_model_override(r"\\Server\Share\FOO.gguf")["max_seq_length"] == 4096
    assert settings.get_model_override("/mnt/C/Models/Bar.gguf")["max_seq_length"] == 2048


def test_a_plain_posix_path_under_mnt_stays_case_sensitive(override_store):
    settings.set_model_override("/mnt/data/models/foo.gguf", max_seq_length = 8192)
    assert settings.get_model_override("/mnt/data/models/Foo.gguf") == {}


def test_an_ambiguous_windows_case_fallback_still_matches_nothing(override_store):
    # Two keys folding to one has no single answer, so the load takes defaults.
    settings.set_model_override(r"c:\models\foo.gguf", max_seq_length = 1024)
    settings.set_model_override("C:/models/FOO.gguf", max_seq_length = 8192)
    assert settings.get_model_override(r"C:\Models\Foo.gguf") == {}


def test_case_fallback_still_covers_repo_ids(override_store):
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
    settings.set_model_override("/models/foo", llama_extra_args = ["--flash-attn"])
    resp = _put("/models/foo:bar.gguf", max_seq_length = 4096)
    assert "llama_extra_args" not in resp.overrides["/models/foo:bar.gguf"]


def test_unknown_quant_label_on_a_gguf_still_carries_flags_over(override_store):
    # A .gguf with no quant token is labelled by its stem, so the UI saves ":custom".
    settings.set_model_override("/models/custom.gguf", llama_extra_args = ["--flash-attn"])
    resp = _put("/models/custom.gguf:custom", max_seq_length = 4096)
    assert resp.overrides["/models/custom.gguf:custom"]["llama_extra_args"] == ["--flash-attn"]


def test_bpw_qualified_variants_still_carry_flags_over(override_store):
    settings.set_model_override("unsloth/Repo-GGUF", llama_extra_args = ["--flash-attn"])
    resp = _put("unsloth/Repo-GGUF:IQ4_XS-3.53bpw", max_seq_length = 4096)
    assert resp.overrides["unsloth/Repo-GGUF:IQ4_XS-3.53bpw"]["llama_extra_args"] == [
        "--flash-attn"
    ]


def test_a_posix_path_variant_folds_while_the_path_does_not(override_store):
    settings.set_model_override("/models/Foo:q4_k_m", max_seq_length = 8192)
    assert settings.get_model_override("/models/Foo:Q4_K_M")["max_seq_length"] == 8192
    assert settings.get_model_override("/models/foo:Q4_K_M") == {}


def test_an_unknown_gguf_label_is_reachable_in_either_casing(override_store):
    settings.set_model_override("/models/CustomModel.gguf:custommodel", max_seq_length = 8192)
    got = settings.get_model_override("/models/CustomModel.gguf:CustomModel")
    assert got["max_seq_length"] == 8192
    assert settings.get_model_override("/models/custommodel.gguf:CustomModel") == {}


def test_a_posix_colon_filename_is_not_folded_as_a_variant(override_store):
    settings.set_model_override("/models/foo:bar.gguf", max_seq_length = 8192)
    assert settings.get_model_override("/models/foo:Bar.gguf") == {}


def test_a_suffix_the_scanner_would_not_derive_carries_nothing_over(override_store):
    # Only the scanner's exact label is accepted, so a stray colon suffix reaches nothing.
    settings.set_model_override("/models/custom.gguf", llama_extra_args = ["--flash-attn"])
    resp = _put("/models/custom.gguf:something-else", max_seq_length = 4096)
    assert "llama_extra_args" not in resp.overrides["/models/custom.gguf:something-else"]


def test_unknown_quant_label_carries_over_for_a_windows_path(override_store):
    settings.set_model_override(r"C:\models\custom.gguf", llama_extra_args = ["--flash-attn"])
    resp = _put(r"C:\models\custom.gguf:custom", max_seq_length = 4096)
    assert resp.overrides[r"C:\models\custom.gguf:custom"]["llama_extra_args"] == ["--flash-attn"]


def test_real_quant_suffix_on_a_path_still_carries_flags_over(override_store):
    settings.set_model_override("/models/x.gguf", llama_extra_args = ["--flash-attn"])
    resp = _put("/models/x.gguf:Q4_K_M", max_seq_length = 4096)
    assert resp.overrides["/models/x.gguf:Q4_K_M"]["llama_extra_args"] == ["--flash-attn"]


def test_load_retries_without_gpu_ids_when_the_loader_rejects_the_pin(monkeypatch):
    # The pre-flight check can't mirror every loader rule, and a stale pin must not block a load.

    backend, rec = _wired(
        monkeypatch, _FakeBackend(None), ("unsloth/B-GGUF", "Q4_K_M", "unsloth/B-GGUF")
    )
    monkeypatch.setattr(
        settings, "get_model_override", lambda mid: {"gpu_ids": [0], "max_seq_length": 4096}
    )

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
    backend, rec = _wired(
        monkeypatch, _FakeBackend(None), ("unsloth/B-GGUF", "Q4_K_M", "unsloth/B-GGUF")
    )
    monkeypatch.setattr(settings, "get_model_override", lambda mid: {"gpu_ids": [0]})

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
    settings.set_model_override("unsloth/B-GGUF:Q4_K_M", max_seq_length = 8192)
    assert settings.get_model_override("unsloth/b-gguf:q4_k_m")["max_seq_length"] == 8192

    _put("unsloth/b-gguf:q4_k_m", remove = True)
    assert settings.get_model_overrides() == {}
    assert settings.get_model_override("unsloth/B-GGUF:Q4_K_M") == {}


def test_save_updates_the_existing_case_variant_instead_of_forking_it(override_store):
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
    settings.set_model_override(
        "/models/Qwen3-8B-Q4_K_M.gguf:q4_k_m",
        max_seq_length = 8192,
    )
    _put("/models/Qwen3-8B-Q4_K_M.gguf", remove = True)
    assert settings.get_model_override("/models/Qwen3-8B-Q4_K_M.gguf:Q4_K_M") == {}
    assert settings.get_model_overrides() == {}


def test_forget_clears_the_bare_repo_entry_the_quant_inherited_from(override_store):
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
    # The bare entry backs other quants, so forgetting Q4 must not strip it while Q8 remains.
    settings.set_model_override("unsloth/B-GGUF", max_seq_length = 8192)
    settings.set_model_override("unsloth/B-GGUF:Q8_0", max_seq_length = 2048)
    _put("unsloth/B-GGUF:Q4_K_M", max_seq_length = 4096)
    _put("unsloth/B-GGUF:Q4_K_M", remove = True)
    assert settings.get_model_override("unsloth/B-GGUF") == {"max_seq_length": 8192}
    assert settings.get_model_override("unsloth/B-GGUF:Q8_0") == {"max_seq_length": 2048}


def test_forget_clears_every_spelling_of_one_model(override_store):
    # Clear both spellings, or the survivor becomes the fold match and reapplies the forgotten flags.
    settings.set_model_override("unsloth/B-GGUF:Q4_K_M", max_seq_length = 8192)
    settings.set_model_override("unsloth/b-gguf:q4_k_m", max_seq_length = 8192)
    _put("unsloth/B-GGUF:Q4_K_M", remove = True)
    assert settings.get_model_overrides() == {}
    assert settings.get_model_override("unsloth/B-GGUF:Q4_K_M") == {}


def test_forget_of_one_windows_spelling_clears_the_other(override_store):
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
    settings.set_model_override("unsloth/b-gguf:q4_k_m", max_seq_length = 8192)
    _put("unsloth/B-GGUF", remove = True)
    assert settings.get_model_override("unsloth/b-gguf:q4_k_m")["max_seq_length"] == 8192


def test_a_tag_that_names_no_quant_resolves_to_the_repo(monkeypatch):
    # org/model:latest must resolve a local GGUF, while a real quant not on disk must still miss.
    from core.inference.local_model_resolver import _LocalGgufEntry

    entry = _LocalGgufEntry("org/model", "/srv/models/org--model", ("Q4_K_M",))
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
    # Every worker exits here, so invalidate for Hub UI downloads too, not just API auto-download.

    class _Registry:
        def cancel_requested(self, key):
            return False

        def drop_process(self, key, proc):
            return True

        def get_job_metadata(self, key):
            return None

        def set_job(self, key, state):
            self.state = state

    resolver._scan = (time.monotonic(), {"already-here": "entry"})
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
    assert stamp < 0.0 and resolver._snapshot_is_trusted(stamp, time.monotonic())
    assert entries == {"already-here": "entry"}


def test_invalidating_keeps_the_entries_it_already_had(monkeypatch):
    # The request path never scans, so keep entries until the rebuild lands; downloads only add.
    entry = resolver._LocalGgufEntry("org/old", "/srv/models/org--old", ("Q4_K_M",))
    monkeypatch.setattr(resolver, "_scan", (time.monotonic(), {"org/old": entry}))
    resolver.invalidate_index()
    assert resolver._scan[0] == 0.0
    assert resolver.resolve_local_gguf("org/old", allow_scan = False) == (
        "/srv/models/org--old",
        "Q4_K_M",
        "org/old",
    )
    assert resolver.resolve_trusted_cached_local_gguf("org/old") is None


def test_trusted_cache_rechecks_snapshot_after_freshness(monkeypatch):
    entry = resolver._LocalGgufEntry("org/old", "/custom/org--old", ("Q4_K_M",))
    snapshot = (time.monotonic(), {"org/old": entry})
    monkeypatch.setattr(resolver, "_scan", snapshot)
    invalidated = False

    def _invalidate_while_deciding_trust():
        nonlocal invalidated
        if not invalidated:
            invalidated = True
            resolver.invalidate_index()
        return snapshot[0]

    monkeypatch.setattr(resolver.time, "monotonic", _invalidate_while_deciding_trust)

    assert resolver.resolve_trusted_cached_local_gguf("org/old") is None


def test_async_scan_folder_routes_offload_storage_and_invalidation(monkeypatch):
    event_loop_thread = threading.get_ident()
    calls = []

    def _add(path, recursive = None):
        calls.append(("add", threading.get_ident()))
        return {"id": 7, "path": path, "created_at": "fake"}, True

    def _remove(folder_id):
        calls.append((f"remove:{folder_id}", threading.get_ident()))
        return True

    def _invalidate():
        calls.append(("invalidate", threading.get_ident()))

    monkeypatch.setattr("storage.studio_db.add_scan_folder_with_status", _add)
    monkeypatch.setattr("storage.studio_db.remove_scan_folder", _remove)
    monkeypatch.setattr(resolver, "invalidate_index", _invalidate)
    monkeypatch.setattr(resolver, "warm_index_soon", lambda: None)

    async def _run():
        folder = await model_routes.add_scan_folder_endpoint(
            SimpleNamespace(path = "/models/custom", recursive = None), current_subject = "tester"
        )
        removed = await model_routes.remove_scan_folder_endpoint(7, current_subject = "tester")
        return folder, removed

    folder, removed = asyncio.run(_run())

    assert folder["path"] == "/models/custom"
    assert removed == {"ok": True}
    assert [name for name, _ in calls] == ["add", "invalidate", "remove:7", "invalidate"]
    assert all(thread_id != event_loop_thread for _, thread_id in calls)


def test_scan_folder_removal_revokes_additions_only_cache_trust(monkeypatch):
    from hub.services.models import local_inventory

    entry = resolver._LocalGgufEntry("org/old", "/custom/org--old", ("Q4_K_M",))
    monkeypatch.setattr(resolver, "_scan", (time.monotonic(), {"org/old": entry}))
    removed = []
    warmed = []

    def _remove(folder_id):
        removed.append(folder_id)
        return True

    monkeypatch.setattr(resolver, "warm_index_soon", lambda: warmed.append(1))
    monkeypatch.setattr("storage.studio_db.remove_scan_folder", _remove)
    monkeypatch.setattr(local_inventory, "remove_scan_folder", _remove)

    resolver._scan = (time.monotonic(), {"org/old": entry})
    resolver.invalidate_index(additions_only = True)
    assert resolver.resolve_trusted_cached_local_gguf("org/old") is not None
    asyncio.run(model_routes.remove_scan_folder_endpoint(7, current_subject = "tester"))
    assert resolver.resolve_trusted_cached_local_gguf("org/old") is None

    resolver._scan = (time.monotonic(), {"org/old": entry})
    resolver.invalidate_index(additions_only = True)
    assert resolver.resolve_trusted_cached_local_gguf("org/old") is not None
    assert local_inventory.remove_scan_folder_response(8) == {"ok": True}
    assert resolver.resolve_trusted_cached_local_gguf("org/old") is None
    assert removed == [7, 8]
    assert warmed == [1, 1]


def test_scan_folder_storage_removals_report_if_a_row_changed(monkeypatch):
    from hub.storage import scan_folders
    class _Connection:
        def __init__(self, rowcount):
            self.rowcount = rowcount
            self.committed = False
            self.closed = False

        def execute(self, _sql, _params):
            return SimpleNamespace(rowcount = self.rowcount)

        def commit(self):
            self.committed = True

        def close(self):
            self.closed = True

    for storage in (studio_db, scan_folders):
        for rowcount, expected in ((1, True), (0, False)):
            connection = _Connection(rowcount)
            monkeypatch.setattr(storage, "get_connection", lambda connection = connection: connection)

            assert storage.remove_scan_folder(7) is expected
            assert connection.committed
            assert connection.closed


def test_out_of_range_scan_folder_ids_remove_nothing(monkeypatch):
    import sqlite3

    from hub.storage import scan_folders
    for storage in (studio_db, scan_folders):
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE TABLE scan_folders (id INTEGER PRIMARY KEY, path TEXT)")
        monkeypatch.setattr(storage, "get_connection", lambda connection = connection: connection)
        for folder_id in (2**63, -(2**63) - 1):
            assert storage.remove_scan_folder(folder_id) is False


def test_noop_scan_folder_removals_do_not_invalidate_the_index(monkeypatch):
    from hub.services.models import local_inventory

    invalidated = []
    warmed = []
    monkeypatch.setattr(resolver, "invalidate_index", lambda: invalidated.append(1))
    monkeypatch.setattr(resolver, "warm_index_soon", lambda: warmed.append(1))
    monkeypatch.setattr("storage.studio_db.remove_scan_folder", lambda _folder_id: False)
    monkeypatch.setattr(local_inventory, "remove_scan_folder", lambda _folder_id: False)

    assert asyncio.run(model_routes.remove_scan_folder_endpoint(404, current_subject = "tester")) == {
        "ok": True
    }
    assert local_inventory.remove_scan_folder_response(404) == {"ok": True}
    assert invalidated == []
    assert warmed == []


def test_a_bare_local_id_takes_the_quant_a_plain_load_would(monkeypatch, tmp_path):
    # Variants sort largest first, so a bare id must not resolve to the head and risk OOM.
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

    monkeypatch.setattr(resolver, "_build_index", dict)
    resolver._index()
    assert not resolver.recently_downloaded("org/fresh")


def test_a_finished_dataset_is_not_recorded_as_a_local_model(monkeypatch):
    # This is shared with dataset downloads, which must not be noted as local models.

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
    # Paths are case-sensitive, so only repo aliases may match case-insensitively.

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
    for spelling in ("/home/me/models/x.gguf", "C:\\models\\x.gguf", "//host/share/x.gguf"):
        assert resolver._is_abs_path_id(spelling) is True, spelling
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

    for repo_id in ("org/Repo-GGUF", "Repo", "org/Repo-GGUF:Q4_K_M"):
        assert resolver._is_abs_path_id(repo_id) is False, repo_id


def test_fill_absent_fields_put_never_replaces_a_newer_server_value(override_store):
    """The one-time localStorage backfill reads the override map once and then
    writes each model in turn, so a save by another tab during that pass was
    overwritten by this browser's older copy. fill_absent_fields writes only what
    the entry lacks, so every value already on the server wins."""
    newer = settings_route.ModelOverridePayload(model_id = "unsloth/B-GGUF", max_seq_length = 8192)
    settings_route.update_openai_auto_switch_override(newer, "tester")

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
    store = _mock_override_store(monkeypatch)

    legacy = settings_route.ModelOverridePayload(
        model_id = "unsloth/B-GGUF:Q4_K_M",
        llama_extra_args = ["--flash-attn"],
        max_seq_length = 8192,
    )
    settings_route.update_openai_auto_switch_override(legacy, "tester")
    store[settings.MODEL_OVERRIDES_SETTING_KEY]["unsloth/B-GGUF:Q4_K_M"]["mlx_kv_bits"] = 8

    backfill = settings_route.ModelOverridePayload(
        model_id = "unsloth/B-GGUF:Q4_K_M",
        max_seq_length = 2048,
        custom_context_length = 32768,
        kv_cache_dtype = "q8_0",
        mlx_kv_quant = "tq-4",
        speculative_type = "ngram",
        gpu_ids = [0, 1],
        fill_absent_fields = True,
    )
    resp = settings_route.update_openai_auto_switch_override(backfill, "tester")
    entry = resp.overrides["unsloth/B-GGUF:Q4_K_M"]
    assert entry["max_seq_length"] == 8192
    assert entry["llama_extra_args"] == ["--flash-attn"]
    assert entry["custom_context_length"] == 32768
    assert entry["kv_cache_dtype"] == "q8_0"
    assert entry["speculative_type"] == "ngram"
    assert entry["gpu_ids"] == [0, 1]
    assert settings.model_override_load_kwargs(entry, is_gguf = False)["mlx_kv_quant"] == "8"
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

    empty = settings_route.ModelOverridePayload(
        model_id = "Unsloth/B-GGUF:Q4_K_M", fill_absent_fields = True
    )
    resp2 = settings_route.update_openai_auto_switch_override(empty, "tester")
    assert resp2.overrides["Unsloth/B-GGUF:Q4_K_M"]["max_seq_length"] == 8192

    with pytest.raises(HTTPException) as excinfo:
        _put("Unsloth/B-GGUF:Q4_K_M", remove = True, fill_absent_fields = True)
    assert excinfo.value.status_code == 400


def test_fill_absent_fields_does_not_break_the_empty_payload_removal(override_store):
    """fill_absent_fields is a write mode, not a saved field: leaving it in the dumped
    payload would make every request look non-empty and silently retire the legacy
    "a payload carrying only model_id forgets this model" contract."""
    stored = settings_route.ModelOverridePayload(model_id = "unsloth/B-GGUF", max_seq_length = 4096)
    settings_route.update_openai_auto_switch_override(stored, "tester")
    empty = settings_route.ModelOverridePayload(model_id = "unsloth/B-GGUF")
    resp = settings_route.update_openai_auto_switch_override(empty, "tester")
    assert "unsloth/B-GGUF" not in resp.overrides


def test_map_entry_fill_reads_and_writes_in_one_transaction(tmp_path, monkeypatch):
    """The real store, not the in-memory stand-in: the read has to share the write's
    transaction, or a concurrent writer still slips between them."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(db, "_schema_ready", set())

    key = "test_map_entry_create"
    assert db.upsert_app_setting_map_entry(key, "a", {"v": 1}) == {"a": {"v": 1}}
    assert db.upsert_app_setting_map_entry(key, "a", {"v": 2}, fill_absent_fields = True) == {
        "a": {"v": 1}
    }
    assert db.get_app_setting(key) == {"a": {"v": 1}}
    assert db.upsert_app_setting_map_entry(key, "a", {"v": 2, "w": 7}, fill_absent_fields = True) == {
        "a": {"v": 1, "w": 7}
    }
    assert db.get_app_setting(key) == {"a": {"v": 1, "w": 7}}
    assert db.upsert_app_setting_map_entry(key, "b", {"v": 3}, fill_absent_fields = True) == {
        "a": {"v": 1, "w": 7},
        "b": {"v": 3},
    }
    # A fill never deletes, even with nothing to store.
    assert db.upsert_app_setting_map_entry(key, "a", None, fill_absent_fields = True) == {
        "a": {"v": 1, "w": 7},
        "b": {"v": 3},
    }
    db.upsert_app_setting_map_entry(key, "a", {"v": 9})
    db.upsert_app_setting_map_entry(key, "b", None)
    assert db.get_app_setting(key) == {"a": {"v": 9}}


def test_a_first_writer_entry_collapses_a_conflicting_claim_inside_the_write(tmp_path, monkeypatch):
    """The credential-provenance rule, decided where the race is. A caller that reads the map,
    sees nothing, and then writes loses to a second caller doing the same with a different
    identity: both see "absent" and the last one stores its own claim over the first. So the
    comparison belongs inside this transaction."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(db, "_schema_ready", set())

    key = "test_map_entry_first_writer"
    first = {"at": 100.0, "by": "identity-a"}
    assert db.upsert_app_setting_map_entry(
        key, "repo", first, keep_first_writer = True, ambiguous_field = "by"
    ) == {"repo": first}

    db.upsert_app_setting_map_entry(
        key,
        "repo",
        {"at": 200.0, "by": "identity-a"},
        keep_first_writer = True,
        ambiguous_field = "by",
    )
    assert db.get_app_setting(key) == {"repo": first}

    db.upsert_app_setting_map_entry(
        key,
        "repo",
        {"at": 300.0, "by": "identity-b"},
        keep_first_writer = True,
        ambiguous_field = "by",
    )
    assert db.get_app_setting(key) == {"repo": {"at": 100.0, "by": None}}
    db.upsert_app_setting_map_entry(
        key,
        "repo",
        {"at": 400.0, "by": "identity-a"},
        keep_first_writer = True,
        ambiguous_field = "by",
    )
    assert db.get_app_setting(key) == {"repo": {"at": 100.0, "by": None}}

    db.upsert_app_setting_map_entry(
        key,
        "other",
        {"at": 500.0, "by": "identity-b"},
        keep_first_writer = True,
        ambiguous_field = "by",
    )
    assert db.get_app_setting(key)["other"] == {"at": 500.0, "by": "identity-b"}
    db.upsert_app_setting_map_entry(key, "other", {"at": 600.0, "by": "identity-c"})
    assert db.get_app_setting(key)["other"] == {"at": 600.0, "by": "identity-c"}


def test_a_fill_never_relabels_a_stored_gpu_pin_with_this_browser_s_index_space(
    tmp_path, monkeypatch
):
    """A pin and the index space it is written in are one value.

    The server holds physical ids with no qualifier, which is what every writer
    before the field meant; this browser's backfill offers Vulkan ordinals. Field
    by field the ids would stay and the qualifier would land, and the row would
    then name devices in a space it was never written in.
    """
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(db, "_schema_ready", set())

    key = "test_map_entry_coupled"
    coupled = (("gpu_ids", "gpu_index_kind"),)
    db.upsert_app_setting_map_entry(key, "m", {"gpu_ids": [0, 1]})
    assert db.upsert_app_setting_map_entry(
        key,
        "m",
        {"gpu_ids": [2], "gpu_index_kind": "vulkan", "max_seq_length": 4096},
        fill_absent_fields = True,
        coupled_fields = coupled,
    ) == {"m": {"gpu_ids": [0, 1], "max_seq_length": 4096}}
    db.upsert_app_setting_map_entry(key, "n", {"max_seq_length": 2048})
    assert db.upsert_app_setting_map_entry(
        key,
        "n",
        {"gpu_ids": [2], "gpu_index_kind": "vulkan"},
        fill_absent_fields = True,
        coupled_fields = coupled,
    )["n"] == {"max_seq_length": 2048, "gpu_ids": [2], "gpu_index_kind": "vulkan"}


def test_gpu_ids_dedupe_is_not_a_scan_of_the_list_being_built():
    """gpu_ids arrives from an authenticated client and normalize_model_override
    de-duplicates it. Testing membership against the growing list walks up to
    MAX_GPU_ID entries per element; a set keeps the pass linear. Order, bounds and
    the bool rejection all have to survive the change."""
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
    assert elapsed < 0.5, elapsed


def test_gpu_ids_payload_is_bounded():
    """A list longer than the number of ids the normalizer can store adds nothing
    but work, so it is rejected at the boundary. A real device list is tiny."""
    import pydantic

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
    assert settings_route.ModelOverridePayload(model_id = "x", gpu_ids = [0, 1]).gpu_ids == [0, 1]


def _switch_with_overrides(monkeypatch, resolves_to, stored, requested):
    """Run the auto-switch hook against a real override map and return the load."""
    backend, rec = _wired(monkeypatch, _FakeBackend(None), resolves_to)
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


def test_two_spellings_of_one_cached_quant_do_not_delete_each_others_save(monkeypatch):
    """A save writes its target key, then reads the map back to retire the other spelling
    of the same cached repo, in a second transaction. This route is a plain `def`, so
    FastAPI runs it in a threadpool: two clients saving one quant, one by repo id and one
    by the snapshot path an upgraded install still holds, can both write before either
    cleanup runs and then retire each other's row. Both calls return 200 and nothing is
    stored. Whichever runs second must retire the first instead."""
    _mock_override_store(monkeypatch)
    repo = "unsloth/Qwen3-8B-GGUF:Q4_K_M"
    snapshot = "/mnt/old-cache/models--unsloth--Qwen3-8B-GGUF/snapshots/abc123:Q4_K_M"

    ready = threading.Barrier(2)
    written = threading.Barrier(2)
    write_lock = threading.Lock()
    wrote = set()
    real_set = settings_route.set_model_override

    def _set_then_sync(model_id, *args, **kwargs):
        # Production writes are one transaction, so keep the fake read-modify-write atomic too.
        with write_lock:
            result = real_set(model_id, *args, **kwargs)
        ident = threading.get_ident()
        if ident not in wrote:
            wrote.add(ident)
            try:
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


def test_mlx_kv_quant_survives_the_whole_override_projection():
    # Dropping this would make an API auto-switch load an MLX model at full precision.
    for quant in ("8", "6", "5", "4", "3", "2", "tq-4", "tq-3.5", "tq-3", "tq-2"):
        assert settings.normalize_model_override({"mlx_kv_quant": quant}) == {"mlx_kv_quant": quant}

    for rejected in ("7", "3.5", "tq-8", "tq-6", "auto", 4, True, None):
        assert settings.normalize_model_override({"mlx_kv_quant": rejected}) == {}
    assert settings.normalize_model_override({"mlx_kv_bits": 8}) == {"mlx_kv_quant": "8"}
    for stored in (7, "invalid", "4", True, [4]):
        assert settings.normalize_model_override({"mlx_kv_bits": stored}) == {}
        assert settings.model_override_load_kwargs({"mlx_kv_bits": stored}, is_gguf = False) == {}
    assert settings.model_override_load_kwargs({"mlx_kv_bits": 8}, is_gguf = False) == {
        "mlx_kv_quant": "8"
    }
    assert (
        settings.model_override_load_kwargs({"mlx_kv_quant": None, "mlx_kv_bits": 8}, is_gguf = False)
        == {}
    )
    both = {"mlx_kv_quant": "auto", "mlx_kv_bits": 4}
    assert settings.normalize_model_override(both) == {}
    assert settings.model_override_load_kwargs(both, is_gguf = False) == {}
    assert LoadRequest(model_path = "unsloth/A", **both).mlx_kv_quant == "auto"

    def _folded(**kw):
        payload = settings_route.ModelOverridePayload(model_id = "m", **kw)
        return payload.mlx_kv_quant, payload.mlx_kv_bits

    assert _folded(mlx_kv_bits = 8) == ("8", None)
    assert _folded(mlx_kv_quant = None, mlx_kv_bits = 8) == (None, None)
    assert _folded(mlx_kv_quant = "tq-4", mlx_kv_bits = 8) == ("tq-4", None)

    for is_gguf in (True, False):
        kwargs = settings.model_override_load_kwargs({"mlx_kv_quant": "tq-4"}, is_gguf = is_gguf)
        assert kwargs["mlx_kv_quant"] == "tq-4"
        assert LoadRequest(model_path = "unsloth/A", **kwargs).mlx_kv_quant == "tq-4"


def test_mlx_int8_prefill_is_stored_only_when_on_and_reaches_the_load(monkeypatch):
    _mock_override_store(monkeypatch)
    assert settings.normalize_model_override({"mlx_int8_prefill": False}) == {}
    _put("org/m", mlx_int8_prefill = True)
    stored = settings.get_model_overrides()["org/m"]
    assert stored == {"mlx_int8_prefill": True}
    kwargs = settings.model_override_load_kwargs(stored, is_gguf = False)
    assert LoadRequest(model_path = "org/m", **kwargs).mlx_int8_prefill is True
    _put("org/m", mlx_int8_prefill = False)
    assert "org/m" not in settings.get_model_overrides()


def _idle_backend(kw, monkeypatch, *, user_loaded):
    """A loaded, long-idle GGUF backend wired into the idle loop."""
    _reset_keepwarm()
    backend = _FakeBackend("unsloth/Idle-GGUF", hf_variant = "Q4_K_M")
    backend._loaded_by_user_action = user_loaded

    def _unload():
        backend.is_loaded = False

    backend.unload_model = _unload
    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 0.005)
    monkeypatch.setattr(settings, "idle_unload_is_configured", lambda: True)
    monkeypatch.setattr(settings, "get_auto_unload_keep_kv", lambda: False)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    return backend


def test_api_only_setting_roundtrip_and_default(monkeypatch):
    store = {}
    monkeypatch.setattr(db, "upsert_app_settings", lambda m: store.update(m))
    monkeypatch.setattr(settings, "_cached_setting", lambda k, d = None: store.get(k, d))

    # Off on an install that never stored it, so existing setups are unchanged.
    assert settings.get_auto_unload_api_only() is False
    assert settings.AUTO_UNLOAD_API_ONLY_SETTING_KEY not in store
    assert settings.set_openai_auto_switch(True, 60, None, None, True)[4] is True
    assert settings.get_auto_unload_api_only() is True
    assert settings.set_openai_auto_switch(True, 60, None, None, None)[4] is True
    with pytest.raises(ValueError, match = "true or false"):
        settings.set_openai_auto_switch(True, 60, None, None, "garbage")


def test_idle_unload_still_frees_a_user_loaded_model_by_default(monkeypatch):
    backend = _idle_backend(kw, monkeypatch, user_loaded = True)
    monkeypatch.setattr(settings, "get_auto_unload_api_only", lambda: False)

    _drive_idle_loop(kw, until = lambda: backend.is_loaded is False)
    assert backend.is_loaded is False


def test_api_only_spares_a_user_loaded_model_but_not_an_api_one(monkeypatch):
    monkeypatch.setattr(settings, "get_auto_unload_api_only", lambda: True)

    pinned = _idle_backend(kw, monkeypatch, user_loaded = True)
    _drive_idle_loop(kw)
    assert pinned.is_loaded is True
    assert kw.get_last_unloaded_model() is None

    api_loaded = _idle_backend(kw, monkeypatch, user_loaded = False)
    _drive_idle_loop(kw, until = lambda: api_loaded.is_loaded is False)
    assert api_loaded.is_loaded is False


def test_an_idle_restored_model_is_api_provenance(monkeypatch):
    # The restore goes through auto-switch, so the model stays idle-unloadable afterwards.

    monkeypatch.setattr(settings, "get_auto_unload_api_only", lambda: True)
    backend = _idle_backend(kw, monkeypatch, user_loaded = True)
    backend.is_loaded = False
    backend.model_identifier = None
    rec = _LoadRecorder(backend)
    _wire_on(
        monkeypatch,
        resolves_to = ("unsloth/B-GGUF", None, "unsloth/B-GGUF"),
        backend = backend,
        recorder = rec,
    )
    _run_hook("unsloth/B-GGUF")
    assert rec.calls and backend._loaded_by_user_action is False

    backend.unload_model = lambda: setattr(backend, "is_loaded", False)
    _drive_idle_loop(kw, until = lambda: backend.is_loaded is False)
    assert backend.is_loaded is False


def test_unload_clears_the_user_load_flag():
    # Otherwise the next API load inherits the pin and never frees its VRAM.

    backend = LlamaCppBackend()
    assert backend._loaded_by_user_action is False
    backend._loaded_by_user_action = True
    backend.unload_model()
    assert backend._loaded_by_user_action is False
    # A bare kill keeps provenance since a respawn replays the same load; check both halves.
    assert "_loaded_by_user_action" not in (
        inspect.getsource(LlamaCppBackend._kill_process)
        + inspect.getsource(LlamaCppBackend._kill_process_body)
    )


def test_the_load_route_pins_and_other_load_surfaces_do_not(monkeypatch):
    backend = _FakeBackend("unsloth/A-GGUF")
    backend._loaded_by_user_action = False
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)

    async def _already_loaded(*_a, **_kw):
        return None

    monkeypatch.setattr(inference_route, "_load_model_impl", _already_loaded)
    request = LoadRequest(model_path = "unsloth/A-GGUF")

    asyncio.run(inference_route.load_model_gated(request, object(), "tester", user_initiated = True))
    assert backend._loaded_by_user_action is True

    # Preview shares this helper and must not pin, so it keeps the default.
    asyncio.run(inference_route.load_model_gated(request, object(), "tester"))
    assert backend._loaded_by_user_action is False
    import routes.preview as preview_route

    assert "user_initiated" not in inspect.getsource(preview_route._serve_chat)


def test_api_only_turned_on_mid_save_still_spares_the_model(monkeypatch, tmp_path):
    on = {"now": False}
    monkeypatch.setattr(settings, "get_auto_unload_api_only", lambda: on["now"])
    backend = _idle_backend(kw, monkeypatch, user_loaded = True)
    monkeypatch.setattr(settings, "get_auto_unload_keep_kv", lambda: True)

    manifest = {"dir": str(tmp_path), "slots": [{"id": 0, "filename": "f.bin", "n_saved": 1}]}
    deleted = []
    monkeypatch.setattr(kw, "_delete_resume_files", lambda m: deleted.append(m))

    def _save(should_abort = None):
        on["now"] = True
        return manifest

    backend.save_slots_for_resume = _save

    _drive_idle_loop(kw, until = lambda: deleted)
    assert backend.is_loaded is True
    assert deleted == [manifest]
    assert kw._kv_resume is None


def _age_resolver_clock(monkeypatch, seconds):
    """Advance the resolver's monotonic clock by *seconds*.

    Ageing a snapshot by rewriting its stamp assumes the host has been up longer
    than the age; a fresh CI runner has not, and the subtraction flips the sign.
    """
    base = time.monotonic()
    monkeypatch.setattr(resolver, "time", types.SimpleNamespace(monotonic = lambda: base + seconds))


def test_additions_only_trust_expires_if_the_rebuild_never_lands(monkeypatch):
    # Retained entries must expire if rebuilds keep failing, or deleted models still trigger switches.
    entry = resolver._LocalGgufEntry("org/a", "/srv/models/org--a", ("Q4_K_M",))
    monkeypatch.setattr(resolver, "_scan", (time.monotonic(), {"org/a": entry}))
    monkeypatch.setattr(resolver, "_last_scan_s", 0.0)
    resolver.invalidate_index(additions_only = True)
    assert resolver.resolve_trusted_cached_local_gguf("org/a") is not None
    _age_resolver_clock(monkeypatch, 600.0)
    assert resolver.resolve_trusted_cached_local_gguf("org/a") is None


def test_additions_only_trust_outlasts_a_scan_slower_than_the_ttl(monkeypatch):
    # The window tracks the rebuild's cost so a slow multi-root scan stays off the request path.
    entry = resolver._LocalGgufEntry("org/a", "/srv/models/org--a", ("Q4_K_M",))
    monkeypatch.setattr(resolver, "_scan", (time.monotonic(), {"org/a": entry}))
    monkeypatch.setattr(resolver, "_last_scan_s", 16.0)
    resolver.invalidate_index(additions_only = True)
    _age_resolver_clock(monkeypatch, resolver._CACHE_TTL_S * 4)
    assert resolver.resolve_trusted_cached_local_gguf("org/a") is not None


def test_a_dead_warm_worker_releases_the_slot(monkeypatch):
    monkeypatch.setattr(resolver, "_scan", (0.0, {}))
    monkeypatch.setattr(resolver, "_warming", False)
    monkeypatch.setattr(resolver, "_warm_pending", False)
    monkeypatch.setattr(resolver, "_last_scan_s", 0.0)

    def _die():
        raise KeyboardInterrupt

    monkeypatch.setattr(resolver, "_build_index", _die)
    resolver.warm_index_soon()
    for _ in range(100):
        if not resolver._warming:
            break
        time.sleep(0.05)
    assert resolver._warming is False
    assert resolver._warm_pending is False


def test_an_invalidated_index_rebuilds_on_a_host_that_just_booted(monkeypatch):
    # Patch the module's time, not the real one: a frozen global clock hangs stray warm threads.
    monkeypatch.setattr(resolver, "time", types.SimpleNamespace(monotonic = lambda: 1.0))
    for kwargs in ({}, {"additions_only": True}):
        monkeypatch.setattr(resolver, "_scan", (1.0, {"org/a": "entry"}))
        resolver.invalidate_index(**kwargs)
        built = []
        monkeypatch.setattr(resolver, "_build_index", lambda: (built.append(1), {})[1])
        resolver._index()
        assert built == [1], kwargs


def _settable_resolver_clock(monkeypatch):
    """Advance resolver time by assigning ``clock.now``."""
    clock = types.SimpleNamespace(now = 1000.0)
    monkeypatch.setattr(resolver, "time", types.SimpleNamespace(monotonic = lambda: clock.now))
    return clock


def _counted_scans(monkeypatch, index = None):
    scans = []
    monkeypatch.setattr(resolver, "_build_index", lambda: scans.append(1) or dict(index or {}))
    monkeypatch.setattr(resolver, "_last_scan_s", 0.0)
    return scans


@pytest.mark.parametrize(
    "loaded, requested",
    [
        ("unsloth/A-GGUF", "claude-haiku-4-5"),
        ("/elsewhere/unscanned.gguf", "/elsewhere/unscanned.gguf"),
        ("/elsewhere/unscanned.gguf", "unscanned"),
    ],
)
def test_a_name_that_is_not_on_disk_rescans_once_not_per_request(monkeypatch, loaded, requested):
    clock = _settable_resolver_clock(monkeypatch)
    monkeypatch.setattr(resolver, "_scan", (clock.now - 60.0, {}))
    scans = _counted_scans(monkeypatch)
    warmed = []

    def _warm_lands():
        warmed.append(1)
        resolver._scan = (clock.now, {})

    monkeypatch.setattr(resolver, "warm_index_soon", _warm_lands)
    real_resolve = resolver.resolve_local_gguf
    backend, rec = _wired(monkeypatch, _FakeBackend(loaded, "Q4_K_M"), None)
    monkeypatch.setattr(resolver, "resolve_local_gguf", real_resolve)

    for _ in range(4):
        clock.now += resolver._CACHE_TTL_S + 1
        _run_hook(requested)

    assert scans == [1]
    assert warmed
    assert rec.calls == []
    assert backend.model_identifier == loaded


def test_a_remembered_miss_still_finds_a_model_that_appears_later(monkeypatch):
    clock = _settable_resolver_clock(monkeypatch)
    monkeypatch.setattr(resolver, "_scan", (clock.now - 60.0, {}))
    scans = _counted_scans(monkeypatch)
    monkeypatch.setattr(resolver, "warm_index_soon", lambda: None)
    assert resolver.resolve_local_gguf_for_switch("org/b") is None
    assert scans == [1]

    added = {"org/b": _entry("org/b", "Q4_K_M")}
    clock.now += resolver._CACHE_TTL_S + 1
    monkeypatch.setattr(resolver, "_scan", (clock.now, added))
    assert resolver.resolve_local_gguf_for_switch("org/b") is not None

    # Stale hits must rescan before switching.
    clock.now += resolver._CACHE_TTL_S + 1
    assert resolver.resolve_local_gguf_for_switch("org/b") is None
    assert scans == [1, 1]


def test_an_invalidation_forgets_a_remembered_miss(monkeypatch):
    # Download completion must make the new model discoverable immediately.
    clock = _settable_resolver_clock(monkeypatch)
    monkeypatch.setattr(resolver, "_scan", (clock.now - 60.0, {}))
    scans = _counted_scans(monkeypatch)
    monkeypatch.setattr(resolver, "warm_index_soon", lambda: None)
    assert resolver.resolve_local_gguf_for_switch("org/b") is None

    monkeypatch.setattr(
        resolver, "_build_index", lambda: scans.append(1) or {"org/b": _entry("org/b")}
    )
    resolver.invalidate_index(additions_only = True)
    assert resolver.resolve_local_gguf_for_switch("org/b") is not None
    assert scans == [1, 1]


def test_a_remembered_miss_expires_and_a_failed_scan_proves_nothing(monkeypatch):
    clock = _settable_resolver_clock(monkeypatch)
    monkeypatch.setattr(resolver, "_scan", (clock.now - 60.0, {}))
    scans = _counted_scans(monkeypatch)
    monkeypatch.setattr(resolver, "warm_index_soon", lambda: None)
    assert resolver.resolve_local_gguf_for_switch("org/b") is None

    # Misses expire if background refresh never completes.
    clock.now += 2 * resolver._duty_window() + 1
    assert resolver.resolve_local_gguf_for_switch("org/b") is None
    assert scans == [1, 1]

    def _fail():
        scans.append(1)
        raise OSError("scan root vanished")

    monkeypatch.setattr(resolver, "_build_index", _fail)
    monkeypatch.setattr(resolver, "_misses", {})
    clock.now += resolver._CACHE_TTL_S + 1
    assert resolver.resolve_local_gguf_for_switch("org/b") is None
    clock.now += resolver._CACHE_TTL_S + 1
    assert resolver.resolve_local_gguf_for_switch("org/b") is None
    assert scans == [1, 1, 1, 1]


def test_an_oversized_name_is_not_remembered(monkeypatch):
    clock = _settable_resolver_clock(monkeypatch)
    monkeypatch.setattr(resolver, "_scan", (clock.now - 60.0, {}))
    _counted_scans(monkeypatch)
    monkeypatch.setattr(resolver, "warm_index_soon", lambda: None)
    assert resolver.resolve_local_gguf_for_switch("x" * (resolver._MAX_MISS_NAME + 1)) is None
    assert resolver.resolve_local_gguf_for_switch("x" * resolver._MAX_MISS_NAME) is None
    assert [len(name) for _scope, name in resolver._misses] == [resolver._MAX_MISS_NAME]


# The resident short circuit must never accept what the pre-existing resident check rejects.
_IDENTITIES = [
    "unsloth/Muse-GGUF",
    "/srv/models/unsloth--Muse-GGUF",
    "/srv/models/Muse.gguf",
    None,
]
_REQUESTS = [
    "unsloth/Muse-GGUF",
    "unsloth/muse-gguf",
    "unsloth/Muse-GGUF:Q4_K_M",
    "unsloth/Muse-GGUF:Q8_0",
    "unsloth/Muse-GGUF:latest",
    "unsloth/Other-GGUF",
    "/srv/models/Muse.gguf",
    "/srv/models/MUSE.gguf",
    "muse.gguf",
    "../../etc/passwd",
    "unsloth/",
    ":Q4_K_M",
    "unsloth/Muse GGUF",
]


@pytest.mark.parametrize("identity", _IDENTITIES)
@pytest.mark.parametrize("requested", _REQUESTS)
@pytest.mark.parametrize("quant", [None, "Q4_K_M"])
def test_the_resident_shortcut_never_answers_where_the_full_check_would_not(
    monkeypatch, identity, requested, quant
):
    backend = _FakeBackend(identity, hf_variant = quant) if identity else _FakeBackend(None)
    backend.is_loaded = identity is not None
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    fast = inference_route._loaded_identity_satisfies(requested)
    assert not (
        fast and not inference_route._loaded_satisfies(requested)
    ), f"shortcut served {requested!r} against {identity!r} (quant={quant!r})"


def test_the_resident_shortcut_refuses_an_explicit_quant_mismatch(monkeypatch):
    backend = _FakeBackend("unsloth/Muse-GGUF", hf_variant = "Q4_K_M")
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    assert inference_route._loaded_identity_satisfies("unsloth/Muse-GGUF:Q8_0") is False
    assert inference_route._loaded_identity_satisfies("unsloth/Muse-GGUF:Q4_K_M") is True


def test_clearing_the_box_for_a_quant_survives_the_next_load(override_store):
    settings.set_model_override("unsloth/B-GGUF", llama_extra_args = ["--numa", "distribute"])
    carried = _put("unsloth/B-GGUF:Q4_K_M", max_seq_length = 4096)
    assert carried.overrides["unsloth/B-GGUF:Q4_K_M"]["llama_extra_args"] == [
        "--numa",
        "distribute",
    ]

    cleared = _put("unsloth/B-GGUF:Q4_K_M", llama_extra_args = [], remove = False)
    assert cleared.overrides["unsloth/B-GGUF:Q4_K_M"] == {"llama_extra_args": []}
    _key, resolved = settings.resolve_override_for_load("unsloth/B-GGUF", variant = "Q4_K_M")
    assert not resolved.get("llama_extra_args")


def test_the_clear_leaves_the_legacy_row_for_the_quants_still_reading_it(override_store):
    # The bare row is the fallback for other quants, so clearing Q4 must not touch Q6's flags.
    settings.set_model_override(
        "unsloth/B-GGUF",
        llama_extra_args = ["--numa", "distribute"],
        max_seq_length = 4096,
    )
    _put("unsloth/B-GGUF:Q4_K_M", llama_extra_args = [], remove = False)
    bare = settings.get_model_override("unsloth/B-GGUF")
    assert bare["llama_extra_args"] == ["--numa", "distribute"]
    assert bare["max_seq_length"] == 4096
    _key, sibling = settings.resolve_override_for_load("unsloth/B-GGUF", variant = "Q6_K")
    assert sibling["llama_extra_args"] == ["--numa", "distribute"]


def test_the_clear_holds_when_another_quant_has_a_row_of_its_own(override_store):
    # A sibling with its own row never reads the bare one, so gating on siblings was wrong.
    settings.set_model_override("unsloth/B-GGUF", llama_extra_args = ["--numa", "distribute"])
    settings.set_model_override("unsloth/B-GGUF:Q8_0", max_seq_length = 4096)
    _put("unsloth/B-GGUF:Q4_K_M", llama_extra_args = [], remove = False)
    _key, resolved = settings.resolve_override_for_load("unsloth/B-GGUF", variant = "Q4_K_M")
    assert not resolved.get("llama_extra_args")
    assert settings.get_model_override("unsloth/B-GGUF")["llama_extra_args"] == [
        "--numa",
        "distribute",
    ]


def test_a_clear_with_no_fallback_behind_it_stores_nothing(override_store):
    # With no fallback to stop, the row is removed rather than leaving a tombstone.
    settings.set_model_override("unsloth/B-GGUF:Q4_K_M", llama_extra_args = ["--numa", "distribute"])
    cleared = _put("unsloth/B-GGUF:Q4_K_M", llama_extra_args = [], remove = False)
    assert "unsloth/B-GGUF:Q4_K_M" not in cleared.overrides


def test_an_explicit_forget_still_removes_the_row(override_store):
    # remove=True must not leave a tombstone the picker would read as a saved setting.
    settings.set_model_override("unsloth/B-GGUF", llama_extra_args = ["--numa", "distribute"])
    settings.set_model_override("unsloth/B-GGUF:Q4_K_M", llama_extra_args = ["--top-k", "20"])
    forgotten = _put("unsloth/B-GGUF:Q4_K_M", llama_extra_args = [], remove = True)
    assert "unsloth/B-GGUF:Q4_K_M" not in forgotten.overrides


def test_a_save_that_did_not_touch_the_box_leaves_the_fallback(override_store):
    # An omitted field means keep the flags, as for every save that never opened the editor.
    settings.set_model_override("unsloth/B-GGUF", llama_extra_args = ["--numa", "distribute"])
    _put("unsloth/B-GGUF:Q4_K_M", max_seq_length = 4096)
    assert settings.get_model_override("unsloth/B-GGUF")["llama_extra_args"] == [
        "--numa",
        "distribute",
    ]


def test_a_fill_pass_never_clears_a_fallback(override_store):
    # The migration only adds what is missing, so it must never clear flags.
    settings.set_model_override("unsloth/B-GGUF", llama_extra_args = ["--numa", "distribute"])
    _put(
        "unsloth/B-GGUF:Q4_K_M",
        llama_extra_args = [],
        remove = False,
        fill_absent_fields = True,
    )
    assert settings.get_model_override("unsloth/B-GGUF")["llama_extra_args"] == [
        "--numa",
        "distribute",
    ]


def test_clearing_one_quant_stops_the_legacy_repo_row_from_answering(monkeypatch):
    _mock_override_store(monkeypatch)
    settings.set_model_override("unsloth/B-GGUF", llama_extra_args = ["--top-k", "40"])
    _put("unsloth/B-GGUF:Q4_K_M", llama_extra_args = [])

    assert settings.get_model_override("unsloth/B-GGUF:Q4_K_M") == {"llama_extra_args": []}
    key, override = settings.resolve_override_for_load("unsloth/B-GGUF", None, "Q4_K_M")
    assert key == "unsloth/B-GGUF:Q4_K_M"
    assert override.get("llama_extra_args") == []
    # The bare row stays as the fallback for every other quant of this repo.
    assert settings.get_model_override("unsloth/B-GGUF")["llama_extra_args"] == ["--top-k", "40"]
    other_key, other = settings.resolve_override_for_load("unsloth/B-GGUF", None, "Q8_0")
    assert other_key == "unsloth/B-GGUF"
    assert other["llama_extra_args"] == ["--top-k", "40"]


def test_an_empty_save_with_nothing_to_suppress_still_stores_nothing(monkeypatch):
    # With no row behind it, an empty row would just show up as configured, so write nothing.
    _mock_override_store(monkeypatch)
    _put("unsloth/B-GGUF:Q4_K_M", llama_extra_args = [])
    assert settings.get_model_override("unsloth/B-GGUF:Q4_K_M") == {}
    assert settings.get_model_overrides() == {}


def test_a_fill_never_writes_the_tombstone(monkeypatch):
    # An empty list in a fill means no request, so writing a row would suppress the fallback.
    _mock_override_store(monkeypatch)
    settings.set_model_override("unsloth/B-GGUF", llama_extra_args = ["--top-k", "40"])
    _put("unsloth/B-GGUF:Q4_K_M", llama_extra_args = [], fill_absent_fields = True)
    assert "llama_extra_args" not in settings.get_model_override("unsloth/B-GGUF:Q4_K_M")
    key, override = settings.resolve_override_for_load("unsloth/B-GGUF", None, "Q4_K_M")
    assert override.get("llama_extra_args") == ["--top-k", "40"]


def test_normalize_keeps_an_explicit_empty_list_only_when_asked(monkeypatch):
    assert settings.normalize_model_override({"llama_extra_args": []}) == {}
    assert settings.normalize_model_override(
        {"llama_extra_args": []}, keep_empty_extra_args = True
    ) == {"llama_extra_args": []}
    # The flag only governs empty lists; non-empty lists are stored, as validation ran upstream.
    for keep in (False, True):
        assert settings.normalize_model_override(
            {"llama_extra_args": ["--top-k", "40"]}, keep_empty_extra_args = keep
        ) == {"llama_extra_args": ["--top-k", "40"]}


def _wire_refusing_switch(monkeypatch):
    backend, rec = _wired(
        monkeypatch, _FakeBackend("org/A-GGUF"), ("/local/B.gguf", "Q8_0", "org/B-GGUF")
    )
    monkeypatch.setattr(inference_route, "_target_is_vision", lambda _p, _v = None, _i = True: False)
    return rec


def _refusal_detail(monkeypatch, **kwargs) -> str:
    rec = _wire_refusing_switch(monkeypatch)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            inference_route._maybe_auto_switch_model(
                "org/B-GGUF", object(), "t", require_vision = True, **kwargs
            )
        )
    assert rec.calls == []
    return json.dumps(exc.value.detail)


def test_the_switch_refusal_names_the_modality_the_request_carried(monkeypatch):
    # Video joins require_vision, so the error must mention video to make sense to the user.
    detail = _refusal_detail(monkeypatch, modality_label = "video")
    assert "video input" in detail
    assert "image or audio" not in detail


def test_the_refusal_lists_every_attached_modality(monkeypatch):
    detail = _refusal_detail(monkeypatch, modality_label = "image or video")
    assert "image or video input" in detail


def test_the_refusal_wording_is_unchanged_for_callers_that_pass_no_label(monkeypatch):
    detail = _refusal_detail(monkeypatch)
    assert "image or audio input" in detail


def test_a_video_request_labels_the_switch_refusal_video(monkeypatch):
    """End to end through the handler: video joins require_vision, so the label
    has to follow or the user who attached a clip is told about image or audio."""

    captured = {}

    async def _capture(
        model,
        request,
        subject,
        *,
        require_vision = False,
        require_image = True,
        modality_label = "image or audio",
        claim_resident = True,
        require_audio_input = False,
        require_video = False,
        audio_preflight = None,
        image_preflight = None,
        tool_images_only = False,
    ):
        captured.update(
            require_vision = require_vision,
            require_image = require_image,
            modality_label = modality_label,
            claim_resident = claim_resident,
            require_video = require_video,
        )
        raise _Reached()

    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _capture)
    payload = _chat_request(model = "org/B-GGUF", video_base64 = "AAAA")
    with pytest.raises(_Reached):
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))
    assert captured == {
        "require_vision": True,
        "require_image": False,
        "modality_label": "video",
        "claim_resident": False,
        "require_video": True,
    }


def test_a_video_request_switches_to_a_non_gguf_target_only_with_a_video_token(
    tmp_path, monkeypatch
):
    """Loaded for video only if the config names a video token; no config is left to the load."""
    from utils.hardware import hardware as hw

    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.MLX, raising = False)
    monkeypatch.setattr("core.inference.mlx_inference._mlx_vlm_decodes_video", lambda: True)

    def _target(name, config):
        target = tmp_path / name
        target.mkdir()
        if config is not None:
            (target / "config.json").write_text(json.dumps(config), encoding = "utf-8")
        return str(target)

    def _accepts(path):
        return inference_route._target_accepts_request_input(
            path, False, True, False, None, False, True
        )

    assert _accepts(_target("image-only", {"model_type": "gemma3", "vision_config": {}})) is False
    assert _accepts(_target("flat", {"model_type": "qwen2_vl", "video_token_id": 151656})) is True
    assert _accepts(_target("nested", {"text_config": {"video_token_index": 7}})) is True
    assert _accepts(_target("unset", {"video_token_id": None})) is False
    assert _accepts(_target("no-config", None)) is True
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.CUDA, raising = False)
    assert _accepts(_target("cuda", {"model_type": "qwen2_vl", "video_token_id": 151656})) is False

    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.MLX, raising = False)
    monkeypatch.setattr("core.inference.mlx_inference._mlx_vlm_decodes_video", lambda: False)
    assert (
        _accepts(_target("old-mlx-vlm", {"model_type": "qwen2_vl", "video_token_id": 151656}))
        is False
    )
    seen = {}

    def _probe(
        path,
        variant = None,
        need_image = True,
    ):
        seen.update(path = path, need_image = need_image)
        return True

    import unittest.mock as _mock

    with _mock.patch.object(inference_route, "_target_is_vision", _probe):
        assert (
            inference_route._target_accepts_request_input(
                "/srv/models/A.gguf", True, True, False, None, False, True
            )
            is True
        )
    assert seen == {"path": "/srv/models/A.gguf", "need_image": False}


def test_count_tokens_switch_marks_new_model_preview_owned(monkeypatch):
    backend, recorder = _wired(
        monkeypatch, _FakeBackend("org/A-GGUF"), ("org/B-GGUF", None, "org/B-GGUF")
    )
    inference_route._set_preview_resident("org/A-GGUF")
    asyncio.run(
        inference_route._maybe_auto_switch_model(
            "org/B-GGUF", object(), "tester", claim_resident = False
        )
    )
    assert len(recorder.calls) == 1
    assert inference_route._is_preview_resident("org/B-GGUF")
    inference_route._set_preview_resident(None)


def test_count_tokens_does_not_own_an_independent_load(monkeypatch):
    state = {"slot": "/outputs/preview-a"}
    identity_checked = threading.Event()
    independent_load_done = threading.Event()

    def loaded_identity_satisfies(_requested):
        identity_checked.set()
        assert independent_load_done.wait(timeout = 2)
        return True

    async def no_model_error(*_args, **_kwargs):
        return 503, "No GGUF model loaded"

    monkeypatch.setattr(inference_route, "_loaded_slot_ident", lambda: state["slot"])
    monkeypatch.setattr(inference_route, "_loaded_identity_satisfies", loaded_identity_satisfies)
    monkeypatch.setattr(inference_route, "_no_model_loaded_error", no_model_error)
    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    monkeypatch.setattr(resolver, "warm_index_soon", lambda: None)
    monkeypatch.setattr(
        inference_route,
        "get_llama_cpp_backend",
        lambda: types.SimpleNamespace(is_loaded = False),
    )

    inference_route._set_preview_resident(state["slot"])
    payload = _anthropic_payload_with_tools(None)
    request = types.SimpleNamespace(scope = {"path": "/v1/messages/count_tokens"})

    async def drive():
        task = asyncio.create_task(
            inference_route.anthropic_count_tokens(payload, request, "tester")
        )
        assert await asyncio.to_thread(identity_checked.wait, 2)
        state["slot"] = "/outputs/studio-b"
        inference_route._set_preview_resident(None)
        independent_load_done.set()
        with pytest.raises(HTTPException) as exc:
            await task
        assert exc.value.status_code == 503

    asyncio.run(drive())
    assert not inference_route._is_preview_resident("/outputs/studio-b")


# Only chat generation uses these entries, so fixtures are causal LMs unless stated.
_CHAT_CONFIG = '{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"}'


def _safetensors_bytes(nbytes = 32):
    """Filler standing in for a .safetensors file. The resolver checks the name, not the
    contents: whether the bytes load is the loader's answer to give."""
    return b"\0" * nbytes


def _local_checkpoint(root, name = "Qwen3-MLX-4bit"):
    """An on-disk non-GGUF checkpoint: config.json and a tokenizer beside safetensors."""
    path = root / name
    path.mkdir()
    (path / "config.json").write_text(_CHAT_CONFIG)
    (path / "model.safetensors").write_bytes(_safetensors_bytes())
    (path / "tokenizer.json").write_text("{}")
    (path / "tokenizer_config.json").write_text("{}")
    return path


def _complete_minimax_pipeline(root):
    pipeline = root / "MiniMax-Music3"
    pipeline.mkdir()
    component_specs = {
        "condition_encoder": ("diffusers", "MiniMaxMusic3ConditionEncoder"),
        "language_model": ("transformers", "Qwen3ForCausalLM"),
        "rvq_depth_decoder": ("diffusers", "MiniMaxMusic3RVQDepthDecoder"),
        "scheduler": ("diffusers", "FlowMatchEulerDiscreteScheduler"),
        "tokenizer": ("transformers", "Qwen2Tokenizer"),
        "transformer": ("diffusers", "MiniMaxMusic3Transformer1DModel"),
        "vocoder": ("diffusers", "MiniMaxMusic3Vocoder"),
    }
    index = {
        "_class_name": "MiniMaxMusic3ModularPipeline",
        "_blocks_class_name": "MiniMaxMusic3Blocks",
        **{
            component: [*component_specs[component], {"subfolder": component}]
            for component in component_specs
        },
    }
    (pipeline / "modular_model_index.json").write_text(json.dumps(index))
    for component in component_specs:
        directory = pipeline / component
        directory.mkdir()
        if component == "scheduler":
            (directory / "scheduler_config.json").write_text(
                json.dumps({"_class_name": "Scheduler"})
            )
        elif component == "tokenizer":
            (directory / "tokenizer_config.json").write_text(
                json.dumps({"tokenizer_class": "PreTrainedTokenizerFast"})
            )
            (directory / "tokenizer.json").write_text(
                json.dumps({"model": {"type": "BPE", "vocab": {}, "merges": []}})
            )
        else:
            (directory / "config.json").write_text(json.dumps({"_class_name": "Component"}))
            stem = "model" if component == "language_model" else "diffusion_pytorch_model"
            (directory / f"{stem}.safetensors").write_bytes(_safetensors_bytes())
    return pipeline


@pytest.mark.parametrize(
    "repo", ["unsloth/Spark-TTS-0.5B", "SparkAudio/Spark-TTS-0.5B", "org/custom-voice"]
)
def test_a_downloaded_spark_repository_resolves_to_its_llm_checkpoint(tmp_path, repo):
    snapshot = tmp_path / "models--unsloth--Spark-TTS-0.5B" / "snapshots" / "revision"
    llm = snapshot / "LLM"
    llm.mkdir(parents = True)
    (llm / "config.json").write_text(
        json.dumps({"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2"})
    )
    (llm / "model.safetensors").write_bytes(_safetensors_bytes())
    tokens = {
        str(i): {"content": f"<|bicodec_{name}_0|>"}
        for i, name in enumerate(("semantic", "global"))
    }
    (llm / "tokenizer_config.json").write_text(json.dumps({"added_tokens_decoder": tokens}))
    info = types.SimpleNamespace(id = repo, model_id = repo, path = str(snapshot), partial = False)
    entry = resolver._local_weights_entry(repo, info)
    assert entry is not None
    assert entry.load_path == str(llm)
    assert entry.is_gguf is False
    assert inference_route._target_speech_audio_type(entry.load_path, False) == "bicodec"
    (llm / "tokenizer_config.json").write_text("{}")
    assert resolver._local_weights_entry("org/text", info) is None


def test_a_diffusers_pipeline_is_not_a_servable_chat_model(tmp_path):
    pipeline = _local_checkpoint(tmp_path, "SomeDiffusionPipeline")
    manifest = {
        "_class_name": "DiffusionPipeline",
        "transformer": ["diffusers", "Transformer2DModel"],
    }
    (pipeline / "model_index.json").write_text(json.dumps(manifest))
    (pipeline / "transformer").mkdir()
    (pipeline / "transformer" / "config.json").write_text("{}")
    (pipeline / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(
        _safetensors_bytes()
    )
    info = SimpleNamespace(id = str(pipeline), path = str(pipeline))
    assert resolver.local_servable_model(info) is None


def test_a_partial_download_is_not_a_servable_chat_model(tmp_path):
    # Advertising an incomplete snapshot hands out an id whose load must fail.

    path = _local_checkpoint(tmp_path)
    info = SimpleNamespace(id = str(path), path = str(path), partial = True)
    assert resolver.local_servable_model(info) is None


def test_an_adapter_only_directory_is_not_a_servable_chat_model(tmp_path):
    path = tmp_path / "adapter"
    path.mkdir()
    (path / "adapter_config.json").write_text("{}")
    (path / "adapter_model.safetensors").write_bytes(_safetensors_bytes())
    info = SimpleNamespace(id = str(path), path = str(path))
    assert resolver.local_servable_model(info) is None


def test_an_installed_mlx_model_is_indexed_and_resolves(tmp_path, monkeypatch):
    # An unloaded MLX model must be visible to the resolver, not 404 as not downloaded.
    from utils import paths

    path = _local_checkpoint(tmp_path)
    (path / "config.json").write_text(
        '{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "quantization": {"group_size": 64, "bits": 4}}'
    )
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.MLX)
    monkeypatch.setattr(models_route, "_scan_hf_cache", lambda *a, **k: [])
    monkeypatch.setattr(models_route, "_scan_lmstudio_dir", lambda *a, **k: [])
    monkeypatch.setattr(paths, "legacy_hf_cache_dir", lambda: None)
    monkeypatch.setattr(paths, "hf_default_cache_dir", lambda: None)
    monkeypatch.setattr(paths, "lmstudio_model_dirs", lambda: [])
    monkeypatch.setattr("storage.studio_db.list_scan_folders", lambda: [{"path": str(tmp_path)}])

    resolver._scan = (0.0, {})
    assert resolver.resolve_local_gguf(str(path)) == (str(path), None, path.name)
    assert resolver.resolve_local_gguf(f"{path.name}:Q4_K_M") is None
    assert resolver.local_target_is_gguf(str(path), path.name) is False
    # An index that no longer carries the entry must not flip the answer to GGUF.
    resolver._scan = (0.0, {})
    assert resolver.local_target_is_gguf(str(path), path.name) is False


# mlx-lm remaps these model types, so find_spec on model_type would hide them.
_REMAPPED_MLX_FAMILIES = ("mistral", "kimi_k2", "llava", "falcon_mamba", "minimax_m2")


@pytest.mark.parametrize("model_type", _REMAPPED_MLX_FAMILIES)
def test_a_family_mlx_lm_remaps_is_listed_in_the_catalog(tmp_path, monkeypatch, model_type):
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.MLX)
    path = _local_checkpoint(tmp_path, f"{model_type}-4bit")
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["MistralForCausalLM"],
                "model_type": model_type,
                "quantization": {"group_size": 64, "bits": 4},
            }
        )
    )
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: SimpleNamespace(active_model_name = None, models = {}),
    )
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _FakeBackend(None))
    info = SimpleNamespace(id = str(path), model_id = str(path), path = str(path))
    assert resolver.local_servable_model(info) == (False, ())
    rows = inference_route._servable_catalog_rows([info])
    assert [(is_gguf, quants) for _i, is_gguf, quants, _r in rows] == [(False, ())]


def test_mlx_lm_resolves_these_families_only_after_the_remap():
    # Read from mlx-lm itself so the catalog cannot drift from its release cadence.
    from importlib.util import find_spec
    mlx_lm_utils = pytest.importorskip("mlx_lm.utils")
    for model_type in _REMAPPED_MLX_FAMILIES:
        remapped = mlx_lm_utils.MODEL_REMAPPING.get(model_type)
        assert remapped, model_type
        assert find_spec(f"mlx_lm.models.{model_type}") is None, model_type
        assert find_spec(f"mlx_lm.models.{remapped}") is not None, model_type


def test_auto_switch_loads_an_unloaded_mlx_model(monkeypatch):
    # The switch must load an MLX model through the orchestrator.
    llama = _FakeBackend(None)

    class _FakeOrchestrator:
        active_model_name = None
        models: dict = {}

    orchestrator = _FakeOrchestrator()
    calls = []

    async def _load(request, *args, **kwargs):
        calls.append(request)
        orchestrator.active_model_name = request.model_path
        return None

    _wire_on(
        monkeypatch,
        resolves_to = ("/srv/models/Qwen3-MLX", None, "unsloth/Qwen3-MLX"),
        backend = llama,
        recorder = _load,
    )
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: orchestrator)
    entry = resolver._LocalGgufEntry(
        "unsloth/Qwen3-MLX", "/srv/models/Qwen3-MLX", (), is_gguf = False
    )
    monkeypatch.setattr(resolver, "_scan", (time.monotonic(), {"unsloth/qwen3-mlx": entry}))
    settings.set_model_override("unsloth/Qwen3-MLX", n_parallel = 8)

    _run_hook("unsloth/Qwen3-MLX")

    assert [c.model_path for c in calls] == ["/srv/models/Qwen3-MLX"]
    assert calls[0].gguf_variant is None
    assert calls[0].n_parallel == 8
    assert orchestrator._openai_advertised_id == "unsloth/Qwen3-MLX"
    assert getattr(llama, "_openai_advertised_id", None) is None
    assert inference_route._openai_model_objects()[0]["id"] == "unsloth/Qwen3-MLX"


def test_auto_switch_does_not_reload_a_resident_mlx_model(monkeypatch):
    llama = _FakeBackend(None)

    class _FakeOrchestrator:
        active_model_name = "/srv/models/Qwen3-MLX"
        models: dict = {}
        _openai_advertised_id = "unsloth/Qwen3-MLX"

    calls = []

    async def _load(request, *args, **kwargs):
        calls.append(request)
        return None

    _wire_on(
        monkeypatch,
        resolves_to = ("/srv/models/Qwen3-MLX", None, "unsloth/Qwen3-MLX"),
        backend = llama,
        recorder = _load,
    )
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: _FakeOrchestrator())
    entry = resolver._LocalGgufEntry(
        "unsloth/Qwen3-MLX", "/srv/models/Qwen3-MLX", (), is_gguf = False
    )
    monkeypatch.setattr(resolver, "_scan", (time.monotonic(), {"unsloth/qwen3-mlx": entry}))

    assert inference_route._loaded_satisfies("unsloth/Qwen3-MLX") is True
    assert inference_route._loaded_identity_satisfies("unsloth/Qwen3-MLX") is True
    _run_hook("unsloth/Qwen3-MLX")
    assert calls == []


def test_a_resident_mlx_alias_counts_as_a_namespaced_identity(monkeypatch):
    # Without the alias an unknown org/model would be answered by the resident MLX model.
    class _FakeOrchestrator:
        active_model_name = "/srv/models/Qwen3-MLX"
        _openai_advertised_id = "unsloth/Qwen3-MLX"

    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _FakeBackend(None))
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: _FakeOrchestrator())
    assert inference_route._resident_id_is_namespaced() is True


def test_a_gguf_only_endpoint_refuses_a_non_gguf_target_before_loading(monkeypatch):
    llama = _FakeBackend("org/A-GGUF", hf_variant = "Q4_K_M")

    class _FakeOrchestrator:
        active_model_name = None
        models: dict = {}

    calls = []

    async def _load(request, *args, **kwargs):
        calls.append(request)
        return None

    _wire_on(
        monkeypatch,
        resolves_to = ("/srv/models/Qwen3-MLX", None, "unsloth/Qwen3-MLX"),
        backend = llama,
        recorder = _load,
    )
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: _FakeOrchestrator())
    entry = resolver._LocalGgufEntry(
        "unsloth/Qwen3-MLX", "/srv/models/Qwen3-MLX", (), is_gguf = False
    )
    monkeypatch.setattr(resolver, "_scan", (time.monotonic(), {"unsloth/qwen3-mlx": entry}))

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            inference_route._maybe_auto_switch_model(
                "unsloth/Qwen3-MLX", object(), "tester", gguf_only = True
            )
        )
    assert excinfo.value.status_code == 400
    assert calls == [], "the resident GGUF was unloaded for a swap the endpoint cannot use"
    assert llama.is_loaded is True


def test_two_scan_roots_sharing_a_basename_do_not_answer_for_each_other(monkeypatch):
    """A scanned directory with no repo alias is advertised under its basename, so
    /root1/model and /root2/model both advertise "model". Accepting that shared alias as
    proof of residency answered an explicit request for the second path with the first
    path's weights, defeating the exact filesystem keys _build_index deliberately holds."""
    llama = _FakeBackend("org/A-GGUF")
    llama.is_loaded = False

    class _FakeOrchestrator:
        active_model_name = "/root1/model"
        models = {"/root1/model": object()}
        _openai_advertised_id = "model"

    orchestrator = _FakeOrchestrator()
    calls = []

    async def _load(request, *args, **kwargs):
        calls.append(request)
        orchestrator.active_model_name = "/root2/model"
        return None

    _wire_on(
        monkeypatch,
        resolves_to = ("/root2/model", None, "model"),
        backend = llama,
        recorder = _load,
    )
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: orchestrator)
    monkeypatch.setattr(inference_route, "_peek_inference_backend", lambda: orchestrator)
    monkeypatch.setattr(inference_route, "_reject_unservable_model", _noop_reject)
    monkeypatch.setattr(resolver, "local_target_is_gguf", lambda *_a, **_kw: False)

    asyncio.run(inference_route._maybe_auto_switch_model("/root2/model", object(), "tester"))
    assert [getattr(c, "model_path", None) for c in calls] == [
        "/root2/model"
    ], "the request named the second path and was answered by the first model"


def test_the_resident_path_still_short_circuits_its_own_request(monkeypatch):
    # the guard above must not cost a reload when the request names the resident path.
    llama = _FakeBackend("org/A-GGUF")
    llama.is_loaded = False

    class _FakeOrchestrator:
        active_model_name = "/root1/model"
        models = {"/root1/model": object()}
        _openai_advertised_id = "model"

    orchestrator = _FakeOrchestrator()
    calls = []

    async def _load(request, *args, **kwargs):
        calls.append(request)
        return None

    _wire_on(
        monkeypatch,
        resolves_to = ("/root1/model", None, "model"),
        backend = llama,
        recorder = _load,
    )
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: orchestrator)
    monkeypatch.setattr(inference_route, "_peek_inference_backend", lambda: orchestrator)
    monkeypatch.setattr(inference_route, "_reject_unservable_model", _noop_reject)
    monkeypatch.setattr(resolver, "local_target_is_gguf", lambda *_a, **_kw: False)

    asyncio.run(inference_route._maybe_auto_switch_model("/root1/model", object(), "tester"))
    assert calls == []


def test_a_repo_id_resident_still_matches_a_path_request_through_the_alias(monkeypatch):
    # A repo-id resident is not a path, so the alias arm answers path requests for the same model.
    llama = _FakeBackend("org/A-GGUF")
    llama.is_loaded = False

    class _FakeOrchestrator:
        active_model_name = "mlx-community/Qwen3-8B-4bit"
        models = {"mlx-community/Qwen3-8B-4bit": object()}
        _openai_advertised_id = None

    orchestrator = _FakeOrchestrator()
    calls = []

    async def _load(request, *args, **kwargs):
        calls.append(request)
        return None

    _wire_on(
        monkeypatch,
        resolves_to = ("/cache/snapshots/abc", None, "mlx-community/Qwen3-8B-4bit"),
        backend = llama,
        recorder = _load,
    )
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: orchestrator)
    monkeypatch.setattr(inference_route, "_peek_inference_backend", lambda: orchestrator)
    monkeypatch.setattr(inference_route, "_reject_unservable_model", _noop_reject)
    monkeypatch.setattr(resolver, "local_target_is_gguf", lambda *_a, **_kw: False)

    asyncio.run(
        inference_route._maybe_auto_switch_model("/cache/snapshots/abc", object(), "tester")
    )
    assert calls == []


def test_an_audio_request_probes_audio_capability_on_a_non_gguf_target(monkeypatch):
    seen = {}

    monkeypatch.setattr(
        inference_route,
        "_target_is_vision",
        lambda path, *_a: seen.setdefault("vision", path) and False,
    )
    monkeypatch.setattr(
        inference_route,
        "_target_accepts_audio_input",
        lambda path: seen.setdefault("audio", path) or True,
    )
    assert (
        inference_route._target_accepts_request_input("/srv/models/Whisper", False, False, True)
        is True
    )
    assert "vision" not in seen
    inference_route._target_accepts_request_input("/srv/models/A.gguf", True, False, True)
    assert seen.get("vision") == "/srv/models/A.gguf"


def test_a_custom_code_checkpoint_is_not_switchable(tmp_path):
    path = _local_checkpoint(tmp_path, "CustomCode")
    (path / "config.json").write_text(
        '{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "auto_map": {"AutoModel": "modeling.MyModel"}}'
    )
    info = SimpleNamespace(id = str(path), path = str(path))
    assert resolver.local_servable_model(info) is None


def test_an_embedding_checkpoint_is_not_switchable(tmp_path):
    # /v1/embeddings is GGUF-only, so a SentenceTransformer must not evict the resident model.

    path = _local_checkpoint(tmp_path, "bge-small")
    info = SimpleNamespace(id = str(path), path = str(path))
    assert resolver.local_servable_model(info) is not None
    (path / "modules.json").write_text("[]")
    assert resolver.local_servable_model(info) is None


def test_streaming_responses_refuses_a_non_gguf_swap(monkeypatch):
    # _responses_stream only reads llama.cpp, so a non-GGUF target must not unload the GGUF.
    captured = {}

    async def _capture(model, request, subject, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("reached the switch")

    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _capture)
    for streaming in (True, False):
        captured.clear()
        payload = _responses_payload(stream = streaming)
        with pytest.raises(RuntimeError):
            asyncio.run(inference_route.openai_responses(payload, object(), "tester"))
        # Assert the key is absent, since the non-streaming branch must not pin GGUF at all.
        if streaming:
            assert captured["gguf_only"] is True
        else:
            assert "gguf_only" not in captured


def test_a_pickle_checkpoint_is_not_switchable(tmp_path):
    # .bin weights are pickle-backed, so an API request must not load one implicitly.

    path = _local_checkpoint(tmp_path, "PickleOnly")
    (path / "model.safetensors").unlink()
    (path / "pytorch_model.bin").write_bytes(b"x" * 32)
    info = SimpleNamespace(id = str(path), path = str(path))
    assert resolver.local_servable_model(info) is None


def test_auto_map_in_a_processor_config_is_not_switchable(tmp_path):
    from utils.security.remote_code_scan import REMOTE_CODE_CONFIG_FILES
    for name in REMOTE_CODE_CONFIG_FILES:
        path = _local_checkpoint(tmp_path, f"custom-{name}")
        info = SimpleNamespace(id = str(path), path = str(path))
        assert resolver.local_servable_model(info) is not None
        blob = '{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "auto_map": {"A": "m.M"}}'
        (path / name).write_text(blob)
        assert resolver.local_servable_model(info) is None, name


def test_a_non_generative_checkpoint_is_not_switchable(tmp_path):
    # Only chat generation uses these entries, so encoder-only or classifier checkpoints are refused.

    path = _local_checkpoint(tmp_path, "bert-base")
    (path / "config.json").write_text('{"architectures": ["BertForSequenceClassification"]}')
    info = SimpleNamespace(id = str(path), path = str(path))
    assert resolver.local_servable_model(info) is None


def test_a_lora_directory_with_a_copied_config_is_not_switchable(tmp_path):
    # Adapters resolve a base model that may need downloading, which this resolver must never do.

    path = _local_checkpoint(tmp_path, "SomeLoRA")
    info = SimpleNamespace(id = str(path), path = str(path))
    assert resolver.local_servable_model(info) is not None
    (path / "adapter_config.json").write_text("{}")
    assert resolver.local_servable_model(info) is None


def test_an_audio_only_request_does_not_demand_vision_from_a_non_gguf_target(monkeypatch):
    # needs_vision for audio applies to GGUF projectors only, not to non-GGUF audio checkpoints.
    monkeypatch.setattr(inference_route, "_target_accepts_audio_input", lambda path: True)
    monkeypatch.setattr(
        inference_route, "_target_is_vision", lambda *_a: pytest.fail("vision probed")
    )
    assert (
        inference_route._target_accepts_request_input(
            "/srv/models/Whisper", False, True, True, None, False
        )
        is True
    )


def test_a_nested_non_gguf_row_is_not_marked_resident(monkeypatch):
    # A non-GGUF model loads from its own dir, so a nested catalog row is different weights.
    class _FakeOrchestrator:
        active_model_name = "/models/A"

    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _FakeBackend(None))
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: _FakeOrchestrator())
    assert inference_route._resolves_to_resident("/models/A") is True
    assert inference_route._resolves_to_resident("/models/A/sub/B") is False


def test_a_text_seq2seq_checkpoint_is_not_switchable(tmp_path):
    path = _local_checkpoint(tmp_path, "t5-base")
    info = SimpleNamespace(id = str(path), path = str(path))
    (path / "config.json").write_text(
        '{"architectures": ["T5ForConditionalGeneration"], "model_type": "t5"}'
    )
    assert resolver.local_servable_model(info) is None
    (path / "config.json").write_text(
        '{"architectures": ["Gemma3ForConditionalGeneration"], "model_type": "gemma3", "vision_config": {}}'
    )
    assert resolver.local_servable_model(info) == (False, ())


def test_a_whisper_checkpoint_is_switchable(tmp_path):
    """Whisper wears ForConditionalGeneration and carries no multimodal sub-config, being
    the audio model rather than wearing one, so requiring that key filtered it out with T5.
    The orchestrator loads it through WhisperForConditionalGeneration and chat routes the
    request to generate_whisper_response, so it is a target this server actually serves."""
    path = _local_checkpoint(tmp_path, "whisper-large-v3")
    info = SimpleNamespace(id = str(path), path = str(path))
    (path / "config.json").write_text(
        '{"architectures": ["WhisperForConditionalGeneration"], "model_type": "whisper",'
        ' "is_encoder_decoder": true}'
    )
    assert resolver.local_servable_model(info) == (False, ())
    assert resolver._model_type_is_audio("whisper") is True
    assert resolver._model_type_is_audio("csm") is True
    assert resolver._model_type_is_audio("t5") is False
    assert resolver._model_type_is_audio(None) is False


@pytest.mark.parametrize(
    ("architecture", "model_type"),
    [
        ("Speech2TextForConditionalGeneration", "speech_to_text"),
        ("MusicgenForConditionalGeneration", "musicgen"),
    ],
)
def test_unsupported_conditional_audio_checkpoints_are_not_switchable(
    tmp_path, architecture, model_type
):
    path = _local_checkpoint(tmp_path, model_type)
    info = SimpleNamespace(id = str(path), path = str(path))
    (path / "config.json").write_text(
        json.dumps({"architectures": [architecture], "model_type": model_type})
    )
    assert resolver.local_servable_model(info) is None


def test_an_mlx_host_does_not_advertise_an_asr_checkpoint(tmp_path, monkeypatch):
    """The MLX worker refuses ASR outright at worker.py:823 and TTS at 1103, so a Whisper
    checkpoint on Apple Silicon can only be loaded to fail. Same host-capability rule as
    _host_has_a_non_gguf_backend, not a prediction about the load."""
    path = _local_checkpoint(tmp_path, "whisper-large-v3")
    info = SimpleNamespace(id = str(path), path = str(path))
    (path / "config.json").write_text(
        '{"architectures": ["WhisperForConditionalGeneration"], "model_type": "whisper",'
        ' "is_encoder_decoder": true}'
    )
    monkeypatch.setattr(resolver, "_host_serves_mlx", lambda: True)
    assert resolver.local_servable_model(info) is None
    # an audio VLM declares a multimodal sub-config, which MLX does serve, so it stays listed.
    (path / "config.json").write_text(
        '{"architectures": ["Gemma3nForConditionalGeneration"], "model_type": "gemma3n",'
        ' "audio_config": {}}'
    )
    assert resolver.local_servable_model(info) == (False, ())


def test_a_conditional_checkpoint_with_no_vision_sub_config_is_switchable(tmp_path, monkeypatch):
    """A conversion that drops the vision tower keeps the parent's multimodal architecture name
    but loses the sub-config, so demanding one withheld a checkpoint both workers load. Shape
    taken from ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit, whose weights hold only language_model.*
    and whose config carries image_token_id and text_config but no vision_config at all."""
    path = _local_checkpoint(tmp_path, "Ornith-MLX-4bit")
    info = SimpleNamespace(id = str(path), path = str(path))
    (path / "config.json").write_text(
        '{"architectures": ["Qwen3_5MoeForConditionalGeneration"],'
        ' "model_type": "qwen3_5_moe", "image_token_id": 151655, "text_config": {}}'
    )
    for mlx_host in (True, False):
        monkeypatch.setattr(resolver, "_host_serves_mlx", lambda mlx_host = mlx_host: mlx_host)
        assert resolver.local_servable_model(info) == (False, ()), mlx_host
    for marker in (
        '"image_token_id": 151655',
        '"image_token_index": 1',
        '"vision_config": {}',
        '"video_token_id": 2',
        '"img_processor": {}',
    ):
        (path / "config.json").write_text(
            '{"architectures": ["Qwen3_5MoeForConditionalGeneration"],'
            ' "model_type": "qwen3_5_moe", %s}' % marker
        )
        assert resolver.local_servable_model(info) == (False, ()), marker
    # A config with no modality marker looks like text seq2seq, so it stays refused.
    (path / "config.json").write_text(
        '{"architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe"}'
    )
    assert resolver.local_servable_model(info) is None


def test_a_multimodal_encoder_decoder_is_not_switchable(tmp_path):
    """Declaring a modality does not make a checkpoint servable here: microsoft/udop-large is an
    encoder-decoder carrying image_size, and the serving path has no AutoModelForSeq2SeqLM branch.
    The flag is what refuses it, since the modality marker is satisfied."""
    path = _local_checkpoint(tmp_path, "udop-large")
    info = SimpleNamespace(id = str(path), path = str(path))
    (path / "config.json").write_text(
        '{"architectures": ["UdopForConditionalGeneration"], "model_type": "udop",'
        ' "is_encoder_decoder": true, "image_size": 224}'
    )
    assert resolver.local_servable_model(info) is None
    (path / "config.json").write_text(
        '{"architectures": ["SomeVlmForConditionalGeneration"], "model_type": "some_vlm",'
        ' "image_size": 224}'
    )
    assert resolver.local_servable_model(info) == (False, ())


def test_a_revision_key_does_not_pass_as_a_modality_marker(tmp_path):
    """The marker is matched on whole words: `revision` ends in one, is common in a saved config,
    and would otherwise admit every text seq2seq that carries it."""
    path = _local_checkpoint(tmp_path, "t5-with-revision")
    info = SimpleNamespace(id = str(path), path = str(path))
    (path / "config.json").write_text(
        '{"architectures": ["T5ForConditionalGeneration"], "model_type": "t5",'
        ' "revision": "main"}'
    )
    assert resolver.local_servable_model(info) is None


@pytest.mark.parametrize("mlx_host", [True, False])
@pytest.mark.parametrize("architecture", ["LlamaForCausalLM", "Qwen3_5MoeForConditionalGeneration"])
def test_a_config_declaring_model_file_is_not_switchable(
    tmp_path, monkeypatch, architecture, mlx_host
):
    """model_file is the other key that runs code out of the checkpoint, and unlike auto_map it
    does not pass through trust_remote_code at all: mlx_lm/utils.py and mlx_vlm/utils.py both
    exec_module the named file before dispatching on model_type. An unattended switch grants no
    approval, so it is refused on the same boundary as auto_map."""
    monkeypatch.setattr(resolver, "_host_serves_mlx", lambda: mlx_host)
    path = _local_checkpoint(tmp_path, "CustomModelFile")
    info = SimpleNamespace(id = str(path), path = str(path))
    (path / "custom.py").write_text("raise SystemExit('should never be executed')")
    base = (
        '{"architectures": ["%s"], "model_type": "qwen3_5_moe", "vision_config": {}%%s}'
        % architecture
    )
    (path / "config.json").write_text(base % "")
    assert resolver.local_servable_model(info) == (False, ())
    (path / "config.json").write_text(base % ', "model_file": "custom.py"')
    assert resolver.local_servable_model(info) is None
    # Empty names no file, so it runs nothing, like the auto_map rule just below.
    (path / "config.json").write_text(base % ', "model_file": ""')
    assert resolver.local_servable_model(info) == (False, ())


def test_a_conditional_family_the_model_picker_refuses_is_not_switchable(tmp_path):
    """The resolver used to re-derive the category from architecture strings and drifted from the
    classifier behind the picker's can_chat, which is how an installed checkpoint could be offered
    in the UI and be unknown to the API. The conditional branch defers to that classifier now, so
    the families it refuses are refused here without being restated.

    Only that branch. The resolver is deliberately not a subset overall: the causal fast path does
    not consult the classifier, and the audio branch serves whisper on a Transformers host though
    the classifier calls it unchattable."""
    from hub.services.models.common import _local_transformers_can_chat

    path = _local_checkpoint(tmp_path, "Shared")
    info = SimpleNamespace(id = str(path), path = str(path))
    refused_by_picker = 0
    for config in (
        '{"architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe",'
        ' "vision_config": {}}',
        '{"architectures": ["MusicgenForConditionalGeneration"], "model_type": "musicgen",'
        ' "audio_encoder": {}}',
        '{"architectures": ["BlipForConditionalGeneration"], "model_type": "blip",'
        ' "vision_config": {}}',
        '{"architectures": ["Gemma3ForConditionalGeneration"], "model_type": "gemma3",'
        ' "vision_config": {}}',
    ):
        (path / "config.json").write_text(config)
        picker_can_chat = _local_transformers_can_chat(path) is True
        servable = resolver.local_servable_model(info) is not None
        if not picker_can_chat:
            refused_by_picker += 1
            assert not servable, config
    # musicgen and blip, so the subset assertion above is not vacuous.
    assert refused_by_picker == 2


def test_an_empty_auto_map_is_not_remote_code(tmp_path):
    """The consent gate reads auto_map for truthiness (_config_has_auto_map in
    utils/security/consent.py), so an empty or null mapping names no implementation and
    executes nothing. Rejecting on the key's presence alone hid a loadable checkpoint
    while adding no approval boundary."""
    path = _local_checkpoint(tmp_path, "EmptyAutoMap")
    info = SimpleNamespace(id = str(path), path = str(path))
    base = '{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "auto_map": %s}'
    for empty in ("{}", "null"):
        (path / "config.json").write_text(base % empty)
        assert resolver.local_servable_model(info) == (False, ()), empty
    (path / "config.json").write_text(base % '{"AutoModel": "modeling.MyModel"}')
    assert resolver.local_servable_model(info) is None


def test_a_host_without_a_non_gguf_backend_advertises_none(tmp_path, monkeypatch):
    path = _local_checkpoint(tmp_path, "NoBackend")
    info = SimpleNamespace(id = str(path), path = str(path))
    monkeypatch.setattr(resolver, "_host_has_a_non_gguf_backend", lambda: False)
    assert resolver.local_servable_model(info) is None


def test_a_nested_row_is_not_resident_against_a_loaded_gguf_directory(monkeypatch):
    # Loading /models/A must not mark a nested /models/A/sub/B as resident via the prefix rule.
    llama = _FakeBackend("/models/A")
    llama.gguf_path = "/models/A"

    class _FakeOrchestrator:
        active_model_name = None

    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: llama)
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: _FakeOrchestrator())
    assert inference_route._resolves_to_resident("/models/A/sub/B", exact_only = True) is False
    assert inference_route._resolves_to_resident("/models/A", exact_only = True) is True


def test_a_non_canonical_safetensors_name_is_not_switchable(tmp_path):
    path = _local_checkpoint(tmp_path, "VariantOnly")
    (path / "model.safetensors").rename(path / "model.fp16.safetensors")
    info = SimpleNamespace(id = str(path), path = str(path))
    assert resolver.local_servable_model(info) is None


def test_a_causal_model_without_the_suffix_stays_switchable(tmp_path):
    path = _local_checkpoint(tmp_path, "gpt2")
    info = SimpleNamespace(id = str(path), path = str(path))
    (path / "config.json").write_text(
        '{"architectures": ["GPT2LMHeadModel"], "model_type": "gpt2"}'
    )
    assert resolver.local_servable_model(info) == (False, ())


def test_a_model_type_without_architectures_is_not_switchable(tmp_path):
    path = _local_checkpoint(tmp_path, "no-arch")
    info = SimpleNamespace(id = str(path), path = str(path))
    for model_type in ("t5", "deberta-v2", "qwen3"):
        (path / "config.json").write_text(json.dumps({"model_type": model_type}))
        assert resolver.local_servable_model(info) is None, model_type

    import transformers  # noqa: F401

    for model_type in ("t5", "deberta-v2", "qwen3"):
        (path / "config.json").write_text(json.dumps({"model_type": model_type}))
        assert resolver.local_servable_model(info) is None, model_type


def test_the_advertised_alias_is_cleared_before_a_replacement_load(tmp_path):
    # Clear the alias before load_model, or the old name briefly matches the new weights.

    src = inspect.getsource(inference_route._load_model_impl)
    # Anchor on line start so llama_backend.load_model is not matched as a substring.
    clear = src.index("\n        backend._openai_advertised_id = None")
    load = src.index("\n                backend.load_model,")
    assert clear < load, "the alias must be cleared before load_model publishes the new model"


def test_a_failed_replacement_keeps_the_surviving_model_advertised():
    """A repair raises SidecarSwapInProgress before the old worker is torn down, so the
    orchestrator keeps active_model_name and that model goes on serving. The alias was
    already cleared for the load that never happened, so without restoring it the still
    resident model loses the id /v1/models and response ids report it under."""

    class _Orchestrator:
        def __init__(self, active):
            self.active_model_name = active
            self._openai_advertised_id = None

    survived = _Orchestrator("/srv/models/Qwen3-8B-MLX")
    inference_route._restore_alias_if_failed_load_left_the_prior_model(
        survived, "mlx-community/Qwen3-8B-4bit", "/srv/models/Qwen3-8B-MLX"
    )
    assert survived._openai_advertised_id == "mlx-community/Qwen3-8B-4bit"

    torn_down = _Orchestrator(None)
    inference_route._restore_alias_if_failed_load_left_the_prior_model(
        torn_down, "mlx-community/Qwen3-8B-4bit", "/srv/models/Qwen3-8B-MLX"
    )
    assert torn_down._openai_advertised_id is None

    # a different model is resident, so the old alias must not be pinned onto it.
    replaced = _Orchestrator("/srv/models/Other")
    inference_route._restore_alias_if_failed_load_left_the_prior_model(
        replaced, "mlx-community/Qwen3-8B-4bit", "/srv/models/Qwen3-8B-MLX"
    )
    assert replaced._openai_advertised_id is None

    cold = _Orchestrator(None)
    inference_route._restore_alias_if_failed_load_left_the_prior_model(cold, None, None)
    assert cold._openai_advertised_id is None


def test_both_failed_load_exits_restore_the_alias():
    src = inspect.getsource(inference_route._load_model_impl)
    assert src.count("_restore_alias_if_failed_load_left_the_prior_model(") == 2
    clear = src.index("\n        backend._openai_advertised_id = None")
    assert src.index("_restore_alias_if_failed_load_left_the_prior_model(") > clear
    assert (
        src.index("_restore_alias_if_failed_load_left_the_prior_model(", clear)
        < src.index("\n        if not success:")
        < src.rindex("_restore_alias_if_failed_load_left_the_prior_model(")
    )


def test_a_stale_alias_after_unload_is_not_a_resident_identity(monkeypatch):
    class _FakeOrchestrator:
        active_model_name = None
        _openai_advertised_id = "unsloth/Qwen3-MLX"

    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _FakeBackend(None))
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: _FakeOrchestrator())
    assert inference_route._resident_id_is_namespaced() is False


def test_the_audio_preflight_only_binds_a_non_gguf_target(monkeypatch):
    import wave

    monkeypatch.setattr(
        inference_route,
        "_decode_audio_base64",
        lambda _b64: pytest.fail("the non-GGUF decoder ran for a GGUF target"),
    )
    monkeypatch.setattr(inference_route, "_audio_decoder_is_available", lambda: False)
    wav = io.BytesIO()
    with wave.open(wav, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * 16)
    preflight = {
        "clips": [f"data:audio/wav;base64,{_b64.b64encode(wav.getvalue()).decode()}"],
        "continue_final": True,
    }
    asyncio.run(inference_route._preflight_audio_for_switch(preflight, True))
    assert preflight["prepared"][0][1] == "wav"
    assert "decoded" not in preflight

    # the non-GGUF branch runs _decode_audio_base64, so it refuses the same input, before the load.
    monkeypatch.setattr(inference_route, "_audio_decoder_is_available", lambda: True)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route._preflight_audio_for_switch(dict(preflight), False))
    assert exc.value.status_code == 400
    assert "continue_final_message" in exc.value.detail
    assert "audio input" in exc.value.detail


def test_a_prior_turn_image_does_not_block_a_non_gguf_audio_switch(monkeypatch):
    llama = _FakeBackend("org/A-GGUF")

    class _FakeOrchestrator:
        active_model_name = None
        models: dict = {}

    orchestrator = _FakeOrchestrator()
    calls = []

    async def _load(request, *_args, **_kwargs):
        calls.append(request)
        orchestrator.active_model_name = request.model_path

    _wire_on(
        monkeypatch,
        resolves_to = ("/srv/models/Whisper", None, "org/Whisper"),
        backend = llama,
        recorder = _load,
    )
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: orchestrator)
    monkeypatch.setattr(inference_route, "_peek_inference_backend", lambda: orchestrator)
    monkeypatch.setattr(resolver, "local_target_is_gguf", lambda *_a, **_kw: False)
    monkeypatch.setattr(inference_route, "_target_accepts_audio_input", lambda _path: True)
    monkeypatch.setattr(
        inference_route,
        "_target_is_vision",
        lambda *_a, **_kw: pytest.fail("a prior-turn image demanded vision"),
    )
    monkeypatch.setattr(inference_route, "_audio_decoder_is_available", lambda: True)
    monkeypatch.setattr(inference_route, "_decode_audio_base64", lambda _b64: "pcm")

    asyncio.run(
        inference_route._maybe_auto_switch_model(
            "org/Whisper",
            object(),
            "tester",
            require_vision = True,
            require_image = True,
            require_audio_input = True,
            audio_preflight = {
                "clips": ["valid"],
                "continue_final": False,
                "has_image": False,
            },
        )
    )

    assert [request.model_path for request in calls] == ["/srv/models/Whisper"]
    assert orchestrator._openai_advertised_id == "org/Whisper"


@pytest.mark.parametrize(
    "image_preflight",
    [
        {"b64": "bm90IGFuIGltYWdl", "multiple": False},
        "truncated image",
        {"b64": None, "multiple": True},
    ],
    ids = ["malformed bytes", "truncated image", "multiple images"],
)
def test_invalid_non_gguf_images_are_rejected_before_switch(monkeypatch, image_preflight):
    if image_preflight == "truncated image":
        from PIL import Image

        encoded = io.BytesIO()
        Image.new("RGB", (8, 8), "red").save(encoded, format = "JPEG")
        image_preflight = {
            "b64": _b64.b64encode(encoded.getvalue()[:-1]).decode(),
            "multiple": False,
        }
    llama = _FakeBackend("org/A-GGUF")

    class _FakeOrchestrator:
        active_model_name = None
        models: dict = {}

    orchestrator = _FakeOrchestrator()
    calls = []

    async def _load(request, *_args, **_kwargs):
        calls.append(request)
        orchestrator.active_model_name = request.model_path

    _wire_on(
        monkeypatch,
        resolves_to = ("/srv/models/Vision", None, "org/Vision"),
        backend = llama,
        recorder = _load,
    )
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: orchestrator)
    monkeypatch.setattr(inference_route, "_peek_inference_backend", lambda: orchestrator)
    monkeypatch.setattr(resolver, "local_target_is_gguf", lambda *_a, **_kw: False)
    monkeypatch.setattr(inference_route, "_target_accepts_request_input", lambda *_a: True)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            inference_route._maybe_auto_switch_model(
                "org/Vision",
                object(),
                "tester",
                require_vision = True,
                image_preflight = image_preflight,
            )
        )

    assert exc.value.status_code == 400
    assert calls == []
    assert llama.model_identifier == "org/A-GGUF"


def test_a_malformed_gguf_image_is_rejected_before_switch(monkeypatch):
    llama = _FakeBackend("org/A-GGUF")
    recorder = _LoadRecorder(llama)
    _wire_on(
        monkeypatch,
        resolves_to = ("/srv/models/B.gguf", "Q4_K_M", "org/B-GGUF"),
        backend = llama,
        recorder = recorder,
    )
    monkeypatch.setattr(resolver, "local_target_is_gguf", lambda *_a, **_kw: True)
    monkeypatch.setattr(inference_route, "_target_accepts_request_input", lambda *_a: True)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            inference_route._maybe_auto_switch_model(
                "org/B-GGUF",
                object(),
                "tester",
                require_vision = True,
                image_preflight = {
                    "b64": "bm90IGFuIGltYWdl",
                    "b64s": ["bm90IGFuIGltYWdl"],
                    "multiple": False,
                },
            )
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "Failed to process image."
    assert recorder.calls == []
    assert llama.model_identifier == "org/A-GGUF"


def test_gguf_image_preflight_allows_multiple_valid_images():
    from PIL import Image

    encoded = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(encoded, format = "PNG")
    image_b64 = _b64.b64encode(encoded.getvalue()).decode()

    asyncio.run(
        inference_route._preflight_image_for_switch(
            {"b64s": [image_b64, image_b64], "multiple": True},
            True,
        )
    )


def test_the_image_preflight_collects_every_local_request_image():
    payload = _chat_request(
        image_base64 = "legacy",
        messages = [
            ChatMessage(
                role = "user",
                content = [
                    ImageContentPart(
                        type = "image_url",
                        image_url = ImageUrl(url = "data:image/png;base64,first"),
                    ),
                    ImageContentPart(
                        type = "image_url",
                        image_url = ImageUrl(url = "https://example.com/remote.png"),
                    ),
                    ImageContentPart(
                        type = "image_url",
                        image_url = ImageUrl(url = "data:image/png;base64,second"),
                    ),
                ],
            )
        ],
    )

    assert inference_route._request_local_image_payloads(payload) == [
        "first",
        "second",
        "legacy",
    ]


def _wire_image_switch_target(monkeypatch, *, target_is_gguf):
    backend = _FakeBackend("org/A-GGUF")
    recorder = _LoadRecorder(backend)
    target_path = "/srv/models/B.gguf" if target_is_gguf else "/srv/models/B"
    _wire_on(
        monkeypatch,
        resolves_to = (target_path, None, "org/B-GGUF"),
        backend = backend,
        recorder = recorder,
    )
    monkeypatch.setattr(resolver, "local_target_is_gguf", lambda *_a, **_kw: target_is_gguf)
    monkeypatch.setattr(inference_route, "_target_accepts_request_input", lambda *_a: True)
    if not target_is_gguf:
        orchestrator = types.SimpleNamespace(active_model_name = None, models = {})
        monkeypatch.setattr(inference_route, "get_inference_backend", lambda: orchestrator)
        monkeypatch.setattr(inference_route, "_peek_inference_backend", lambda: orchestrator)
    return backend, recorder


def _chat_image_request(*urls):
    return _chat_request(
        model = "org/B-GGUF",
        messages = [
            ChatMessage(
                role = "user",
                content = [
                    ImageContentPart(type = "image_url", image_url = ImageUrl(url = url)) for url in urls
                ],
            )
        ],
    )


def test_chat_rejects_an_empty_data_url_before_non_gguf_switch(monkeypatch):
    backend, recorder = _wire_image_switch_target(monkeypatch, target_is_gguf = False)
    payload = _chat_image_request("data:image/png;base64,")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))

    assert exc.value.status_code == 400
    assert exc.value.detail == "Failed to decode image"
    assert recorder.calls == []
    assert backend.model_identifier == "org/A-GGUF"


@pytest.mark.parametrize("count", [1, 2])
def test_chat_loads_a_non_gguf_target_for_remote_images(monkeypatch, count):
    # The loaded model fetches the URLs as a resident one does, so the switch is not refused.
    _, recorder = _wire_image_switch_target(monkeypatch, target_is_gguf = False)
    monkeypatch.setattr(inference_route, "_local_target_may_take_several_images", lambda *_a: True)
    recorder.fail = True
    urls = [f"https://example.com/{index}.png" for index in range(count)]

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            inference_route.openai_chat_completions(_chat_image_request(*urls), object(), "tester")
        )

    assert exc.value.detail == "load failed"
    assert len(recorder.calls) == 1


@pytest.mark.parametrize(
    ("turns", "detail"),
    [
        ([["http://example.com/0.png"]], "Unsupported image URL scheme ('http:')"),
        (
            [["https://example.com/0.png", "http://example.com/1.png"]],
            "one image per message",
        ),
        (
            [["http://example.com/0.png"], ["https://example.com/1.png"]],
            "Unsupported image URL scheme ('http:')",
        ),
    ],
    ids = ["alone", "beside https", "earlier turn"],
)
def test_chat_refuses_an_unfetchable_scheme_before_non_gguf_switch(monkeypatch, turns, detail):
    backend, recorder = _wire_image_switch_target(monkeypatch, target_is_gguf = False)
    monkeypatch.setattr(inference_route, "_local_target_may_take_several_images", lambda *_a: True)
    payload = _chat_request(
        model = "org/B-GGUF",
        messages = [_chat_image_request(*urls).messages[0] for urls in turns],
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))

    assert detail in exc.value.detail
    assert recorder.calls == []
    assert backend.model_identifier == "org/A-GGUF"


def test_chat_leaves_an_unread_older_image_to_the_loaded_model(monkeypatch):
    # The newer remote image is the one the model reads, so the older one is not validated.
    _, recorder = _wire_image_switch_target(monkeypatch, target_is_gguf = False)
    recorder.fail = True
    turns = ["data:image/png;base64,Zm9v", "https://example.com/0.png"]
    payload = _chat_request(
        model = "org/B-GGUF",
        messages = [_chat_image_request(url).messages[0] for url in turns],
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))

    assert exc.value.detail == "load failed"
    assert len(recorder.calls) == 1


def test_chat_validates_a_legacy_image_beside_a_remote_one_before_non_gguf_switch(monkeypatch):
    backend, recorder = _wire_image_switch_target(monkeypatch, target_is_gguf = False)
    monkeypatch.setattr(inference_route, "_local_target_may_take_several_images", lambda *_a: True)
    payload = _chat_image_request("https://example.com/0.png")
    payload.image_base64 = "Zm9v"

    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))

    assert exc.value.detail == "Failed to decode image"
    assert recorder.calls == []
    assert backend.model_identifier == "org/A-GGUF"


def test_chat_refuses_a_remote_image_beside_a_legacy_one_before_non_gguf_switch(monkeypatch):
    backend, recorder = _wire_image_switch_target(monkeypatch, target_is_gguf = False)
    reply = _chat_image_request("https://example.com/0.png").messages[0]
    reply.role = "assistant"
    payload = _chat_request(
        model = "org/B-GGUF",
        messages = [
            ChatMessage(role = "user", content = "hi"),
            reply,
            ChatMessage(role = "user", content = "and?"),
        ],
        image_base64 = "aGVsbG8=",
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))

    assert "one image per message" in exc.value.detail
    assert recorder.calls == []
    assert backend.model_identifier == "org/A-GGUF"


def _responses_image_payload(*images, stream):
    return ResponsesRequest(
        model = "org/B-GGUF",
        stream = stream,
        input = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image}",
                    }
                    for image in images
                ],
            }
        ],
    )


def test_streaming_responses_rejects_a_malformed_gguf_image_before_switch(monkeypatch):
    backend, recorder = _wire_image_switch_target(monkeypatch, target_is_gguf = True)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            inference_route.openai_responses(
                _responses_image_payload("bm90IGFuIGltYWdl", stream = True),
                object(),
                "tester",
            )
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "Failed to process image."
    assert recorder.calls == []
    assert backend.model_identifier == "org/A-GGUF"


def test_nonstreaming_responses_defers_multiple_image_preflight_to_chat(monkeypatch):
    backend, recorder = _wire_image_switch_target(monkeypatch, target_is_gguf = False)
    request = types.SimpleNamespace(
        state = types.SimpleNamespace(skip_api_monitor = True),
        url = types.SimpleNamespace(path = "/v1/responses"),
        method = "POST",
        scope = {},
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            inference_route.openai_responses(
                _responses_image_payload("first", "second", stream = False),
                request,
                "tester",
            )
        )

    assert exc.value.status_code == 400
    assert "one image per message" in exc.value.detail
    assert recorder.calls == []
    assert backend.model_identifier == "org/A-GGUF"


def test_nonstreaming_responses_defers_malformed_gguf_preflight_to_chat(monkeypatch):
    backend, recorder = _wire_image_switch_target(monkeypatch, target_is_gguf = True)
    request = types.SimpleNamespace(
        state = types.SimpleNamespace(skip_api_monitor = True),
        url = types.SimpleNamespace(path = "/v1/responses"),
        method = "POST",
        scope = {},
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            inference_route.openai_responses(
                _responses_image_payload("bm90IGFuIGltYWdl", stream = False),
                request,
                "tester",
            )
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "Failed to process image."
    assert recorder.calls == []
    assert backend.model_identifier == "org/A-GGUF"


@pytest.mark.parametrize(
    "source",
    [
        {
            "type": "base64",
            "media_type": "image/png",
            "data": "bm90IGFuIGltYWdl",
        },
        {
            "type": "url",
            "url": "data:image/png;base64,bm90IGFuIGltYWdl",
        },
        {
            "type": "base64",
            "media_type": "image/png",
            "data": "",
        },
    ],
    ids = ["base64 source", "data url source", "empty base64 source"],
)
def test_anthropic_rejects_a_malformed_gguf_image_before_switch(monkeypatch, source):
    from models.inference import AnthropicMessage, AnthropicMessagesRequest

    backend, recorder = _wire_image_switch_target(monkeypatch, target_is_gguf = True)
    payload = AnthropicMessagesRequest(
        model = "org/B-GGUF",
        max_tokens = 16,
        messages = [
            AnthropicMessage(
                role = "user",
                content = [
                    {
                        "type": "image",
                        "source": source,
                    }
                ],
            )
        ],
    )

    with pytest.raises(HTTPException) as exc:
        asyncio.run(inference_route.anthropic_messages(payload, object(), "tester"))

    assert exc.value.status_code == 400
    assert exc.value.detail == "Failed to process image."
    assert recorder.calls == []
    assert backend.model_identifier == "org/A-GGUF"


def test_mixed_audio_and_image_is_rejected_before_a_non_gguf_switch(monkeypatch):
    llama = _FakeBackend("org/A-GGUF")

    class _FakeOrchestrator:
        active_model_name = None
        models: dict = {}

    recorder = _LoadRecorder(llama)
    _wire_on(
        monkeypatch,
        resolves_to = ("/srv/models/Audio-VLM", None, "org/Audio-VLM"),
        backend = llama,
        recorder = recorder,
    )
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: _FakeOrchestrator())
    monkeypatch.setattr(inference_route, "_peek_inference_backend", lambda: _FakeOrchestrator())
    monkeypatch.setattr(resolver, "local_target_is_gguf", lambda *_a, **_kw: False)
    monkeypatch.setattr(inference_route, "_target_accepts_request_input", lambda *_a: True)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            inference_route._maybe_auto_switch_model(
                "org/Audio-VLM",
                object(),
                "tester",
                require_vision = True,
                require_image = True,
                require_audio_input = True,
                audio_preflight = {
                    "clips": ["AAAA"],
                    "continue_final": False,
                    "has_image": True,
                },
            )
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == inference_route._AUDIO_IMAGE_INPUT_DETAIL
    assert recorder.calls == []
    assert llama.is_loaded is True


def test_audio_beside_a_clip_is_rejected_before_a_non_gguf_switch(monkeypatch):
    """The clip conflict is the first audio rule served, so the switch orders it first."""
    llama = _FakeBackend("org/A-GGUF")

    class _FakeOrchestrator:
        active_model_name = None
        models: dict = {}

    recorder = _LoadRecorder(llama)
    _wire(
        monkeypatch,
        enabled = True,
        resolves_to = ("/srv/models/Audio-VLM", None, "org/Audio-VLM"),
        backend = llama,
        recorder = recorder,
    )
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: _FakeOrchestrator())
    monkeypatch.setattr(inference_route, "_peek_inference_backend", lambda: _FakeOrchestrator())
    monkeypatch.setattr(resolver, "local_target_is_gguf", lambda *_a, **_kw: False)
    monkeypatch.setattr(inference_route, "_target_accepts_request_input", lambda *_a: True)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            inference_route._maybe_auto_switch_model(
                "org/Audio-VLM",
                object(),
                "tester",
                require_vision = True,
                require_image = False,
                require_audio_input = True,
                require_video = True,
                audio_preflight = {
                    "clips": ["AAAA"],
                    "continue_final": True,
                    "has_image": True,
                    "has_video": True,
                },
            )
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == inference_route._AUDIO_VIDEO_INPUT_DETAIL
    assert recorder.calls == []
    assert llama.is_loaded is True


def test_the_gguf_audio_preflight_takes_the_base64_llama_cpp_takes():
    """The preflight must refuse exactly what _prepare_audio_for_llama refuses.

    That helper decodes with the lenient default, which drops whitespace, so a
    validate = True preflight 400'd MIME-wrapped and newline-padded uploads the same
    server served before the preflight existed."""
    import wave

    wav = io.BytesIO()
    with wave.open(wav, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * 16)
    payload = wav.getvalue()
    plain = _b64.b64encode(payload).decode()
    for name, encoded in (
        ("plain", plain),
        ("trailing newline", plain + "\n"),
        ("surrounding spaces", f" {plain} "),
        ("mime wrapped", _b64.encodebytes(payload).decode()),
        ("data uri", f"data:audio/wav;base64,{plain}"),
    ):
        try:
            asyncio.run(
                inference_route._preflight_audio_for_switch(
                    {"clips": [encoded], "continue_final": False}, True
                )
            )
        except HTTPException:
            pytest.fail(f"the preflight refused {name} base64 that llama.cpp decodes")
    for bad in ("abc", "!!!!", "AA!!AA==", "   ", "data:audio/wav;base64", "\n\n"):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(
                inference_route._preflight_audio_for_switch(
                    {"clips": [bad], "continue_final": False}, True
                )
            )
        assert exc.value.status_code == 400, bad


def test_non_audio_bytes_are_rejected_before_a_gguf_switch(monkeypatch):
    llama = _FakeBackend("org/A-GGUF")
    recorder = _LoadRecorder(llama)
    _wire_on(
        monkeypatch,
        resolves_to = ("/srv/models/B.gguf", "Q4_K_M", "org/B-GGUF"),
        backend = llama,
        recorder = recorder,
    )
    monkeypatch.setattr(inference_route, "_target_accepts_request_input", lambda *_a: True)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            inference_route._maybe_auto_switch_model(
                "org/B-GGUF",
                object(),
                "tester",
                require_audio_input = True,
                audio_preflight = {
                    "clips": [_b64.b64encode(b"not audio").decode()],
                    "continue_final": False,
                },
            )
        )

    assert exc.value.status_code == 400
    assert "decoded as audio" in json.dumps(exc.value.detail)
    assert recorder.calls == []
    assert llama.model_identifier == "org/A-GGUF"


def test_a_non_gguf_audio_target_is_refused_without_a_decoder(monkeypatch):
    monkeypatch.setattr(inference_route, "_audio_decoder_is_available", lambda: False)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            inference_route._preflight_audio_for_switch(
                {"clips": ["AAAA"], "continue_final": False}, False
            )
        )
    assert exc.value.status_code == 400
    assert "torchaudio" in exc.value.detail["error"]["message"]


def test_non_audio_bytes_are_rejected_before_a_non_gguf_switch(monkeypatch):
    monkeypatch.setattr(inference_route, "_audio_decoder_is_available", lambda: True)

    def _decode(b64):
        if b64 != "GOOD":
            raise ValueError("not audio")
        return "pcm"

    monkeypatch.setattr(inference_route, "_decode_audio_base64", _decode)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            inference_route._preflight_audio_for_switch(
                {"clips": ["AAAA"], "continue_final": False}, False
            )
        )
    assert exc.value.status_code == 400

    preflight = {"clips": ["GOOD"], "continue_final": False}
    asyncio.run(inference_route._preflight_audio_for_switch(preflight, False))
    assert preflight["decoded"] == ["pcm"]


def test_a_gguf_only_host_does_not_need_torchaudio_to_accept_audio(monkeypatch):
    reached = {}

    async def _capture(*_a, **_kw):
        reached["switch"] = True
        raise RuntimeError("reached the switch")

    monkeypatch.setattr(settings, "get_openai_auto_switch_enabled", lambda: True)
    monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _capture)
    monkeypatch.setattr(inference_route, "_audio_decoder_is_available", lambda: False)
    monkeypatch.setattr(
        inference_route,
        "_decode_audio_base64",
        lambda _b64: pytest.fail("torchaudio decoder ran on a GGUF-only host"),
    )
    payload = _chat_request(model = "org/B-GGUF", audio_base64 = "AAAA")
    with pytest.raises(RuntimeError):
        asyncio.run(inference_route.openai_chat_completions(payload, object(), "tester"))
    assert reached.get("switch")


def test_a_host_with_no_accelerator_serves_gguf_only(monkeypatch):
    # unsloth's get_device_type raises without a GPU, so a CPU wheel proves nothing.

    monkeypatch.setattr(resolver, "_host_serves_mlx", lambda: False)
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.CPU, raising = False)
    assert _REAL_HOST_HAS_NON_GGUF_BACKEND() is False
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.CUDA, raising = False)
    assert _REAL_HOST_HAS_NON_GGUF_BACKEND() is True


def test_a_serving_hf_cache_row_is_reported_loaded(tmp_path, monkeypatch):
    repo = tmp_path / "models--org--Chat"
    snapshot = repo / "snapshots" / "abc"
    snapshot.mkdir(parents = True)
    (snapshot / "config.json").write_text(_CHAT_CONFIG)
    (snapshot / "tokenizer.json").write_text("{}")
    (snapshot / "tokenizer_config.json").write_text('{"chat_template": "{{ messages }}"}')
    (snapshot / "model.safetensors").write_bytes(_safetensors_bytes())

    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: SimpleNamespace(active_model_name = str(snapshot), models = {}),
    )
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _FakeBackend(None))
    info = SimpleNamespace(id = "org/Chat", model_id = "org/Chat", path = str(repo))
    rows = inference_route._servable_catalog_rows([info])
    assert [(is_gguf, quants, resident) for _i, is_gguf, quants, resident in rows] == [
        (False, (), True)
    ]


def _speech_case(
    monkeypatch,
    audio_type = "snac",
    gguf = True,
):
    backend = _FakeBackend("org/A-GGUF")
    recorder = _LoadRecorder(backend)
    target = ("/local/B.gguf", "Q8_0", "org/B-GGUF") if gguf else ("/local/B", None, "org/B-GGUF")
    _wire(monkeypatch, enabled = True, resolves_to = target, backend = backend, recorder = recorder)
    monkeypatch.setattr(inference_route, "_target_speech_audio_type", lambda *_a: audio_type)
    return backend, recorder


@pytest.mark.parametrize(
    "audio_type,context,text,instructions,status",
    [
        (None, 8192, "hi", "", 400),
        ("snac", 8192, "hi", "", None),
        ("snac", 512, "x" * 1000, "", 400),
        ("snac", None, "hi", "", None),
        ("higgs_tts2", 512, "hi", "x" * 1000, 400),
        ("minimax_music3", None, "hi", "", 400),
        ("minimax_music3", None, "hi", "a warm ballad", None),
    ],
)
def test_speech_switch_admission(monkeypatch, audio_type, context, text, instructions, status):
    backend, recorder = _speech_case(monkeypatch, audio_type, audio_type in (None, "snac"))
    monkeypatch.setattr(inference_route, "_target_effective_context_length", lambda *_a: context)
    call = inference_route._maybe_auto_switch_model(
        "org/B-GGUF",
        object(),
        "tester",
        require_speech = True,
        speech_budget = {"text": text, "instructions": instructions},
    )
    if status:
        with pytest.raises(HTTPException) as error:
            asyncio.run(call)
        assert error.value.status_code == status
        assert recorder.calls == [] and backend.model_identifier == "org/A-GGUF"
    else:
        asyncio.run(call)
        assert len(recorder.calls) == 1


@pytest.mark.parametrize(
    "workflow, speech_type, workflows, admitted",
    [
        ("separate", None, ["separate"], True),
        ("separate", None, [], False),
        ("clone", "audiocpp_tts", ["speak"], False),
        ("clone", "audiocpp_tts", ["speak", "clone"], True),
    ],
)
def test_audio_workflow_switch_admission(monkeypatch, workflow, speech_type, workflows, admitted):
    backend, recorder = _speech_case(monkeypatch, speech_type, gguf = False)
    seen = []
    monkeypatch.setattr(
        inference_route, "_target_audio_workflows", lambda *a: seen.append(a) or workflows
    )
    call = inference_route._maybe_auto_switch_model(
        "org/B-GGUF",
        object(),
        "tester",
        require_speech = speech_type is not None,
        require_audio_workflow = workflow,
    )
    if admitted:
        asyncio.run(call)
        assert len(recorder.calls) == 1
    else:
        with pytest.raises(HTTPException) as error:
            asyncio.run(call)
        assert error.value.status_code == 400
        assert error.value.detail["error"]["param"] == "model"
        # Refused before the resident model is evicted.
        assert recorder.calls == [] and backend.model_identifier == "org/A-GGUF"
    assert seen == [("/local/B", speech_type)]


def test_target_audio_workflows_reads_audio_cpp_targets_from_the_cache(monkeypatch):
    from core.inference import audio_cpp_models

    sep = SimpleNamespace(workflows = {"separate": None})
    monkeypatch.setattr(audio_cpp_models, "resolve", lambda target, network: sep)
    workflows = inference_route._target_audio_workflows
    assert workflows("audio-cpp/audio.cpp-gguf/HTDemucs-6stems-GGUF", None) == ["separate"]
    assert workflows("/local/csm", "csm") == ["speak"]
    assert workflows("/local/chat", None) == []


@pytest.mark.parametrize("audio_type", ["snac", "bicodec", "dac", "higgs_tts2"])
@pytest.mark.parametrize("token", [None, "caller-token"])
def test_speech_switch_preserves_staged_assets_and_identity(monkeypatch, audio_type, token):
    from starlette.requests import Request

    _, recorder = _speech_case(monkeypatch, audio_type, audio_type != "higgs_tts2")
    staged = inference_route._SpeechCodecPreflightResult(
        {"HF_HUB_CACHE": "/captured/hub"}, "/captured/codec"
    )
    seen = []
    monkeypatch.setattr(
        inference_route, "_preflight_speech_codec_for_switch", lambda *a: seen.append(a) or staged
    )
    headers = [(b"x-unsloth-hf-token", token.encode())] if token else []
    request = Request({"type": "http", "path": "/v1/audio/speech", "headers": headers})
    asyncio.run(
        inference_route._maybe_auto_switch_model(
            "org/B-GGUF", request, "tester", require_speech = True
        )
    )
    assert seen[0][-1] == token
    assert recorder.cache_environment == staged.cache_environment
    assert recorder.speech_codec_path == staged.codec_path
    assert recorder.anonymous_hf_access is (token is None)
    assert recorder.calls[0].hf_token == token


@pytest.mark.parametrize("condition", ["complete", "missing", "unsafe"])
@pytest.mark.parametrize("token", [None, "caller-token"])
def test_higgs_companion_preflight_before_eviction(tmp_path, monkeypatch, condition, token):
    import huggingface_hub
    from core.inference import native_audio
    from utils import hf_cache_settings, security, utils
    from starlette.requests import Request

    preflight = inference_route._preflight_speech_codec_for_switch
    backend, recorder = _speech_case(monkeypatch, "higgs_tts2", False)
    monkeypatch.setattr(inference_route, "_preflight_speech_codec_for_switch", preflight)
    codec = tmp_path / "codec"
    codec.mkdir()
    (codec / "config.json").write_text(json.dumps({"model_type": "higgs_audio_v2_tokenizer"}))
    if condition != "missing":
        (codec / "model.safetensors").write_bytes(_safetensors_bytes())
    cache = hf_cache_settings.HuggingFaceCachePaths(
        tmp_path, tmp_path / "hub", tmp_path / "xet", "studio"
    )
    seen = []
    monkeypatch.setattr(hf_cache_settings, "get_hf_cache_paths", lambda: cache)
    monkeypatch.setattr(utils, "hf_env_offline", lambda: True)
    monkeypatch.setattr(
        native_audio, "native_audio_security_targets", lambda p, *_a: [p, "org/codec"]
    )
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda repo, **kw: seen.append(kw) or str(codec)
    )
    monkeypatch.setattr(
        security,
        "evaluate_file_security",
        lambda *_a, **_k: types.SimpleNamespace(blocked = condition == "unsafe", reason = "unsafe"),
    )
    headers = [(b"x-unsloth-hf-token", token.encode())] if token else []
    request = Request({"type": "http", "path": "/v1/audio/speech", "headers": headers})
    call = inference_route._maybe_auto_switch_model(
        "org/B-GGUF", request, "tester", require_speech = True
    )
    if condition == "complete":
        asyncio.run(call)
        assert len(recorder.calls) == 1
    else:
        with pytest.raises(HTTPException) as error:
            asyncio.run(call)
        assert error.value.status_code == 503
        assert recorder.calls == [] and backend.is_loaded
    assert seen == [
        {"token": token or False, "cache_dir": str(cache.hub_cache), "local_files_only": True}
    ]


def test_speech_budget_rechecks_changing_load_override(monkeypatch):
    backend, recorder = _speech_case(monkeypatch)
    reads = []

    def override(*_a):
        value = 8192 if not reads else 512
        reads.append(value)
        return "org/B-GGUF:Q8_0", {"max_seq_length": value}

    monkeypatch.setattr(settings, "resolve_override_for_load", override)
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            inference_route._maybe_auto_switch_model(
                "org/B-GGUF",
                object(),
                "tester",
                require_speech = True,
                speech_budget = {"text": "x" * 1000},
            )
        )
    assert reads == [8192, 512] and error.value.status_code == 400
    assert recorder.calls == [] and backend.model_identifier == "org/A-GGUF"


@pytest.mark.parametrize("fault", [None, "weights", "metadata", "remote-code"])
def test_minimax_pipeline_completeness(tmp_path, monkeypatch, fault):
    pipeline = _complete_minimax_pipeline(tmp_path)
    monkeypatch.setattr(resolver, "_host_can_serve_minimax_music3", lambda: True)
    if fault == "weights":
        (pipeline / "vocoder" / "diffusion_pytorch_model.safetensors").unlink()
    elif fault == "metadata":
        (pipeline / "tokenizer" / "tokenizer.json").write_text("not-json")
    elif fault == "remote-code":
        (pipeline / "language_model" / "config.json").write_text(
            json.dumps({"auto_map": {"AutoModel": "modeling.Custom"}})
        )
    info = types.SimpleNamespace(id = "MiniMaxAI/MiniMax-Music3", path = str(pipeline))
    assert resolver.local_servable_model(info) == ((False, ()) if fault is None else None)


@pytest.mark.parametrize(
    "audio_type,allowed",
    [
        ("moss_tts_local", False),
        ("moss_tts_nano", False),
        ("higgs_tts3", False),
        ("higgs_tts2", True),
    ],
)
def test_speech_probe_refuses_remote_code(monkeypatch, audio_type, allowed):
    from utils.models import model_config
    monkeypatch.setattr(model_config, "detect_audio_type", lambda *_a, **_kw: audio_type)
    assert inference_route._target_speech_audio_type("/local/model", False) == (
        audio_type if allowed else None
    )


def test_chat_withheld_model_named_by_its_advertised_alias(monkeypatch):
    # /v1/models shows "tiny" for unsloth/whisper-tiny, so that name must read as downloaded.
    _wire_withheld_chat(
        monkeypatch,
        objects = [
            {"id": "org/A-GGUF"},
            {"id": "tiny", "task": "automatic-speech-recognition"},
        ],
        downloaded = ("org/A-GGUF", "unsloth/whisper-tiny"),
    )
    status, detail = _chat_error(_chat_request(model = "tiny"))
    assert status == 404
    assert "cannot serve it here" in detail
    assert "not downloaded on this server" not in detail


def test_chat_absent_model_with_only_resolver_withheld_checkpoints(monkeypatch):
    # Withheld checkpoints never enter the catalog, so testing catalog_objects would prove nothing.
    _wire_withheld_chat(
        monkeypatch,
        objects = [],
        downloaded = ("org/has-auto-map",),
    )
    status, detail = _chat_error(_chat_request(model = "org/nope-GGUF"))
    assert status == 404
    assert "none of the downloaded models is a chat model" in detail
    assert "no models are downloaded yet" not in detail


def test_the_settings_memo_is_cleared_around_every_test():
    # Drive the fixture directly, since xdist may split ordered tests across workers.
    import time

    import sys

    fixture = next(
        getattr(module, "_drop_the_settings_memo_between_tests")
        for module in list(sys.modules.values())
        if module is not None and hasattr(module, "_drop_the_settings_memo_between_tests")
    )

    key = (settings.OWNER.account_id, settings.OPENAI_AUTO_DOWNLOAD_SETTING_KEY)
    run = fixture.__wrapped__()
    settings._cache[key] = (time.monotonic(), True)
    next(run)
    assert settings._cache == {}, "the memo was not cleared before the test body"

    settings._cache[key] = (time.monotonic(), True)
    next(run, None)
    assert settings._cache == {}, "the memo was not cleared after the test body"


def test_withheld_refusal_survives_a_leaked_auto_download_flag(monkeypatch):
    # Seed the process-wide auto-download memo as a neighbour would; the 404 must not change.
    import time

    _wire_withheld_chat(monkeypatch, objects = [], downloaded = ("org/has-auto-map",))
    settings._cache[(settings.OWNER.account_id, settings.OPENAI_AUTO_DOWNLOAD_SETTING_KEY)] = (
        time.monotonic(),
        True,
    )
    status, detail = _chat_error(_chat_request(model = "org/nope-GGUF"))
    assert status == 404, f"a leaked auto-download flag turned the refusal into a {status}"
    assert "none of the downloaded models is a chat model" in detail


def test_a_stale_idle_reload_stash_diverts_a_refusal_into_a_reload(monkeypatch):
    # The idle stash reloads what was freed, which is only wrong when another test left it behind.
    asked: list = []

    _wire_withheld_chat(
        monkeypatch,
        objects = [
            {"id": "org/A-GGUF"},
            {"id": "tiny", "task": "automatic-speech-recognition"},
        ],
        downloaded = ("org/A-GGUF", "unsloth/whisper-tiny"),
    )

    def _load_model(config = None, **_kw):
        asked.append(getattr(config, "model_name", None) or config)
        raise _Reached()

    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: type(
            "_B",
            (),
            {"active_model_name": None, "models": {}, "load_model": staticmethod(_load_model)},
        )(),
    )
    kw._last_unloaded_model = ("unsloth/Idle-GGUF", "Q4_K_M", "unsloth/Idle-GGUF")
    import utils.models.model_config as model_config

    monkeypatch.setattr(
        model_config, "detect_gguf_model_remote", lambda identifier, hf_token = None: None
    )

    with pytest.raises(Exception):
        asyncio.run(
            inference_route.openai_chat_completions(_chat_request(model = "tiny"), object(), "tester")
        )

    assert any("Idle-GGUF" in str(one) for one in asked), (
        "the stale stash did not divert the request, so this test no longer covers the leak "
        f"the fixture exists to stop (asked: {asked})"
    )


def test_the_idle_reload_stash_is_cleared_around_every_test(tmp_path):
    # Driven directly because xdist ordering is luck; the manifest names real files to unlink.
    import sys

    fixture = next(
        getattr(module, "_drop_the_idle_reload_stash_between_tests")
        for module in list(sys.modules.values())
        if module is not None and hasattr(module, "_drop_the_idle_reload_stash_between_tests")
    )

    def _saved(name):
        slot = tmp_path / name
        slot.write_bytes(b"kv")
        return {"dir": str(tmp_path), "slots": [{"id": 0, "filename": name, "n_saved": 42}]}, slot

    run = fixture.__wrapped__()
    kw._last_unloaded_model = ("unsloth/Idle-GGUF", "Q4_K_M", "unsloth/Idle-GGUF")
    kw._kv_resume, before = _saved("before.bin")
    next(run)
    assert kw._last_unloaded_model is None, "the stash was not cleared before the test body"
    assert kw._kv_resume is None, "the KV manifest was not cleared before the test body"
    assert not before.exists(), "the KV slot file outlived the manifest that named it"

    kw._last_unloaded_model = ("unsloth/Idle-GGUF", "Q4_K_M", "unsloth/Idle-GGUF")
    kw._kv_resume, after = _saved("after.bin")
    next(run, None)
    assert kw._last_unloaded_model is None, "the stash was not cleared after the test body"
    assert kw._kv_resume is None, "the KV manifest was not cleared after the test body"
    assert not after.exists(), "the KV slot file outlived the manifest that named it"


def test_chat_withheld_model_does_not_send_the_caller_to_load_it(monkeypatch):
    # A truthy model_file is exec'd by MLX loaders, so do not suggest a manual load.
    _wire_withheld_chat(
        monkeypatch,
        objects = [
            {"id": "org/A-GGUF"},
            {"id": "org/custom-code", "task": "automatic-speech-recognition"},
        ],
        downloaded = ("org/A-GGUF", "org/custom-code"),
    )
    status, detail = _chat_error(_chat_request(model = "org/custom-code"))
    assert status == 404
    assert "cannot serve it here" in detail
    assert "Unsloth Studio" not in detail


def test_preset_reasoning_budget_rejects_booleans():
    from routes.chat_history import ChatPresetLoadConfig
    with pytest.raises(ValueError, match = "Expected a number, got a boolean"):
        ChatPresetLoadConfig(reasoningBudget = True)
    assert ChatPresetLoadConfig(reasoningBudget = 0).reasoningBudget == 0


@pytest.mark.parametrize(
    "image_preflight",
    [{"b64": None, "multiple": True}, {"b64": None, "multiple": False, "remote": True}],
    ids = ["multiple images", "remote url"],
)
def test_a_saved_managed_engine_target_skips_the_default_image_preflight(
    monkeypatch, image_preflight
):
    from utils import openai_auto_switch_settings as settings

    llama = _FakeBackend("org/A-GGUF")

    class _FakeOrchestrator:
        active_model_name = None
        models: dict = {}

    orchestrator = _FakeOrchestrator()
    calls = []

    async def _load(request, *_args, **_kwargs):
        calls.append(request)
        orchestrator.active_model_name = request.model_path

    _wire_on(
        monkeypatch,
        resolves_to = ("/srv/models/Vision", None, "org/Vision"),
        backend = llama,
        recorder = _load,
    )
    monkeypatch.setattr(inference_route, "get_inference_backend", lambda: orchestrator)
    monkeypatch.setattr(inference_route, "_peek_inference_backend", lambda: orchestrator)
    monkeypatch.setattr(resolver, "local_target_is_gguf", lambda *_a, **_kw: False)
    monkeypatch.setattr(inference_route, "_target_accepts_request_input", lambda *_a: True)
    monkeypatch.setattr(
        settings,
        "resolve_override_for_load",
        lambda *_a: ("org/Vision", {"engine": "vllm", "engine_precision": "auto"}),
    )

    try:
        asyncio.run(
            inference_route._maybe_auto_switch_model(
                "org/Vision",
                object(),
                "tester",
                require_vision = True,
                image_preflight = image_preflight,
            )
        )
    except HTTPException as exc:
        assert exc.status_code != 400 or "image" not in str(exc.detail).lower(), exc.detail
    assert calls and calls[0].engine == "vllm"


def test_the_legacy_bare_delete_clears_a_managed_engine_choice(override_store):
    settings.set_model_override("org/Model", engine = "vllm", engine_precision = "int4")

    _put("org/Model")
    assert settings.get_model_override("org/Model") == {}
