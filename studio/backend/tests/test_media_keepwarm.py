# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the idle auto-unload of the image and video backends.

The real backends are replaced with fakes that publish the same status /
loading_repo_ids / generate_progress surface, so these verify only the idle
decision -- no torch, GPU, or model download.
"""

from __future__ import annotations

import asyncio
import sys
import time

import pytest

import core.inference.gpu_arbiter as arb
import core.inference.media_keepwarm as mk
import utils.openai_auto_switch_settings as settings


class _FakeEngine:
    """Minimal stand-in for the diffusers / sd.cpp / video backends."""

    def __init__(
        self,
        repo_id = "unsloth/FLUX.1-dev",
        loaded = True,
        **build,
    ):
        self.repo_id = repo_id
        self.loaded = loaded
        self.loading: tuple[str, ...] = ()
        self.active = False
        self.unloads = 0
        # The rest of the build identity the real backends publish (H3 task, quants).
        self.build = dict(build)

    def status(self):
        rest = self.build if self.loaded else dict.fromkeys(self.build)
        return {
            "loaded": self.loaded,
            "repo_id": self.repo_id if self.loaded else None,
            "gguf_variant": None,
            **rest,
        }

    def loading_repo_ids(self):
        return self.loading

    def generate_progress(self):
        return {"active": self.active}

    def unload(self):
        self.unloads += 1
        self.loaded = False
        return self.status()


@pytest.fixture
def media(monkeypatch):
    """Both trackers reset, both engines faked, the arbiter left unowned."""
    monkeypatch.setattr(arb, "_owner", None)
    engines = {arb.DIFFUSION: _FakeEngine(), arb.VIDEO: _FakeEngine("unsloth/Wan2.2")}
    for owner, engine in engines.items():
        monkeypatch.setitem(mk._ENGINES, owner, lambda e = engine: e)
        # The real evictors tear down live backends; ownership sequencing is all these need.
        monkeypatch.setitem(arb._EVICTORS, owner, lambda: None)
        tracker = mk._TRACKERS[owner]
        monkeypatch.setattr(tracker, "_inflight", 0)
        monkeypatch.setattr(tracker, "_pending", 0)
        monkeypatch.setattr(tracker, "_last_active", time.monotonic())
        monkeypatch.setattr(tracker, "seen", None)
    return engines


_BOTH = (arb.DIFFUSION, arb.VIDEO)


def _idle(*owners):
    """Backdate the trackers as if nothing had touched these backends for an hour."""
    for owner in owners:
        mk._TRACKERS[owner]._last_active = time.monotonic() - 3600.0


def _step(*idle_owners):
    """One idle tick, with the named backends backdated past the TTL first."""
    _idle(*idle_owners)
    asyncio.run(mk.idle_unload_step())


# ── the TTL setting ─────────────────────────────────────────────────


@pytest.fixture
def store(monkeypatch):
    """The app settings map in memory, read back through the real stored readers."""
    values: dict = {}
    monkeypatch.setattr(
        settings, "_cached_setting", lambda key, default = None: values.get(key, default)
    )
    for var in (settings.MODEL_IDLE_TTL_ENV_VAR, settings.MEDIA_IDLE_TTL_ENV_VAR):
        monkeypatch.delenv(var, raising = False)
    return values


def test_the_chat_ttl_alone_does_not_unload_media(store):
    # The consent line. "Model auto-switch (OpenAI API)" never mentions Images or Video,
    # so a user who turned that on gets nothing new here on upgrade: the media TTL is its
    # own setting and its default is off.
    store[settings.OPENAI_AUTO_SWITCH_SETTING_KEY] = True
    store[settings.AUTO_UNLOAD_IDLE_SETTING_KEY] = 600
    assert settings.get_auto_unload_idle_seconds() == 600
    assert settings.get_media_auto_unload_idle_seconds() == 0


def test_the_media_ttl_unloads_media_without_touching_chat(store):
    # The other direction: a whole setting, not a modifier on the chat one, so it works
    # with auto-switch and the chat TTL both off.
    store[settings.MEDIA_AUTO_UNLOAD_IDLE_SETTING_KEY] = 600
    assert settings.get_media_auto_unload_idle_seconds() == 600
    assert settings.get_auto_unload_idle_seconds() == 0
    # Floored like the chat one, for a value persisted before the minimum existed.
    store[settings.MEDIA_AUTO_UNLOAD_IDLE_SETTING_KEY] = 5
    assert settings.get_media_auto_unload_idle_seconds() == settings.MIN_AUTO_UNLOAD_IDLE_SECONDS
    # And it is not gated on auto-switch: that flag is about serving /v1 requests.
    store[settings.OPENAI_AUTO_SWITCH_SETTING_KEY] = False
    store[settings.MEDIA_AUTO_UNLOAD_IDLE_SETTING_KEY] = 600
    assert settings.get_media_auto_unload_idle_seconds() == 600


def test_media_ttl_env_behaves_like_the_chat_env(store, monkeypatch):
    # UNSLOTH_MEDIA_IDLE_TTL stands in the same relationship to the media setting that
    # UNSLOTH_MODEL_IDLE_TTL has to the chat one: the startup default while nothing is
    # stored, floored the same way, and outranked by an explicit value.
    monkeypatch.setenv(settings.MEDIA_IDLE_TTL_ENV_VAR, "900")
    assert settings.get_media_auto_unload_idle_seconds() == 900
    assert settings.get_stored_media_auto_unload_idle_seconds() == 900
    monkeypatch.setenv(settings.MEDIA_IDLE_TTL_ENV_VAR, "5")
    assert settings.get_media_auto_unload_idle_seconds() == settings.MIN_AUTO_UNLOAD_IDLE_SECONDS
    monkeypatch.setenv(settings.MEDIA_IDLE_TTL_ENV_VAR, "900")
    store[settings.MEDIA_AUTO_UNLOAD_IDLE_SETTING_KEY] = 0
    assert settings.get_media_auto_unload_idle_seconds() == 0
    store[settings.MEDIA_AUTO_UNLOAD_IDLE_SETTING_KEY] = 600
    assert settings.get_media_auto_unload_idle_seconds() == 600
    # The chat env var is not the media one.
    del store[settings.MEDIA_AUTO_UNLOAD_IDLE_SETTING_KEY]
    monkeypatch.delenv(settings.MEDIA_IDLE_TTL_ENV_VAR)
    monkeypatch.setenv(settings.MODEL_IDLE_TTL_ENV_VAR, "900")
    assert settings.get_media_auto_unload_idle_seconds() == 0


def test_api_only_disables_the_media_ttl(store, monkeypatch):
    # "Only unload models loaded by the API" promises a model the user loaded from
    # Studio stays resident, and nothing but the user ever loads an image or video
    # model: /images/load and /video/load are the only entry points, and
    # /v1/images/generations 503s rather than loading one. So with the setting on
    # there is nothing here the idle unload is allowed to free.
    store[settings.MEDIA_AUTO_UNLOAD_IDLE_SETTING_KEY] = 600
    store[settings.AUTO_UNLOAD_API_ONLY_SETTING_KEY] = True
    assert settings.get_media_auto_unload_idle_seconds() == 0
    # The env-backed TTL is vetoed too, exactly as residency vetoes it.
    del store[settings.MEDIA_AUTO_UNLOAD_IDLE_SETTING_KEY]
    monkeypatch.setenv(settings.MEDIA_IDLE_TTL_ENV_VAR, "900")
    assert settings.get_media_auto_unload_idle_seconds() == 0
    # The stored seconds survive it, so turning the veto off brings them back.
    store[settings.AUTO_UNLOAD_API_ONLY_SETTING_KEY] = False
    assert settings.get_media_auto_unload_idle_seconds() == 900


def test_residency_vetoes_the_media_ttl(store, monkeypatch):
    monkeypatch.setattr(settings, "_residency_vetoes_unload", lambda: True)
    store[settings.MEDIA_AUTO_UNLOAD_IDLE_SETTING_KEY] = 900
    assert settings.get_media_auto_unload_idle_seconds() == 0
    assert settings.get_stored_media_auto_unload_idle_seconds() == 900
    del store[settings.MEDIA_AUTO_UNLOAD_IDLE_SETTING_KEY]
    monkeypatch.setenv(settings.MEDIA_IDLE_TTL_ENV_VAR, "900")
    assert settings.get_media_auto_unload_idle_seconds() == 0
    monkeypatch.setattr(settings, "_residency_vetoes_unload", lambda: False)
    assert settings.get_media_auto_unload_idle_seconds() == 900


# ── the idle decision ───────────────────────────────────────────────


def test_idle_load_is_unloaded_after_the_ttl(media, monkeypatch):
    monkeypatch.setattr(settings, "get_media_auto_unload_idle_seconds", lambda: 60)
    arb.acquire_for(arb.DIFFUSION)
    _step()  # the loop has now seen both models, so only the TTL is left
    _step(*_BOTH)
    assert media[arb.DIFFUSION].unloads == 1
    assert media[arb.VIDEO].unloads == 1
    # The arbiter claim went with it, so a later chat load has nothing to evict.
    assert arb.current_owner() is None
    # Freed once, not once per tick.
    _step(*_BOTH)
    assert media[arb.DIFFUSION].unloads == 1


def test_an_in_flight_generation_is_not_unloaded(media, monkeypatch):
    monkeypatch.setattr(settings, "get_media_auto_unload_idle_seconds", lambda: 60)
    for owner in _BOTH:
        media[owner].active = True
    _step()
    _step(*_BOTH)
    assert media[arb.DIFFUSION].unloads == 0
    assert media[arb.VIDEO].unloads == 0
    # The generation counted as activity, so the TTL restarts from its end rather than
    # freeing the pipeline the moment the last step lands.
    for owner in _BOTH:
        media[owner].active = False
    _step()
    assert media[arb.DIFFUSION].unloads == 0
    assert media[arb.VIDEO].unloads == 0


def test_an_in_flight_load_is_not_unloaded(media, monkeypatch):
    monkeypatch.setattr(settings, "get_media_auto_unload_idle_seconds", lambda: 60)
    engine = media[arb.DIFFUSION]
    _step()
    # A superseding load in flight over the resident model.
    engine.loading = ("unsloth/FLUX.1-schnell",)
    _step(arb.DIFFUSION)
    assert engine.unloads == 0
    # Once it lands, the same state IS collectable: the load was what spared it.
    engine.loading = ()
    _step(arb.DIFFUSION)
    assert engine.unloads == 1


def test_a_request_in_flight_is_not_unloaded(media, monkeypatch):
    monkeypatch.setattr(settings, "get_media_auto_unload_idle_seconds", lambda: 60)
    _step()

    async def _drive():
        await mk.begin_request(arb.DIFFUSION)
        _idle(arb.DIFFUSION)
        await mk.idle_unload_step()
        assert media[arb.DIFFUSION].unloads == 0
        mk.end_request(arb.DIFFUSION)

    asyncio.run(_drive())
    # The completed request stamped activity, so the next tick still spares it.
    _step()
    assert media[arb.DIFFUSION].unloads == 0


def test_a_load_that_just_finished_survives_one_ttl(media, monkeypatch):
    # The server has been idle far longer than the TTL and a model then lands: the
    # first tick that sees it stamps activity, so it is not freed out from under the
    # user who just loaded it.
    monkeypatch.setattr(settings, "get_media_auto_unload_idle_seconds", lambda: 60)
    _step(*_BOTH)
    _step()
    assert media[arb.DIFFUSION].unloads == 0
    assert media[arb.VIDEO].unloads == 0


def test_reload_after_an_idle_unload_works(media, monkeypatch):
    monkeypatch.setattr(settings, "get_media_auto_unload_idle_seconds", lambda: 60)
    engine = media[arb.DIFFUSION]
    _step()
    _step(arb.DIFFUSION)
    assert engine.unloads == 1 and not engine.status()["loaded"]
    # The user comes back and loads again: the reload sticks, and the tick that finds it
    # treats the load as activity instead of freeing it straight back off the stale stamp.
    engine.loaded = True
    _step()
    _step()
    assert engine.unloads == 1 and engine.status()["loaded"]


def test_a_different_model_restarts_the_ttl(media, monkeypatch):
    monkeypatch.setattr(settings, "get_media_auto_unload_idle_seconds", lambda: 60)
    engine = media[arb.DIFFUSION]
    _step()
    engine.repo_id = "unsloth/FLUX.1-schnell"
    _step(arb.DIFFUSION)
    assert engine.unloads == 0


def test_api_only_spares_a_model_the_user_loaded(media, store):
    # The whole feature is off while the setting is on, and off means today's behaviour:
    # nothing resolved, nothing unloaded.
    store[settings.MEDIA_AUTO_UNLOAD_IDLE_SETTING_KEY] = 60
    store[settings.AUTO_UNLOAD_API_ONLY_SETTING_KEY] = True
    _step()
    _step(*_BOTH)
    assert media[arb.DIFFUSION].unloads == 0
    assert media[arb.VIDEO].unloads == 0
    # Turned off again, the same idle models are collectable: the setting was the only
    # thing sparing them, so this does not cost the feature anything else.
    store[settings.AUTO_UNLOAD_API_ONLY_SETTING_KEY] = False
    _step()
    _step(*_BOTH)
    assert media[arb.DIFFUSION].unloads == 1
    assert media[arb.VIDEO].unloads == 1


def test_a_cached_reload_of_another_h3_partition_is_not_unloaded(media, monkeypatch):
    # MiniMax-H3 keeps its identity in more than the repo id: fl2va and ref2va are
    # different denoiser partitions, and the quants are part of the build too. A cached
    # reload between two ticks lands with the old timestamp already expired, so an
    # identity that cannot tell the partitions apart frees it the moment it arrives.
    monkeypatch.setattr(settings, "get_media_auto_unload_idle_seconds", lambda: 60)
    engine = media[arb.VIDEO]
    engine.repo_id = "MiniMaxAI/MiniMax-H3"
    engine.build = {"h3_task": "fl2va", "transformer_quant": "fp8", "text_encoder_quant": None}
    _step()
    engine.build["h3_task"] = "ref2va"
    _step(arb.VIDEO)
    assert engine.unloads == 0
    # A quant swap is a rebuild as well.
    engine.build["transformer_quant"] = None
    _step(arb.VIDEO)
    assert engine.unloads == 0
    # Unchanged and idle, it is still collectable.
    _step(arb.VIDEO)
    assert engine.unloads == 1


def test_disabled_ttl_never_touches_the_backends(media, monkeypatch):
    # Today's behaviour, and the default: nothing is resolved, nothing is unloaded.
    monkeypatch.setattr(settings, "get_media_auto_unload_idle_seconds", lambda: 0)
    resolved = []
    for owner in _BOTH:
        monkeypatch.setitem(mk._ENGINES, owner, lambda o = owner: resolved.append(o))
    arb.acquire_for(arb.VIDEO)
    _step(*_BOTH)
    _step(*_BOTH)
    assert resolved == []
    assert media[arb.DIFFUSION].unloads == 0
    assert arb.current_owner() == arb.VIDEO


def test_a_chat_ttl_alone_leaves_the_media_backends_alone(media, store, monkeypatch):
    # Same consent line as the settings test, one level down: an install that had chat
    # idle-unload on before this landed must tick exactly as it did before.
    store[settings.OPENAI_AUTO_SWITCH_SETTING_KEY] = True
    store[settings.AUTO_UNLOAD_IDLE_SETTING_KEY] = 600
    resolved = []
    for owner in _BOTH:
        monkeypatch.setitem(mk._ENGINES, owner, lambda o = owner: resolved.append(o))
    _step(*_BOTH)
    _step(*_BOTH)
    assert resolved == []
    assert media[arb.DIFFUSION].unloads == 0
    assert media[arb.VIDEO].unloads == 0
    # Turning the media TTL on is what starts it, and only that.
    store[settings.MEDIA_AUTO_UNLOAD_IDLE_SETTING_KEY] = 60
    _step(*_BOTH)
    assert resolved == list(_BOTH)


def test_the_off_tick_does_not_import_the_media_modules(store, monkeypatch):
    # Off is the default and has to stay free: the tick runs every 15s from startup, so
    # importing diffusion or video to find out there is nothing loaded would drag torch
    # into a Studio that never opened either page. No engine fakes here on purpose --
    # this is the real resolution path.
    store[settings.OPENAI_AUTO_SWITCH_SETTING_KEY] = True
    store[settings.AUTO_UNLOAD_IDLE_SETTING_KEY] = 600
    media_modules = {
        "core.inference.diffusion",
        "core.inference.sd_cpp_backend",
        "core.inference.video",
    }
    for module in media_modules:
        monkeypatch.delitem(sys.modules, module, raising = False)
    assert settings.get_media_auto_unload_idle_seconds() == 0
    asyncio.run(mk.idle_unload_step())
    asyncio.run(mk.idle_unload_step())
    assert not media_modules & set(sys.modules)


def test_a_failing_unload_does_not_stop_the_other_backend(media, monkeypatch):
    monkeypatch.setattr(settings, "get_media_auto_unload_idle_seconds", lambda: 60)

    def _boom():
        raise RuntimeError("cuda teardown failed")

    media[arb.DIFFUSION].unload = _boom
    _step()
    _step(*_BOTH)
    assert media[arb.VIDEO].unloads == 1


def test_an_unimported_backend_is_not_imported_to_check_it(monkeypatch):
    # The tick runs every 15s from startup; resolving the engines would drag torch in
    # on a Studio that has never opened the Image or Video page.
    for module in ("core.inference.diffusion", "core.inference.sd_cpp_backend"):
        monkeypatch.delitem(sys.modules, module, raising = False)
    monkeypatch.delitem(sys.modules, "core.inference.video", raising = False)
    assert mk._diffusion_engine() is None
    assert mk._video_engine() is None


# ── the request middleware ──────────────────────────────────────────


def test_generate_routes_map_to_their_backend():
    assert mk.owner_for_path("/api/inference/images/generate") == arb.DIFFUSION
    assert mk.owner_for_path("/v1/images/generations") == arb.DIFFUSION
    assert mk.owner_for_path("/api/inference/video/generate") == arb.VIDEO
    # Progress polling while the user watches is not activity, and neither is chat.
    assert mk.owner_for_path("/api/inference/images/generate-progress") is None
    assert mk.owner_for_path("/api/inference/images/generate/cancel") is None
    assert mk.owner_for_path("/v1/chat/completions") is None


def test_load_routes_map_to_their_backend():
    # A load registers with the backend only PART WAY through its POST, so the route has
    # to hold the gate for the whole of it: sampling loading_repo_ids() cannot see a load
    # the route has been accepted for but not yet started.
    assert mk.owner_for_path("/api/inference/images/load") == arb.DIFFUSION
    assert mk.owner_for_path("/api/inference/video/load") == arb.VIDEO
    # Progress polling is not a load, and neither is planning a download.
    assert mk.owner_for_path("/api/inference/images/load-progress") is None
    assert mk.owner_for_path("/api/inference/video/load-progress") is None
    assert mk.owner_for_path("/api/inference/images/download-plan") is None


def test_a_load_that_has_not_registered_yet_is_not_unloaded(media, monkeypatch):
    # The check/start race. The tick reads the backend as idle with no load in flight, the
    # user's load is accepted a moment later, and the unload that tick issues bumps the load
    # token and signals the fresh cancel event: the worker exits without publishing an error
    # and the page silently rolls the pick back. The window has to be closed, not narrowed,
    # so the tick is pinned to the exact moment the route has started and the backend still
    # reports nothing loading.
    from core.inference.llama_keepwarm import LlamaKeepWarmMiddleware

    monkeypatch.setattr(settings, "get_media_auto_unload_idle_seconds", lambda: 60)
    _step()
    engine = media[arb.VIDEO]
    seen = {}

    async def _app(scope, receive, send):
        # Inside the load route, before begin_load has registered anything.
        assert engine.loading == ()
        _idle(arb.VIDEO)
        await mk.idle_unload_step()
        seen["unloads"] = engine.unloads
        # begin_load registers only now; from here loading_repo_ids() covers it.
        engine.loading = ("MiniMaxAI/MiniMax-H3",)
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    scope = {"type": "http", "method": "POST", "path": "/api/inference/video/load"}
    asyncio.run(LlamaKeepWarmMiddleware(_app)(scope, None, lambda message: _noop()))
    assert seen["unloads"] == 0
    assert engine.unloads == 0
    assert mk._TRACKERS[arb.VIDEO]._inflight == 0
    # The accepted load kept it: once the load is in flight the existing guard has it.
    _step(arb.VIDEO)
    assert engine.unloads == 0


def test_the_middleware_counts_a_generation_against_its_backend(media, monkeypatch):
    from core.inference.llama_keepwarm import LlamaKeepWarmMiddleware

    monkeypatch.setattr(settings, "get_media_auto_unload_idle_seconds", lambda: 60)
    _step()
    _idle(arb.DIFFUSION)
    seen = {}

    async def _app(scope, receive, send):
        # Mid-request: the idle tick must see this backend as busy and spare it.
        await mk.idle_unload_step()
        seen["unloads"] = media[arb.DIFFUSION].unloads
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"{}", "more_body": False})

    scope = {"type": "http", "method": "POST", "path": "/api/inference/images/generate"}
    asyncio.run(LlamaKeepWarmMiddleware(_app)(scope, None, lambda message: _noop()))
    assert seen["unloads"] == 0
    assert mk._TRACKERS[arb.DIFFUSION]._inflight == 0


async def _noop():
    return None


def test_an_unauthenticated_probe_does_not_keep_the_pipeline_warm(media, monkeypatch):
    from core.inference.llama_keepwarm import LlamaKeepWarmMiddleware

    monkeypatch.setattr(settings, "get_media_auto_unload_idle_seconds", lambda: 60)
    _step()
    _idle(arb.DIFFUSION)

    async def _app(scope, receive, send):
        await send({"type": "http.response.start", "status": 401})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    scope = {"type": "http", "method": "POST", "path": "/v1/images/generations"}
    asyncio.run(LlamaKeepWarmMiddleware(_app)(scope, None, lambda message: _noop()))
    # The 401 never reached the backend, so the model is still idle and gets freed.
    _step(arb.DIFFUSION)
    assert media[arb.DIFFUSION].unloads == 1
