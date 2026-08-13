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
    ):
        self.repo_id = repo_id
        self.loaded = loaded
        self.loading: tuple[str, ...] = ()
        self.active = False
        self.unloads = 0

    def status(self):
        return {
            "loaded": self.loaded,
            "repo_id": self.repo_id if self.loaded else None,
            "gguf_variant": None,
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


def test_media_ttl_follows_the_chat_ttl(monkeypatch):
    monkeypatch.delenv(settings.MEDIA_IDLE_TTL_ENV_VAR, raising = False)
    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 600)
    assert settings.get_media_auto_unload_idle_seconds() == 600


def test_media_ttl_env_overrides_and_can_disable(monkeypatch):
    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 600)
    monkeypatch.setenv(settings.MEDIA_IDLE_TTL_ENV_VAR, "900")
    assert settings.get_media_auto_unload_idle_seconds() == 900
    # 0 keeps pipelines resident while chat still idle-unloads.
    monkeypatch.setenv(settings.MEDIA_IDLE_TTL_ENV_VAR, "0")
    assert settings.get_media_auto_unload_idle_seconds() == 0
    # Below the shared minimum is floored, not honoured.
    monkeypatch.setenv(settings.MEDIA_IDLE_TTL_ENV_VAR, "5")
    assert settings.get_media_auto_unload_idle_seconds() == settings.MIN_AUTO_UNLOAD_IDLE_SECONDS


def test_residency_vetoes_the_media_ttl(monkeypatch):
    monkeypatch.setenv(settings.MEDIA_IDLE_TTL_ENV_VAR, "900")
    monkeypatch.setattr(settings, "_residency_vetoes_unload", lambda: True)
    assert settings.get_media_auto_unload_idle_seconds() == 0


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
