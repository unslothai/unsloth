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
        # The terminal record the video backend holds after a job (None on the image side).
        self.terminal: dict | None = None
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
        if self.active:
            return {"active": True}
        return {"active": False, **(self.terminal or {})}

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
        monkeypatch.setattr(tracker, "was_busy", False, raising = False)
        monkeypatch.setattr(tracker, "completed", None, raising = False)
    monkeypatch.setattr(mk, "_LOAD_ORIGINS", {})
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


def test_api_only_does_not_veto_the_media_ttl(store, monkeypatch):
    # Media auto-switch gives an API request its own way to load a pipeline, so "only
    # unload models loaded by the API" is a per-model rule here (see the tick tests
    # below) rather than something that holds the whole TTL off.
    store[settings.MEDIA_AUTO_UNLOAD_IDLE_SETTING_KEY] = 600
    store[settings.AUTO_UNLOAD_API_ONLY_SETTING_KEY] = True
    assert settings.get_media_auto_unload_idle_seconds() == 600
    del store[settings.MEDIA_AUTO_UNLOAD_IDLE_SETTING_KEY]
    monkeypatch.setenv(settings.MEDIA_IDLE_TTL_ENV_VAR, "900")
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
    # Once it lands, the same state IS collectable: the load was what spared it. One tick
    # later, though -- the tick that finds the load done starts the TTL from there, since a
    # load that outlives its POST stamps no activity of its own when it finishes.
    engine.loading = ()
    _step(arb.DIFFUSION)
    assert engine.unloads == 0
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


def test_the_ttl_starts_when_the_background_work_ends(media, monkeypatch):
    # A video generation outlives its POST: the response is sent at once and the job runs on
    # in a worker, so after that only the busy polls stamp activity. Dating the TTL from the
    # last of those spends up to a whole poll interval of the keep-warm window the user
    # configured before the model was even free. The tick that finds the work done starts it.
    monkeypatch.setattr(settings, "get_media_auto_unload_idle_seconds", lambda: 60)
    engine = media[arb.VIDEO]
    engine.active = True
    _step()
    # The job ends just after that tick, so the newest stamp is already a poll old -- here,
    # far older than the TTL, which is the same thing with the clock wound on.
    _idle(arb.VIDEO)
    engine.active = False
    _step()
    assert engine.unloads == 0
    # A restart, not a one-tick reprieve: the whole TTL runs from the end of the work.
    _step()
    assert engine.unloads == 0
    _step(arb.VIDEO)
    assert engine.unloads == 1


def test_a_job_that_lives_between_two_polls_still_gets_the_full_ttl(media, monkeypatch):
    # A video job can start and finish inside one 15s poll interval, so no tick ever samples
    # it as busy. Its POST returned near the START of the generation, and that response is
    # the only activity it stamps, so the TTL was spent while the job was still running: a
    # 74s TTL could free the model after about 60s of real idleness. The terminal record the
    # backend publishes is the only proof the job ran, so the tick that first sees it starts
    # the TTL there.
    monkeypatch.setattr(settings, "get_media_auto_unload_idle_seconds", lambda: 60)
    engine = media[arb.VIDEO]
    _step()
    engine.terminal = {"phase": "completed", "video": {"id": "clip-1"}}
    _idle(arb.VIDEO)  # the POST's stamp is already older than the TTL
    _step()
    assert engine.unloads == 0
    # A restart, not a one-tick reprieve.
    _step()
    assert engine.unloads == 0
    # The record itself keeps nothing warm: it is still published on that last tick, and
    # only a record this tracker has not seen before counts as work having finished.
    _step(arb.VIDEO)
    assert engine.unloads == 1


def test_a_veto_applied_during_the_step_stops_the_next_teardown(media, monkeypatch):
    # One step tears down both backends and freeing several GB takes seconds. Reading the
    # effective TTL once for the whole step let a residency veto turned on during the
    # diffusion unload be ignored by the video one, so Unsloth freed a model its own settings
    # response already reported as pinned.
    ttl = {"value": 60}
    monkeypatch.setattr(settings, "get_media_auto_unload_idle_seconds", lambda: ttl["value"])
    diffusion, video = media[arb.DIFFUSION], media[arb.VIDEO]
    real_unload = diffusion.unload

    def _slow_unload():
        # Model Memory residency (or API-only, or a TTL of 0) applied mid-teardown.
        ttl["value"] = 0
        return real_unload()

    diffusion.unload = _slow_unload
    _step()
    _step(*_BOTH)
    assert diffusion.unloads == 1
    assert video.unloads == 0


def test_a_ttl_raised_during_the_step_spares_the_next_teardown(media, monkeypatch):
    # The same window, with the setting moved rather than vetoed: a TTL the backend is no
    # longer past must be honoured by the teardown that has not happened yet.
    ttl = {"value": 60}
    monkeypatch.setattr(settings, "get_media_auto_unload_idle_seconds", lambda: ttl["value"])
    diffusion, video = media[arb.DIFFUSION], media[arb.VIDEO]
    real_unload = diffusion.unload

    def _slow_unload():
        ttl["value"] = 7200  # further out than the hour _idle backdates by
        return real_unload()

    diffusion.unload = _slow_unload
    _step()
    _step(*_BOTH)
    assert diffusion.unloads == 1
    assert video.unloads == 0


def test_a_request_landing_during_the_pin_read_is_not_unloaded_out_from_under(media, monkeypatch):
    # A request may register _pending during the off-loop pin read, invalidating prior idleness.
    monkeypatch.setattr(settings, "get_media_auto_unload_idle_seconds", lambda: 60)

    def _pinned_while_a_request_lands(owner, *_args, **_kwargs):
        mk._TRACKERS[owner].note_pending()
        return False

    monkeypatch.setattr(mk, "_user_pinned", _pinned_while_a_request_lands)
    _step()  # Both models are seen; only the TTL remains.
    _step(*_BOTH)
    assert media[arb.DIFFUSION].unloads == 0
    assert media[arb.VIDEO].unloads == 0


def test_a_different_model_restarts_the_ttl(media, monkeypatch):
    monkeypatch.setattr(settings, "get_media_auto_unload_idle_seconds", lambda: 60)
    engine = media[arb.DIFFUSION]
    _step()
    engine.repo_id = "unsloth/FLUX.1-schnell"
    _step(arb.DIFFUSION)
    assert engine.unloads == 0


def test_api_only_spares_a_model_the_user_loaded(media, store):
    # Unknown provenance reads as user-loaded, so an install that never recorded one is
    # spared exactly as it was before media auto-switch existed.
    store[settings.MEDIA_AUTO_UNLOAD_IDLE_SETTING_KEY] = 60
    store[settings.AUTO_UNLOAD_API_ONLY_SETTING_KEY] = True
    mk.note_load_origin(arb.DIFFUSION, "unsloth/FLUX.1-dev", None, user_action = True)
    _step()
    _step(*_BOTH)
    assert media[arb.DIFFUSION].unloads == 0
    assert media[arb.VIDEO].unloads == 0
    # Turned off again, the same idle models are collectable.
    store[settings.AUTO_UNLOAD_API_ONLY_SETTING_KEY] = False
    _step()
    _step(*_BOTH)
    assert media[arb.DIFFUSION].unloads == 1
    assert media[arb.VIDEO].unloads == 1


def test_api_only_still_frees_a_model_the_api_loaded(media, store):
    # The other half of the per-model rule: auto-switch marks its own load, and that one
    # is what the setting exists to collect.
    store[settings.MEDIA_AUTO_UNLOAD_IDLE_SETTING_KEY] = 60
    store[settings.AUTO_UNLOAD_API_ONLY_SETTING_KEY] = True
    mk.note_load_origin(arb.DIFFUSION, "unsloth/FLUX.1-dev", None, user_action = False)
    mk.note_load_origin(arb.VIDEO, "unsloth/Wan2.2", None, user_action = True)
    _step()
    _step(*_BOTH)
    assert media[arb.DIFFUSION].unloads == 1
    assert media[arb.VIDEO].unloads == 0


def test_a_failed_api_load_does_not_unpin_the_resident_user_model(media, store):
    # A load is recorded when it is accepted, and it can still fail with the previous model
    # resident. Reading that failed load's origin off the surviving model would evict a
    # pipeline the setting promises to keep.
    store[settings.MEDIA_AUTO_UNLOAD_IDLE_SETTING_KEY] = 60
    store[settings.AUTO_UNLOAD_API_ONLY_SETTING_KEY] = True
    mk.note_load_origin(arb.DIFFUSION, "unsloth/FLUX.1-dev", None, user_action = True)
    mk.note_load_origin(arb.DIFFUSION, "unsloth/Z-Image-Turbo", None, user_action = False)
    _step()
    _step(*_BOTH)
    assert media[arb.DIFFUSION].unloads == 0


def test_a_failed_api_load_of_another_quant_does_not_unpin_the_user_build(media, store):
    # Same repo, different quant: the path alone is not the build, so a failed API load of Q8
    # would otherwise mark the user's resident Q4 as API-loaded and free it.
    store[settings.MEDIA_AUTO_UNLOAD_IDLE_SETTING_KEY] = 60
    store[settings.AUTO_UNLOAD_API_ONLY_SETTING_KEY] = True
    media[arb.DIFFUSION].build["gguf_variant"] = "Q4_K_M"
    mk.note_load_origin(arb.DIFFUSION, "unsloth/FLUX.1-dev", "Q4_K_M", user_action = True)
    mk.note_load_origin(arb.DIFFUSION, "unsloth/FLUX.1-dev", "Q8_0", user_action = False)
    _step()
    _step(*_BOTH)
    assert media[arb.DIFFUSION].unloads == 0


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
    # into an Unsloth that never opened either page. No engine fakes here on purpose --
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
    # on an Unsloth that has never opened the Image or Video page.
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


def test_a_path_that_is_not_a_mounted_route_is_not_tracked():
    # A recognised prefix and a recognised tail is not a route. FastAPI answers these with a
    # 404 without running an endpoint, and _finish() excludes only 401/403 from stamping
    # activity, so an unauthenticated caller could hold a multi-GB pipeline resident forever
    # by repeating one below the TTL.
    assert mk.owner_for_path("/v1/not-a-route/images/generations") is None
    assert mk.owner_for_path("/api/inference/nope/video/generate") is None
    # The Unsloth routes are mounted under /api/inference only; /v1 carries the OpenAI shape.
    assert mk.owner_for_path("/v1/images/generate") is None
    assert mk.owner_for_path("/v1/video/load") is None
    assert mk.owner_for_path("/v1/videos/video_abc") is None
    assert mk.owner_for_path("/v1/videos/video_abc/content") is None


def test_every_tracked_path_is_a_route_that_is_actually_mounted():
    # Exact matching costs this: a renamed route would silently stop being tracked, and an
    # untracked generate is one an idle tick can tear the pipeline down under. So pin the
    # list to the routers main.py mounts, in both directions.
    from routes.inference import router as inference_router
    from routes.inference import studio_router
    from routes.video import openai_router as video_openai_router
    from routes.video import router as video_router

    mounted = {
        prefix + route.path
        for router, prefixes in (
            (inference_router, ("/api/inference", "/v1")),
            (studio_router, ("/api/inference",)),
            (video_router, ("/api/inference",)),
            (video_openai_router, ("/api/inference", "/v1")),
        )
        for route in router.routes
        for prefix in prefixes
    }
    assert not set(mk._TRACKED_PATHS) - mounted
    for path in ("/v1/videos", "/api/inference/videos"):
        assert path in mounted, path
        assert mk.owner_for_path(path) == arb.VIDEO, path
    for path in mounted:
        if ("/images/" in path or "/video/" in path) and path.rsplit("/", 1)[-1] in (
            "generate",
            "generations",
            "load",
        ):
            assert mk.owner_for_path(path) is not None, path


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


def test_a_cancelled_wait_on_the_media_gate_leaves_no_chat_request_behind(media, monkeypatch):
    # The generate routes are counted on BOTH sides, and the media gate is held for the
    # length of a teardown. A client that disconnects while waiting on it used to leave the
    # process-wide chat count positive for good: chat idle unload would never fire again and
    # every training start would go on being told an inference request was running.
    import core.inference.llama_keepwarm as lk
    from core.inference.llama_keepwarm import LlamaKeepWarmMiddleware

    monkeypatch.setattr(lk, "_inflight", 0)
    monkeypatch.setattr(lk, "_pending", 0)
    tracker = mk._TRACKERS[arb.DIFFUSION]

    async def _app(scope, receive, send):
        raise AssertionError("the request was cancelled before it could reach the app")

    async def _run():
        # Stand in for the tick: the gate is taken for the whole check-and-unload.
        tracker.gate.acquire()
        try:
            scope = {
                "type": "http",
                "method": "POST",
                "path": "/api/inference/images/generate",
            }
            task = asyncio.ensure_future(
                LlamaKeepWarmMiddleware(_app)(scope, None, lambda message: _noop())
            )
            await asyncio.sleep(0.1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            tracker.gate.release()

    asyncio.run(_run())
    assert lk._inflight == 0
    assert lk._pending == 0
    assert tracker._inflight == 0
    assert tracker._pending == 0


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


_BEARER = [(b"authorization", b"Bearer sk-unsloth-test")]


def _stalled_media_request(path, headers):
    """Open a tracked media POST that never sends a body, tick, and cancel it.

    Stands in for a client that opens the connection and drips: the count is taken in the
    middleware, ahead of the body parsing every one of these routes does before its auth
    dependency runs, so nothing downstream ever produces a status for it."""

    async def _run():
        from core.inference.llama_keepwarm import LlamaKeepWarmMiddleware

        started = asyncio.Event()

        async def _app(scope, receive, send):
            started.set()
            await asyncio.sleep(3600)  # the body never arrives

        scope = {"type": "http", "method": "POST", "path": path, "headers": headers}
        task = asyncio.ensure_future(
            LlamaKeepWarmMiddleware(_app)(scope, None, lambda message: _noop())
        )
        await started.wait()
        _idle(arb.DIFFUSION)
        await mk.idle_unload_step()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run())


def test_an_unauthenticated_stalled_request_cannot_pin_the_pipeline(media, monkeypatch):
    # An exposed server: a client opens a POST to a tracked media route and withholds its
    # body. It is counted before FastAPI authenticates or parses anything, and it produces
    # no status, so the 401/403 exclusion never runs -- one held connection kept a multi-GB
    # pipeline resident for the life of the process, which is the whole feature denied.
    monkeypatch.setattr(settings, "get_media_auto_unload_idle_seconds", lambda: 60)
    _step()
    _stalled_media_request("/api/inference/images/generate", [])
    assert media[arb.DIFFUSION].unloads == 1
    assert mk._TRACKERS[arb.DIFFUSION]._inflight == 0
    assert mk._TRACKERS[arb.DIFFUSION]._pending == 0


def test_an_authenticated_request_is_still_counted_before_its_body(media, monkeypatch):
    # The other direction, which matters more: a real client's generation is protected from
    # the moment its request arrives, body or no body.
    monkeypatch.setattr(settings, "get_media_auto_unload_idle_seconds", lambda: 60)
    _step()
    _stalled_media_request("/api/inference/images/generate", _BEARER)
    assert media[arb.DIFFUSION].unloads == 0


def test_the_openai_videos_route_never_claims_the_llama_slot():
    """/v1/videos runs the video backend, exactly like /video/generate.

    Adding "/videos" to the inference suffixes made it a tracked path; without the
    matching non-LLM entry its completion called _claim_non_preview_slot(), clearing
    preview ownership so the next preview of a different checkpoint 503s on the guard.
    """
    from core.inference import llama_keepwarm as kw

    for path in ("/v1/videos", "/api/inference/videos"):
        assert kw._is_inference_path(path), path
        assert path.endswith(kw._NON_LLM_SLOT_SUFFIXES), path
    # Matched whole. As an endswith suffix it also caught unrouted paths, and those
    # 404 before auth -- which this middleware does not exclude -- so each probe would
    # have refreshed the chat model's idle timer and kept it resident for free.
    for path in ("/v1/anything/videos", "/api/inference/nope/videos", "/v1/videosx"):
        assert not kw._is_inference_path(path), path
    for path in ("/v1/videos/", "/api/inference/videos/"):
        assert not kw._is_inference_path(path), path
