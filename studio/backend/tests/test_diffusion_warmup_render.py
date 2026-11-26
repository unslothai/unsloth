# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Recipe time of a load's first render includes compile / tuning, so the recipe flags it (warmup_render)."""

from __future__ import annotations

import inspect
import types

from core.inference import diffusion


W = diffusion._is_warmup_render


def test_a_compiled_first_render_is_a_warmup():
    assert W(0, False, True, False) is True


def test_the_render_that_engages_the_deferred_compile_inline_is_a_warmup():
    assert W(2, True, True, False) is True


def test_a_render_that_waited_for_the_background_compile_is_a_warmup():
    assert W(5, False, False, True) is True


def test_eager_renders_are_not_warmups():
    # speed off / eager first render: nothing compiled
    assert W(0, False, False, False) is False
    # deferred profile engaged but compiling in the background: this render ran eager
    assert W(2, True, False, False) is False
    # declined or failed deferred profile stays eager
    assert W(2, False, False, False) is False


def test_steady_state_compiled_renders_are_not_warmups():
    assert W(1, False, True, False) is False
    assert W(5, False, True, False) is False


def _deferred_state(pipe):
    return types.SimpleNamespace(
        speed_deferred = True,
        pipe = pipe,
        resolved = {"speed_mode": {"value": "auto", "source": "auto"}},
        deferred_static_plan = None,
        transformer_cache = None,
    )


def test_a_streamed_unet_declines_the_deferred_profile(monkeypatch):
    backend = object.__new__(diffusion.DiffusionBackend)
    state = _deferred_state(types.SimpleNamespace(_unsloth_unet_streamed = True))
    assert backend._engage_deferred_speed(state) is False
    assert state.speed_deferred is False


def test_a_deferred_profile_that_does_not_fit_is_declined(monkeypatch):
    monkeypatch.setattr(diffusion, "_deferred_profile_headroom_short", lambda state: "2 GiB short")
    backend = object.__new__(diffusion.DiffusionBackend)
    state = _deferred_state(types.SimpleNamespace())
    assert backend._engage_deferred_speed(state) is False


def test_an_engaged_deferred_profile_reports_true(monkeypatch):
    monkeypatch.setattr(diffusion, "_deferred_profile_headroom_short", lambda state: None)
    monkeypatch.setattr(diffusion, "invalidate_cfg_replica", lambda *a, **k: None)
    monkeypatch.setattr(diffusion, "drop_displaced_cfg_router", lambda *a, **k: None)
    backend = object.__new__(diffusion.DiffusionBackend)
    monkeypatch.setattr(
        diffusion.DiffusionBackend, "_engage_deferred_profile", lambda self, state: None
    )
    state = _deferred_state(types.SimpleNamespace())
    assert backend._engage_deferred_speed(state) is True


def test_generate_marks_the_deferred_render_only_when_it_engaged():
    src = inspect.getsource(diffusion.DiffusionBackend.generate)
    assert "deferred_engaged = self._engage_deferred_speed(state) is True" in src
    assert (
        "_is_warmup_render(\n                        warmup_count_before,\n                        deferred_engaged,"
        in src
    )
    assert "waited_for_bg = bg_was_compiling and paused_bg is None" in src


def test_generate_reports_it_and_the_gallery_keeps_it():
    from models.inference import GalleryImage
    assert '"warmup_render": _is_warmup_render(' in inspect.getsource(
        diffusion.DiffusionBackend.generate
    )
    assert "warmup_render" in GalleryImage.model_fields


def test_both_generate_routes_persist_it():
    from routes import inference as routes
    src = inspect.getsource(routes)
    assert src.count('**({"warmup_render": True} if result.get("warmup_render") else {})') == 2
