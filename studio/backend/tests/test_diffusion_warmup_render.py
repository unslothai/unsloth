# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Recipe time of a load's first render includes compile / tuning, so the recipe flags it (warmup_render)."""

from __future__ import annotations

import inspect

from core.inference import diffusion


def test_first_render_of_a_load_is_a_warmup():
    assert diffusion._is_warmup_render(0, False, False) is True
    assert diffusion._is_warmup_render(0, True, True) is True


def test_the_render_that_engages_the_deferred_compile_is_a_warmup():
    assert diffusion._is_warmup_render(2, True, False) is True


def test_steady_state_renders_are_not_warmups():
    assert diffusion._is_warmup_render(1, False, False) is False
    assert diffusion._is_warmup_render(5, False, False) is False
    assert diffusion._is_warmup_render(1, True, True) is False  # still eager, before the 3rd image


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
