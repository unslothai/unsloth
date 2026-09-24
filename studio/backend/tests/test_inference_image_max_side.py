# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The chat-inference image cap is configurable: default 1024 (or the model's own size), a pixel value, or 0 for no resize."""

import pytest
from PIL import Image

from utils.inference_image import (
    DEFAULT_INFERENCE_IMAGE_MAX_SIDE,
    INFERENCE_IMAGE_MAX_SIDE_ENV,
    inference_image_max_side,
    resize_for_inference,
)


def _page():
    return Image.new("RGB", (1148, 1288))


FALLBACK_PAGE_SIZE = (912, 1023)  # _page() under the 1024 fallback


def test_fallback_cap_is_1024_when_the_model_declares_nothing(monkeypatch):
    monkeypatch.delenv(INFERENCE_IMAGE_MAX_SIDE_ENV, raising = False)
    assert inference_image_max_side() is None
    assert DEFAULT_INFERENCE_IMAGE_MAX_SIDE == 1024
    # the arithmetic truncates (int()), so a side can land one pixel under the cap
    assert resize_for_inference(_page()).size == FALLBACK_PAGE_SIZE


def test_env_cap_keeps_a_page_below_it_untouched(monkeypatch):
    monkeypatch.setenv(INFERENCE_IMAGE_MAX_SIDE_ENV, "1540")
    page = _page()
    assert resize_for_inference(page) is page


def test_env_cap_downscales_keeping_aspect(monkeypatch):
    monkeypatch.setenv(INFERENCE_IMAGE_MAX_SIDE_ENV, "1000")
    assert resize_for_inference(_page()).size == (891, 1000)


def test_zero_disables_resizing(monkeypatch):
    monkeypatch.setenv(INFERENCE_IMAGE_MAX_SIDE_ENV, "0")
    big = Image.new("RGB", (4000, 3000))
    assert resize_for_inference(big) is big


def test_never_upscales(monkeypatch):
    monkeypatch.setenv(INFERENCE_IMAGE_MAX_SIDE_ENV, "1540")
    small = Image.new("RGB", (300, 200))
    assert resize_for_inference(small) is small


def test_explicit_max_side_overrides_env(monkeypatch):
    monkeypatch.setenv(INFERENCE_IMAGE_MAX_SIDE_ENV, "0")
    assert max(resize_for_inference(_page(), max_side = 500).size) == 500


def test_none_passes_through(monkeypatch):
    assert resize_for_inference(None) is None


@pytest.mark.parametrize("raw", ["abc", "12.5", "-1"])
def test_malformed_value_raises(monkeypatch, raw):
    monkeypatch.setenv(INFERENCE_IMAGE_MAX_SIDE_ENV, raw)
    with pytest.raises(ValueError, match = INFERENCE_IMAGE_MAX_SIDE_ENV):
        inference_image_max_side()


# ── model default: the loaded model's own image size ────────────────────────────────────────────
from types import SimpleNamespace

from utils.inference_image import (
    MAX_SAFE_SIDE,
    PIXEL_BUDGET_MAX_SIDE,
    effective_image_limit,
    native_image_limit,
)


def _processor(**image_processor_attrs):
    return SimpleNamespace(image_processor = SimpleNamespace(**image_processor_attrs))


def test_native_limit_longest_edge():
    assert native_image_limit(_processor(size = {"longest_edge": 1540})) == {"max_side": 1540}


def test_native_limit_fixed_height_width_takes_the_larger_side():
    assert native_image_limit(_processor(size = {"height": 896, "width": 640})) == {"max_side": 896}


def test_native_limit_pixel_budget_wins_over_qwen_style_longest_edge():
    # Qwen-VL processors spell their pixel budget as size["longest_edge"] too — never a side length
    proc = _processor(
        max_pixels = 12_845_056, size = {"shortest_edge": 3136, "longest_edge": 12_845_056}
    )
    assert native_image_limit(proc) == {
        "max_side": PIXEL_BUDGET_MAX_SIDE,
        "max_pixels": 12_845_056,
    }


def test_native_limit_falls_back_to_vision_config_image_size():
    model = SimpleNamespace(
        config = SimpleNamespace(vision_config = SimpleNamespace(image_size = [448, 448]))
    )
    assert native_image_limit(None, model) == {"max_side": 448}


def test_native_limit_is_clamped():
    assert native_image_limit(_processor(size = {"longest_edge": 100_000})) == {
        "max_side": MAX_SAFE_SIDE
    }
    assert MAX_SAFE_SIDE == 2048


def test_native_limit_none_when_nothing_declared():
    assert native_image_limit(None, None) is None
    assert native_image_limit(_processor(size = 224)) is None


def test_model_limit_is_the_default_and_env_overrides_it(monkeypatch):
    monkeypatch.delenv(INFERENCE_IMAGE_MAX_SIDE_ENV, raising = False)
    assert effective_image_limit({"max_side": 1540}) == {"max_side": 1540}
    assert effective_image_limit(None) == {"max_side": DEFAULT_INFERENCE_IMAGE_MAX_SIDE}
    monkeypatch.setenv(INFERENCE_IMAGE_MAX_SIDE_ENV, "600")
    assert effective_image_limit({"max_side": 1540}) == {"max_side": 600}
    monkeypatch.setenv(INFERENCE_IMAGE_MAX_SIDE_ENV, "0")
    assert effective_image_limit({"max_side": 1540}) is None


def test_page_under_the_model_size_is_left_alone(monkeypatch):
    monkeypatch.delenv(INFERENCE_IMAGE_MAX_SIDE_ENV, raising = False)
    page = _page()
    assert resize_for_inference(page, model_limit = {"max_side": 1540}) is page


def test_pixel_budget_model_gets_1024_on_a_phone_photo(monkeypatch):
    monkeypatch.delenv(INFERENCE_IMAGE_MAX_SIDE_ENV, raising = False)
    limit = native_image_limit(_processor(max_pixels = 12_845_056))
    photo = Image.new("RGB", (4000, 3000))
    assert resize_for_inference(photo, model_limit = limit).size == (1024, 768)


def test_a_small_pixel_budget_stays_the_tighter_bound(monkeypatch):
    monkeypatch.delenv(INFERENCE_IMAGE_MAX_SIDE_ENV, raising = False)
    out = resize_for_inference(
        Image.new("RGB", (4000, 3000)), model_limit = {"max_side": 1024, "max_pixels": 300_000}
    )
    assert out.size[0] * out.size[1] <= 300_000
    assert abs(out.size[0] / out.size[1] - 4 / 3) < 0.01


# ── wiring: worker derives the limit, the parent mirrors it and resizes by it ─────────────────────
def test_worker_reads_the_active_models_processor():
    from core.inference.worker import _active_image_limit

    backend = SimpleNamespace(
        active_model_name = "m",
        models = {"m": {"processor": _processor(size = {"longest_edge": 1540}), "model": None}},
    )
    assert _active_image_limit(backend) == {"max_side": 1540}
    assert _active_image_limit(SimpleNamespace(active_model_name = None, models = {})) is None


def test_parent_mirrors_the_limit_and_resizes_by_it(monkeypatch):
    from core.inference.orchestrator import InferenceOrchestrator, _mirrored_model_entry

    monkeypatch.delenv(INFERENCE_IMAGE_MAX_SIDE_ENV, raising = False)
    entry = _mirrored_model_entry({"image_limit": {"max_side": 1540}}, "m")
    assert entry["image_limit"] == {"max_side": 1540}
    orch = InferenceOrchestrator.__new__(InferenceOrchestrator)
    orch.models, orch.active_model_name = {"m": entry}, "m"
    page = _page()
    assert orch.resize_image(page) is page  # 1148×1288 fits the model's 1540
    orch.models, orch.active_model_name = {}, None
    assert orch.resize_image(page).size == FALLBACK_PAGE_SIZE  # no model info → 1024 fallback
