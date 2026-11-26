# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Status payloads name each family's refused transformer_quant schemes, so the Precision picker disables them."""

from __future__ import annotations

import pytest

from core.inference import diffusion_transformer_quant as tq


@pytest.fixture(autouse = True)
def _fresh(monkeypatch):
    tq._DENIALS_CACHE.clear()
    yield
    tq._DENIALS_CACHE.clear()


def test_wan_and_hunyuan_nvfp4_carry_the_measured_gate_failure(monkeypatch):
    monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "1")
    denials = tq.transformer_quant_denials(["wan2.2-ti2v-5b", "hunyuanvideo-1.5", "ltx-2"])
    assert set(denials) == {"wan2.2-ti2v-5b", "hunyuanvideo-1.5"}
    wan = denials["wan2.2-ti2v-5b"]["nvfp4"]
    assert "failed the NVFP4 accuracy gate" in wan and "LPIPS" in wan
    assert "LPIPS" in denials["hunyuanvideo-1.5"]["nvfp4"]
    # Only denied schemes: fp8 / int8 stay selectable.
    assert set(denials["wan2.2-ti2v-5b"]) == {"nvfp4"}


def test_every_entry_is_what_an_explicit_load_would_refuse(monkeypatch):
    monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "1")
    denials = tq.transformer_quant_denials()
    assert "mxfp8" in denials["qwen-image"]
    for family, reasons in denials.items():
        for scheme, reason in reasons.items():
            assert tq.family_denies_scheme(family, scheme)
            assert reason == tq.explain_unusable_scheme(family, scheme)


def test_a_gate_record_that_lifts_nvfp4_for_a_base_leaves_nvfp4_enabled(monkeypatch):
    monkeypatch.setenv("UNSLOTH_NVFP4_DIFFUSION", "1")
    monkeypatch.setattr(
        tq,
        "nvfp4_gated_base",
        lambda family: "some/base" if family == "wan2.2-ti2v-5b" else None,
    )
    assert "wan2.2-ti2v-5b" not in tq.transformer_quant_denials(["wan2.2-ti2v-5b"])


def test_cached_answer_is_a_copy(monkeypatch):
    first = tq.transformer_quant_denials(["qwen-image"])
    first["qwen-image"].clear()
    assert tq.transformer_quant_denials(["qwen-image"])["qwen-image"]


def test_image_and_video_status_payloads_carry_the_denials(monkeypatch):
    from core.inference import diffusion_engine_router as router
    from core.inference import video as video_mod

    monkeypatch.setattr(router, "_supported_family_capabilities", lambda: ("qwen-image", "z-image"))
    out = router.annotate_status({"loaded": False, "family": None})
    assert "mxfp8" in out["transformer_quant_denials"]["qwen-image"]
    assert "z-image" not in out["transformer_quant_denials"]
    assert "nvfp4" in video_mod._transformer_quant_denials(("wan2.2-ti2v-5b",))["wan2.2-ti2v-5b"]


def test_the_status_response_models_pass_the_field_through():
    from models.inference import DiffusionStatusResponse, VideoStatusResponse
    payload = {"loaded": False, "transformer_quant_denials": {"wan2.2-ti2v-5b": {"nvfp4": "why"}}}
    for model in (DiffusionStatusResponse, VideoStatusResponse):
        fields = model.model_fields
        assert "transformer_quant_denials" in fields
        assert (
            model.model_validate(payload).transformer_quant_denials
            == payload["transformer_quant_denials"]
        )
