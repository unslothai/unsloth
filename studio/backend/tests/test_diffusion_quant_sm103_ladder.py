# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Auto transformer quant on sm_103 (B300 / GB300): fp8 first where fp8 passed its accuracy gate, int8 first for
every other family, and every other capability exactly as before. CPU only: the card is mocked."""

from __future__ import annotations

import pytest

import core.inference.diffusion_transformer_quant as tq
from core.inference.diffusion_transformer_quant import TQ_FP8, TQ_INT8, TQ_MXFP8, TQ_NVFP4

SM103 = (10, 3)

# Frozen here, not read from the module: the test then collects on any tree and catches a family dropped by accident.
FP8_FIRST = (
    "flux.1",
    "flux.2-klein",
    "flux.2-dev",
    "flux.3-klein",
    "flux.3-klein-base",
    "z-image",
    "krea-2",
    "lumina-2",
    "qwen-image",
    "wan2.2-ti2v-5b",
)
# FP8-first families with an mxfp8 row on record: mxfp8 before int8 when fp8 fails. The rest go fp8, int8, mxfp8.
MXFP8_BEFORE_INT8 = (
    "flux.1",
    "flux.2-klein",
    "flux.2-dev",
    "flux.3-klein",
    "flux.3-klein-base",
    "z-image",
    "krea-2",
)

# The ladder before the sm_103 rung: one Blackwell rung for every cap >= (10, 0).
_OLD_LADDER = (
    ((10, 0), (TQ_INT8, TQ_FP8, TQ_MXFP8)),
    ((8, 9), (TQ_INT8, TQ_FP8)),
    ((8, 0), (TQ_INT8,)),
)

FAMILIES = (
    None,
    "flux.1",
    "flux.1-kontext",
    "flux.2-klein",
    "flux.2-dev",
    "flux.3-klein",
    "flux.3-klein-base",
    "qwen-image",
    "qwen-image-edit",
    "qwen-image-layered",
    "qwen-image-2.1",
    "z-image",
    "krea-2",
    "lumina-2",
    "chroma",
    "anima",
    "moss-image",
    "moss-image-base",
    "hunyuanimage-2.1",
    "hidream-i1",
    "ideogram-4",
    "minimax-h3-image",
    "sdxl",
    "minimax-h3",
    "ltx-2.5",
    "ltx-2",
    "wan2.2-ti2v-5b",
    "wan2.2-t2v-a14b",
    "wan2.1-t2v-1.3b",
    "hunyuanvideo-1.5",
    "hunyuanvideo-1.5-720p",
    "cosmos3-nano",
)


def _allowed(family, cap):
    return tuple(
        s
        for s in tq._auto_scheme_order(family, "cuda", cap)
        if not tq._family_denied(family, s, None)
    )


@pytest.mark.parametrize("family", FP8_FIRST)
def test_sm103_puts_fp8_first_for_families_whose_fp8_passed(family):
    order = _allowed(family, SM103)
    if family == "qwen-image":  # qwen denies mxfp8
        expected = (TQ_FP8, TQ_INT8)
    elif family in MXFP8_BEFORE_INT8:
        expected = (TQ_FP8, TQ_MXFP8, TQ_INT8)
    else:
        expected = (TQ_FP8, TQ_INT8, TQ_MXFP8)
    assert order == expected


@pytest.mark.parametrize("family", [f for f in FAMILIES if f not in FP8_FIRST])
def test_sm103_keeps_the_old_order_without_passing_fp8_evidence(family, monkeypatch):
    new = _allowed(family, SM103)
    monkeypatch.setattr(tq, "_AUTO_LADDER", _OLD_LADDER)
    assert new == _allowed(family, SM103)


@pytest.mark.parametrize(
    "cap",
    [(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (10, 1), (10, 4), (10, 7), (11, 0), (12, 0), (12, 1)],
)
@pytest.mark.parametrize("family", FAMILIES)
def test_every_other_capability_is_unchanged(family, cap, monkeypatch):
    new = tq._auto_scheme_order(family, "cuda", cap)
    monkeypatch.setattr(tq, "_AUTO_LADDER", _OLD_LADDER)
    assert new == tq._auto_scheme_order(family, "cuda", cap)


def test_failed_or_skipped_fp8_families_stay_off_fp8_first():
    # fp8 over the bar (HunyuanVideo-1.5 0.35-0.37, LTX-2.5 0.366, H3-Image 0.255) or auto-skipped (Chroma, MOSS).
    for family in (
        "hunyuanvideo-1.5",
        "ltx-2.5",
        "minimax-h3-image",
        "hidream-i1",
        "hunyuanimage-2.1",
    ):
        assert _allowed(family, SM103)[0] == TQ_INT8
    assert _allowed("chroma", SM103) == ()
    assert _allowed("moss-image", SM103) == (TQ_INT8,)


def test_a_gated_int8_first_family_head_also_puts_int8_last_on_sm103(monkeypatch):
    from core.inference import diffusion_nvfp4_flag as flag

    monkeypatch.setenv(flag.NVFP4_DIFFUSION_ENV, "1")
    monkeypatch.setattr(tq, "_nvfp4_gate_passed", lambda family, base_repo: True)
    monkeypatch.setattr(tq, "_nvfp4_gate_backend_ok", lambda family, base_repo, device: True)
    monkeypatch.setattr(tq, "_is_consumer_gpu", lambda device = None: False)
    assert tq._auto_scheme_order("flux.1", "cuda", SM103) == (TQ_FP8, TQ_NVFP4, TQ_MXFP8, TQ_INT8)
    assert tq._auto_scheme_order("flux.1", "cuda", (10, 7)) == (TQ_INT8, TQ_FP8, TQ_NVFP4, TQ_MXFP8)
    assert tq._auto_scheme_order("flux.1", "cuda", (10, 0)) == (TQ_INT8, TQ_FP8, TQ_NVFP4, TQ_MXFP8)
    # Unmeasured on fp8: the head is kept as it is.
    assert tq._auto_scheme_order("qwen-image-2.1", "cuda", SM103)[0] == TQ_INT8


def test_a_mocked_sm103_card_selects_fp8_and_falls_back(monkeypatch):
    target = type("T", (), {"device": "cuda", "dtype": "bfloat16", "ordinal": 0})()
    monkeypatch.setattr(tq, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(tq, "_capability", lambda ordinal = None: SM103)
    monkeypatch.setattr(tq, "_scheme_supported", lambda scheme, device, *a, **k: True)
    assert tq.auto_scheme_candidates(target, "flux.1") == (TQ_FP8, TQ_MXFP8, TQ_INT8)
    assert tq.auto_scheme_candidates(target, "lumina-2") == (TQ_FP8, TQ_INT8, TQ_MXFP8)
    assert tq.auto_scheme_candidates(target, "ltx-2.5")[0] == TQ_INT8
    assert tq.select_transformer_quant_scheme(target, "auto", family = "z-image") == TQ_FP8
    # fp8's smoke probe fails on this card: auto moves on to the next rung entry, not to bf16.
    monkeypatch.setattr(tq, "_scheme_supported", lambda scheme, device, *a, **k: scheme != TQ_FP8)
    assert tq.select_transformer_quant_scheme(target, "auto", family = "z-image") == TQ_MXFP8
    # No mxfp8 row for these: int8 is the fallback, not mxfp8.
    assert tq.select_transformer_quant_scheme(target, "auto", family = "lumina-2") == TQ_INT8
    assert tq.select_transformer_quant_scheme(target, "auto", family = "wan2.2-ti2v-5b") == TQ_INT8
    monkeypatch.setattr(tq, "_capability", lambda ordinal = None: (10, 7))
    monkeypatch.setattr(tq, "_scheme_supported", lambda scheme, device, *a, **k: True)
    assert tq.select_transformer_quant_scheme(target, "auto", family = "z-image") == TQ_INT8
    monkeypatch.setattr(tq, "_capability", lambda ordinal = None: (10, 0))
    monkeypatch.setattr(tq, "_scheme_supported", lambda scheme, device, *a, **k: True)
    assert tq.select_transformer_quant_scheme(target, "auto", family = "z-image") == TQ_INT8


@pytest.mark.parametrize("cap", [(10, 4), (10, 7)])
def test_only_the_exact_sm103_capability_reorders(cap):
    for family in FP8_FIRST:
        assert _allowed(family, cap)[0] == TQ_INT8


def test_mixed_host_live_and_cached_answers_read_the_target_card(monkeypatch):
    """Current device sm_103 (ordinal 0), target an sm_100 at ordinal 1: every reader answers for the target."""
    caps = {0: SM103, 1: (10, 0)}
    monkeypatch.setattr(tq, "dense_transformer_supported", lambda target: True)
    monkeypatch.setattr(
        tq, "_capability", lambda ordinal = None: caps[0 if ordinal is None else int(ordinal)]
    )
    monkeypatch.setattr(tq, "_scheme_supported", lambda scheme, device, *a, **k: True)
    monkeypatch.setattr(
        tq, "_smoke_cache_device_key", lambda device, ordinal = None: f"cuda:{ordinal or 0}"
    )
    monkeypatch.setattr(tq, "_SMOKE_CACHE", {})
    for ordinal, first in ((1, TQ_INT8), (0, TQ_FP8)):
        target = type(
            "T", (), {"device": f"cuda:{ordinal}", "dtype": "bfloat16", "ordinal": ordinal}
        )()
        live = tq.select_transformer_quant_scheme(target, "auto", family = "z-image")
        candidates = tq.auto_scheme_candidates(target, "z-image")
        cached = tq.auto_scheme_candidates_cached(target, "z-image")
        assert live == candidates[0] == cached[0] == first
