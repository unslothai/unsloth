# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An int8 image load on an fp16-only NVIDIA card (Turing) reads the hosted int8 ConvRot DiT as native W8A8 Linears in
the card's dtype instead of quantising the dense shards at load. bf16 cards, weight-only builds, families without a
hosted ConvRot build and the kill switch keep the previous routes."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from core.inference import diffusion_fp16_hosted_int8 as FH  # noqa: E402
from core.inference import diffusion_rocm_hosted_int8 as RH  # noqa: E402
from core.inference import diffusion_transformer_quant as tq  # noqa: E402
from core.inference.diffusion_families import detect_family  # noqa: E402

QWEN21 = ("Qwen/Qwen-Image-2.1", "qwen-image-2.1")
FLUX3 = ("black-forest-labs/flux-3-image-klein", "flux.3-klein")
T4 = types.SimpleNamespace(device = "cuda", dtype = torch.float16, ordinal = 0)
L4 = types.SimpleNamespace(device = "cuda", dtype = torch.bfloat16, ordinal = 0)


def _fam(pair = QWEN21):
    fam = detect_family(pair[0], override = pair[1])
    assert fam is not None and fam.name == pair[1]
    return fam


@pytest.fixture
def turing(monkeypatch):
    for env in (
        FH.FP16_CARD_HOSTED_INT8_ENV,
        "UNSLOTH_NATIVE_INT8_ACT",
        "UNSLOTH_NATIVE_INT8_ROT",
        "UNSLOTH_DIFFUSION_INT8_CONVROT",
    ):
        monkeypatch.delenv(env, raising = False)
    monkeypatch.setattr(tq, "torch_is_rocm", lambda: False)
    monkeypatch.setattr(tq, "is_stubbed", lambda name: False)
    monkeypatch.setattr(tq, "_capability", lambda ordinal = None: (7, 5))


@pytest.mark.parametrize("pair", [QWEN21, FLUX3])
def test_turing_w8a8_build_reads_the_hosted_file(turing, pair):
    assert FH.fp16_card_hosted_int8_reason(_fam(pair), "int8", T4) is not None


def test_unscored_families_keep_the_at_load_build(turing, monkeypatch):
    # A family outside the scored set keeps the at-load build even when it hosts a ConvRot file.
    monkeypatch.setattr(FH, "FP16_CARD_HOSTED_INT8_FAMILIES", frozenset({"qwen-image-2.1"}))
    assert FH.fp16_card_hosted_int8_reason(_fam(FLUX3), "int8", T4) is None


def test_bf16_cards_keep_their_routes(turing):
    assert FH.fp16_card_hosted_int8_reason(_fam(), "int8", L4) is None


@pytest.mark.parametrize("scheme", [None, "fp8", "nvfp4"])
def test_other_native_builds_keep_the_at_load_build(turing, scheme):
    assert FH.fp16_card_hosted_int8_reason(_fam(), scheme, T4) is None


def test_weight_only_turing_build_keeps_the_dense_shards(turing, monkeypatch):
    # The hosted file is W8A8: a weight-only build (activations kept float) is not it.
    monkeypatch.setattr(tq, "fp16_card_int8_act", lambda target: False)
    assert FH.fp16_card_hosted_int8_reason(_fam(), "int8", T4) is None


def test_family_without_a_hosted_convrot_build_is_untouched(turing, monkeypatch):
    monkeypatch.setattr(RH, "_hosted_convrot_file", lambda fam: None)
    assert FH.fp16_card_hosted_int8_reason(_fam(), "int8", T4) is None


def test_kill_switch(turing, monkeypatch):
    monkeypatch.setenv(FH.FP16_CARD_HOSTED_INT8_ENV, "0")
    assert FH.fp16_card_hosted_int8_reason(_fam(), "int8", T4) is None


def test_reader_asks_for_the_cards_dtype(turing, monkeypatch):
    from core.inference import diffusion_denoiser_prequant as DP

    source = types.SimpleNamespace(kind = "repo", location = "unsloth/Qwen-Image-2.1-FP8")
    monkeypatch.setattr(DP, "denoiser_prequant_source", lambda *a, **k: source)
    seen = {}
    module = object()

    def reader(*a, **k):
        seen.update(k)
        return module

    monkeypatch.setattr(RH, "load_rocm_hosted_transformer", reader)
    got = FH.fp16_card_hosted_int8_transformer(
        _fam(), T4, object, base = QWEN21[0], config_source = "/snap"
    )
    assert got is module
    assert seen["compute_dtype"] is torch.float16
    assert seen["config_source"] == "/snap"
    assert seen["tag"] == "diffusion.fp16_hosted_int8"


def test_no_hosted_source_keeps_the_at_load_build(turing, monkeypatch):
    from core.inference import diffusion_denoiser_prequant as DP
    monkeypatch.setattr(DP, "denoiser_prequant_source", lambda *a, **k: None)
    assert (
        FH.fp16_card_hosted_int8_transformer(_fam(), T4, object, base = QWEN21[0], config_source = None)
        is None
    )
