# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.1 T5 sequence length follows ComfyUI (floor 256, cap 512) instead of diffusers' fixed 512."""

from __future__ import annotations

import types

import pytest

from core.inference.diffusion_text_length import (
    flux_t5_kwarg,
    flux_t5_sequence_length,
)


class _WordTokenizer:
    """One id per whitespace word plus EOS, like T5TokenizerFast on plain words."""

    def __call__(
        self,
        text,
        add_special_tokens = True,
        **_,
    ):
        ids = [5] * len(text.split())
        return {"input_ids": ids + ([1] if add_special_tokens else [])}


def _words(n):
    return " ".join(["w"] * n)


@pytest.mark.parametrize(
    "words, expected",
    [(3, 256), (255, 256), (256, 512), (300, 512), (511, 512), (700, 512)],
)
def test_length_is_256_up_to_256_tokens_else_the_512_bucket(words, expected):
    assert flux_t5_sequence_length(_WordTokenizer(), [_words(words)]) == expected


def test_longest_prompt_of_a_list_sets_the_length():
    assert flux_t5_sequence_length(_WordTokenizer(), [["a", _words(200)]]) == 256
    assert flux_t5_sequence_length(_WordTokenizer(), [["a", _words(400)]]) == 512


def test_no_tokenizer_keeps_the_pipeline_default():
    assert flux_t5_sequence_length(None, ["a"]) is None


def test_only_flux1_families_and_only_when_accepted():
    pipe = types.SimpleNamespace(tokenizer_2 = _WordTokenizer())
    params = {"max_sequence_length": None}
    assert flux_t5_kwarg("flux.1", pipe, params, {"prompt": "a"}) == 256
    assert flux_t5_kwarg("flux.1-kontext", pipe, params, {"prompt": "a"}) == 256
    assert flux_t5_kwarg("flux.2-klein", pipe, params, {"prompt": "a"}) is None
    assert flux_t5_kwarg("flux.1", pipe, {}, {"prompt": "a"}) is None
    assert flux_t5_kwarg("flux.1", pipe, params, {"prompt": "a", "max_sequence_length": 77}) is None


def test_negative_counts_only_under_true_cfg():
    pipe = types.SimpleNamespace(tokenizer_2 = _WordTokenizer())
    params = {"max_sequence_length": None}
    kw = {"prompt": "a", "negative_prompt": _words(300)}
    assert flux_t5_kwarg("flux.1", pipe, params, kw) == 256
    assert flux_t5_kwarg("flux.1", pipe, params, {**kw, "true_cfg_scale": 4.0}) == 512
    assert flux_t5_kwarg("flux.1", pipe, params, {"prompt": "a", "prompt_2": _words(280)}) == 512


def test_ideogram4_comfy_guidance_switches_where_sigma_falls_to_0_3():
    from core.inference.diffusion_text_length import (
        ideogram4_comfy_guidance_schedule,
        ideogram4_sigmas,
    )

    sigmas = ideogram4_sigmas(20, 1024, 1024, 0.0, 1.75)
    assert sigmas == sorted(sigmas, reverse = True) and sigmas[0] > 0.999  # clamped at logSNR -15
    schedule = ideogram4_comfy_guidance_schedule(20, 1024, 1024)
    assert schedule == [7.0] * 17 + [3.0] * 3
    assert sigmas[16] > 0.3 >= sigmas[17]
