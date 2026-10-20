# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every denoiser class Studio loads has a prompt-length policy (diffusion_dynamic_text.PROMPT_LENGTH_POLICY): either
the dynamo sources that make the first compile cover every prompt length, or the pipeline pads every prompt to one
length. A class with neither recompiled on the first new prompt length (generation 2) on the max tier."""

from __future__ import annotations

import re

from core.inference import diffusion_dynamic_text as dt
from core.inference import diffusion_families as df
from core.inference import video_families as vf


def _classes() -> set[str]:
    out: set[str] = set()
    for fam in (*df._FAMILIES, *vf._FAMILIES):
        for attr in ("transformer_class", "transformer2_class"):
            name = getattr(fam, attr, None)
            if name:
                out.add(str(name))
    return out


def test_every_family_class_has_a_prompt_length_policy():
    missing = sorted(_classes() - set(dt.PROMPT_LENGTH_POLICY))
    assert not missing, f"no prompt-length policy for {missing}"


def test_policies_are_sources_or_fixed():
    for name, policy in dt.PROMPT_LENGTH_POLICY.items():
        assert policy == dt.FIXED_LENGTH or (isinstance(policy, tuple) and policy), name
        if isinstance(policy, tuple):
            for source in policy:
                assert source.startswith("L[") or re.compile(source), (name, source)
            assert dt._FAMILY_SOURCES[name] == policy


def test_variable_length_pipelines_are_armed():
    # the pipelines that hand the transformer a prompt-dependent text length
    for name in (
        "FluxTransformer2DModel",
        "Flux3Transformer2DModel",
        "QwenImageTransformer2DModel",
        "QwenImage21Transformer2DModel",
        "ZImageTransformer2DModel",
        "MiniMaxH3Transformer3DModel",
    ):
        assert isinstance(dt.PROMPT_LENGTH_POLICY[name], tuple), name


def test_the_sources_reach_the_compile_config():
    class Flux:
        pass

    Flux.__name__ = "FluxTransformer2DModel"
    assert dt.sources_for(Flux()) == tuple(
        s for s in dt._FLUX1_SOURCES if s not in set(dt.unbacked_sources_for(Flux()))
    )


def test_qwen_image_arms_its_sources_on_the_dynamic_tier_too():
    class Qwen:
        pass

    class Flux3:
        pass

    Qwen.__name__ = "QwenImageTransformer2DModel"
    Flux3.__name__ = "Flux3Transformer2DModel"
    if not dt.supported():
        return
    assert dt._armed_when_dynamic(Qwen()) and not dt._armed_when_dynamic(Flux3())
    # the compile-cache key follows what the compile arms
    assert dt.fingerprint(Qwen(), True) == dt.fingerprint(Qwen(), None)
    assert dt.fingerprint(Flux3(), True) != dt.fingerprint(Flux3(), None)
