# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every denoiser class Studio loads has a prompt-length policy (diffusion_dynamic_text.PROMPT_LENGTH_POLICY): either
the text length follows the prompt (the dynamo sources that carry it), or the pipeline pads every prompt to one
length. The whole-step CUDA graph keys on input shapes, so it reads this table to tell the two apart."""

from __future__ import annotations

import re

from core.inference import diffusion_dynamic_text as dt
from core.inference import diffusion_families as df
from core.inference import video_families as vf

VARIABLE = (
    "FluxTransformer2DModel",
    "Flux3Transformer2DModel",
    "QwenImageTransformer2DModel",
    "QwenImage21Transformer2DModel",
    "ZImageTransformer2DModel",
    "MiniMaxH3Transformer3DModel",
    "MOSSImageTransformer",
)


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


def test_variable_length_pipelines_are_classified():
    for name in VARIABLE:
        assert isinstance(dt.PROMPT_LENGTH_POLICY[name], tuple), name
    fixed = sorted(n for n, p in dt.PROMPT_LENGTH_POLICY.items() if p == dt.FIXED_LENGTH)
    assert fixed == sorted(set(dt.PROMPT_LENGTH_POLICY) - set(VARIABLE))


def test_prompt_length_varies_reads_the_policy():
    for name, policy in dt.PROMPT_LENGTH_POLICY.items():
        cls = type(name, (), {})
        assert dt.prompt_length_varies(cls) is isinstance(policy, tuple), name
        assert dt.prompt_length_varies(cls()) is isinstance(policy, tuple), name
    assert dt.prompt_length_varies(type("SomeNewTransformer", (), {})()) is False


def test_the_compile_arms_only_its_own_sources():
    # the classification adds no compile sources: FLUX.1 and Z-Image compile as before
    assert set(dt._FAMILY_SOURCES) == {
        "Flux3Transformer2DModel",
        "QwenImage21Transformer2DModel",
        "QwenImageTransformer2DModel",
        "MiniMaxH3Transformer3DModel",
        # MOSS-Image never pads its prompt (true text length per branch), so it is armed from day one.
        "MOSSImageTransformer",
    }
    for name, sources in dt._FAMILY_SOURCES.items():
        assert dt.PROMPT_LENGTH_POLICY[name] == sources, name


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
    assert dt.fingerprint(Qwen(), True) == dt.fingerprint(Qwen(), None)
    assert dt.fingerprint(Flux3(), True) != dt.fingerprint(Flux3(), None)
