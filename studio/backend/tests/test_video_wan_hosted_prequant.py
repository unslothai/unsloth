# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Wan2.2 video families resolve their hosted fp8 / int8 denoiser artifacts, so a conventional load seeds them
instead of quantising the released fp32 shards at load. Hermetic: registry reads only, no torch, no Hub."""

from __future__ import annotations

import pytest

from core.inference.video_denoiser_prequant import denoiser_prequant_sources
from core.inference.video_families import (
    detect_video_family,
    video_family_prequant_repo,
    video_family_prequant_resident_gb,
    video_family_prequant_schemes,
)

TI2V_5B = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
T2V_A14B = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"


def _names(source) -> tuple[str, ...]:
    return (source.filename,) + tuple(getattr(source, "fallback_filenames", ()) or ())


@pytest.mark.parametrize("scheme", ["fp8", "int8"])
def test_ti2v_5b_resolves_its_hosted_artifact(scheme):
    fam = detect_video_family(TI2V_5B)
    assert fam is not None and fam.name == "wan2.2-ti2v-5b"
    assert video_family_prequant_repo(fam, scheme, TI2V_5B) == "unsloth/Wan2.2-TI2V-5B-FP8"
    assert scheme in video_family_prequant_schemes(fam)
    sources = denoiser_prequant_sources(fam, scheme, TI2V_5B)
    assert sources is not None and set(sources) == {"transformer"}
    names = _names(sources["transformer"])
    stem = f"Wan2.2-TI2V-5B-{scheme.upper()}"
    # A safetensors twin, once hosted, is tried first; today's pickle is next.
    assert names[:2] == (f"{stem}.safetensors", f"{stem}.pt")


@pytest.mark.parametrize("scheme", ["fp8", "int8"])
def test_t2v_a14b_seeds_both_experts_from_their_own_files(scheme):
    fam = detect_video_family(T2V_A14B)
    assert fam is not None and fam.name == "wan2.2-t2v-a14b"
    sources = denoiser_prequant_sources(fam, scheme, T2V_A14B)
    assert sources is not None and set(sources) == {"transformer", "transformer_2"}
    tag = scheme.upper()
    assert f"Wan2.2-T2V-A14B-{tag}.pt" in _names(sources["transformer"])
    # Expert 2 is served by its own row only: falling back to expert 1's name would load the wrong weights.
    assert _names(sources["transformer_2"]) == (f"Wan2.2-T2V-A14B-{tag}-2.pt",)
    assert sources["transformer_2"].location == "unsloth/Wan2.2-T2V-A14B-FP8"


@pytest.mark.parametrize(
    "repo, scheme, gb",
    [
        (TI2V_5B, "fp8", 5.1),
        (TI2V_5B, "int8", 5.0),
        (T2V_A14B, "fp8", 29.2),
        (T2V_A14B, "int8", 28.8),
    ],
)
def test_resident_sizes_price_the_artifact(repo, scheme, gb):
    assert video_family_prequant_resident_gb(detect_video_family(repo), scheme) == pytest.approx(gb)


def test_nvfp4_rows_are_unchanged():
    # Raw rows: the NVFP4 switch hides the scheme from the resolver, not from the table.
    assert ("nvfp4", "unsloth/Wan2.2-TI2V-5B-NVFP4") in detect_video_family(TI2V_5B).prequant_repos
    a14b = detect_video_family(T2V_A14B)
    assert ("nvfp4", "unsloth/Wan2.2-T2V-A14B-NVFP4") in a14b.prequant_repos
    assert (
        "nvfp4",
        "transformer_2",
        "Wan2.2-T2V-A14B-transformer_2-NVFP4.pt",
    ) in a14b.prequant_filenames


@pytest.mark.parametrize(
    "repo",
    [
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
    ],
)
@pytest.mark.parametrize("scheme", ["fp8", "int8"])
def test_families_without_a_hosted_artifact_resolve_none(repo, scheme):
    """No fp8 / int8 denoiser is hosted for HunyuanVideo-1.5, so nothing may be seeded (it would 404 after the plan
    dropped the released shards)."""
    fam = detect_video_family(repo)
    assert video_family_prequant_repo(fam, scheme, repo) is None
    assert denoiser_prequant_sources(fam, scheme, repo) is None
