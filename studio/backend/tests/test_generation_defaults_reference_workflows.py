# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Default steps / guidance match the reference workflows; frontend tables match the backend."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.inference import diffusion_families as df
from core.inference import video_families as vf

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "src" / "features"
_ROW = re.compile(r'\{\s*match:\s*"([^"]+)",\s*steps:\s*([\d.]+),\s*guidance:\s*([\d.]+)\s*\}')


def _frontend_rows(relative: str) -> dict[str, tuple[int, float]]:
    text = (_FRONTEND / relative).read_text(encoding = "utf-8")
    return {m[1]: (int(m[2]), float(m[3])) for m in _ROW.finditer(text)}


@pytest.mark.parametrize(
    "repo, want",
    [
        ("OpenMOSS-Team/MOSS-Image-SFT", (25, 2.5)),
        ("OpenMOSS-Team/MOSS-Image-Base", (25, 2.5)),
        ("unsloth/MOSS-Image-SFT-FP8", (25, 2.5)),
        ("stabilityai/sdxl-turbo", (1, 0.0)),
        ("unsloth/sdxl-turbo", (1, 0.0)),
        # Unchanged: already the reference values (Z-Image's diffusers g is the reference cfg minus 1).
        ("stabilityai/stable-diffusion-xl-base-1.0", (25, 7.0)),
        ("Tongyi-MAI/Z-Image", (25, 3.0)),
        ("black-forest-labs/FLUX.3-Image-Klein-Base", (50, 3.5)),
        ("flux.3-klein", (16, 1.0)),
    ],
)
def test_image_defaults(repo, want):
    assert df.default_generation_params(repo) == want


def test_minimax_h3_default_is_the_non_lora_reference_recipe():
    fam = vf.detect_video_family("MiniMaxAI/MiniMax-H3")
    assert fam is not None and fam.name == "minimax-h3"
    assert (fam.default_steps, fam.default_guidance) == (20, 1.0)
    assert vf.default_video_generation_params(
        "MiniMaxAI/MiniMax-H3", fallback = (fam.default_steps, fam.default_guidance), family = fam.name
    ) == (20, 1.0)
    assert vf.default_video_generation_params(
        "someone/MiniMax-H3-PDD", fallback = (20, 1.0), family = fam.name
    ) == (8, 1.0)


def test_ltx2_keeps_its_single_stage_recipe():
    fam = vf.detect_video_family("Lightricks/LTX-2")
    assert (fam.default_steps, fam.default_guidance) == (40, 4.0)


def test_images_page_table_matches_backend():
    rows = _frontend_rows("images/image-generation-defaults.ts")
    for key in (
        "moss-image",
        "moss_image",
        "mossimage",
        "sdxl-turbo",
        "sdxl",
        "z-image",
        "flux.3-klein-base",
    ):
        assert rows[key] == df.default_generation_params(key), key
    backend = {k: (s, g) for k, s, g in df._GENERATION_DEFAULTS}
    for key, value in rows.items():
        if key in backend:
            assert value == backend[key], key


def test_video_page_minimax_row_matches_backend():
    rows = _frontend_rows("video/video-page.tsx")
    fam = vf.detect_video_family("MiniMaxAI/MiniMax-H3")
    assert rows["minimax-h3"] == (fam.default_steps, fam.default_guidance)
