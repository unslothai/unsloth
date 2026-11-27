# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``_same_base_model``: the base check every hosted prequant reader runs. A fine-tune whose repo name equals the
official one (``someone/Qwen-Image-2.1``) is refused; the official repo, its unsloth mirrors, the retired full copies
and a local copy of the base still match every ``base_model_id`` the hosted files record."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.inference.diffusion_families import (  # noqa: E402
    _ALL_MIRROR_PAIRS,
    _RETIRED_FULL_COPIES,
    canonical_base,
)
from core.inference.diffusion_prequant import _same_base_model  # noqa: E402

# Every base_model_id in the unsloth_metadata of the root .safetensors of the unsloth repos Studio names
# (header reads of the live Hub repos, Nov 2026).
HOSTED_BASE_MODEL_IDS = (
    "Alpha-VLLM/Lumina-Image-2.0",
    "HiDream-ai/HiDream-I1-Full",
    "Lightricks/LTX-2",
    "Lightricks/LTX-2.3",
    "Lightricks/LTX-2.5-Diffusers",
    "MiniMaxAI/MiniMax-H3",
    "MiniMaxAI/MiniMax-H3-Image",
    "OpenMOSS-Team/MOSS-Image-Base",
    "OpenMOSS-Team/MOSS-Image-SFT",
    "Qwen/Qwen-Image",
    "Qwen/Qwen-Image-2.1",
    "Qwen/Qwen-Image-2.1-Turbo",
    "Qwen/Qwen-Image-2512",
    "Tongyi-MAI/Z-Image-Turbo",
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "black-forest-labs/FLUX.1-Krea-dev",
    "black-forest-labs/FLUX.1-dev",
    "black-forest-labs/FLUX.1-schnell",
    "black-forest-labs/FLUX.2-dev",
    "black-forest-labs/FLUX.2-klein-4B",
    "black-forest-labs/FLUX.2-klein-9B",
    "black-forest-labs/flux-3-image-klein",
    "black-forest-labs/flux-3-image-klein-base",
    "hunyuanvideo-community/HunyuanImage-2.1-Diffusers",
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
    "ideogram-ai/ideogram-4-fp8",
    "krea/Krea-2-Turbo",
    "unsloth/Meta-Llama-3.1-8B-Instruct",
)


def _loading_ids(recorded):
    """Every id Studio can load the base under: itself, any case, its declared mirror and retired copies."""
    ids = {recorded, recorded.lower(), recorded.upper()}
    ids.update(m for u, m in _ALL_MIRROR_PAIRS if u.lower() == recorded.lower())
    ids.update(c for c, u in _RETIRED_FULL_COPIES.items() if u.lower() == recorded.lower())
    return sorted(ids)


@pytest.mark.parametrize("recorded", HOSTED_BASE_MODEL_IDS)
def test_every_hosted_base_model_id_matches_its_official_and_mirror_ids(recorded):
    for base in _loading_ids(recorded):
        assert _same_base_model(recorded, base), (recorded, base)
        assert _same_base_model(base, recorded), (base, recorded)


@pytest.mark.parametrize("upstream, mirror", _ALL_MIRROR_PAIRS)
def test_every_declared_mirror_pair_still_matches(upstream, mirror):
    assert _same_base_model(upstream, mirror)
    assert _same_base_model(mirror, upstream)
    assert _same_base_model(mirror.lower(), upstream)
    # a mirror is a copy of its own upstream only
    others = [u for u, _m in _ALL_MIRROR_PAIRS if canonical_base(u).lower() != upstream.lower()]
    assert not any(_same_base_model(mirror, other) for other in others)


@pytest.mark.parametrize("copy_id, upstream", sorted(_RETIRED_FULL_COPIES.items()))
def test_retired_full_copies_still_match(copy_id, upstream):
    assert _same_base_model(copy_id, upstream)
    assert _same_base_model(upstream, copy_id)


@pytest.mark.parametrize(
    "recorded, base",
    [
        ("Qwen/Qwen-Image-2.1", "community/Qwen-Image-2.1"),
        ("Qwen/Qwen-Image-2.1", "someone/qwen-image-2.1"),
        ("Lightricks/LTX-2.5-Diffusers", "some-other-owner/LTX-2.5-Diffusers"),
        ("black-forest-labs/FLUX.1-dev", "someone/FLUX.1-dev"),
        ("black-forest-labs/flux-3-image-klein", "someone/flux-3-image-klein"),
        # a mirror id does not vouch for a fine-tune either
        ("unsloth/Qwen-Image-2.1", "community/Qwen-Image-2.1"),
    ],
)
def test_a_same_name_fine_tune_is_not_the_official_base(recorded, base):
    assert not _same_base_model(recorded, base)
    assert not _same_base_model(base, recorded)


@pytest.mark.parametrize(
    "local",
    [
        "/models/Qwen-Image-2.1",
        "/models/Qwen-Image-2.1/",
        "C:\\models\\Qwen-Image-2.1",
        "./Qwen-Image-2.1",
    ],
)
def test_a_local_copy_keeps_the_name_compare(local):
    assert _same_base_model("Qwen/Qwen-Image-2.1", local)
    assert not _same_base_model("Qwen/Qwen-Image-2.1-Turbo", local)


def test_an_existing_relative_dir_is_a_local_copy(tmp_path, monkeypatch):
    (tmp_path / "community" / "Qwen-Image-2.1").mkdir(parents = True)
    monkeypatch.chdir(tmp_path)
    assert _same_base_model("Qwen/Qwen-Image-2.1", "community/Qwen-Image-2.1")


def test_te_and_training_readers_refuse_a_same_name_fine_tune():
    from core.inference.diffusion_te_prequant import te_base_equivalent
    from core.training.diffusion_fp8_prequant import checkpoint_problem

    assert te_base_equivalent("Qwen/Qwen-Image", "unsloth/Qwen-Image")
    assert not te_base_equivalent("Qwen/Qwen-Image", "community/Qwen-Image")

    def header(base):
        info = {"scheme": "fp8", "fp8_granularity": "per_row", "base_model_id": base}
        from core.training import diffusion_fp8_prequant as fp8
        return {
            "__metadata__": {"unsloth_format": fp8._FORMAT, "unsloth_metadata": json.dumps(info)}
        }

    assert "baked from" not in (
        checkpoint_problem(header("Qwen/Qwen-Image"), "unsloth/Qwen-Image") or ""
    )
    assert "baked from" in (
        checkpoint_problem(header("Qwen/Qwen-Image"), "community/Qwen-Image") or ""
    )
