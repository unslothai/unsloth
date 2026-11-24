# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hosted LTX-2 (2.0) DiT checkpoints (unsloth/LTX-2-FP8): only the 2.0 base seeds them, only fp8 resolves.
Network-free and torch-free; metadata copied from the published safetensors headers."""

from __future__ import annotations

import logging

import pytest

from core.inference.diffusion_prequant import (
    PREQUANT_FORMAT,
    _validate_checkpoint,
    resolve_prequant_source,
)
from core.inference.video_families import (
    detect_video_family,
    video_family_prequant_repo,
    video_family_prequant_schemes,
)

BASE = "Lightricks/LTX-2"
REPO = "unsloth/LTX-2-FP8"

# unsloth_metadata of LTX-2-FP8.safetensors and LTX-2-INT8.safetensors (revision 35b7c2304c), verbatim.
_FP8_META = {
    "base_model_id": "Lightricks/LTX-2",
    "family": "ltx-2",
    "scheme": "fp8",
    "min_features": 512,
    "exclude_name_tokens": [],
    "require_bf16": True,
    "fast_accum": False,
    "torch_dtype": "bfloat16",
    "quant_backend": "torchao",
    "transformer_class": "LTX2VideoTransformer3DModel",
    "torch_version": "2.10.0+cu130",
    "torchao_version": "0.17.0",
    "diffusers_version": "0.39.0",
    "fp8_granularity": "per_row",
}
_INT8_META = {
    "base_model_id": "Lightricks/LTX-2",
    "family": "ltx-2",
    "scheme": "int8",
    "min_features": 512,
    "exclude_name_tokens": [
        "norm",
        "_mod",
        "modulation",
        "timestep_embed",
        "guidance_embed",
        "time_text_embed",
        "pooled",
        "time_embed",
    ],
    "require_bf16": False,
    "fast_accum": None,
    "torch_dtype": "bfloat16",
    "quant_backend": "torchao",
    "transformer_class": "LTX2VideoTransformer3DModel",
    "torch_version": "2.10.0+cu130",
    "torchao_version": "0.17.0",
    "diffusers_version": "0.39.0",
}


@pytest.fixture
def fam():
    family = detect_video_family(BASE)
    assert family is not None and family.name == "ltx-2"
    return family


def test_the_20_base_resolves_the_hosted_fp8_file(fam):
    assert video_family_prequant_repo(fam, "fp8", BASE) == REPO
    assert video_family_prequant_repo(fam, "fp8", REPO) == REPO
    src = resolve_prequant_source(fam, "fp8", base_repo = BASE)
    assert (src.kind, src.location, src.filename) == ("repo", REPO, "LTX-2-FP8.safetensors")
    assert "LTX-2-FP8.pt" in src.fallback_filenames


def test_int8_is_not_wired(fam):
    # LTX-2-INT8.safetensors predates the LTX-2 int8 excludes; the loader would refuse it.
    assert video_family_prequant_repo(fam, "int8", BASE) is None
    assert resolve_prequant_source(fam, "int8", base_repo = BASE) is None


@pytest.mark.parametrize("base", ["Lightricks/LTX-2.3", None, "", "someone/LTX-2-fork"])
def test_no_other_base_takes_the_20_file(fam, base):
    assert video_family_prequant_repo(fam, "fp8", base) != REPO


def test_23_keeps_its_own_file(fam):
    src = resolve_prequant_source(fam, "fp8", base_repo = "Lightricks/LTX-2.3")
    assert (src.location, src.filename) == ("unsloth/LTX-2.3-FP8", "LTX-2.3-FP8.safetensors")
    assert set(video_family_prequant_schemes(fam)) == {"fp8"}


def test_the_published_fp8_metadata_passes_the_loader():
    ckpt = {"format": PREQUANT_FORMAT, "metadata": dict(_FP8_META)}
    assert _validate_checkpoint(
        ckpt,
        "fp8",
        BASE,
        logging.getLogger("t"),
        min_features = 512,
        component = "transformer",
        metadata_only = True,
    )
    for other in ("Lightricks/LTX-2.3", "Lightricks/LTX-2.5-Diffusers"):
        assert not _validate_checkpoint(
            ckpt, "fp8", other, logging.getLogger("t"), min_features = 512, metadata_only = True
        )


def test_the_published_int8_metadata_is_refused():
    ckpt = {"format": PREQUANT_FORMAT, "metadata": dict(_INT8_META)}
    assert not _validate_checkpoint(
        ckpt, "int8", BASE, logging.getLogger("t"), min_features = 512, metadata_only = True
    )


def test_an_explicit_fp8_pipeline_load_plans_the_seed(fam, monkeypatch):
    from core.inference import video as vid
    from core.inference import video_denoiser_prequant as vdp

    monkeypatch.setattr(vdp, "hosted_denoiser_refused", lambda *a, **k: False)
    sources = vdp.denoiser_prequant_sources(fam, "fp8", BASE)
    assert set(sources) == {"transformer"} and sources["transformer"].location == REPO
    assert vdp.denoiser_prequant_sources(fam, "auto", BASE) is None
    assert vdp.denoiser_prequant_sources(fam, "int8", BASE) is None
    assert vid._planned_denoiser_components(fam, "pipeline") == ("transformer",)


def _api(siblings):
    import types

    asked: list = []

    def model_info(repo_id, **_kwargs):
        asked.append(repo_id)
        return types.SimpleNamespace(
            siblings = [types.SimpleNamespace(rfilename = n, size = s) for n, s in siblings.items()]
        )

    return types.SimpleNamespace(token = None, model_info = model_info, asked = asked)


def test_the_plan_credits_the_hosted_file(fam, monkeypatch):
    import core.inference.diffusion_prequant as pq
    from core.inference import video_denoiser_prequant as vdp
    from core.inference.video import VideoBackend

    monkeypatch.setattr(
        pq, "restricted_prequant_load_supported", lambda scheme = None, filename = None: True
    )
    monkeypatch.setattr(vdp, "hosted_denoiser_refused", lambda *a, **k: False)
    published = {
        "LTX-2-FP8.pt": 18_913_229_341,
        "LTX-2-FP8.safetensors": 18_913_466_264,
        "LTX-2-INT8.safetensors": 19_091_964_368,
    }
    api = _api(published)
    repo, files = VideoBackend._denoiser_prequant_hub_files(fam, "fp8", BASE, api)
    assert (repo, files) == (REPO, [("LTX-2-FP8.safetensors", 18_913_466_264)])
