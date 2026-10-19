# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""unsloth/flux-3-image-klein[-base] are retired full copies: a load naming one becomes the pick of the base it copies,
which Studio fetches through the -FP8 repo."""

import pytest

from core.inference import diffusion_families as df
from models.inference import DiffusionLoadRequest

_COPIES = {
    "unsloth/flux-3-image-klein": ("black-forest-labs/flux-3-image-klein", "unsloth/flux-3-image-klein-FP8"),
    "unsloth/flux-3-image-klein-base": (
        "black-forest-labs/flux-3-image-klein-base",
        "unsloth/flux-3-image-klein-base-FP8",
    ),
}


@pytest.mark.parametrize("copy", sorted(_COPIES))
def test_each_retired_copy_points_at_a_mirrored_base_of_the_same_family(copy):
    base, mirror = _COPIES[copy]
    assert df.retired_full_copy_base(copy) == base
    assert df.retired_full_copy_base(copy.upper()) == base
    assert df.mirror_repo(base) == mirror
    assert df.detect_family(base) is df.detect_family(copy)


@pytest.mark.parametrize("copy", sorted(_COPIES))
def test_a_pipeline_pick_of_a_retired_copy_loads_its_base(copy):
    base, _mirror = _COPIES[copy]
    request = DiffusionLoadRequest(model_path = copy, model_kind = "pipeline", base_repo = copy)
    assert request.model_path == base
    assert request.base_repo == base
    assert request.model_kind == "pipeline"
    assert DiffusionLoadRequest(model_path = f" {copy} ").model_path == base


def test_a_single_file_pick_of_the_root_bf16_transformer_is_the_pipeline_load():
    request = DiffusionLoadRequest(
        model_path = "unsloth/flux-3-image-klein",
        gguf_filename = "model.safetensors",
        model_kind = "single_file",
    )
    assert (request.model_path, request.gguf_filename, request.model_kind) == (
        "black-forest-labs/flux-3-image-klein",
        None,
        "pipeline",
    )


def test_other_picks_are_left_as_named():
    # Anything that is not the base pipeline or its root transformer keeps naming the copy (it still exists).
    kept = DiffusionLoadRequest(
        model_path = "unsloth/flux-3-image-klein",
        gguf_filename = "model.safetensors",
        model_kind = "single_file",
        vae_file = "unsloth/flux-3-image-klein-FP8/vae/diffusion_pytorch_model.safetensors",
    )
    assert kept.model_path == "unsloth/flux-3-image-klein"
    assert kept.gguf_filename == "model.safetensors"
    for repo in ("unsloth/flux-3-image-klein-FP8", "black-forest-labs/flux-3-image-klein", "unsloth/FLUX.1-dev"):
        assert DiffusionLoadRequest(model_path = repo).model_path == repo
