# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every image and video request reads seed -1 as random (A1111, sd.cpp) and keeps the JSON-safe bound."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.inference.generate_image_tool import _build_request
from models.inference import (
    DiffusionGenerateRequest,
    ImageGenerationRequest,
    VideoGenerateRequest,
    VideoJobCreateRequest,
)

MEDIA_REQUESTS = (
    DiffusionGenerateRequest,
    ImageGenerationRequest,
    VideoJobCreateRequest,
    VideoGenerateRequest,
)


@pytest.mark.parametrize("model", MEDIA_REQUESTS)
@pytest.mark.parametrize(
    "sent, kept",
    [(-1, None), ("-1", None), (None, None), (0, 0), (2**53 - 1, 2**53 - 1)],
)
def test_seed_minus_one_means_random(model, sent, kept):
    assert model(prompt = "p", seed = sent).seed == kept


@pytest.mark.parametrize("model", MEDIA_REQUESTS)
@pytest.mark.parametrize("seed", [-2, 2**53])
def test_seed_outside_the_range_is_rejected(model, seed):
    with pytest.raises(ValidationError):
        model(prompt = "p", seed = seed)


def test_generate_image_tool_reads_minus_one_as_random():
    assert _build_request({"seed": -1}, "p", {}).seed is None


@pytest.mark.parametrize("sent, kept", [("7", 7), (7.0, 7)])
def test_generate_image_tool_passes_the_raw_seed_to_the_validator(sent, kept):
    assert _build_request({"seed": sent}, "p", {}).seed == kept


def test_generate_image_tool_rejects_a_fractional_seed_instead_of_rounding_it_to_random():
    with pytest.raises(ValidationError):
        _build_request({"seed": -1.5}, "p", {})
