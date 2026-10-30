# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


def test_video_families_are_flagged_through_the_response_model():
    """The images loader refuses video bases, so the Train panel hides Deploy to Create on this flag."""
    from core.training.diffusion_train_common import (
        TRAINABLE_VIDEO_FAMILIES,
        family_train_infos,
    )
    from models.training import DiffusionTrainableFamily

    infos = {info["name"]: info for info in family_train_infos()}
    assert {"ltx-2", "minimax-h3"} <= TRAINABLE_VIDEO_FAMILIES
    for name, info in infos.items():
        expected = name in TRAINABLE_VIDEO_FAMILIES
        assert info["is_video"] is expected, name
        assert DiffusionTrainableFamily(**info).is_video is expected, name
    assert infos["z-image"]["is_video"] is False
