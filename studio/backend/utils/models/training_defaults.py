# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


def model_training_recommendations(
    config: dict, *, is_vision: bool, is_audio: bool, is_embedding: bool, is_decision: bool
) -> dict:
    # Model-picker recommendations, not user opt-outs; data/objective restrictions apply later.
    chat = not (is_embedding or is_decision or (is_audio and not is_vision))
    return {
        **config,
        "training": {
            **(config.get("training") or {}),
            "packing": chat,
            "train_on_completions": chat,
        },
    }
