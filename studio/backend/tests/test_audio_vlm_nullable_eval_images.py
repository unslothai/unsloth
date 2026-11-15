# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from .test_trainer_audio_vlm_image_and_audio import (
    _NoAudioCast,
    _dataset,
    audio_vlm_trainer,  # noqa: F401
    datasets,
)


@pytest.mark.parametrize("image_col", ["image", "photo"])
@pytest.mark.parametrize("empty_value", [None, []], ids = ["null", "empty_list"])
def test_all_null_eval_images_keep_the_shared_image_column(
    audio_vlm_trainer, image_col, empty_value
):
    mapping = (
        {"audio": "audio", "text": "text", image_col: "image"} if image_col == "photo" else None
    )
    audio_vlm_trainer._format_audio_vlm_dataset(_dataset([image_col]), mapping)
    evaluation = _NoAudioCast(
        datasets.Dataset.from_dict(
            {
                "audio": [{"array": [0.0] * 160, "sampling_rate": 16000}] * 2,
                "text": ["one", "two"],
                image_col: [empty_value, empty_value],
            }
        )
    )

    formatted = audio_vlm_trainer._format_audio_vlm_eval_split(evaluation, mapping)

    assert formatted is not None
    assert audio_vlm_trainer._audio_vlm_image_col == image_col
    assert audio_vlm_trainer._audio_vlm_audio_col == "audio"
    assert list(formatted[image_col]) == [empty_value, empty_value]
    for row in formatted:
        user = next(message for message in row["messages"] if message["role"] == "user")
        assert [part["type"] for part in user["content"]] == ["audio", "text"]
    if mapping is not None:
        assert mapping == {"audio": "audio", "text": "text", image_col: "image"}


def test_a_different_eval_image_column_still_disables_evaluation(audio_vlm_trainer, monkeypatch):
    audio_vlm_trainer._format_audio_vlm_dataset(_dataset(["image"]))
    warnings = []
    monkeypatch.setattr(audio_vlm_trainer, "_record_warning", warnings.append)

    formatted = audio_vlm_trainer._format_audio_vlm_eval_split(_dataset(["photo"]), None)

    assert formatted is None
    assert audio_vlm_trainer._audio_vlm_image_col == "image"
    assert audio_vlm_trainer._audio_vlm_audio_col == "audio"
    assert any("image column" in warning for warning in warnings)


def test_non_image_eval_values_do_not_gain_an_image_mapping(audio_vlm_trainer, monkeypatch):
    audio_vlm_trainer._format_audio_vlm_dataset(_dataset(["image"]))
    evaluation = _NoAudioCast(
        datasets.Dataset.from_dict(
            {
                "audio": [{"array": [0.0] * 160, "sampling_rate": 16000}] * 2,
                "text": ["one", "two"],
                "image": ["metadata", "metadata"],
            }
        )
    )
    warnings = []
    monkeypatch.setattr(audio_vlm_trainer, "_record_warning", warnings.append)

    assert audio_vlm_trainer._format_audio_vlm_eval_split(evaluation, None) is None
    assert any("image column" in warning for warning in warnings)
    assert not any("could not be prepared" in warning for warning in warnings)
