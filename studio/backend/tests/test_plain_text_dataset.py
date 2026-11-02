# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One string column is raw training text: the preview and the formatter must agree on it."""

import sys
from pathlib import Path

import pytest
from datasets import Dataset

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from hub.utils import dataset_format  # noqa: E402
from utils.datasets import format_detection  # noqa: E402
from utils.datasets.dataset_utils import format_dataset  # noqa: E402


def _text_dataset(column: str = "text"):
    return Dataset.from_list(
        [
            {column: "## What happened\nWrote the fix."},
            {column: "## What happened\nReviewed the diff."},
        ]
    )


@pytest.mark.parametrize("column", ["text", "content", "body"], ids = ["text", "content", "body"])
@pytest.mark.parametrize("is_vlm", [False, True], ids = ["text_model", "vision_model"])
def test_single_string_column_needs_no_mapping(column, is_vlm):
    result = dataset_format.check_dataset_format(_text_dataset(column), is_vlm = is_vlm)

    assert result["requires_manual_mapping"] is False
    assert result["detected_format"] == "raw"
    assert result["suggested_mapping"] is None
    assert result["columns"] == [column]


def test_text_column_keeps_detection_flags_for_other_branches():
    result = dataset_format.check_dataset_format(_text_dataset())

    assert result["is_image"] is False
    assert result["is_audio"] is False
    assert result["detected_image_column"] is None


def test_media_named_text_column_is_not_routed_as_media():
    result = dataset_format.check_dataset_format(_text_dataset("speech"), is_vlm = True)

    assert result["detected_format"] == "raw"
    assert result["is_audio"] is False
    assert result["detected_audio_column"] is None
    assert result["is_image"] is False


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param([{"text": "a", "id": 1}], id = "text_plus_metadata"),
        pytest.param([{"text": "   "}], id = "blank_text"),
        pytest.param([{"text": 5}], id = "non_string_text"),
    ],
)
def test_other_shapes_still_require_mapping(rows):
    result = dataset_format.check_dataset_format(Dataset.from_list(rows))

    assert result["detected_format"] == "unknown"
    assert result["requires_manual_mapping"] is True


@pytest.mark.parametrize(
    "row", [{"image": "photos/sloth.jpg"}, {"audio": "clips/sloth.wav"}], ids = ["image", "audio"]
)
def test_media_path_column_is_not_raw_text(row):
    result = dataset_format.check_dataset_format(Dataset.from_list([row]))

    assert result["detected_format"] != "raw"
    assert result["requires_manual_mapping"] is True


def test_plain_text_dataset_helper_rejects_other_shapes():
    assert format_detection.is_plain_text_dataset(_text_dataset()) is True
    assert format_detection.is_plain_text_dataset(_text_dataset("content")) is True
    assert (
        format_detection.is_plain_text_dataset(Dataset.from_list([{"text": "a", "id": 1}])) is False
    )


@pytest.mark.parametrize("column", ["text", "content"], ids = ["text", "content"])
def test_auto_resolves_a_single_string_column_to_raw_text(column):
    info = format_dataset(_text_dataset(column), format_type = "auto")

    assert info["final_format"] == "raw_text"
    assert info["requires_manual_mapping"] is False
    assert info["dataset"].column_names == ["text"]
    assert info["dataset"]["text"] == [
        "## What happened\nWrote the fix.",
        "## What happened\nReviewed the diff.",
    ]


def test_renamed_column_is_reported():
    info = format_dataset(_text_dataset("content"), format_type = "auto")

    assert any("content" in warning for warning in info["warnings"])


def test_auto_leaves_a_mapped_dataset_alone():
    info = format_dataset(
        Dataset.from_list([{"text": "a", "id": 1}]),
        format_type = "auto",
        custom_format_mapping = {"text": "user"},
    )

    assert info["final_format"] != "raw_text"
