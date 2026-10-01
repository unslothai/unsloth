# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
from pathlib import Path

import pytest
from datasets import Dataset
from PIL import Image

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from hub.utils import dataset_format  # noqa: E402
from utils.datasets import format_detection  # noqa: E402


def _llava_row():
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "index": 0, "text": None},
                    {"type": "text", "index": None, "text": "What is in the picture?"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "index": None, "text": "A red square."}],
            },
        ],
        "images": [Image.new("RGB", (8, 8), "red")],
    }


def _image_part_row():
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "/data/photos/a.jpg", "text": None},
                    {"type": "text", "image": None, "text": "Describe this."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "image": None, "text": "A dog."}],
            },
        ]
    }


def _text_parts_row():
    return {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
        ]
    }


@pytest.mark.parametrize(
    "row, expected_vlm_format",
    [
        pytest.param(_llava_row(), "vlm_messages_llava", id = "images-list"),
        pytest.param(_image_part_row(), "vlm_messages", id = "image-parts"),
    ],
)
def test_images_in_a_list_or_in_messages_are_detected(row, expected_vlm_format):
    dataset = Dataset.from_list([row])

    assert dataset_format.check_dataset_format(dataset, is_vlm = False)["is_image"] is True
    vlm = dataset_format.check_dataset_format(dataset, is_vlm = True)
    assert vlm["is_image"] is True
    assert vlm["detected_format"] == expected_vlm_format
    assert vlm["requires_manual_mapping"] is False
    assert format_detection.detect_multimodal_dataset(dataset)["is_image"] is True


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(_text_parts_row(), id = "text-parts"),
        pytest.param(
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Yo"},
                ]
            },
            id = "chatml",
        ),
        pytest.param({"tags": ["news", "sports"], "text": "Final score 2-1."}, id = "string-list"),
    ],
)
def test_text_only_datasets_stay_text(row):
    dataset = Dataset.from_list([row])

    assert dataset_format.check_dataset_format(dataset, is_vlm = False)["is_image"] is False
    assert format_detection.detect_multimodal_dataset(dataset)["is_image"] is False
