# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from datasets import Dataset

from utils.datasets import (
    convert_chatml_to_alpaca,
    convert_sharegpt_with_images_to_vlm_format,
)


def test_falsy_scalar_content_is_trained_not_dropped():
    # A numeric answers column types content as int64; 0 is a valid answer, not a missing cell.
    dataset = Dataset.from_dict(
        {
            "messages": [
                [
                    {"role": "user", "content": 1},
                    {"role": "assistant", "content": 0},
                ],
                [
                    {"role": "user", "content": 2},
                    {"role": "assistant", "content": 3},
                ],
            ]
        }
    )

    stats = {}
    converted = convert_chatml_to_alpaca(dataset, num_proc = 1, stats = stats)

    assert converted.to_list() == [
        {"instruction": "1", "input": "", "output": "0"},
        {"instruction": "2", "input": "", "output": "3"},
    ]
    assert stats == {"conversations": 2, "dropped": 0}


def test_sharegpt_value_fallback_still_applies_for_missing_content():
    dataset = Dataset.from_dict(
        {
            "conversations": [
                [
                    {"from": "human", "value": 4},
                    {"from": "gpt", "value": 0},
                ]
            ]
        }
    )

    converted = convert_chatml_to_alpaca(dataset, num_proc = 1)

    assert converted.to_list() == [{"instruction": "4", "input": "", "output": "0"}]



@pytest.mark.parametrize("value, expected", [(False, "False"), (0, "0")])
def test_primary_scalar_wins_when_the_fallback_has_text(value, expected):
    dataset = Dataset.from_list([{"messages": [
        {"role": "user", "content": value, "value": "fallback"},
        {"role": "assistant", "content": value, "value": "fallback"},
    ]}])

    converted = convert_chatml_to_alpaca(dataset, num_proc = 1)
    assert converted.to_list() == [{"instruction": expected, "input": "", "output": expected}]


def test_an_empty_primary_string_still_falls_back():
    dataset = Dataset.from_list([{"messages": [
        {"role": "user", "content": "", "value": "question"},
        {"role": "assistant", "content": "", "value": "answer"},
    ]}])

    converted = convert_chatml_to_alpaca(dataset, num_proc = 1)
    assert converted.to_list() == [{"instruction": "question", "input": "", "output": "answer"}]


def test_sharegpt_image_conversion_keeps_falsy_turn_text():
    from PIL import Image

    image = Image.new("RGB", (4, 4))
    dataset = Dataset.from_dict(
        {
            "image": [image],
            "conversations": [
                [
                    {"from": "human", "value": 5},
                    {"from": "gpt", "value": 0},
                ]
            ],
        }
    )

    converted = convert_sharegpt_with_images_to_vlm_format(dataset)

    assert converted == [
        {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "5"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "0"}]},
            ]
        }
    ]
