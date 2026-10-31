# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from datasets import Dataset
from PIL import Image

from utils.datasets import format_conversion


def test_rows_with_a_null_caption_are_left_out_of_vision_training():
    ds = Dataset.from_dict(
        {
            "image": [Image.new("RGB", (4, 4)) for _ in range(3)],
            "text": ["a cat", None, "a dog"],
        }
    )

    out = format_conversion.convert_to_vlm_format(ds, instruction = "Describe.")

    answers = [sample["messages"][1]["content"][0]["text"] for sample in out]
    assert answers == ["a cat", "a dog"]
