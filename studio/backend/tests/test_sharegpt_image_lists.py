# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Separate image lists follow ShareGPT placeholders in conversation order."""

from io import BytesIO
from pathlib import Path
import sys

import pytest
from datasets import Dataset
from PIL import Image

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from hub.utils.dataset_format import check_dataset_format  # noqa: E402
from utils.datasets import format_conversion  # noqa: E402
from utils.datasets.dataset_utils import format_and_template_dataset  # noqa: E402
from utils.datasets.format_conversion import convert_sharegpt_with_images_to_vlm_format  # noqa: E402


def _images():
    return [Image.new("RGB", (2, 2), color) for color in ("red", "blue")]


def _encoded(image):
    buffer = BytesIO()
    image.save(buffer, format = "PNG")
    return {"bytes": buffer.getvalue(), "path": None}


def _pixels(messages):
    return [
        part["image"].getpixel((0, 0))
        for message in messages
        for part in message["content"]
        if part["type"] == "image"
    ]


@pytest.mark.parametrize("chatml", [False, True], ids = ["sharegpt", "chatml"])
def test_detected_image_lists_convert_through_public_formatting(chatml):
    messages_column = "messages" if chatml else "conversations"
    role_key, text_key = ("role", "content") if chatml else ("from", "value")
    rows = [
        {
            "image": _images(),
            messages_column: [
                {
                    role_key: "user" if chatml else "human",
                    text_key: "Compare <image> with <image>.",
                },
                {role_key: "assistant" if chatml else "gpt", text_key: "Red became blue."},
            ],
        }
    ]

    detected = check_dataset_format(rows, is_vlm = True)
    result = format_and_template_dataset(rows, model_name = "test", tokenizer = None, is_vlm = True)

    assert detected["detected_format"] == "sharegpt_with_images"
    assert detected["requires_manual_mapping"] is False
    assert result["success"], result["errors"]
    assert result["final_format"] == "vlm_messages"
    messages = result["dataset"][0]["messages"]
    assert _pixels(messages) == [(255, 0, 0), (0, 0, 255)]
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[1]["content"] == [{"type": "text", "text": "Red became blue."}]


def test_encoded_image_lists_follow_markers_across_turns():
    rows = Dataset.from_list(
        [
            {
                "image": [_encoded(image) for image in _images()],
                "conversations": [
                    {"from": "human", "value": "Describe <image>."},
                    {"from": "gpt", "value": "It is red."},
                    {"from": "human", "value": "Compare it to <image>."},
                ],
            }
        ]
    )

    converted = convert_sharegpt_with_images_to_vlm_format(rows)

    assert _pixels(converted[0]["messages"]) == [(255, 0, 0), (0, 0, 255)]
    assert converted[0]["messages"][1]["content"] == [{"type": "text", "text": "It is red."}]


@pytest.mark.parametrize("image_column", ["image", "images"])
def test_one_element_image_list(image_column):
    rows = [{image_column: _images()[:1], "conversations": [{"from": "human", "value": "<image>"}]}]

    converted = convert_sharegpt_with_images_to_vlm_format(rows, image_column = image_column)

    assert _pixels(converted[0]["messages"]) == [(255, 0, 0)]


def test_scalar_image_can_still_fill_repeated_markers():
    rows = [
        {
            "image": _images()[0],
            "conversations": [
                {"from": "human", "value": "<image> and <image>"},
                {"from": "human", "value": "Again <image>"},
            ],
        }
    ]

    converted = convert_sharegpt_with_images_to_vlm_format(rows)

    assert _pixels(converted[0]["messages"]) == [(255, 0, 0)] * 3


@pytest.mark.parametrize(
    "images, text", [(0, "<image>"), (1, "<image><image>"), (2, "<image>"), (1, "No marker")]
)
def test_mismatched_image_lists_skip_only_the_bad_row(images, text):
    rows = [
        {"image": _images()[:images], "conversations": [{"from": "human", "value": text}]},
        {"image": _images()[:1], "conversations": [{"from": "human", "value": "Valid <image>"}]},
    ]

    converted = convert_sharegpt_with_images_to_vlm_format(rows)

    assert len(converted) == 1
    assert _pixels(converted[0]["messages"]) == [(255, 0, 0)]
    assert converted[0]["messages"][0]["content"][0] == {"type": "text", "text": "Valid"}


def test_empty_and_missing_images_do_not_prevent_later_valid_rows():
    conversation = [{"from": "human", "value": "<image>"}]
    rows = [
        {"image": [], "conversations": conversation},
        {"conversations": conversation},
        {"image": _images()[0], "conversations": conversation},
    ]

    converted = convert_sharegpt_with_images_to_vlm_format(rows)

    assert len(converted) == 1
    assert _pixels(converted[0]["messages"]) == [(255, 0, 0)]


def test_url_image_lists_use_the_image_transport(monkeypatch):
    urls = ["https://example.com/red.png", "https://example.com/blue.png"]
    images = dict(zip(urls, _images()))
    fetched = []

    def fetch(url):
        fetched.append(url)
        return images[url]

    monkeypatch.setattr(format_conversion, "_fetch_row_image", fetch)
    rows = Dataset.from_list(
        [{"image": urls, "conversations": [{"from": "human", "value": "<image> then <image>"}]}]
    )

    converted = convert_sharegpt_with_images_to_vlm_format(rows)

    assert _pixels(converted[0]["messages"]) == [(255, 0, 0), (0, 0, 255)]
    assert fetched == urls


@pytest.mark.parametrize(
    "bad_first_row, scalar_image",
    [
        ("none", False),
        ("empty", False),
        ("mismatch", False),
        ("empty", True),
        ("null", False),
        ("null_item", False),
    ],
)
def test_repo_filename_lists_resolve_each_image(monkeypatch, tmp_path, bad_first_row, scalar_image):
    import huggingface_hub

    filenames = ["images/red.png", "images/blue.png"]
    for filename, image in zip(filenames, _images()):
        image.save(tmp_path / Path(filename).name)
    listed = []
    downloaded = []

    def list_files(self, repo, **kwargs):
        listed.append((repo, kwargs["repo_type"]))
        return filenames

    def download(repo, filename, **kwargs):
        downloaded.append((repo, filename, kwargs["repo_type"]))
        return str(tmp_path / Path(filename).name)

    monkeypatch.setattr(huggingface_hub.HfApi, "list_repo_files", list_files)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    examples = [
        {
            "image": "red.png" if scalar_image else ["red.png", "images/blue.png"],
            "conversations": [
                {"from": "human", "value": "<image>" if scalar_image else "<image><image>"}
            ],
        }
    ]
    if bad_first_row != "none":
        examples.insert(
            0,
            {
                "image": {
                    "empty": [],
                    "mismatch": ["missing.png"],
                    "null": None,
                    "null_item": [None],
                }[bad_first_row],
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image><image>" if bad_first_row == "mismatch" else "<image>",
                    }
                ],
            },
        )
    rows = examples if scalar_image else Dataset.from_list(examples)

    converted = convert_sharegpt_with_images_to_vlm_format(rows, dataset_name = "org/vision")

    assert len(converted) == 1
    expected_colors = [(255, 0, 0)] if scalar_image else [(255, 0, 0), (0, 0, 255)]
    assert _pixels(converted[0]["messages"]) == expected_colors
    assert listed == [("org/vision", "dataset")]
    expected_files = filenames[:1] if scalar_image else filenames
    assert downloaded == [("org/vision", filename, "dataset") for filename in expected_files]
