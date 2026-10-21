# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import io
import sys
from pathlib import Path

import pytest
from datasets import Dataset, IterableDataset
from PIL import Image

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from hub.utils import dataset_format  # noqa: E402
from utils.datasets.format_conversion import convert_llava_to_vlm_format  # noqa: E402
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


def _llava_row_without_index():
    row = _llava_row()
    for message in row["messages"]:
        for part in message["content"]:
            part.pop("index", None)
    return row


def _llava_rows():
    yield _llava_row_without_index()


def _image_part_row():
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.new("RGB", (8, 8), "red"), "text": None},
                    {"type": "text", "image": None, "text": "Describe this."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "image": None, "text": "A dog."}],
            },
        ]
    }


def _image_part_row_after_system_message():
    row = _image_part_row()
    row["messages"].insert(
        0,
        {
            "role": "system",
            "content": [{"type": "text", "image": None, "text": "Answer accurately."}],
        },
    )
    return row


def _text_parts_row():
    return {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
        ]
    }


def _encoded_image_row():
    encoded = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(encoded, format = "PNG")
    return {"media": [{"bytes": encoded.getvalue(), "path": None}]}


def _encoded_audio_row():
    return {"clips": [{"bytes": b"RIFF\x00\x00\x00\x00WAVEfmt ", "path": None}]}


@pytest.mark.parametrize(
    "row, expected_vlm_format",
    [
        pytest.param(_llava_row(), "vlm_messages_llava", id = "images-list"),
        pytest.param(
            _llava_row_without_index(),
            "vlm_messages_llava",
            id = "images-list-without-index",
        ),
        pytest.param(_image_part_row(), "vlm_messages", id = "image-parts"),
        pytest.param(
            _image_part_row_after_system_message(),
            "vlm_messages",
            id = "image-parts-after-system-message",
        ),
    ],
)
def test_images_in_a_list_or_in_messages_are_detected(row, expected_vlm_format):
    dataset = [row] if expected_vlm_format == "vlm_messages" else Dataset.from_list([row])

    assert dataset_format.check_dataset_format(dataset, is_vlm = False)["is_image"] is True
    vlm = dataset_format.check_dataset_format(dataset, is_vlm = True)
    assert vlm["is_image"] is True
    assert vlm["detected_format"] == expected_vlm_format
    assert vlm["requires_manual_mapping"] is False
    assert format_detection.detect_multimodal_dataset(dataset)["is_image"] is True


def test_unindexed_llava_images_are_consumed_in_order():
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image"}, {"type": "image"}],
                    }
                ],
                "images": [
                    Image.new("RGB", (8, 8), "red"),
                    Image.new("RGB", (8, 8), "blue"),
                ],
            }
        ]
    )

    converted = convert_llava_to_vlm_format(dataset)
    images = [part["image"] for part in converted[0]["messages"][0]["content"]]

    assert [image.getpixel((0, 0)) for image in images] == [(255, 0, 0), (0, 0, 255)]


def test_unindexed_llava_images_skip_explicit_indices():
    dataset = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image", "index": 0}, {"type": "image"}],
                }
            ],
            "images": [
                Image.new("RGB", (8, 8), "red"),
                Image.new("RGB", (8, 8), "blue"),
            ],
        }
    ]

    converted = convert_llava_to_vlm_format(dataset)
    images = [part["image"] for part in converted[0]["messages"][0]["content"]]

    assert [image.getpixel((0, 0)) for image in images] == [(255, 0, 0), (0, 0, 255)]


def test_llava_conversion_rejects_missing_unindexed_image():
    dataset = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "image"}],
                }
            ],
            "images": [Image.new("RGB", (8, 8), "red")],
        }
    ]

    with pytest.raises(ValueError, match = "image index 1"):
        convert_llava_to_vlm_format(dataset)


def test_llava_conversion_rejects_undecoded_image_dictionary():
    row = _llava_row_without_index()
    row["images"] = [_encoded_image_row()["media"][0]]

    with pytest.raises(ValueError, match = "Unsupported Llava image value"):
        convert_llava_to_vlm_format([row])


def test_llava_conversion_preserves_string_turns():
    row = _llava_row_without_index()
    row["messages"].insert(0, {"role": "system", "content": "Answer accurately."})
    dataset = [row]

    assert dataset_format.detect_vlm_dataset_structure(dataset)["format"] == "vlm_messages_llava"
    assert format_detection.detect_vlm_dataset_structure(dataset)["format"] == "vlm_messages_llava"

    converted = convert_llava_to_vlm_format(dataset)

    assert converted[0]["messages"][0] == {
        "role": "system",
        "content": [{"type": "text", "text": "Answer accurately."}],
    }


def test_llava_conversion_stays_lazy_for_streaming_dataset():
    dataset = IterableDataset.from_generator(_llava_rows)

    converted = convert_llava_to_vlm_format(dataset)

    assert isinstance(converted, IterableDataset)
    first = next(iter(converted))
    assert first["messages"][0]["content"][0]["image"].getpixel((0, 0)) == (255, 0, 0)


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
        pytest.param(
            {
                "messages": [
                    {"role": "user", "content": "Which files changed?"},
                    {"role": "assistant", "content": "The entry point and the logo."},
                ],
                "files": ["src/main.py", "docs/logo.png"],
            },
            id = "file-list",
        ),
    ],
)
def test_text_only_datasets_stay_text(row):
    dataset = Dataset.from_list([row])

    assert dataset_format.check_dataset_format(dataset, is_vlm = False)["is_image"] is False
    assert format_detection.detect_multimodal_dataset(dataset)["is_image"] is False


@pytest.mark.parametrize("detector", [dataset_format, format_detection])
def test_image_list_detection_requires_values_the_converter_can_open(detector, tmp_path):
    png_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "red").save(png_path)
    svg_path = tmp_path / "image.svg"
    svg_path.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')

    encoded_image_row = _encoded_image_row()
    image_dataset = Dataset.from_list([encoded_image_row])
    image_bytes_dataset = Dataset.from_list([{"media": [encoded_image_row["media"][0]["bytes"]]}])
    audio_dataset = Dataset.from_list([_encoded_audio_row()])
    local_image_dataset = Dataset.from_list([{"media": [str(png_path)]}])
    remote_image_dataset = Dataset.from_list([{"media": ["https://example.com/image.png"]}])
    svg_image_dataset = Dataset.from_list([{"media": [str(svg_path)]}])

    assert detector.detect_multimodal_dataset(image_dataset)["is_image"] is False
    assert detector.detect_multimodal_dataset(image_bytes_dataset)["is_image"] is False
    assert detector.detect_multimodal_dataset(audio_dataset)["is_image"] is False
    assert detector.detect_multimodal_dataset(local_image_dataset)["is_image"] is True
    assert detector.detect_multimodal_dataset(remote_image_dataset)["is_image"] is False
    assert detector.detect_multimodal_dataset(svg_image_dataset)["is_image"] is False


@pytest.mark.parametrize("detector", [dataset_format, format_detection])
def test_llava_placeholders_require_convertible_images(detector):
    row = _llava_row_without_index()
    row["images"] = [_encoded_image_row()["media"][0]]
    dataset = Dataset.from_list([row])

    result = detector.detect_vlm_dataset_structure(dataset)

    assert result["format"] != "vlm_messages_llava"


@pytest.mark.parametrize("detector", [dataset_format, format_detection])
def test_embedded_images_require_convertible_values(detector):
    row = _image_part_row()
    row["messages"][0]["content"][0]["image"] = _encoded_image_row()["media"][0]
    dataset = Dataset.from_list([row])

    result = detector.detect_vlm_dataset_structure(dataset)

    assert result["format"] != "vlm_messages"


@pytest.mark.parametrize("detector", [dataset_format, format_detection])
def test_embedded_image_paths_require_conversion(detector):
    row = _image_part_row()
    row["messages"][0]["content"][0]["image"] = "/data/photos/a.jpg"

    result = detector.detect_vlm_dataset_structure([row])

    assert result["format"] != "vlm_messages"


@pytest.mark.parametrize("detector", [dataset_format, format_detection])
def test_embedded_images_do_not_hide_top_level_placeholders(detector):
    row = _image_part_row()
    row["messages"][0]["content"].append({"type": "image", "image": None, "text": None})
    row["images"] = [Image.new("RGB", (8, 8), "blue")]

    result = detector.detect_vlm_dataset_structure([row])

    assert result["format"] not in {"vlm_messages", "vlm_messages_llava"}
