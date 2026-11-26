# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from datasets import Dataset
from PIL import Image

from utils.datasets.dataset_utils import format_and_template_dataset

PINNED = "0123456789abcdef0123456789abcdef01234567"
RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)


@pytest.fixture
def hub(monkeypatch, tmp_path):
    images = {}
    for i, (key, color) in enumerate(
        (
            ((PINNED, "images/a.png"), RED),
            ((PINNED, "images/b.png"), GREEN),
            ((None, "images/a.png"), BLUE),
            ((None, "images/b.png"), BLUE),
        )
    ):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (4, 4), color).save(path)
        images[key] = str(path)
    seen = []

    def _list(
        self,
        repo_id,
        *args,
        revision = None,
        **kwargs,
    ):
        seen.append(("list", revision))
        return ["README.md", "images/a.png", "images/b.png"]

    def _download(
        repo_id,
        filename,
        *args,
        revision = None,
        **kwargs,
    ):
        seen.append(("download", revision))
        return images[revision, filename]

    monkeypatch.setattr("huggingface_hub.HfApi.list_repo_files", _list)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _download)
    return seen


def _images(result):
    assert result["success"], result["errors"]
    return [
        part["image"]
        for row in result["dataset"]
        for message in row["messages"]
        for part in message["content"]
        if part.get("type") == "image"
    ]


def _format(dataset, **kwargs):
    kwargs = {"dataset_name": "org/ds", "dataset_revision": PINNED, **kwargs}
    return format_and_template_dataset(
        dataset,
        model_name = "org/vlm",
        tokenizer = None,
        is_vlm = True,
        vlm_instruction = "Describe.",
        **kwargs,
    )


def test_image_text_rows_read_images_from_the_pinned_revision(hub):
    ds = Dataset.from_dict({"image": ["a.png", "images/b.png"], "text": ["a", "b"]})

    images = _images(_format(ds))

    assert [image.getpixel((0, 0)) for image in images] == [RED, GREEN]
    assert set(hub) == {("list", PINNED), ("download", PINNED)}


def test_mapped_image_text_columns_read_images_from_the_pinned_revision(hub):
    ds = Dataset.from_dict({"picture": ["a.png"], "caption": ["a"]})

    images = _images(_format(ds, custom_format_mapping = {"picture": "image", "caption": "text"}))

    assert [image.getpixel((0, 0)) for image in images] == [RED]
    assert set(hub) == {("list", PINNED), ("download", PINNED)}


@pytest.mark.parametrize(
    "repo, build_name, expected",
    [("org/ds", PINNED, PINNED), ("org/other", PINNED, None), ("org/ds", "a1b2c3d4e5f60789", None)],
)
def test_processed_cache_rows_read_images_from_their_build_commit(
    hub, monkeypatch, tmp_path, repo, build_name, expected
):
    build = tmp_path / "datasets" / "org___ds" / "default" / "0.0.0" / build_name
    build.mkdir(parents = True)
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "datasets"))
    Dataset.from_dict({"image": ["a.png"], "text": ["a"]}).save_to_disk(str(tmp_path / "saved"))
    arrow = next((tmp_path / "saved").glob("*.arrow"))
    arrow.rename(build / "ds-train.arrow")
    ds = Dataset.from_file(str(build / "ds-train.arrow"))

    images = _images(_format(ds, dataset_name = repo, dataset_revision = None))

    assert [image.getpixel((0, 0)) for image in images] == [RED if expected else BLUE]
    assert set(hub) == {("list", expected), ("download", expected)}


def test_sharegpt_rows_read_images_from_the_pinned_revision(hub):
    ds = Dataset.from_dict(
        {
            "image": ["images/a.png"],
            "conversations": [
                [
                    {"from": "human", "value": "<image> What is this?"},
                    {"from": "gpt", "value": "A square."},
                ]
            ],
        }
    )

    images = _images(_format(ds))

    assert [image.getpixel((0, 0)) for image in images] == [RED]
    assert set(hub) == {("list", PINNED), ("download", PINNED)}
