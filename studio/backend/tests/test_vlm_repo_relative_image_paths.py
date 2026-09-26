# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import huggingface_hub
from datasets import Dataset
from PIL import Image

from utils.datasets import format_conversion


def _fake_hub(monkeypatch, tmp_path, repo_files):
    local = tmp_path / "a.png"
    Image.new("RGB", (4, 4)).save(local)
    fetched = []

    def _download(repo_id, filename, **kwargs):
        fetched.append(filename)
        return str(local)

    monkeypatch.setattr(
        huggingface_hub.HfApi, "list_repo_files", lambda self, *a, **k: list(repo_files)
    )
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _download)
    # The all-failed path asks the helper model for a friendlier message; keep it offline.
    from utils.datasets import llm_assist

    monkeypatch.setattr(llm_assist, "llm_generate_dataset_warning", lambda *a, **k: None)
    return fetched


def test_simple_image_text_resolves_a_repo_relative_image_path(monkeypatch, tmp_path):
    # convert_sharegpt_with_images_to_vlm_format keys its lookup by the full relative path too;
    # the simple {image, text} converter has to resolve the same "images/a.png" value.
    fetched = _fake_hub(monkeypatch, tmp_path, ["README.md", "images/a.png"])
    ds = Dataset.from_dict({"image": ["images/a.png"], "text": ["a cat"]})

    out = format_conversion.convert_to_vlm_format(
        ds, instruction = "Describe.", dataset_name = "org/ds"
    )

    assert fetched == ["images/a.png"]
    assert len(out) == 1
    image = next(
        part["image"] for part in out[0]["messages"][0]["content"] if part.get("type") == "image"
    )
    assert image.size == (4, 4)


def test_simple_image_text_still_resolves_a_bare_filename(monkeypatch, tmp_path):
    fetched = _fake_hub(monkeypatch, tmp_path, ["images/a.png"])
    ds = Dataset.from_dict({"image": ["a.png"], "text": ["a cat"]})

    out = format_conversion.convert_to_vlm_format(
        ds, instruction = "Describe.", dataset_name = "org/ds"
    )

    assert fetched == ["images/a.png"]
    assert len(out) == 1
