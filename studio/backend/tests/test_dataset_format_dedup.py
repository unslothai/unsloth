# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The hub preview and the trainer run one dataset-format detector, not drifting copies."""

import sys
from pathlib import Path

from datasets import Dataset

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from hub.utils import dataset_format  # noqa: E402
from utils.datasets import format_detection  # noqa: E402


def test_hub_detectors_are_the_canonical_ones():
    for name in (
        "detect_custom_format_heuristic",
        "detect_multimodal_dataset",
        "detect_vlm_dataset_structure",
    ):
        assert getattr(dataset_format, name) is getattr(format_detection, name)


def test_first_turn_gate_is_uniform(tmp_path):
    dataset = Dataset.from_list(
        [
            {
                "conversations": [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": "<image> what is this"},
                ],
                "image": str(tmp_path / "a.png"),
            }
        ]
    )
    hub_result = dataset_format.detect_vlm_dataset_structure(dataset)
    utils_result = format_detection.detect_vlm_dataset_structure(dataset)
    assert hub_result == utils_result


def test_resolvable_path_beats_metadata_named_column(tmp_path):
    photo = tmp_path / "photo.png"
    photo.write_bytes(b"")
    dataset = Dataset.from_list(
        [
            {
                "image_filename": "photo.png",
                "photo": str(photo),
                "caption": "a cat",
            }
        ]
    )
    hub_result = dataset_format.detect_vlm_dataset_structure(dataset)
    utils_result = format_detection.detect_vlm_dataset_structure(dataset)
    assert hub_result == utils_result
    assert utils_result["image_column"] == "photo"
