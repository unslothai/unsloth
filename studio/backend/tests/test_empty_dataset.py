# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Empty datasets report "needs mapping" instead of dying on a bare StopIteration."""

import sys
from pathlib import Path

from datasets import Dataset, IterableDataset

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from utils.datasets.dataset_utils import format_dataset  # noqa: E402
from utils.datasets.format_conversion import standardize_chat_format  # noqa: E402
from utils.datasets.format_detection import (  # noqa: E402
    detect_custom_format_heuristic,
    detect_multimodal_dataset,
)


def test_format_dataset_empty_dataset():
    info = format_dataset(Dataset.from_dict({"text": [], "label": []}))
    assert info["requires_manual_mapping"] is True
    assert info["final_format"] == "unknown"
    assert any("no rows" in warning for warning in info["warnings"])


def test_format_dataset_empty_iterable_dataset():
    info = format_dataset(IterableDataset.from_generator(lambda: iter([])))
    assert info["requires_manual_mapping"] is True


def test_detectors_return_safe_results_on_empty_datasets():
    empty = Dataset.from_dict({"conversations": []})
    multimodal = detect_multimodal_dataset(empty)
    assert multimodal["is_image"] is False
    assert multimodal["is_audio"] is False
    assert detect_custom_format_heuristic(empty) is None
    assert standardize_chat_format(empty) is empty
