# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The trainer trains on the audio and text columns the dataset check accepted."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402
from datasets import Dataset  # noqa: E402

from hub.utils.dataset_format import check_dataset_format  # noqa: E402
from utils.datasets.audio_rows import resolve_audio_columns  # noqa: E402


_AUDIO = {"array": [0.0] * 160, "sampling_rate": 16000}


@pytest.mark.parametrize(
    "row, expected",
    [
        ({"audio": _AUDIO, "normalized_text": "hi"}, ("audio", "normalized_text", None)),
        ({"audio": _AUDIO, "prompt": "hi"}, ("audio", "prompt", None)),
        ({"audio": _AUDIO, "label": "hi"}, ("audio", "label", None)),
        ({"wav": _AUDIO, "text": "hi"}, ("wav", "text", None)),
        ({"clip": _AUDIO, "text": "hi", "speaker": "a"}, ("clip", "text", "speaker")),
        ({"audio": _AUDIO, "text": "hi", "speaker": "a"}, ("audio", "text", "speaker")),
    ],
)
def test_resolves_the_columns_the_check_accepted(row, expected):
    dataset = Dataset.from_list([row])
    check = check_dataset_format(dataset, is_vlm = False)
    assert check["requires_manual_mapping"] is False

    resolved = resolve_audio_columns(dataset, None)

    assert (resolved["audio_col"], resolved["text_col"], resolved["speaker_col"]) == expected
    assert resolved["audio_col"] == check["detected_audio_column"]


def test_exact_names_and_explicit_mapping_are_unchanged():
    dataset = Dataset.from_list(
        [{"audio": _AUDIO, "text": "hi", "prompt": "p", "speaker": "a", "source": "0"}]
    )

    assert resolve_audio_columns(dataset, None) == {
        "audio_col": "audio",
        "text_col": "text",
        "speaker_col": "source",
    }
    assert resolve_audio_columns(dataset, {"audio": "audio", "prompt": "text"}) == {
        "audio_col": "audio",
        "text_col": "prompt",
        "speaker_col": None,
    }


def test_text_only_dataset_still_resolves_nothing():
    dataset = Dataset.from_list([{"instruction": "a", "output": "b"}])

    assert resolve_audio_columns(dataset, None) == {
        "audio_col": None,
        "text_col": None,
        "speaker_col": None,
    }
