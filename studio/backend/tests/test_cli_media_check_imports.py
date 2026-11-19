# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`unsloth train` sniffs media cells with the backend detectors; a lost import reads every dataset as text."""

import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND_ROOT.parents[1]
for root in (_BACKEND_ROOT, _REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

pytest.importorskip("typer")
from datasets import Dataset  # noqa: E402

from unsloth_cli.commands.train import _media_columns_hold_media  # noqa: E402
from utils.datasets.format_detection import detect_multimodal_dataset  # noqa: E402


def test_cli_media_check_finds_image_and_audio_cells():
    dataset = Dataset.from_dict(
        {
            "image": [{"bytes": b"\x89PNG\r\n\x1a\n" + bytes(8), "path": "a.png"}],
            "audio": [{"bytes": b"RIFF\0\0\0\0WAVEfmt ", "path": "a.wav"}],
            "text": ["hi"],
        }
    )
    assert _media_columns_hold_media(dataset, detect_multimodal_dataset(dataset)) == (True, True)
