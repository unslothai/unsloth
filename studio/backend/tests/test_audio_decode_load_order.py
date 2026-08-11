# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The training worker installs the soundfile decoder before it reads any row.

Deliberately dependency-free, unlike test_audio_dataset_decode.py: that module
importorskips soundfile and librosa, so on a host with neither it skips, and a
host with neither is exactly where this ordering is load-bearing.
"""

from __future__ import annotations

from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]


def _load_and_format_dataset_body() -> str:
    text = (_BACKEND / "core/training/trainer.py").read_text(encoding = "utf-8")
    return text[text.index("    def load_and_format_dataset(") :]


def test_the_shim_is_installed_before_the_first_row_is_read():
    # The format-check endpoint installs it in the API process. This worker is a
    # separate process and starts without it, so an Audio column decoding inside
    # load_dataset() raised `datasets`' own "please install 'torchcodec'" before
    # the audio branches further down could install anything.
    body = _load_and_format_dataset_body()
    assert body.index("ensure_audio_decoding()") < body.index("= load_dataset(")


def test_the_audio_branches_are_still_covered():
    # They call it themselves and report the FFmpeg-naming failure, which stays the
    # message a caller sees when neither backend works. This only has to precede them.
    body = _load_and_format_dataset_body()
    assert body.index("ensure_audio_decoding()") < body.index(
        "# ========== AUDIO MODELS: custom preprocessing =========="
    )


def test_the_import_is_module_level():
    # A local import inside the audio branch would leave the early call a NameError.
    text = (_BACKEND / "core/training/trainer.py").read_text(encoding = "utf-8")
    assert "\nfrom utils.datasets.audio_decode import ensure_audio_decoding\n" in text
