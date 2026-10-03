# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which Audio page workflows a model or a clip belongs to.

Mirrors ``studio/frontend/src/features/audio/workflows.ts``. This module reads the audio.cpp
constants; ``audio_cpp_models`` never imports it.
"""

from __future__ import annotations

from typing import Optional

from core.inference.audio_cpp_models import AUDIO_CPP_MUSIC_AUDIO_TYPE, HUB_TASKS

AUDIO_WORKFLOW_IDS = ("speak", "clone", "convert", "music", "transcribe")

# Generation audio types that make music; every other generation audio type speaks.
MUSIC_AUDIO_TYPES = frozenset(("minimax_music3", AUDIO_CPP_MUSIC_AUDIO_TYPE))

# Audio types of a loaded model that transcribes rather than generates.
_TRANSCRIBE_AUDIO_TYPES = frozenset(("whisper",))


def workflow_for_audio_type(audio_type: Optional[str]) -> str:
    """The generation workflow (``speak`` or ``music``) that a model or clip of this type serves."""
    return "music" if audio_type in MUSIC_AUDIO_TYPES else "speak"


def status_audio_workflows(is_audio: bool, audio_type: Optional[str]) -> list[str]:
    """Workflows the loaded model serves; empty for a model that is not an audio model."""
    if not is_audio:
        return []
    if audio_type in _TRANSCRIBE_AUDIO_TYPES:
        return ["transcribe"]
    return [workflow_for_audio_type(audio_type)]


def inventory_audio_workflows(
    task: Optional[str], audio_type: Optional[str]
) -> Optional[list[str]]:
    """Workflows a cached or local model row serves, from its pipeline task first.

    audio.cpp music rows carry ``text-to-audio`` with no audio_type, so the task decides before
    the audio_type does. None for a row that is not an audio model.
    """
    if task == HUB_TASKS["music"]:
        return ["music"]
    if task == HUB_TASKS["asr"]:
        return ["transcribe"]
    if task == HUB_TASKS["tts"]:
        return [workflow_for_audio_type(audio_type)]
    return None
