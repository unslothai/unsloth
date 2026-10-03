# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which Audio page workflows an inventory row or a gallery clip belongs to."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.inference import audio_workflows as aw
from core.inference.audio_cpp_models import AUDIO_CPP_MUSIC_AUDIO_TYPE, AUDIO_CPP_TTS_AUDIO_TYPE
from hub.schemas.inventory import CachedGgufRepo, CachedModelRepo, LocalModelInfo


def test_music_audio_types_match_the_frontend_set():
    source = (
        Path(__file__).resolve().parents[2] / "frontend/src/features/audio/workflows.ts"
    ).read_text(encoding = "utf-8")
    block = re.search(r"MUSIC_AUDIO_TYPES[^=]*=\s*new Set\(\[(.*?)\]\)", source, re.S).group(1)
    names = {"AUDIO_CPP_MUSIC_AUDIO_TYPE": AUDIO_CPP_MUSIC_AUDIO_TYPE}
    frontend = {
        names.get(item.strip(), item.strip().strip('"'))
        for item in block.split(",")
        if item.strip()
    }
    assert frontend == set(aw.MUSIC_AUDIO_TYPES) == {"minimax_music3", "audiocpp_music"}


@pytest.mark.parametrize(
    "task, audio_type, workflows",
    [
        ("text-to-audio", None, ["music"]),
        ("automatic-speech-recognition", None, ["transcribe"]),
        ("text-to-speech", AUDIO_CPP_TTS_AUDIO_TYPE, ["speak"]),
        ("text-to-speech", "minimax_music3", ["music"]),
        ("text-to-speech", AUDIO_CPP_MUSIC_AUDIO_TYPE, ["music"]),
        ("text-to-speech", None, ["speak"]),
        ("text-generation", None, None),
        (None, None, None),
    ],
)
def test_inventory_rows_derive_workflows_from_the_task_first(task, audio_type, workflows):
    assert aw.inventory_audio_workflows(task, audio_type) == workflows
    gguf = CachedGgufRepo(repo_id = "someone/m-GGUF", task = task, audio_type = audio_type)
    model = CachedModelRepo(repo_id = "someone/m", task = task, audio_type = audio_type)
    local = LocalModelInfo(
        id = "m", display_name = "m", path = "/x", source = "models_dir", task = task, audio_type = audio_type
    )
    assert gguf.audio_workflows == model.audio_workflows == local.audio_workflows == workflows
    assert gguf.model_dump()["audio_workflows"] == workflows


def test_workflow_for_audio_type_is_speak_or_music():
    assert aw.workflow_for_audio_type(AUDIO_CPP_MUSIC_AUDIO_TYPE) == "music"
    assert aw.workflow_for_audio_type("minimax_music3") == "music"
    assert aw.workflow_for_audio_type("snac") == "speak"
    assert aw.workflow_for_audio_type("unknown") == "speak"
    assert aw.workflow_for_audio_type(None) == "speak"
