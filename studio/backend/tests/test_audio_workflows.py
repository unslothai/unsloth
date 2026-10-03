# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which Audio page workflows a loaded model, an inventory row or a gallery clip belongs to."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.inference import audio_workflows as aw
from core.inference.audio_cpp_models import AUDIO_CPP_MUSIC_AUDIO_TYPE, AUDIO_CPP_TTS_AUDIO_TYPE
from hub.schemas.inventory import CachedGgufRepo, CachedModelRepo, LocalModelInfo
from models.inference import InferenceStatusResponse, LoadResponse


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


def test_a_status_reports_no_workflows_for_a_model_that_is_not_audio():
    status = InferenceStatusResponse(active_model = "unsloth/Qwen3-0.6B-GGUF")
    assert status.audio_workflows == []
    assert InferenceStatusResponse().audio_workflows == []


@pytest.mark.parametrize(
    "audio_type, workflows",
    [
        (AUDIO_CPP_TTS_AUDIO_TYPE, ["speak"]),
        (AUDIO_CPP_MUSIC_AUDIO_TYPE, ["music"]),
        ("minimax_music3", ["music"]),
        ("snac", ["speak"]),
        ("higgs_tts2", ["speak"]),
        ("whisper", ["transcribe"]),
        (None, ["speak"]),
    ],
)
def test_a_status_reports_the_workflow_of_its_audio_type(audio_type, workflows):
    status = InferenceStatusResponse(is_audio = True, audio_type = audio_type)
    assert status.audio_workflows == workflows
    assert status.model_dump()["audio_workflows"] == workflows


def test_a_load_response_derives_workflows_and_keeps_an_explicit_value():
    load = LoadResponse(
        status = "loaded",
        model = "audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF",
        display_name = "Kokoro",
        inference = {},
        is_audio = True,
        audio_type = AUDIO_CPP_TTS_AUDIO_TYPE,
    )
    assert load.audio_workflows == ["speak"]
    explicit = InferenceStatusResponse(is_audio = True, audio_workflows = ["music"])
    assert explicit.audio_workflows == ["music"]


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
    assert aw.AUDIO_WORKFLOW_IDS == ("speak", "clone", "convert", "music", "transcribe")


def test_workflow_ids_match_the_frontend_order():
    source = (
        Path(__file__).resolve().parents[2] / "frontend/src/features/audio/workflows.ts"
    ).read_text(encoding = "utf-8")
    block = source.split("export const AUDIO_WORKFLOWS", 1)[1]
    assert tuple(re.findall(r'^\s*id: "(\w+)"', block, re.M)) == aw.AUDIO_WORKFLOW_IDS


def test_status_carries_the_clone_fields():
    clone = {"audio_workflows": ["clone"], "audio_reference_text": "required"}
    status = InferenceStatusResponse(is_audio = True, audio_type = "audiocpp_tts", **clone)
    assert status.model_dump().items() >= clone.items()


def test_status_and_load_carry_the_convert_fields():
    caps = {
        "modes": ["speech", "singing"],
        "target": "audio",
        "builtin_voices": [],
        "pitch": {"singing": {"auto": True}},
        "style": False,
        "route_reloads": True,
        "source_max_seconds": 300,
    }
    fields = dict(
        is_audio = True,
        audio_type = AUDIO_CPP_TTS_AUDIO_TYPE,
        audio_workflows = ["convert"],
        audio_options_by_workflow = {"convert": [{"name": "length_adjust", "type": "float"}]},
        audio_workflow_tasks = {"convert": "vc", "convert:singing": "svc"},
        audio_server_task = "svc",
        audio_convert = caps,
        audio_convert_route = "v1_svc",
    )
    dumped = InferenceStatusResponse(**fields).model_dump()
    assert dumped["audio_convert"] == caps
    assert dumped["audio_workflow_tasks"]["convert:singing"] == "svc"
    assert (dumped["audio_server_task"], dumped["audio_convert_route"]) == ("svc", "v1_svc")
    assert dumped["audio_options_by_workflow"]["convert"][0]["name"] == "length_adjust"
    load = LoadResponse(status = "loaded", model = "m", display_name = "m", inference = {}, **fields)
    assert load.audio_convert == caps and load.audio_server_task == "svc"
    plain = InferenceStatusResponse(is_audio = True, audio_type = AUDIO_CPP_TTS_AUDIO_TYPE)
    assert plain.audio_convert is None and plain.audio_workflow_tasks is None


@pytest.mark.parametrize(
    "folder, family, workflows",
    [
        ("Chatterbox-GGUF", "chatterbox", ["clone", "convert"]),
        ("Chatterbox-Turbo-GGUF", "chatterbox_turbo", ["speak"]),
        ("VoxCPM2-GGUF", "voxcpm2", ["speak", "clone"]),
        ("Qwen3-TTS-12Hz-0.6B-Base-GGUF", "qwen3_tts", ["clone"]),
        ("Kokoro-82M-GGUF", "kokoro_tts", ["speak"]),
        ("RVC-GGUF", "rvc", ["convert"]),
        ("SeedVC-MLX-GGUF", "seed_vc", ["convert"]),
        ("MeanVC2-GGUF", "meanvc2", ["convert"]),
        ("Vevo2-GGUF", "vevo2", ["clone", "convert"]),
    ],
)
def test_inventory_rows_of_clone_families_list_clone(tmp_path, folder, family, workflows):
    from hub.services.models import catalog_classification as cc

    from .test_audio_cpp_models import _gguf_bytes

    path = tmp_path / folder / f"{folder.lower().removesuffix('-gguf')}-q8_0.gguf"
    path.parent.mkdir(parents = True)
    path.write_bytes(_gguf_bytes(family = family))
    found = cc._gguf_path_audio_workflows(path.parent, (f"someone/{folder}",))
    row = CachedGgufRepo(repo_id = "r", task = "text-to-speech", audio_workflows = found)
    assert found == row.audio_workflows == workflows
    # A local folder row (models dir, LM Studio, custom folder) gets the same answer.
    local = LocalModelInfo(
        id = str(path.parent), display_name = folder, path = str(path.parent), source = "models_dir"
    )
    assert cc.local_audio_workflows(local, "audiocpp_tts") == workflows
    assert cc.local_audio_workflows(local, "orpheus") is None


def test_an_umbrella_snapshot_lists_every_downloaded_familys_workflows(tmp_path):
    # audio-cpp/audio.cpp-gguf is one cached row: its workflows were the first GGUF's alone.
    from hub.services.models import catalog_classification as cc

    from .test_audio_cpp_models import _gguf_bytes

    for folder, family in (
        ("Qwen3-TTS-12Hz-0.6B-Base-GGUF", "qwen3_tts"),
        ("Kokoro-82M-GGUF", "kokoro_tts"),
    ):
        path = tmp_path / folder / f"{folder.lower()}-q8_0.gguf"
        path.parent.mkdir(parents = True)
        path.write_bytes(_gguf_bytes(family = family))
    assert cc._gguf_path_audio_workflows(tmp_path, ("audio-cpp/audio.cpp-gguf",)) == [
        "speak",
        "clone",
    ]
