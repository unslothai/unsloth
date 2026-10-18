# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json

import pytest

from core.inference.audio_cpp_models import AUDIO_CPP_REPO
from core.inference.native_audio import NATIVE_AUDIO_MODEL_IDS

from .mcp_harness import CONFIG, DIFFUSION, SENTINEL_ROOTS, poison, run_tool

TRAINABLE = "Trainable text-to-speech models: unsloth/orpheus-3b-0.1-ft, sesame/csm-1b"


def _detect(monkeypatch, answer):
    seen = []

    def detect(
        name,
        hf_token = None,
        **_kw,
    ):
        seen.append((name, hf_token))
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr("utils.models.model_config.detect_audio_type_checked", detect)
    return seen


def _validate(
    monkeypatch,
    config,
    kind = "llm",
):
    result, studio = run_tool(
        monkeypatch, {}, "start_training", {"config": config, "kind": kind, "validate_only": True}
    )
    assert studio.state.calls == []
    return result


def test_a_text_model_passes_with_its_resolved_config(monkeypatch):
    seen = _detect(monkeypatch, (None, True))
    result = _validate(monkeypatch, CONFIG)
    out = result["structuredContent"]
    assert out["status"] == "valid" and out["job_id"] is None
    assert out["message"] == "The config is valid; nothing was started."
    config = out["config"]
    assert config["model_name"] == "unsloth/Qwen3-0.6B"
    assert config["max_steps"] == 30 and config["is_dataset_audio"] is False
    # The config's hf_token is used for detection, and never echoed; no start id is made up.
    assert seen == [("unsloth/Qwen3-0.6B", "hf_cfg")]
    assert "hf_token" not in config and "start_request_id" not in config
    assert "hf_cfg" not in json.dumps(result)


def test_the_echo_carries_no_poisoned_path_or_secret(monkeypatch):
    _detect(monkeypatch, (None, True))
    config = {
        **CONFIG,
        "project_name": "demo",
        "wandb_token": "wandb-secret",
        "custom_format_mapping": poison({"question": "user"}),
        "tensorboard_dir": f"{SENTINEL_ROOTS[0]}/tb",
        "resume_from_checkpoint": f"{SENTINEL_ROOTS[1]}\\ckpt",
        "wandb_project": poison("proj"),
    }
    result = _validate(monkeypatch, config)
    text = json.dumps(result)
    for root in SENTINEL_ROOTS:
        assert root not in text and root.replace("\\", "\\\\") not in text
    assert "wandb-secret" not in text and "hf_cfg" not in text
    shown = result["structuredContent"]["config"]
    assert "tensorboard_dir" not in shown and "resume_from_checkpoint" not in shown
    assert shown["project_name"] == "demo"
    assert shown["wandb_project"].startswith("proj")


def test_a_local_model_path_is_not_echoed(monkeypatch):
    _detect(monkeypatch, (None, True))
    result = _validate(monkeypatch, {**CONFIG, "model_name": f"{SENTINEL_ROOTS[0]}/models/qwen"})
    assert "model_name" not in result["structuredContent"]["config"]
    assert SENTINEL_ROOTS[0] not in json.dumps(result)


def test_a_codec_tts_model_gets_is_dataset_audio(monkeypatch):
    _detect(monkeypatch, ("snac", True))
    result = _validate(monkeypatch, {**CONFIG, "model_name": "unsloth/orpheus-3b-0.1-ft"})
    out = result["structuredContent"]
    assert out["config"]["is_dataset_audio"] is True
    assert "is_dataset_audio is set to true" in out["message"]


def test_an_explicit_is_dataset_audio_needs_no_note(monkeypatch):
    _detect(monkeypatch, ("csm", True))
    result = _validate(
        monkeypatch, {**CONFIG, "model_name": "sesame/csm-1b", "is_dataset_audio": True}
    )
    out = result["structuredContent"]
    assert out["config"]["is_dataset_audio"] is True
    assert out["message"] == "The config is valid; nothing was started."


@pytest.mark.parametrize(
    "model_name, extra, answer, message",
    [
        ("unsloth/Qwen3-0.6B-GGUF", {}, None, "GGUF models are inference-only"),
        ("unsloth/Qwen3-0.6B", {"model_format": "gguf"}, None, "GGUF models are inference-only"),
        (
            f"{AUDIO_CPP_REPO}/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF",
            {},
            None,
            "runs on audio.cpp, which Unsloth Studio can run but not train.",
        ),
        (
            "someone/audio-cpp-tts",
            {},
            ("audiocpp_tts", True),
            "runs on audio.cpp, which Unsloth Studio can run but not train.",
        ),
        (
            next(iter(NATIVE_AUDIO_MODEL_IDS)),
            {},
            RuntimeError("no network"),
            "is a text-to-speech model Unsloth Studio can run but not train.",
        ),
        (
            "someone/higgs-finetune",
            {},
            ("higgs_tts2", True),
            "is a text-to-speech model Unsloth Studio can run but not train.",
        ),
    ],
)
def test_untrainable_models_are_refused(monkeypatch, model_name, extra, answer, message):
    seen = _detect(monkeypatch, answer or AssertionError("detection should not run"))
    result = _validate(monkeypatch, {**CONFIG, "model_name": model_name, **extra})
    assert result["isError"] is True
    text = result["content"][0]["text"]
    assert message in text and TRAINABLE in text
    # Names and curated ids are refused without asking the Hub.
    if answer is None or isinstance(answer, Exception):
        assert seen == []


def test_a_failed_detection_is_no_verdict(monkeypatch):
    seen = _detect(monkeypatch, OSError("hub unreachable"))
    result = _validate(monkeypatch, CONFIG)
    assert result.get("isError") is not True
    assert result["structuredContent"]["status"] == "valid"
    assert result["structuredContent"]["config"]["is_dataset_audio"] is False
    assert seen == [("unsloth/Qwen3-0.6B", "hf_cfg")]


def test_diffusion_echo_drops_folders(monkeypatch):
    _detect(monkeypatch, AssertionError("diffusion runs no detection"))
    result = _validate(monkeypatch, {**DIFFUSION, "hf_token": "hf_x"}, kind = "diffusion")
    out = result["structuredContent"]
    assert out["status"] == "valid"
    shown = out["config"]
    assert shown["base_model"] == DIFFUSION["base_model"] and shown["train_steps"] == 500
    assert not {"data_dir", "output_dir", "hf_token"} & set(shown)


@pytest.mark.parametrize("answer", [("higgs_tts2", True), ("snac", True)])
def test_a_ref_model_name_never_comes_back_as_its_path(monkeypatch, answer):
    from hub.utils.host_paths import cache_reference

    seen = _detect(monkeypatch, answer)
    ref = cache_reference(f"{SENTINEL_ROOTS[0]}/models/my-tts")
    result = _validate(monkeypatch, {**CONFIG, "model_name": ref})
    # The validator resolved the ref for detection, but the text names the ref.
    assert seen and seen[0][0] == f"{SENTINEL_ROOTS[0]}/models/my-tts"
    text = json.dumps(result)
    assert SENTINEL_ROOTS[0] not in text and "my-tts" not in text
    assert ref in text


def test_a_real_start_sets_is_dataset_audio_for_codec_tts(monkeypatch):
    _detect(monkeypatch, ("snac", True))
    started = {"job_id": "job_1", "status": "queued"}
    config = {**CONFIG, "model_name": "unsloth/orpheus-3b-0.1-ft"}
    result, studio = run_tool(
        monkeypatch, {("POST", "/api/train/start"): started}, "start_training", {"config": config}
    )
    assert result.get("isError") is not True, result
    (_method, _path, _headers, body) = next(
        c for c in studio.state.calls if c[1] == "/api/train/start"
    )
    assert json.loads(body)["is_dataset_audio"] is True


def test_a_real_start_refuses_an_untrainable_tts_model_before_posting(monkeypatch):
    _detect(monkeypatch, ("higgs_tts2", True))
    config = {**CONFIG, "model_name": "someone/higgs-finetune"}
    result, studio = run_tool(monkeypatch, {}, "start_training", {"config": config}, fallback = {})
    assert result["isError"] is True and TRAINABLE in result["content"][0]["text"]
    assert [c for c in studio.state.calls if c[1] == "/api/train/start"] == []
