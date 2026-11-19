# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import io
import json
import wave
from types import SimpleNamespace

import pytest

from core.inference import audio_cpp_clip_sidecar as clip
from core.inference import audio_cpp_models as acm
from core.inference import turn_audiocpp_sidecar as turn


def _wav(seconds: float) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))
    return buf.getvalue()


def _model(family: str, folder: str) -> acm.AudioCppModel:
    policy = acm.family_policy(family)
    return acm.AudioCppModel(
        id = f"{acm.AUDIO_CPP_REPO}/{folder}",
        repo_id = acm.AUDIO_CPP_REPO,
        folder = folder,
        display_name = folder,
        family = policy.family,
        task = policy.task,
        server_task = policy.default_server_task,
        variant = acm.AudioCppVariant("F32", (), "x.gguf"),
        variants = (),
        default_variant = "F32",
    )


def test_the_answer_is_the_probability_the_runtime_gave():
    answer = {"custom_schema_output": {"data": {"probability": 0.9855, "complete": True}}}
    assert turn.completion_from_result(answer) == 0.9855
    assert (
        turn.completion_from_result({"custom_schema_output": {"data": {"probability": 7}}}) == 1.0
    )
    for empty in (
        {},
        {"custom_schema_output": None},
        {"custom_schema_output": {"data": {"probability": True}}},
    ):
        with pytest.raises(clip.ClipUnavailableError, match = "gave no answer"):
            turn.completion_from_result(empty)


@pytest.mark.parametrize(
    "language, bar",
    [
        ("en", 0.6),
        ("en-GB", 0.6),
        ("DE_de", 0.6),
        ("es-419", 0.9),
        ("auto", 0.9),
        (None, 0.9),
        ("vi", None),
        ("ar-EG", None),
    ],
)
def test_the_bar_follows_the_language(language, bar):
    assert turn.complete_at(language) == bar


def test_a_check_reads_the_end_of_the_clip_on_the_cpu_with_four_threads(monkeypatch, tmp_path):
    from core.inference import stt_sidecar

    entry = _model("smart_turn", "Smart-Turn-v3-GGUF")
    resolved = []
    monkeypatch.setattr(turn, "resolve_turn_model", lambda *a, **k: resolved.append(1) or entry)
    monkeypatch.setattr(clip.audio_cpp_files, "materialize", lambda e: tmp_path / "turn.gguf")
    monkeypatch.setattr(stt_sidecar, "_training_active", lambda: False)
    launched, heard = [], []

    class Server:
        model_id = "studio-turn"
        process = SimpleNamespace(poll = lambda: None)

        def stop(self):
            pass

        def post_json(self, path, body, timeout):
            assert (path, body["model"]) == ("/v1/tasks/run", "studio-turn")
            with wave.open(body["request"]["audio"], "rb") as w:
                heard.append(w.getnframes() / w.getframerate())
            p = 0.97 if len(heard) == 1 else 0.7
            return 200, json.dumps({"custom_schema_output": {"data": {"probability": p}}}).encode()

    def start(
        model,
        path,
        force_cpu = False,
        max_threads = None,
    ):
        launched.append((model.family, force_cpu, max_threads))
        return Server()

    monkeypatch.setattr(clip.AudioCppServer, "start", staticmethod(start))
    sidecar = turn.AudioCppTurnSidecar()
    assert sidecar.detect(_wav(12.0), language = "en-US") == {
        "p_complete": 0.97,
        "finished": True,
        "bar": 0.6,
        "model": entry.id,
    }
    assert sidecar.detect(_wav(1.0))["finished"] is False
    assert sidecar.detect(_wav(1.0), language = "zh-CN") == {
        "p_complete": None,
        "finished": False,
        "bar": None,
        "model": None,
    }
    assert heard == [8.0, 1.0]
    assert launched == [("smart_turn", True, 4)] and resolved == [1]
    assert sidecar.status() == {
        "available": sidecar.status()["available"],
        "loaded_model": entry.id,
        "loading": None,
        "default_model": turn.DEFAULT_TURN_MODEL,
        "ask_after_ms": 250,
    }
    assert sidecar.unload() is True and sidecar.loaded_model is None


def test_a_clip_too_large_to_check_is_refused():
    with pytest.raises(clip.ClipError, match = "too long"):
        turn.AudioCppTurnSidecar().detect(b"x" * (clip.MAX_CLIP_BYTES + 1))


def test_model_access_and_the_runtime_update_cover_the_turn_model(monkeypatch):
    import asyncio

    from routes import inference

    asked = []
    monkeypatch.setattr(inference.account_access, "managed_account", lambda: True)
    monkeypatch.setattr(inference.account_access, "require_model_access", asked.append)
    asyncio.run(inference._require_turn_model_access(None))
    asyncio.run(inference._require_turn_model_access("someone/Turn-GGUF"))
    assert asked == [acm.AUDIO_CPP_REPO, "someone/Turn-GGUF"]
    for exc, status in (
        (clip.ClipModelNotDownloadedError("x"), 409),
        (clip.ClipUnavailableError("x"), 501),
        (clip.ClipError("x"), 422),
    ):
        assert inference._clip_http_error(exc, "Turn detection").status_code == status
