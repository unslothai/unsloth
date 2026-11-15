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
    # No number is no answer: the caller keeps its silence wait rather than reading it as "unfinished".
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


def test_only_a_turn_model_resolves(monkeypatch):
    smart_turn = _model("smart_turn", "Smart-Turn-v3-GGUF")
    assert (smart_turn.task, smart_turn.unsupported, dict(smart_turn.workflows)) == (
        "turn",
        None,
        {},
    )
    asked = []
    monkeypatch.setattr(
        clip, "resolve", lambda base, variant, network: asked.append(base) or smart_turn
    )
    assert turn.resolve_turn_model(None) is smart_turn
    assert asked == [turn.DEFAULT_TURN_MODEL]
    monkeypatch.setattr(clip, "resolve", lambda *a, **k: _model("pulsevad", "PulseVAD-GGUF"))
    with pytest.raises(clip.ClipError, match = "not a turn detection model"):
        turn.resolve_turn_model(f"{acm.AUDIO_CPP_REPO}/PulseVAD-GGUF")
    # It has no page, so nothing else loads it: not as a voice, a transcriber or a detector.
    for task, refusal in (
        ("tts", "not a speech or music"),
        ("asr", "not a speech-to-text"),
        ("vad", "not a voice activity"),
    ):
        with pytest.raises(acm.AudioCppModelError, match = refusal):
            acm.require_runnable(smart_turn, task)
    monkeypatch.setattr(clip, "resolve", lambda *a, **k: None)
    with pytest.raises(clip.ClipModelNotDownloadedError):
        turn.resolve_turn_model(None)


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
    # A language nobody measured gets the careful bar, and 0.7 is under it: the loop keeps waiting.
    assert sidecar.detect(_wav(1.0))["finished"] is False
    # Where the model cuts speakers off too often it is not asked at all.
    assert sidecar.detect(_wav(1.0), language = "zh-CN") == {
        "p_complete": None,
        "finished": False,
        "bar": None,
        "model": None,
    }
    assert heard == [8.0, 1.0]
    # One start, on the CPU whatever training is doing, and nothing re-resolved for the second clip.
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


def test_the_thread_cap_reaches_the_runtimes_own_pool(monkeypatch, tmp_path):
    """--threads alone leaves the runtime's OpenMP pool on every core."""
    from core.inference import audio_cpp_server as srv

    seen = {}

    class Stop(Exception):
        pass

    def popen(command, env, **kwargs):
        seen.update(command = command, omp = env.get("OMP_NUM_THREADS"))
        raise Stop

    monkeypatch.setattr(srv, "ensure_binary", lambda: "/bin/audiocpp_server")
    monkeypatch.setattr(srv, "model_runtime_problem", lambda model, binary = None: None)
    monkeypatch.setattr(srv, "select_backend", lambda binary, force_cpu: "cpu")
    monkeypatch.setattr(srv, "child_env", lambda binary: {})
    monkeypatch.setattr(srv.subprocess, "Popen", popen)
    monkeypatch.setattr(srv.tempfile, "mkdtemp", lambda prefix: str(tmp_path))
    monkeypatch.delenv("UNSLOTH_CPU_THREADS", raising = False)
    entry = _model("smart_turn", "Smart-Turn-v3-GGUF")
    with pytest.raises(Stop):
        srv.AudioCppServer.start(entry, "/m.gguf", force_cpu = True, max_threads = 4)
    assert seen["command"][-2:] == ["--threads", "4"] and seen["omp"] == "4"
    # A lower budget the user set for every engine still wins.
    monkeypatch.setenv("UNSLOTH_CPU_THREADS", "2")
    with pytest.raises(Stop):
        srv.AudioCppServer.start(entry, "/m.gguf", force_cpu = True, max_threads = 4)
    assert seen["command"][-2:] == ["--threads", "2"] and seen["omp"] == "2"
    # Without a cap the launch is as it was.
    monkeypatch.delenv("UNSLOTH_CPU_THREADS")
    with pytest.raises(Stop):
        srv.AudioCppServer.start(entry, "/m.gguf", force_cpu = True)
    assert "--threads" not in seen["command"] and seen["omp"] is None


def test_model_access_and_the_runtime_update_cover_the_turn_model(monkeypatch):
    import asyncio

    from routes import inference

    asked = []
    monkeypatch.setattr(inference.account_access, "managed_account", lambda: True)
    monkeypatch.setattr(inference.account_access, "require_model_access", asked.append)
    asyncio.run(inference._require_turn_model_access(None))
    asyncio.run(inference._require_turn_model_access("someone/Turn-GGUF"))
    assert asked == [acm.AUDIO_CPP_REPO, "someone/Turn-GGUF"]
    # Shapes the browser acts on: not downloaded, cannot run here, bad request.
    for exc, status in (
        (clip.ClipModelNotDownloadedError("x"), 409),
        (clip.ClipUnavailableError("x"), 501),
        (clip.ClipError("x"), 422),
    ):
        assert inference._clip_http_error(exc, "Turn detection").status_code == status
