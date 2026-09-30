# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The audio.cpp catalog, its file lookup in the HF cache, and the worker backend's request shaping."""

import base64
import io
import json
import os
import sys
import wave
from types import SimpleNamespace

import pytest

from core.inference import audio_cpp_backend, audio_cpp_files
from core.inference.audio_cpp_models import (
    AUDIO_CPP_REPO,
    AUDIO_CPP_AUDIO_TYPES,
    all_models,
    lookup,
    stt_models,
)


def test_ids_keys_and_tasks_are_unique_and_consistent():
    models = all_models()
    assert len({m.id.lower() for m in models}) == len(models)
    assert len({m.key for m in models}) == len(models)
    for m in models:
        # Three segments, so a curated id can never collide with a real owner/name repo.
        assert m.id.startswith(AUDIO_CPP_REPO + "/") and m.id.count("/") >= 2
        assert m.task in ("tts", "music", "asr")
        assert m.files and m.files[0].endswith(".gguf")
        assert all(f.startswith(m.id.split("/")[2] + "/") for f in m.files)
        assert (m.audio_type in AUDIO_CPP_AUDIO_TYPES) == (m.task != "asr")
    assert all(m.task == "asr" for m in stt_models())


def test_lookup_by_id_key_case_and_trailing_slash():
    m = lookup("audiocpp-qwen3-asr-0.6b")
    assert m is not None and m.family == "qwen3_asr"
    assert lookup(m.id.upper()) is m
    assert lookup(m.id + "/") is m
    assert lookup("audio-cpp/audio.cpp-gguf") is None
    assert lookup("Qwen/Qwen3-ASR-0.6B") is None
    assert lookup(None) is None and lookup("") is None


def test_music_models_run_the_gen_task():
    music = [m for m in all_models() if m.task == "music"]
    assert music and all(m.server_task == "gen" for m in music)


def _snapshot(tmp_path, sha = "a" * 40):
    root = tmp_path / "hub"
    snap = root / ("models--" + AUDIO_CPP_REPO.replace("/", "--")) / "snapshots" / sha
    snap.mkdir(parents = True)
    return root, snap


def test_cached_files_require_every_file(tmp_path):
    root, snap = _snapshot(tmp_path)
    m = lookup("audiocpp-canary-180m-flash")
    assert audio_cpp_files.cached_files(m, hub_cache = root) is None
    target = snap / m.gguf_file
    target.parent.mkdir(parents = True)
    target.write_bytes(b"GGUF")
    assert audio_cpp_files.cached_files(m, hub_cache = root) == [target]
    served = audio_cpp_files.materialize(m, hub_cache = root)
    if sys.platform == "win32":
        # Windows always serves the short per-model farm path (MAX_PATH), linked to the cache file.
        assert os.path.samefile(served, target) and m.key in served
    else:
        # A real file (a cache without symlinks) is handed over as is.
        assert served == str(target)


def test_windows_refuses_a_model_path_the_server_cannot_open(tmp_path, monkeypatch):
    from core.inference.audio_cpp_server import AudioCppUnavailableError

    root, snap = _snapshot(tmp_path)
    m = lookup("audiocpp-canary-180m-flash")
    target = snap / m.gguf_file
    target.parent.mkdir(parents = True)
    target.write_bytes(b"GGUF")
    monkeypatch.setattr(audio_cpp_files.sys, "platform", "win32")
    monkeypatch.setattr(audio_cpp_files, "_WINDOWS_MAX_MODEL_PATH", 10)
    with pytest.raises(AudioCppUnavailableError, match = "shorter"):
        audio_cpp_files.materialize(m, hub_cache = root)


def test_farm_layout_keeps_package_subfolders(tmp_path, monkeypatch):
    root, snap = _snapshot(tmp_path)
    m = lookup("audiocpp-pocket-tts-en")
    gguf = snap / m.gguf_file
    (gguf.parent / "embeddings").mkdir(parents = True)
    gguf.write_bytes(b"GGUF")
    for i in range(m.min_glob_matches):
        (gguf.parent / "embeddings" / f"v{i}.safetensors").write_bytes(b"x")
    monkeypatch.setattr(audio_cpp_files.sys, "platform", "win32")
    served = audio_cpp_files.materialize(m, hub_cache = root)
    farm = audio_cpp_files._link_farm_root(root) / m.key
    # The server finds embeddings/ beside the GGUF, as in the package.
    assert served == str(farm / "english" / "pocket-tts-english-q8_0.gguf")
    assert len(list((farm / "english" / "embeddings").iterdir())) == m.min_glob_matches


def test_pinned_revision_snapshot_is_preferred(tmp_path):
    from core.inference.audio_cpp_models import AUDIO_CPP_REVISION

    root, pinned = _snapshot(tmp_path, AUDIO_CPP_REVISION)
    other = pinned.parent / ("f" * 40)
    m = lookup("audiocpp-canary-180m-flash")
    for snap in (other, pinned):
        (snap / m.gguf_file).parent.mkdir(parents = True, exist_ok = True)
        (snap / m.gguf_file).write_bytes(b"GGUF")
    os.utime(pinned, (1, 1))  # older than the other snapshot, still first
    assert audio_cpp_files.cached_files(m, hub_cache = root) == [pinned / m.gguf_file]


def test_prune_removes_legacy_and_unknown_farm_dirs(tmp_path):
    root, _snap = _snapshot(tmp_path)
    farm = audio_cpp_files._link_farm_root(root)
    legacy = farm / ("a" * 40) / "Canary-180M-Flash-GGUF"
    legacy.mkdir(parents = True)
    (legacy / "canary.gguf").write_bytes(b"x")
    assert audio_cpp_files.prune_link_farm(root) == 1
    assert not (farm / ("a" * 40)).exists()


def test_one_failed_transcription_does_not_hide_the_engine(monkeypatch):
    from core.inference import stt_audiocpp_sidecar as s

    monkeypatch.setattr(s.audio_cpp_server, "is_available", lambda: True)
    s.note_runtime_inference_failure("HTTP 500 on a bad clip")
    try:
        assert s.is_available() is True
    finally:
        s.clear_runtime_inference_failure()


def test_child_home_is_private_and_per_user(tmp_path, monkeypatch):
    from core.inference import audio_cpp_server as srv
    import utils.paths.storage_roots as roots

    monkeypatch.setattr(roots, "studio_root", lambda: tmp_path / "studio")
    assert srv._child_home_dir() == tmp_path / "studio" / "cache" / "audiocpp-home"

    def broken():
        raise OSError("no studio root")

    monkeypatch.setattr(roots, "studio_root", broken)
    monkeypatch.setattr(srv.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    home = srv._child_home_dir()
    assert home.parent == tmp_path / "tmp" and home.name.startswith("unsloth-audiocpp-home-")


def test_glob_files_need_every_published_match(tmp_path):
    root, snap = _snapshot(tmp_path)
    m = lookup("audiocpp-pocket-tts-en")
    gguf = snap / m.gguf_file
    gguf.parent.mkdir(parents = True)
    gguf.write_bytes(b"GGUF")
    assert audio_cpp_files.cached_files(m, hub_cache = root) is None
    (gguf.parent / "embeddings").mkdir()
    # An interrupted download holding one voice is not a complete model.
    (gguf.parent / "embeddings" / "alba.safetensors").write_bytes(b"x")
    assert audio_cpp_files.cached_files(m, hub_cache = root) is None
    for i in range(m.min_glob_matches - 1):
        (gguf.parent / "embeddings" / f"voice{i}.safetensors").write_bytes(b"x")
    assert len(audio_cpp_files.cached_files(m, hub_cache = root)) == 1 + m.min_glob_matches


def test_concurrent_first_materialize_does_not_race(tmp_path):
    import threading

    root, snap = _snapshot(tmp_path)
    blobs = snap.parent.parent / "blobs"
    blobs.mkdir()
    blob = blobs / "cafe"
    blob.write_bytes(b"GGUF" * 1000)
    m = lookup("audiocpp-canary-180m-flash")
    link = snap / m.gguf_file
    link.parent.mkdir(parents = True)
    try:
        link.symlink_to(os.path.relpath(blob, link.parent))
    except OSError:
        pytest.skip("this filesystem or account cannot create symlinks")
    errors = []

    def worker():
        try:
            audio_cpp_files.materialize(m, hub_cache = root)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    for _ in range(10):
        threads = [threading.Thread(target = worker) for _ in range(6)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        farm = audio_cpp_files._link_farm_root(root)
        for p in farm.rglob("*"):
            if p.is_file():
                p.unlink()
    assert errors == []


def test_prune_drops_farm_entries_whose_blob_was_deleted(tmp_path):
    root, snap = _snapshot(tmp_path)
    blobs = snap.parent.parent / "blobs"
    blobs.mkdir()
    blob = blobs / "beef"
    blob.write_bytes(b"GGUF")
    m = lookup("audiocpp-canary-180m-flash")
    link = snap / m.gguf_file
    link.parent.mkdir(parents = True)
    try:
        link.symlink_to(os.path.relpath(blob, link.parent))
    except OSError:
        pytest.skip("this filesystem or account cannot create symlinks")
    path = audio_cpp_files.materialize(m, hub_cache = root)
    assert audio_cpp_files.prune_link_farm(root) == 0  # still referenced
    link.unlink()
    blob.unlink()
    assert audio_cpp_files.prune_link_farm(root) == 1
    assert not os.path.exists(path)


def test_a_copied_file_is_not_recopied(tmp_path, monkeypatch):
    root, snap = _snapshot(tmp_path)
    blobs = snap.parent.parent / "blobs"
    blobs.mkdir()
    blob = blobs / "f00d"
    blob.write_bytes(b"GGUF")
    m = lookup("audiocpp-canary-180m-flash")
    link = snap / m.gguf_file
    link.parent.mkdir(parents = True)
    try:
        link.symlink_to(os.path.relpath(blob, link.parent))
    except OSError:
        pytest.skip("this filesystem or account cannot create symlinks")

    def no_link(*_a, **_k):
        raise OSError("cross-device link")

    copies = []
    real_copy = audio_cpp_files.shutil.copyfile
    monkeypatch.setattr(audio_cpp_files.os, "link", no_link)
    monkeypatch.setattr(
        audio_cpp_files.shutil, "copyfile", lambda a, b: copies.append(a) or real_copy(a, b)
    )
    for _ in range(3):
        audio_cpp_files.materialize(m, hub_cache = root)
    assert len(copies) == 1


def test_qwen3_packages_get_their_task_and_defaults():
    design = lookup("audio-cpp/audio.cpp-gguf/Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF")
    custom = lookup("audio-cpp/audio.cpp-gguf/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF")
    assert design.server_task == "vdes" and design.request_defaults["options"]["instruct"]
    assert custom.server_task == "tts" and custom.request_defaults["voice"] == "vivian"


def test_espeak_models_need_an_espeak_build(tmp_path, monkeypatch):
    from core.inference import audio_cpp_server as srv

    binary = tmp_path / srv.BINARY_NAME
    binary.write_bytes(b"x")
    kokoro = lookup("audiocpp-kokoro-82m")
    assert "eSpeak-ng" in srv.model_runtime_problem(kokoro, str(binary))
    assert srv.model_runtime_problem(lookup("audiocpp-canary-180m-flash"), str(binary)) is None
    (tmp_path / "espeak-ng-data.bin").write_bytes(b"x")
    assert srv.model_runtime_problem(kokoro, str(binary)) is None
    monkeypatch.setattr(srv, "find_audio_cpp_server_binary", lambda: None)
    assert "not installed" in srv.model_runtime_problem(kokoro)


def test_child_home_is_outside_the_managed_install(monkeypatch, tmp_path):
    from core.inference import audio_cpp_server as srv

    monkeypatch.setattr(srv, "managed_audio_cpp_dir", lambda: tmp_path / "audio.cpp")
    env = srv.child_env(str(tmp_path / srv.BINARY_NAME))
    homes = {v for k, v in env.items() if k in ("HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA")}
    assert homes and not any(str(tmp_path / "audio.cpp") in h for h in homes)


def test_a_binary_that_cannot_spawn_is_a_typed_error_and_leaks_nothing(monkeypatch, tmp_path):
    import tempfile as _tempfile

    from core.inference import audio_cpp_server as srv

    made = []
    real_mkdtemp = _tempfile.mkdtemp
    monkeypatch.setattr(
        srv.tempfile,
        "mkdtemp",
        lambda **kw: made.append(real_mkdtemp(dir = tmp_path, **kw)) or made[-1],
    )
    monkeypatch.setattr(srv, "ensure_binary", lambda: str(tmp_path / "missing" / srv.BINARY_NAME))
    monkeypatch.setattr(srv, "model_runtime_problem", lambda model, binary = None: None)
    with pytest.raises(srv.AudioCppUnavailableError):
        srv.AudioCppServer.start(lookup("audiocpp-canary-180m-flash"), str(tmp_path / "m.gguf"))
    assert made and not os.path.exists(made[0])


def test_speech_retry_keeps_package_defaults_and_covers_500():
    b, srv = _backend_with(
        "audio-cpp/audio.cpp-gguf/Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF", ("audio/wav", _wav())
    )
    from core.inference.audio_cpp_server import AudioCppRequestError

    calls = []

    def post_json(path, payload, **kwargs):
        calls.append(json.loads(json.dumps(payload)))
        if len(calls) == 1:
            raise AudioCppRequestError(500, "unknown option: language")
        return ("audio/wav", _wav())

    srv.post_json = post_json
    b.generate_audio_response("hi", language = "en")
    assert calls[0]["options"]["language"] == "en" and calls[0]["options"]["instruct"]
    # The retry drops the request's hints, never the package's own instruction.
    assert (
        calls[1]["options"]
        == lookup("audio-cpp/audio.cpp-gguf/Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF").request_defaults[
            "options"
        ]
    )


def test_a_busy_server_is_not_retried():
    b, srv = _backend_with("audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF", ("audio/wav", _wav()))
    from core.inference.audio_cpp_server import AudioCppRequestError

    def post_json(path, payload, **kwargs):
        raise AudioCppRequestError(503, "server_busy")

    srv.post_json = post_json
    with pytest.raises(RuntimeError, match = "server_busy"):
        b.generate_audio_response("hi", instructions = "warm")


def test_speech_api_music_default_is_thirty_seconds():
    import routes.inference as ri
    from models.inference import ChatCompletionRequest

    payload = ChatCompletionRequest(messages = [{"role": "user", "content": "x"}], max_tokens = 8192)
    assert (
        ri._tts_max_new_tokens(
            payload, None, audio_type = "audiocpp_music", speech_api_default_max_tokens = True
        )
        == 750
    )
    assert ri._tts_max_new_tokens(payload, None, audio_type = "audiocpp_music") == 240 * 25


def test_public_id_of_an_audio_cpp_model_is_the_id_itself():
    from core.inference.model_ids import public_model_id
    for mid in (
        "audio-cpp/audio.cpp-gguf/ACE-Step1.5-GGUF/turbo",
        "audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF",
    ):
        assert public_model_id(mid) == mid
    # Unrelated three-segment paths are still paths.
    assert public_model_id("some/local/dir") != "some/local/dir"


def test_registry_keeps_the_resident_engine_when_audiocpp_runtime_is_missing(monkeypatch):
    from core.inference import stt_audiocpp_sidecar, stt_registry

    monkeypatch.setattr(stt_audiocpp_sidecar, "is_available", lambda: False)
    monkeypatch.setattr(stt_audiocpp_sidecar, "is_model_downloaded", lambda model: True)
    assert stt_registry._model_is_downloaded("audiocpp", "audiocpp-canary-180m-flash") is False


def test_files_split_across_snapshots_are_found_in_the_one_that_holds_them(tmp_path):
    root, main_snap = _snapshot(tmp_path, "a" * 40)
    (main_snap.parent.parent / "refs").mkdir()
    (main_snap.parent.parent / "refs" / "main").write_text("a" * 40)
    other = main_snap.parent / ("b" * 40)
    m = lookup("audiocpp-canary-180m-flash")
    (other / m.gguf_file).parent.mkdir(parents = True)
    (other / m.gguf_file).write_bytes(b"GGUF")
    assert audio_cpp_files.cached_files(m, hub_cache = root) == [other / m.gguf_file]


def test_symlinked_cache_entries_are_hardlinked_under_their_real_names(tmp_path):
    root, snap = _snapshot(tmp_path)
    blobs = snap.parent.parent / "blobs"
    blobs.mkdir()
    blob = blobs / "deadbeef"
    blob.write_bytes(b"GGUF-weights")
    m = lookup("audiocpp-canary-180m-flash")
    link = snap / m.gguf_file
    link.parent.mkdir(parents = True)
    try:
        link.symlink_to(os.path.relpath(blob, link.parent))
    except OSError:
        pytest.skip("this filesystem or account cannot create symlinks")
    path = audio_cpp_files.materialize(m, hub_cache = root)
    assert path.endswith(".gguf") and not os.path.islink(path)
    assert os.path.samefile(path, blob)
    # Idempotent.
    assert audio_cpp_files.materialize(m, hub_cache = root) == path


def test_materialize_of_a_missing_model_raises(tmp_path):
    root, _ = _snapshot(tmp_path)
    with pytest.raises(FileNotFoundError):
        audio_cpp_files.materialize(lookup("audiocpp-canary-180m-flash"), hub_cache = root)


# Worker backend request shaping, against a recording fake server.


def _wav(seconds = 0.1, rate = 24000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\0\0" * int(seconds * rate))
    return buf.getvalue()


class _FakeServer:
    def __init__(self, model, reply):
        self.model = model
        self.model_id = "studio-test"
        self.backend = "cuda"
        self.reply = reply
        self.calls = []

    def alive(self):
        return True

    def post_json(self, path, payload, **kwargs):
        self.calls.append((path, payload))
        return self.reply

    def stop(self):
        pass


def _backend_with(model_id, reply):
    b = audio_cpp_backend.AudioCppBackend()
    m = lookup(model_id)
    b._server = _FakeServer(m, reply)
    b.models = {m.id: {"is_audio": True, "audio_type": m.audio_type}}
    b.active_model_name = m.id
    return b, b._server


def test_speech_never_forwards_generic_sampling():
    b, srv = _backend_with(
        "audio-cpp/audio.cpp-gguf/MOSS-TTS-Nano-100M-GGUF", ("audio/wav", _wav())
    )
    wav, rate = b.generate_audio_response("hello", temperature = 0.6, top_p = 0.9, seed = 7)
    path, body = srv.calls[0]
    assert path == "/v1/audio/speech" and rate == 24000 and wav[:4] == b"RIFF"
    assert body == {"model": "studio-test", "input": "hello", "seed": "7"}


def test_speech_passes_instruction_language_and_default_voice():
    b, srv = _backend_with("audio-cpp/audio.cpp-gguf/PocketTTS-GGUF/english", ("audio/wav", _wav()))
    b.generate_audio_response("hi", instructions = " warm ", language = "en")
    body = srv.calls[0][1]
    assert body["voice"] == "alba"
    assert body["options"] == {"instruct": "warm", "language": "en"}


def test_music_maps_description_lyrics_and_duration():
    reply = ("application/json", json.dumps({"audio": base64.b64encode(_wav()).decode()}).encode())
    b, srv = _backend_with("audio-cpp/audio.cpp-gguf/ACE-Step1.5-GGUF/turbo", reply)
    wav, _ = b.generate_audio_response(
        "[verse] la la", instructions = "synth pop", max_new_tokens = 1500
    )
    path, body = srv.calls[0]
    assert path == "/v1/tasks/run" and wav[:4] == b"RIFF"
    assert body["request"] == {
        "text": "synth pop",
        "duration_seconds": 60.0,
        "lyrics": "[verse] la la",
    }


def test_music_without_a_description_uses_the_prompt_and_stable_audio_drops_lyrics():
    reply = ("application/json", json.dumps({"audio": base64.b64encode(_wav()).decode()}).encode())
    b, srv = _backend_with("audio-cpp/audio.cpp-gguf/Stable-Audio-3-Small-Music-GGUF", reply)
    b.generate_audio_response("uplifting house", max_new_tokens = 0)
    assert srv.calls[0][1]["request"] == {"text": "uplifting house", "duration_seconds": 30.0}


def test_task_response_without_audio_is_an_error():
    with pytest.raises(RuntimeError, match = "no audio"):
        audio_cpp_backend._audio_from_task_response("application/json", b'{"text": "x"}')


def test_generation_needs_an_active_model():
    b = audio_cpp_backend.AudioCppBackend()
    with pytest.raises(RuntimeError, match = "No active audio model"):
        b.generate_audio_response("x")


def test_stt_routing_forces_the_audiocpp_engine_for_its_models():
    import routes.inference as ri

    assert ri._stt_engine_for_model("audiocpp-canary-180m-flash") == "audiocpp"
    assert ri._stt_engine_for_model(lookup("audiocpp-canary-180m-flash").id) == "audiocpp"
    # Speech models are not dictation models.
    assert ri._stt_engine_for_model("audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF") is None
    for alias in ("audiocpp", "audio_cpp", "audio.cpp", " AudioCpp "):
        assert ri._resolve_stt_engine(alias) == "audiocpp"
    assert ri._stt_repo_reference("audiocpp-canary-180m-flash", "audiocpp") == AUDIO_CPP_REPO
    assert ri._stt_resolved_model_id(lookup("audiocpp-canary-180m-flash").id, "audiocpp") == (
        "audiocpp-canary-180m-flash"
    )


def test_stt_sidecar_rejects_speech_and_unknown_ids():
    from core.inference import stt_audiocpp_sidecar as s
    from core.inference.stt_sidecar import SttModelIdError

    assert s.resolve_audio_cpp_stt_model_id(None) == "audiocpp-qwen3-asr-0.6b"
    with pytest.raises(SttModelIdError):
        s.resolve_audio_cpp_stt_model("audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF")
    with pytest.raises(SttModelIdError):
        s.resolve_audio_cpp_stt_model("small")


def test_model_config_answers_curated_ids_without_the_hub(monkeypatch):
    from utils.models import model_config

    def refuse(*args, **kwargs):
        raise AssertionError("an audio.cpp id must not reach a Hub probe")

    monkeypatch.setattr(model_config, "detect_gguf_model_remote", refuse)
    monkeypatch.setattr(model_config, "is_vision_model", refuse)
    config = model_config.ModelConfig.from_identifier("audio-cpp/audio.cpp-gguf/Supertonic-3-GGUF")
    assert config.identifier == "audio-cpp/audio.cpp-gguf/Supertonic-3-GGUF"
    assert config.is_audio and config.audio_type == "audiocpp_tts"
    assert not config.is_gguf and not config.is_lora and not config.is_vision
    # A dictation id is not a main-slot model.
    assert model_config.ModelConfig._from_audio_cpp_identifier("audiocpp-canary-180m-flash") is None


def test_native_audio_helpers_point_at_the_umbrella_repo():
    from core.inference import native_audio

    tts = "audio-cpp/audio.cpp-gguf/ACE-Step1.5-GGUF/turbo"
    assert native_audio.is_native_audio_model(tts)
    assert native_audio.audio_cpp_audio_type(tts) == "audiocpp_music"
    assert native_audio.native_audio_security_targets(tts) == [AUDIO_CPP_REPO]
    assert "audiocpp_tts" in native_audio.NATIVE_AUDIO_TYPES
    assert "audiocpp_music" not in native_audio.REMOTE_CODE_AUDIO_TYPES


def test_umbrella_repo_is_hidden_from_chat():
    from utils.hidden_models import is_hidden_model
    assert is_hidden_model(AUDIO_CPP_REPO)
    assert is_hidden_model("Audio-CPP/audio.cpp-GGUF")


def test_resident_estimate_scales_with_the_package():
    import routes.inference as ri

    small = ri.audio_cpp_resident_gb(lookup("audiocpp-canary-180m-flash").size_bytes)
    big = ri.audio_cpp_resident_gb(
        lookup("audio-cpp/audio.cpp-gguf/ACE-Step1.5-GGUF/turbo").size_bytes
    )
    assert 1.5 < small < 3 and 15 < big < 18


def test_load_rejects_speech_to_text_ids():
    b = audio_cpp_backend.AudioCppBackend()
    with pytest.raises(RuntimeError, match = "not a curated audio.cpp speech or music model"):
        b.load_model(SimpleNamespace(identifier = "audiocpp-qwen3-asr-0.6b"))


def _dir_link(link, target):
    """A directory symlink, or a Windows junction when this account cannot create symlinks."""
    try:
        os.symlink(target, link, target_is_directory = True)
    except OSError:
        if sys.platform != "win32":
            pytest.skip("this filesystem or account cannot create symlinks")
        import _winapi
        _winapi.CreateJunction(str(target), str(link))


def test_prune_never_follows_a_link_planted_in_the_farm(tmp_path):
    root, _snap = _snapshot(tmp_path)
    farm = audio_cpp_files._link_farm_root(root)
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "notes.txt").write_text("keep", encoding = "utf-8")
    farm.mkdir(parents = True)
    # A top-level link named like no model, and one inside a (not downloaded) model folder.
    _dir_link(farm / "not-a-model", victim)
    (farm / "audiocpp-canary-180m-flash").mkdir()
    _dir_link(farm / "audiocpp-canary-180m-flash" / "sub", victim)
    audio_cpp_files.prune_link_farm(root)
    assert (victim / "notes.txt").read_text(encoding = "utf-8") == "keep"
    assert not (farm / "not-a-model").exists()


def test_materialize_does_not_write_through_a_planted_model_folder_link(tmp_path, monkeypatch):
    root, snap = _snapshot(tmp_path)
    m = lookup("audiocpp-canary-180m-flash")
    (snap / m.gguf_file).parent.mkdir(parents = True)
    (snap / m.gguf_file).write_bytes(b"GGUF")
    farm = audio_cpp_files._link_farm_root(root)
    farm.mkdir(parents = True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _dir_link(farm / m.key, elsewhere)
    monkeypatch.setattr(audio_cpp_files.sys, "platform", "win32")  # always goes through the farm
    served = audio_cpp_files.materialize(m, hub_cache = root)
    assert list(elsewhere.iterdir()) == []
    assert os.path.samefile(served, snap / m.gguf_file)


def test_prune_keeps_the_empty_folders_of_a_downloaded_model(tmp_path):
    # A concurrent materialize makes the folder before it links the file into it.
    root, snap = _snapshot(tmp_path)
    m = lookup("audiocpp-canary-180m-flash")
    (snap / m.gguf_file).parent.mkdir(parents = True)
    (snap / m.gguf_file).write_bytes(b"GGUF")
    in_progress = audio_cpp_files._link_farm_root(root) / m.key
    in_progress.mkdir(parents = True)
    audio_cpp_files.prune_link_farm(root)
    assert in_progress.is_dir()


def test_download_plan_of_a_downloaded_model_needs_no_hub(tmp_path, monkeypatch):
    from core.inference import native_audio

    root, snap = _snapshot(tmp_path)
    m = lookup("audiocpp-canary-180m-flash")
    (snap / m.gguf_file).parent.mkdir(parents = True)
    (snap / m.gguf_file).write_bytes(b"GGUF")
    monkeypatch.setattr(audio_cpp_files, "_hub_cache", lambda: root)

    def offline(*_args, **_kwargs):
        raise OSError("offline")

    monkeypatch.setattr(audio_cpp_files, "expand_repo_files", offline)
    plan = native_audio._audio_cpp_download_plan(m.id, None)
    assert plan["entries"] == [] and plan["total_bytes"] == 0 and plan["required_bytes"] == 4


def test_clearing_the_hub_cache_prunes_the_link_farm(tmp_path, monkeypatch):
    from utils import cache_inventory

    root, snap = _snapshot(tmp_path)
    m = lookup("audiocpp-canary-180m-flash")
    (snap / m.gguf_file).parent.mkdir(parents = True)
    (snap / m.gguf_file).write_bytes(b"GGUF")
    monkeypatch.setattr(audio_cpp_files.sys, "platform", "win32")
    served = audio_cpp_files.materialize(m, hub_cache = root)
    pruned = []
    real = audio_cpp_files.prune_link_farm
    monkeypatch.setattr(
        audio_cpp_files, "prune_link_farm", lambda hub = None: pruned.append(hub) or real(hub)
    )
    monkeypatch.setattr(cache_inventory, "_training_refusal", lambda key: None)
    monkeypatch.setattr(cache_inventory, "_inference_refusal", lambda key: None)
    monkeypatch.setattr(cache_inventory, "_link_mode_refusal", lambda key: None)
    monkeypatch.setattr(cache_inventory, "_reserve_downloads", lambda key: ([], None))
    monkeypatch.setattr(cache_inventory, "_release_downloads", lambda reserved: None)
    monkeypatch.setattr(cache_inventory, "_resolve_roots", lambda definition: [root])

    def empty(hub, **_kwargs):
        import shutil
        shutil.rmtree(hub / ("models--" + AUDIO_CPP_REPO.replace("/", "--")))
        return cache_inventory.PurgeOutcome()

    monkeypatch.setattr(cache_inventory, "empty_cache_root", empty)
    cache_inventory.purge_cache("hf_hub")
    assert pruned == [root]
    assert not os.path.exists(served)


def test_deleting_the_umbrella_repo_prunes_the_farm_of_the_cache_it_was_in(tmp_path, monkeypatch):
    import asyncio

    from hub.services.models import deletion
    from hub.utils import hf_cache_state

    other_root, _snap = _snapshot(tmp_path / "old")  # a remembered, non-active cache
    repo_folder = other_root / ("models--" + AUDIO_CPP_REPO.replace("/", "--"))
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda: [other_root])
    monkeypatch.setattr(deletion.account_access, "require_installation_owner", lambda: None)
    for guard in (
        "_llama_cpp_blocks_delete",
        "_inference_backend_blocks_delete",
        "_audio_cpp_blocks_delete",
    ):
        monkeypatch.setattr(deletion, guard, lambda *a: False)
    for guard in ("_diffusion_blocks_delete", "_video_blocks_delete"):
        monkeypatch.setattr(deletion, guard, lambda *a: None)
    monkeypatch.setattr(deletion, "resolve_cached_repo_id_case", lambda repo_id, repo_type: repo_id)
    monkeypatch.setattr(deletion.downloads.registry, "begin_delete", lambda *a: True)
    monkeypatch.setattr(deletion.downloads.registry, "end_delete", lambda *a: None)
    monkeypatch.setattr(
        deletion, "_delete_cached_model_blocking", lambda *a, **k: {"deleted": True}
    )
    monkeypatch.setattr(deletion.cache_inventory, "invalidate_hf_cache_scans", lambda: None)
    pruned = []
    monkeypatch.setattr(audio_cpp_files, "prune_link_farm", lambda hub = None: pruned.append(hub))
    asyncio.run(deletion.delete_cached_model_response(AUDIO_CPP_REPO, cache_path = str(repo_folder)))
    # The Hub row sends the repo folder; the farm sits beside that folder's hub root.
    assert pruned == [other_root.resolve()]
