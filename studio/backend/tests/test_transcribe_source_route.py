# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""POST /audio/transcribe/source with stub engines; the account boundary is real."""

from __future__ import annotations

import asyncio
import json
import sys
import wave
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import policy, storage as auth_storage
from auth.authentication import authenticated_via_api_key, get_current_subject
from core.inference import (
    audio_cpp_files,
    audio_gallery,
    audio_inputs,
    transcript_gallery,
)
from core.inference import audio_cpp_models as acm
from routes import inference
from utils.account_context import AccountContext, bind_account, reset_account, run_as

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_audio_inputs import _chunks, encode, wav_bytes  # noqa: E402

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")
REPO = acm.AUDIO_CPP_REPO
MOSS = f"{REPO}/MOSS-Transcribe-Diarize-GGUF"
QWEN3 = f"{REPO}/Qwen3-ASR-0.6B-GGUF"
VIBEVOICE = f"{REPO}/VibeVoice-ASR-GGUF"


@pytest.fixture(autouse = True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(auth_storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(auth_storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(auth_storage, "_bootstrap_password", None)
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setattr(acm, "_hub_cache", lambda: hub)
    monkeypatch.setattr(audio_cpp_files, "_hub_cache", lambda: hub)
    monkeypatch.setattr(acm, "runtime_spec", lambda family: None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    acm.forget()
    policy.invalidate_account_cache()
    connection = auth_storage.get_connection()
    with connection:
        for account in (ALICE, BOB):
            connection.execute(
                "INSERT INTO auth_user (username, password_salt, password_hash, jwt_secret,"
                " account_id, role, is_active) VALUES (?, 'salt', 'hash', 'secret', ?, 'user', 1)",
                (account.username, account.account_id),
            )
    connection.close()
    yield
    policy.invalidate_account_cache()
    acm.forget()


class _Sidecar:
    loaded_model = None

    def __init__(self, result):
        self.result = result
        self.paths: list[dict] = []
        self.bytes: list[bytes] = []
        self.events: list[tuple] = []

    def needs_reload_for(self, model, timestamps):
        return True

    def ensure_aligner(self, model, on_phase):
        self.events.append(("aligner", model))

    def transcribe_path(self, path, model, language, *, timestamps, cancel_event, on_phase):
        on_phase("transcribing")
        with wave.open(str(path)) as w:
            rate = w.getframerate()
        self.paths.append(
            {
                "path": Path(path),
                "rate": rate,
                "timestamps": timestamps,
                "language": language,
            }
        )
        return {**self.result, "model": model}

    def transcribe(
        self,
        raw,
        model,
        language,
        fast,
        cancel_event = None,
        on_progress = None,
    ):
        self.bytes.append(raw)
        return {
            "text": "plain words",
            "language": language,
            "duration": 1.0,
            "model": model,
        }

    def cancel_transcription(self, cancel_event):
        cancel_event.set()
        return True


MOSS_RESULT = {
    "text": "Hello there. General Kenobi.",
    "language": None,
    "duration": 2.0,
    "segments": [
        {"start": 0.12, "end": 1.0, "text": "Hello there.", "speaker": "S01"},
        {"start": 1.1, "end": 2.0, "text": "General Kenobi.", "speaker": "S02"},
    ],
    "speakers": ["S01", "S02"],
}


@pytest.fixture
def stub(monkeypatch):
    sidecar = _Sidecar(MOSS_RESULT)
    loads = []

    def load(
        model,
        engine,
        cancel_event = None,
        device = None,
        timestamps = False,
    ):
        loads.append((model, engine, device))
        sidecar.events.append(("load", timestamps))

    monkeypatch.setattr(inference, "_stt_lifecycle", lambda: (load, lambda *a, **k: []))
    monkeypatch.setattr(inference, "_stt_sidecar_for", lambda engine: sidecar)
    monkeypatch.setattr(inference.account_access, "require_model_access", lambda *a, **k: None)
    sidecar.loads = loads
    return sidecar


def _client(account):
    app = FastAPI()

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[get_current_subject] = subject
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    app.include_router(inference.studio_router, prefix = "/api/inference")
    return TestClient(app)


def _input(account, seconds = 1.0) -> str:
    data = encode("webm", "libopus", 48000, "stereo", seconds)

    def save():
        record, _ = asyncio.run(audio_inputs.save_stream(_chunks([data]), "meeting.webm"))
        return record["id"]

    return run_as(account, save)


def _clip(account) -> dict:
    meta = {
        "prompt": "A history clip",
        "model": "m",
        "audio_type": "audiocpp_tts",
        "sample_rate": 24000,
        "duration_s": 1.0,
        "created_at": "2026-10-02T00:00:00Z",
    }
    return run_as(account, audio_gallery.save, wav_bytes(1.0, 24000), meta)


def _events(response):
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def _post(client, **body):
    payload = {"model": MOSS, "engine": "audiocpp", **body}
    return client.post("/api/inference/audio/transcribe/source", json = payload)


@pytest.mark.parametrize(
    "body",
    [
        {"source": {"input_id": "a" * 32}, "path": "/etc/passwd"},
        {"source": {"input_id": "a" * 32}, "file": "x.wav"},
        {"source": {"input_id": "a" * 32}, "url": "http://x"},
        {"source": {"path": "/etc/passwd"}},
        {"source": {"input_id": "a" * 32, "audio": "/etc/passwd"}},
        {"source": {"input_id": "a" * 32, "trim": {"start_s": 1}}},
        {"source": {"input_id": "a" * 32, "clip_id": "b" * 32}},
        {"source": {}},
        {"source": {"input_id": "../../etc/passwd"}},
        {"source": {"input_id": "a" * 32}, "title": "x" * 256},
        {"source": {"input_id": "a" * 32}, "device": "tpu"},
    ],
)
def test_client_paths_and_unknown_fields_are_422(stub, body):
    with _client(ALICE) as client:
        response = _post(client, **body)
    assert response.status_code == 422, response.text
    assert not stub.paths and not stub.bytes


def test_another_accounts_input_or_clip_is_404(stub):
    input_id = _input(ALICE)
    clip = _clip(ALICE)
    with _client(BOB) as client:
        for source in (
            {"input_id": input_id},
            {"clip_id": clip["id"]},
            {"voice_id": "c" * 32},
        ):
            response = _post(client, source = source)
            assert response.status_code == 404, response.text
    assert not stub.paths and not stub.bytes


def test_audio_cpp_reads_a_16k_copy_in_the_account_and_the_record_names_the_source(stub, tmp_path):
    input_id = _input(ALICE)
    with _client(ALICE) as client:
        response = _post(client, source = {"input_id": input_id}, speakers = True, language = "en")
        assert response.status_code == 200, response.text
        events = _events(response)
        (call,) = stub.paths
        inputs_root = tmp_path / "accounts" / ALICE.account_id / "audio" / "inputs"
        assert call["path"].parent == inputs_root and call["rate"] == 16000
        assert call["path"].name.startswith(f"{input_id}.16000.mono")
        assert (call["timestamps"], call["language"]) == (False, "en")
        assert not stub.bytes
        complete = events[-1]
        assert complete["type"] == "complete"
        assert {"type": "progress", "text": "", "phase": "loading"} in events
        assert {"type": "progress", "text": "", "phase": "transcribing"} in events
        assert complete["speakers"] == [
            {"id": "S01", "label": "Speaker 1"},
            {"id": "S02", "label": "Speaker 2"},
        ]
        assert [s["speaker"] for s in complete["segments"]] == ["S01", "S02"]
        assert complete["source"] == {"kind": "input", "id": input_id, "name": "meeting.webm"}
        assert complete["timestamps"] is True
        record = complete["record"]
        assert record["title"] == "meeting.webm" and record["source"] == complete["source"]
        saved = client.get(f"/api/inference/audio/transcripts/{record['id']}").json()
        assert (
            saved["segments"] == complete["segments"] and saved["speakers"] == complete["speakers"]
        )
        # No server path anywhere the client or history can see.
        stored = run_as(
            ALICE,
            lambda: (transcript_gallery.gallery_dir() / f"{record['id']}.json").read_text(),
        )
        for text in (response.text, stored):
            assert str(tmp_path) not in text and "/inputs/" not in text


def test_vibevoice_sources_are_prepared_at_24k(stub):
    clip = _clip(ALICE)
    with _client(ALICE) as client:
        response = _post(client, model = VIBEVOICE, source = {"clip_id": clip["id"]})
        assert response.status_code == 200, response.text
        assert _events(response)[-1]["source"]["kind"] == "clip"
    (call,) = stub.paths
    assert call["rate"] == 24000 and call["path"].name.startswith(f"c-{clip['id']}.24000.mono")


def test_speakers_off_strips_them(stub):
    input_id = _input(ALICE)
    with _client(ALICE) as client:
        complete = _events(_post(client, source = {"input_id": input_id}))[-1]
    assert "speakers" not in complete
    assert all("speaker" not in s for s in complete["segments"])
    assert "speakers" not in complete["record"]


def test_qwen3_timestamps_are_asked_of_the_sidecar(stub):
    stub.result = {"text": "Concord returned.", "language": "English", "duration": 3.5}
    input_id = _input(ALICE)
    with _client(ALICE) as client:
        complete = _events(
            _post(client, model = QWEN3, source = {"input_id": input_id}, timestamps = True)
        )[-1]
    assert stub.paths[0]["timestamps"] is True
    # The aligner is fetched first and the server starts with it: one load, not two.
    assert stub.events == [("aligner", QWEN3), ("load", True)]
    assert complete["timestamps"] is False and "segments" not in complete


def test_another_engine_gets_the_prepared_bytes(stub):
    input_id = _input(ALICE)
    with _client(ALICE) as client:
        response = _post(
            client, model = "small", engine = "transformers", source = {"input_id": input_id}
        )
        assert response.status_code == 200, response.text
        complete = _events(response)[-1]
    (raw,) = stub.bytes
    assert raw[:4] == b"RIFF" and not stub.paths
    with wave.open(__import__("io").BytesIO(raw)) as w:
        assert (w.getframerate(), w.getnchannels()) == (16000, 1)
    assert complete["text"] == "plain words" and complete["timestamps"] is False
    assert "segments" not in complete and complete["source"]["id"] == input_id
    assert stub.loads == [("small", "transformers", None)]


def test_a_prepared_source_is_not_held_to_the_encoded_upload_cap(stub, monkeypatch):
    # 30 minutes of 16 kHz PCM is ~58 MB, past the 25 MB cap for encoded uploads.
    monkeypatch.setattr(inference, "_MAX_AUDIO_RAW_BYTES", 1024)
    input_id = _input(ALICE)
    with _client(ALICE) as client:
        response = _post(
            client, model = "small", engine = "transformers", source = {"input_id": input_id}
        )
        assert response.status_code == 200, response.text
        assert _events(response)[-1]["text"] == "plain words"
    assert len(stub.bytes[0]) > 1024


@pytest.mark.parametrize(
    "model,engine,flag",
    [
        ("small", "transformers", "timestamps"),
        ("qwen3-asr-0.6b", "mtmd", "timestamps"),
        (f"{REPO}/Nemotron-3.5-ASR-Streaming-0.6B-GGUF", "audiocpp", "timestamps"),
        (QWEN3, "audiocpp", "speakers"),
        ("small", "transformers", "speakers"),
    ],
)
def test_asking_an_unsupported_model_for_timestamps_or_speakers_is_422(stub, model, engine, flag):
    input_id = _input(ALICE)
    with _client(ALICE) as client:
        response = _post(
            client,
            model = model,
            engine = engine,
            source = {"input_id": input_id},
            **{flag: True},
        )
    assert response.status_code == 422
    expected = "cannot add timestamps" if flag == "timestamps" else "cannot tell speakers apart"
    assert expected in response.json()["detail"]
    assert not stub.paths and not stub.bytes


def test_capabilities_route_answers_for_any_model(stub):
    with _client(ALICE) as client:
        moss = client.get("/api/inference/audio/stt/capabilities", params = {"model": MOSS}).json()
        unknown = client.get(
            "/api/inference/audio/stt/capabilities",
            params = {"model": "who/knows", "engine": "bogus"},
        )
    assert moss == {
        "engine": "audiocpp",
        "family": "moss_transcribe_diarize",
        "timestamps": "always",
        "speakers": True,
        "aligner": None,
        "cpu_only": False,
    }
    assert unknown.status_code == 200 and unknown.json()["timestamps"] == "unsupported"


def test_transcript_routes_get_rename_and_archive(stub):
    input_id = _input(ALICE)
    with _client(ALICE) as client:
        record = _events(_post(client, source = {"input_id": input_id}, speakers = True))[-1]["record"]
        url = f"/api/inference/audio/transcripts/{record['id']}"
        (row,) = client.get("/api/inference/audio/transcripts").json()["transcripts"]
        assert "segments" not in row and row["segment_count"] == 2 and row["has_words"] is False
        renamed = client.patch(url, json = {"speaker_names": {"S01": "Alice"}})
        assert renamed.status_code == 200
        assert (
            renamed.json()["speaker_names"] == {"S01": "Alice"} and "segments" not in renamed.json()
        )
        assert client.get(url).json()["speaker_names"] == {"S01": "Alice"}
        assert client.patch(url, json = {"speaker_names": {"S09": "Ghost"}}).status_code == 422
        assert client.patch(url, json = {"speaker_names": {"S01": "x" * 41}}).status_code == 422
        assert client.patch(url, json = {"pinned": True}).status_code == 422
        missing = client.patch(url, json = {})
        assert missing.status_code == 422
        assert missing.json()["detail"] == "Specify whether to archive this transcript."
        cleared = client.patch(url, json = {"speaker_names": {"S01": None}, "archived": True}).json()
        assert "speaker_names" not in cleared and cleared["archived"] is True
        assert client.get("/api/inference/audio/transcripts/" + "e" * 32).status_code == 404
