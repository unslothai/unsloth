# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""POST /audio/transcribe/source with stub engines; the account boundary is real."""

from __future__ import annotations

import asyncio
import io
import json
import sys
import wave
from pathlib import Path

import pytest

from core.inference import audio_gallery, audio_inputs, transcript_gallery
from core.inference.audio_cpp_models import AUDIO_CPP_REPO as REPO
from routes import inference
from utils.account_context import run_as

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_account_media_isolation import ALICE, _client, isolated  # noqa: E402, F401
from test_audio_cpp_models import hub  # noqa: E402, F401
from test_audio_inputs import _chunks, encode, wav_bytes  # noqa: E402

MOSS = f"{REPO}/MOSS-Transcribe-Diarize-GGUF"
QWEN3 = f"{REPO}/Qwen3-ASR-0.6B-GGUF"
SEGMENTS = [
    {"start": 0.12, "end": 1.0, "text": "Hello there.", "speaker": "S01"},
    {"start": 1.1, "end": 2.0, "text": "General Kenobi.", "speaker": "S02"},
]


class _Sidecar:
    loaded_model = None

    def __init__(self):
        self.result = {"text": "Hello there. General Kenobi.", "language": None, "duration": 2.0}
        self.result.update(segments = SEGMENTS, speakers = ["S01", "S02"])
        self.paths, self.bytes, self.events, self.loads = [], [], [], []

    def ensure_aligner(self, model, on_phase):
        self.events.append(("aligner", model))

    def transcribe_path(self, path, model, language, *, timestamps, cancel_event, on_phase):
        on_phase("transcribing")
        with wave.open(str(path)) as w:
            self.paths.append((Path(path), w.getframerate(), timestamps, language))
        return {**self.result, "model": model}

    def transcribe(self, raw, model, language, *_args, **_kwargs):
        self.bytes.append(raw)
        return {"text": "plain words", "language": language, "duration": 1.0, "model": model}

    def cancel_transcription(self, cancel_event):
        cancel_event.set()
        return True


@pytest.fixture
def stub(monkeypatch, hub):
    sidecar = _Sidecar()

    def load(
        model,
        engine,
        *_args,
        timestamps = False,
        on_phase = None,
        **_kwargs,
    ):
        sidecar.loads.append((model, engine))
        sidecar.events.append(("load", timestamps))
        if on_phase is not None:
            on_phase("loading")

    monkeypatch.setattr(inference, "_stt_lifecycle", lambda: (load, lambda *a, **k: []))
    monkeypatch.setattr(inference, "_stt_sidecar_for", lambda engine: sidecar)
    monkeypatch.setattr(inference.account_access, "require_model_access", lambda *a, **k: None)
    return sidecar


def _input(account) -> str:
    data = encode("webm", "libopus", 48000, "stereo", 1.0)
    save = audio_inputs.save_stream(_chunks([data]), "meeting.webm")
    return run_as(account, lambda: asyncio.run(save)[0]["id"])


def _post(client, **body):
    body = {"model": MOSS, "engine": "audiocpp", **body}
    response = client.post("/api/inference/audio/transcribe/source", json = body)
    return response, [json.loads(line) for line in response.text.splitlines() if line.strip()]


@pytest.mark.parametrize(
    "body",
    [
        {"source": {"input_id": "a" * 32}, "path": "/etc/passwd"},
        {"source": {"path": "/etc/passwd"}},
        {"source": {"input_id": "a" * 32, "clip_id": "b" * 32}},
        {"source": {}},
        {"source": {"input_id": "../../etc/passwd"}},
        {"source": {"input_id": "a" * 32}, "title": "x" * 256},
        {"source": {"input_id": "a" * 32}, "device": "tpu"},
    ],
)
def test_client_paths_and_unknown_fields_are_422(stub, body):
    with _client(ALICE) as client:
        assert _post(client, **body)[0].status_code == 422
    assert not stub.paths and not stub.bytes


def test_audio_cpp_reads_a_16k_copy_in_the_account_and_the_record_names_the_source(stub, tmp_path):
    input_id = _input(ALICE)
    with _client(ALICE) as client:
        response, events = _post(
            client, source = {"input_id": input_id}, speakers = True, language = "en"
        )
        ((path, rate, timestamps, language),) = stub.paths
        assert path.parent == tmp_path / "accounts" / ALICE.account_id / "audio" / "inputs"
        assert path.name.startswith(f"{input_id}.16000.mono") and rate == 16000
        assert (timestamps, language) == (False, "en") and not stub.bytes
        assert [e["phase"] for e in events if "phase" in e] == ["loading", "transcribing"]
        complete = events[-1]
        labels = [{"id": "S01", "label": "Speaker 1"}, {"id": "S02", "label": "Speaker 2"}]
        assert (complete["speakers"], complete["segments"]) == (labels, SEGMENTS)
        assert complete["source"] == {"kind": "input", "id": input_id, "name": "meeting.webm"}
        assert complete["timestamps"] is True
        record = complete["record"]
        assert record["title"] == "meeting.webm" and record["source"] == complete["source"]
        saved = client.get(f"/api/inference/audio/transcripts/{record['id']}").json()
        assert (saved["segments"], saved["speakers"]) == (SEGMENTS, labels)
    stored = run_as(
        ALICE, lambda: (transcript_gallery.gallery_dir() / f"{record['id']}.json").read_text()
    )
    for text in (response.text, stored):
        assert str(tmp_path) not in text and "/inputs/" not in text


def test_vibevoice_sources_are_prepared_at_24k_and_speakers_off_strips_them(stub):
    meta = {"prompt": "p", "model": "m", "audio_type": "audiocpp_tts", "sample_rate": 24000}
    meta.update(duration_s = 1.0, created_at = "2026-10-02T00:00:00Z")
    clip_id = run_as(ALICE, audio_gallery.save, wav_bytes(1.0, 24000), meta)["id"]
    with _client(ALICE) as client:
        _, events = _post(client, model = f"{REPO}/VibeVoice-ASR-GGUF", source = {"clip_id": clip_id})
    ((path, rate, _, _),) = stub.paths
    assert rate == 24000 and path.name.startswith(f"c-{clip_id}.24000.mono")
    complete = events[-1]
    assert "speakers" not in complete and "speakers" not in complete["record"]
    assert all("speaker" not in s for s in complete["segments"])


def test_qwen3_timestamps_are_asked_of_the_sidecar(stub):
    stub.result = {"text": "Concord returned.", "language": "English", "duration": 3.5}
    source = {"input_id": _input(ALICE)}
    with _client(ALICE) as client:
        complete = _post(client, model = QWEN3, source = source, timestamps = True)[1][-1]
    assert stub.paths[0][2] is True
    # The aligner is fetched first and the server starts with it: one load, not two.
    assert stub.events == [("aligner", QWEN3), ("load", True)]
    assert complete["timestamps"] is False and "segments" not in complete


def test_another_engine_gets_the_prepared_bytes_past_the_encoded_upload_cap(stub, monkeypatch):
    # 30 minutes of 16 kHz PCM is ~58 MB, past the 25 MB cap for encoded uploads.
    monkeypatch.setattr(inference, "_MAX_AUDIO_RAW_BYTES", 1024)
    source = {"input_id": _input(ALICE)}
    with _client(ALICE) as client:
        response, events = _post(client, model = "small", engine = "transformers", source = source)
    assert response.status_code == 200, response.text
    (raw,) = stub.bytes
    assert len(raw) > 1024 and not stub.paths and stub.loads == [("small", "transformers")]
    with wave.open(io.BytesIO(raw)) as w:
        assert (w.getframerate(), w.getnchannels()) == (16000, 1)
    assert (events[-1]["text"], events[-1]["timestamps"]) == ("plain words", False)


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
    body = {"model": model, "engine": engine, "source": {"input_id": _input(ALICE)}, flag: True}
    with _client(ALICE) as client:
        response, _ = _post(client, **body)
    assert response.status_code == 422 and not stub.paths and not stub.bytes
    expected = "cannot add timestamps" if flag == "timestamps" else "cannot tell speakers apart"
    assert expected in response.json()["detail"]


def test_capabilities_route_answers_for_any_model(stub):
    url = "/api/inference/audio/stt/capabilities"
    with _client(ALICE) as client:
        assert client.get(url, params = {"model": MOSS}).json()["speakers"] is True
        unknown = client.get(url, params = {"model": "who/knows", "engine": "bogus"})
    assert unknown.status_code == 200 and unknown.json()["timestamps"] == "unsupported"


def test_transcript_routes_get_rename_and_archive(stub):
    with _client(ALICE) as client:
        record = _post(client, source = {"input_id": _input(ALICE)}, speakers = True)[1][-1]["record"]
        url = f"/api/inference/audio/transcripts/{record['id']}"
        renamed = client.patch(url, json = {"speaker_names": {"S01": "Alice"}}).json()
        assert renamed["speaker_names"] == {"S01": "Alice"} and "segments" not in renamed
        assert client.get(url).json()["speaker_names"] == {"S01": "Alice"}
        for bad in ({"S09": "Ghost"}, {"S01": "x" * 41}):
            assert client.patch(url, json = {"speaker_names": bad}).status_code == 422
        for bad in ({"pinned": True}, {}):
            assert client.patch(url, json = bad).status_code == 422
        cleared = client.patch(url, json = {"speaker_names": {"S01": None}, "archived": True}).json()
        assert "speaker_names" not in cleared and cleared["archived"] is True
        assert client.get("/api/inference/audio/transcripts/" + "e" * 32).status_code == 404
