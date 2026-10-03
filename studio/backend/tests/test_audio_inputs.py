# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Audio page inputs: streamed upload, PyAV decode, caps, dedup, expiry and prepared copies.

Every clip here is encoded with PyAV inside the test (a browser's webm/opus recording, an mp3,
a wav), so the decode path is the one Studio ships, not a fixture file.
"""

from __future__ import annotations

import asyncio
import io
import json
import math
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import core.inference.audio_gallery as gallery
from auth.authentication import get_current_subject
from core.inference import audio_inputs


@pytest.fixture(autouse = True)
def _tmp_gallery(monkeypatch, tmp_path):
    monkeypatch.setattr(gallery, "studio_root", lambda: tmp_path)


def encode(
    fmt: str,
    codec: str,
    rate: int,
    layout: str,
    seconds: float,
    freq: float = 440.0,
) -> bytes:
    """A sine tone encoded with PyAV into ``fmt``/``codec``."""
    import av

    channels = 2 if layout == "stereo" else 1
    count = int(round(rate * seconds))
    t = np.arange(count) / rate
    tone = (0.3 * np.sin(2 * math.pi * freq * t) * 32767).astype(np.int16)
    packed = np.repeat(tone, channels).reshape(1, -1)
    buf = io.BytesIO()
    with av.open(buf, mode = "w", format = fmt) as out:
        stream = out.add_stream(codec, rate = rate, layout = layout)
        # Fed in 20 ms frames, as a recorder delivers them.
        step = rate // 50
        for start in range(0, count, step):
            chunk = packed[:, start * channels : min(count, start + step) * channels]
            frame = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(chunk), format = "s16", layout = layout
            )
            frame.sample_rate = rate
            frame.pts = start
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    return buf.getvalue()


def wav_bytes(
    rate = 16000,
    seconds = 1.0,
    channels = 1,
) -> bytes:
    return encode("wav", "pcm_s16le", rate, "stereo" if channels == 2 else "mono", seconds)


def _client() -> TestClient:
    from routes import inference

    app = FastAPI()
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    app.include_router(inference.studio_router, prefix = "/api/inference")
    return TestClient(app)


def _upload(
    client,
    data: bytes,
    name = "clip.webm",
):
    return client.post(
        "/api/inference/audio/inputs",
        params = {"name": name},
        content = data,
        headers = {"Content-Type": "application/octet-stream"},
    )


@pytest.mark.parametrize(
    "fmt, codec, rate, layout, name",
    [
        ("webm", "libopus", 48000, "stereo", "recording.webm"),
        ("mp3", "libmp3lame", 44100, "mono", "voice.mp3"),
        ("wav", "pcm_s16le", 16000, "mono", "voice.wav"),
    ],
)
def test_an_upload_decodes_to_a_canonical_wav(fmt, codec, rate, layout, name):
    data = encode(fmt, codec, rate, layout, 2.0)
    with _client() as client:
        response = _upload(client, data, f"C:\\Users\\me\\{name}")
        assert response.status_code == 201, response.text
        record = response.json()
        assert set(record) == {
            "id",
            "name",
            "duration_s",
            "sample_rate",
            "channels",
            "url",
            "expires_at",
        }
        # Only the file's own name is kept, never the client's folders.
        assert record["name"] == name
        assert record["sample_rate"] == rate
        assert record["channels"] == (2 if layout == "stereo" else 1)
        assert abs(record["duration_s"] - 2.0) < 0.1
        assert record["url"] == f"/api/inference/audio/inputs/{record['id']}/file"
        served = client.get(record["url"])
        assert served.status_code == 200 and served.headers["content-type"] == "audio/wav"
        with wave.open(io.BytesIO(served.content)) as w:
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (
                rate,
                record["channels"],
                2,
            )
    directory = audio_inputs.inputs_dir()
    sidecar = json.loads((directory / f"{record['id']}.json").read_text(encoding = "utf-8"))
    assert len(sidecar["sha256"]) == 64 and sidecar["bytes"] == len(data)
    assert directory == gallery.gallery_dir() / "inputs"
    # No partial upload is left behind.
    assert not [p for p in directory.iterdir() if p.name.endswith(".tmp")]


def test_more_than_two_channels_keep_two():
    data = encode("wav", "pcm_s16le", 24000, "5.1", 0.5)
    record, created = asyncio.run(audio_inputs.save_stream(_chunks([data]), "surround.wav"))
    assert created and record["channels"] == 2


async def _chunks(parts, seen = None):
    for part in parts:
        if seen is not None:
            seen.append(len(part))
        yield part


def test_an_oversize_body_is_refused_while_streaming():
    seen: list[int] = []
    parts = [b"\x00" * 1024] * 100
    with pytest.raises(audio_inputs.AudioInputError) as refused:
        asyncio.run(audio_inputs.save_stream(_chunks(parts, seen), "big.wav", max_bytes = 4096))
    assert (refused.value.status, refused.value.detail) == (413, "Audio is too large.")
    # Refused at the chunk that crossed the cap: the rest of the body was never read.
    assert len(seen) == 5
    assert not list(audio_inputs.inputs_dir().iterdir())


def test_the_route_refuses_an_oversize_upload_with_413(monkeypatch):
    import utils.upload_limits as limits

    monkeypatch.setattr(limits, "AUDIO_INPUT_MAX_BYTES", 2048)
    monkeypatch.setattr(audio_inputs, "AUDIO_INPUT_MAX_BYTES", 2048)
    with _client() as client:
        response = _upload(client, wav_bytes(seconds = 1.0))
    assert response.status_code == 413 and response.json()["detail"] == "Audio is too large."


def test_the_body_middleware_passes_the_upload_through_at_its_own_cap():
    import main
    from utils.upload_limits import AUDIO_INPUT_MAX_BYTES

    assert AUDIO_INPUT_MAX_BYTES == 200 * 1024 * 1024
    assert "/api/inference/audio/inputs" in main._BODY_UPLOAD_PASSTHROUGH_EXACT_PATHS
    for path in ("/api/inference/audio/inputs", "/api/inference/audio/inputs/"):
        assert main._get_upload_passthrough_request_max_bytes(path) == AUDIO_INPUT_MAX_BYTES
    # The JSON sub-routes keep the ordinary cap.
    assert (
        "/api/inference/audio/inputs/x/transcribe" not in main._BODY_UPLOAD_PASSTHROUGH_EXACT_PATHS
    )


def test_audio_longer_than_the_cap_is_refused(monkeypatch):
    monkeypatch.setattr(audio_inputs, "MAX_SECONDS", 1)
    with _client() as client:
        response = _upload(client, wav_bytes(seconds = 2.0), "long.wav")
    assert response.status_code == 413
    assert response.json()["detail"] == "Audio is longer than 0 minutes."
    assert not list(audio_inputs.inputs_dir().glob("*.wav"))


@pytest.mark.parametrize(
    "data, detail",
    [
        (b"", "Audio is empty."),
        (b"this is not audio at all" * 50, "This file is not audio Studio can read."),
    ],
)
def test_non_audio_is_a_400(data, detail):
    with _client() as client:
        response = _upload(client, data, "notes.txt")
    assert response.status_code == 400 and response.json()["detail"] == detail


def test_a_video_without_audio_has_no_audio_stream(tmp_path):
    import av

    buf = io.BytesIO()
    with av.open(buf, mode = "w", format = "matroska") as out:
        stream = out.add_stream("ffv1", rate = 1)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), dtype = np.uint8), format = "rgb24")
        for packet in stream.encode(frame):
            out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    with _client() as client:
        response = _upload(client, buf.getvalue(), "silent.mkv")
    assert response.status_code == 400
    assert response.json()["detail"] == "This file has no audio stream."


def test_the_same_audio_twice_returns_the_existing_record():
    data = wav_bytes(seconds = 0.5)
    with _client() as client:
        first = _upload(client, data, "a.wav")
        second = _upload(client, data, "b.wav")
    assert (first.status_code, second.status_code) == (201, 200)
    assert second.json()["id"] == first.json()["id"] and second.json()["name"] == "a.wav"
    assert len(list(audio_inputs.inputs_dir().glob("*.json"))) == 1


def test_an_input_expires_after_the_ttl_and_the_sweep_removes_it(monkeypatch):
    with _client() as client:
        record = _upload(client, wav_bytes(seconds = 0.5)).json()
        prepared = audio_inputs.prepared_path(record["id"])
        assert prepared.is_file()
        later = audio_inputs._now() + audio_inputs.TTL_SECONDS + 1
        monkeypatch.setattr(audio_inputs, "_now", lambda: later)
        assert client.get(record["url"]).status_code == 404
        assert audio_inputs.input_path(record["id"]) is None
        assert audio_inputs.sweep() == 1
    assert not list(audio_inputs.inputs_dir().glob(f"{record['id']}*"))


def test_the_sweep_drops_the_oldest_inputs_past_the_byte_cap(monkeypatch):
    clock = [1_000_000.0]
    monkeypatch.setattr(audio_inputs, "_now", lambda: clock[0])
    ids = []
    for freq in (300.0, 500.0, 700.0):
        data = encode("wav", "pcm_s16le", 16000, "mono", 0.5, freq)
        record, _ = asyncio.run(audio_inputs.save_stream(_chunks([data]), f"{freq}.wav"))
        ids.append(record["id"])
        clock[0] += 10
    one = (audio_inputs.inputs_dir() / f"{ids[0]}.wav").stat().st_size
    assert audio_inputs.sweep(byte_cap = int(one * 2.5)) == 1
    assert audio_inputs.input_path(ids[0]) is None
    assert (
        audio_inputs.input_path(ids[1]) is not None and audio_inputs.input_path(ids[2]) is not None
    )


def test_a_prepared_reference_is_24k_mono_with_the_right_frame_count():
    data = encode("webm", "libopus", 48000, "stereo", 3.0)
    record, _ = asyncio.run(audio_inputs.save_stream(_chunks([data]), "rec.webm"))
    source = audio_inputs.inputs_dir() / f"{record['id']}.wav"
    with wave.open(str(source)) as w:
        source_seconds = w.getnframes() / w.getframerate()
    path = audio_inputs.prepared_path(record["id"])
    info = audio_inputs.wav_info(path)
    assert (info["sample_rate"], info["channels"]) == (24000, 1)
    # Within one frame per second of the source.
    assert abs(info["frames"] - source_seconds * 24000) <= math.ceil(source_seconds)
    assert path.parent == audio_inputs.inputs_dir() and path.name.startswith(record["id"])
    # Cached: asking again reuses the file.
    mtime = path.stat().st_mtime_ns
    assert audio_inputs.prepared_path(record["id"]).stat().st_mtime_ns == mtime
    trimmed = audio_inputs.prepared_path(record["id"], trim = {"start_s": 0.5, "end_s": 2.0})
    assert audio_inputs.wav_info(trimmed)["frames"] == 36000
    capped = audio_inputs.prepared_path(record["id"], max_seconds = 1.0)
    assert audio_inputs.wav_info(capped)["frames"] == 24000
    with pytest.raises(audio_inputs.AudioInputError):
        audio_inputs.prepared_path(record["id"], trim = {"start_s": 5.0, "end_s": 6.0})


def test_a_reference_is_cut_to_thirty_seconds(monkeypatch):
    record, _ = asyncio.run(
        audio_inputs.save_stream(_chunks([wav_bytes(rate = 8000, seconds = 31.0)]), "long.wav")
    )
    source, path = audio_inputs.prepare_reference({"input_id": record["id"]})
    assert source.kind == "input" and source.name == "long.wav"
    assert audio_inputs.wav_info(path)["frames"] == 30 * 24000
    _source, trimmed = audio_inputs.prepare_reference(
        {"input_id": record["id"], "trim": {"start_s": 20.0, "end_s": None}}
    )
    assert audio_inputs.wav_info(trimmed)["frames"] == 11 * 24000


@pytest.mark.parametrize("bad", ["../escape", "a/b", "..", "x" * 129, "name.wav", ""])
def test_unsafe_ids_resolve_to_nothing(bad):
    assert audio_inputs.input_path(bad) is None
    assert audio_inputs.delete(bad) is False
    with pytest.raises(audio_inputs.AudioInputError) as refused:
        audio_inputs.resolve_source({"input_id": bad} if bad else {"input_id": None})
    assert refused.value.status in (400, 404)


def test_unsafe_ids_are_404_on_the_routes():
    with _client() as client:
        for bad in ("..%2Fescape", "a%2Fb", "x" * 129):
            assert client.get(f"/api/inference/audio/inputs/{bad}/file").status_code == 404
            assert client.delete(f"/api/inference/audio/inputs/{bad}").status_code == 404


def test_delete_removes_the_input_and_its_prepared_copies():
    with _client() as client:
        record = _upload(client, wav_bytes(seconds = 0.5)).json()
        audio_inputs.prepared_path(record["id"])
        assert client.delete(f"/api/inference/audio/inputs/{record['id']}").json() == {
            "removed": True
        }
        assert client.get(record["url"]).status_code == 404
        assert client.delete(f"/api/inference/audio/inputs/{record['id']}").status_code == 404
    assert not list(audio_inputs.inputs_dir().glob(f"{record['id']}*"))


def test_transcribe_sends_a_16k_mono_copy_and_saves_nothing(monkeypatch):
    from routes import inference

    calls = []

    async def fake_transcribe(
        raw,
        model,
        language,
        fast,
        engine = None,
        request = None,
        device = None,
        on_progress = None,
    ):
        with wave.open(io.BytesIO(raw)) as w:
            calls.append((w.getframerate(), w.getnchannels(), model, language, engine, device))
        return {"text": " Okay, I'm Cemo. ", "language": "English"}

    monkeypatch.setattr(inference, "_transcribe_audio_result", fake_transcribe)
    with _client() as client:
        record = _upload(client, encode("webm", "libopus", 48000, "stereo", 1.0)).json()
        response = client.post(
            f"/api/inference/audio/inputs/{record['id']}/transcribe",
            json = {"model": "audio-cpp/audio.cpp-gguf/Qwen3-ASR-0.6B-GGUF", "device": "cpu"},
        )
        assert response.status_code == 200, response.text
        assert response.json() == {
            "text": "Okay, I'm Cemo.",
            "language": "English",
            "model": "audio-cpp/audio.cpp-gguf/Qwen3-ASR-0.6B-GGUF",
        }
        assert calls == [
            (16000, 1, "audio-cpp/audio.cpp-gguf/Qwen3-ASR-0.6B-GGUF", None, None, "cpu")
        ]
        # A history clip by query, with the path id "source".
        clip = gallery.save(
            wav_bytes(),
            {
                "prompt": "p",
                "model": "m",
                "audio_type": "snac",
                "sample_rate": 16000,
                "duration_s": 1.0,
                "created_at": "2026-10-02T00:00:00Z",
            },
        )
        by_clip = client.post(
            "/api/inference/audio/inputs/source/transcribe",
            params = {"clip_id": clip["id"]},
            json = {"model": "m"},
        )
        assert by_clip.status_code == 200
        # Two sources, or none, are refused.
        both = client.post(
            f"/api/inference/audio/inputs/{record['id']}/transcribe",
            params = {"clip_id": clip["id"]},
            json = {"model": "m"},
        )
        assert both.status_code == 400
        assert (
            client.post(
                "/api/inference/audio/inputs/source/transcribe", json = {"model": "m"}
            ).status_code
            == 400
        )
        # A path where an id belongs is not a field.
        assert (
            client.post(
                f"/api/inference/audio/inputs/{record['id']}/transcribe",
                json = {"model": "m", "path": "/etc/passwd"},
            ).status_code
            == 422
        )
    # Nothing landed in history beyond the clip this test saved.
    assert [r["id"] for r in gallery.list_audio()] == [clip["id"]]


def test_a_convert_source_is_transcribed_past_the_clone_reference_cap(monkeypatch):
    from routes import inference

    seconds = []

    async def fake_transcribe(raw, *args, **kwargs):
        with wave.open(io.BytesIO(raw)) as w:
            seconds.append(round(w.getnframes() / w.getframerate()))
        return {"text": "words"}

    monkeypatch.setattr(inference, "_transcribe_audio_result", fake_transcribe)
    with _client() as client:
        record = _upload(client, encode("wav", "pcm_s16le", 16000, "mono", 40.0)).json()
        url = f"/api/inference/audio/inputs/{record['id']}/transcribe"
        for purpose in ("reference", "convert"):
            response = client.post(url, json = {"model": "m", "purpose": purpose})
            assert response.status_code == 200, response.text
    assert seconds == [30, 40]
