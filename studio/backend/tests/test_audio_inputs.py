# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import io
import math
import os
import time
import wave
from urllib.parse import quote

import av
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import core.inference.audio_gallery as gallery
import utils.upload_limits as limits
from auth.authentication import get_current_subject
from core.inference import audio_inputs
from utils.api_errors import install_api_error_handlers

INPUTS = "/api/inference/audio/inputs"


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
    channels = {"stereo": 2, "5.1": 6}.get(layout, 1)
    count = int(round(rate * seconds))
    tone = (0.3 * np.sin(2 * math.pi * freq * np.arange(count) / rate) * 32767).astype(np.int16)
    packed = np.repeat(tone, channels).reshape(1, -1)
    buf = io.BytesIO()
    with av.open(buf, mode = "w", format = fmt) as out:
        stream = out.add_stream(codec, rate = rate, layout = layout)
        step = rate // 50  # 20 ms frames, as a recorder delivers them
        for start in range(0, count, step):
            chunk = packed[:, start * channels : min(count, start + step) * channels]
            arr = np.ascontiguousarray(chunk)
            frame = av.AudioFrame.from_ndarray(arr, format = "s16", layout = layout)
            frame.sample_rate, frame.pts = rate, start
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    return buf.getvalue()


def wav_bytes(seconds = 1.0, rate = 16000) -> bytes:
    return encode("wav", "pcm_s16le", rate, "mono", seconds)


async def _chunks(parts, seen = None):
    for part in parts:
        if seen is not None:
            seen.append(len(part))
        yield part


def _save(data: bytes, name = "clip.wav") -> dict:
    return asyncio.run(audio_inputs.save_stream(_chunks([data]), name))[0]


def _clip(rate = 16000, audio_type = "snac") -> dict:
    meta = {"prompt": "p", "model": "m", "audio_type": audio_type, "sample_rate": rate}
    meta.update(duration_s = 1.0, created_at = "2026-10-02T00:00:00Z")
    return gallery.save(wav_bytes(rate = rate), meta)


def _frames(path) -> tuple[int, int, int]:
    with wave.open(str(path)) as w:
        return w.getframerate(), w.getnchannels(), w.getnframes()


@pytest.fixture
def client():
    from routes import inference

    app = FastAPI()
    install_api_error_handlers(app)
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    app.include_router(inference.router, prefix = "/api/inference")
    app.include_router(inference.studio_router, prefix = "/api/inference")
    app.include_router(inference.router, prefix = "/v1")
    yield TestClient(app)


def _upload(client, data: bytes, name: str):
    headers = {"Content-Type": "application/octet-stream"}
    return client.post(INPUTS, params = {"name": name}, content = data, headers = headers)


def _video_without_audio() -> bytes:
    buf = io.BytesIO()
    with av.open(buf, mode = "w", format = "matroska") as out:
        stream = out.add_stream("ffv1", rate = 1)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), dtype = np.uint8), format = "rgb24")
        for packet in [*stream.encode(frame), *stream.encode(None)]:
            out.mux(packet)
    return buf.getvalue()


@pytest.mark.parametrize(
    "fmt, codec, rate, layout, name, channels",
    [
        ("webm", "libopus", 48000, "stereo", "recording.webm", 2),
        ("mp3", "libmp3lame", 44100, "mono", "voice.mp3", 1),
        ("wav", "pcm_s16le", 16000, "mono", "voice.wav", 1),
        ("wav", "pcm_s16le", 24000, "5.1", "surround.wav", 2),
        ("flac", "flac", 96000, "stereo", "hires.flac", 2),
    ],
)
def test_an_upload_decodes_dedupes_and_deletes(client, fmt, codec, rate, layout, name, channels):
    data = encode(fmt, codec, rate, layout, 2.0)
    response = _upload(client, data, f"C:\\Users\\me\\{name}")
    assert response.status_code == 201, response.text
    record = response.json()
    assert set(record) == set("id name duration_s sample_rate channels url expires_at".split())
    got = (record["name"], record["sample_rate"], record["channels"], round(record["duration_s"]))
    rate = min(rate, audio_inputs.MAX_RATE)
    assert got == (name, rate, channels, 2)
    assert record["url"] == f"{INPUTS}/{record['id']}/file"
    served = client.get(record["url"])
    assert served.status_code == 200 and served.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(served.content)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (rate, channels, 2)
    assert (directory := audio_inputs.inputs_dir()) == gallery.gallery_dir() / "inputs"
    again = _upload(client, data, "other.wav")
    assert again.status_code == 200
    # A re-upload refreshes the expiry, which can cross a second boundary.
    reused = again.json()
    assert reused.pop("expires_at") >= record["expires_at"]
    assert reused == {k: v for k, v in record.items() if k != "expires_at"}
    assert len(list(directory.glob("*.json"))) == 1
    assert audio_inputs.prepare_reference({"input_id": record["id"]})[1].is_file()
    assert client.delete(f"{INPUTS}/{record['id']}").json() == {"removed": True}
    assert client.get(record["url"]).status_code == 404
    assert client.delete(f"{INPUTS}/{record['id']}").status_code == 404
    assert not list(directory.iterdir())


def test_an_oversize_body_is_refused_while_streaming():
    seen: list[int] = []
    with pytest.raises(audio_inputs.AudioInputError) as refused:
        body = _chunks([b"\x00" * 1024] * 100, seen)
        asyncio.run(audio_inputs.save_stream(body, "big.wav", max_bytes = 4096))
    assert (refused.value.status, refused.value.detail) == (413, "Audio is too large.")
    # Refused at the chunk that crossed the cap: the rest of the body was never read.
    assert len(seen) == 5
    assert not list(audio_inputs.inputs_dir().iterdir())


@pytest.mark.parametrize(
    "patches, make, status, detail",
    [
        ({"AUDIO_INPUT_MAX_BYTES": 2048}, lambda: wav_bytes(2.0), 413, "Audio is too large."),
        ({"MAX_SECONDS": 1}, lambda: wav_bytes(2.0), 413, "Audio is longer than 0 minutes."),
        ({}, lambda: b"", 400, "Audio is empty."),
        ({}, lambda: b"not audio " * 120, 400, "This file is not audio Studio can read."),
        ({}, _video_without_audio, 400, "This file has no audio stream."),
    ],
)
def test_bad_or_over_cap_uploads_are_refused(client, monkeypatch, patches, make, status, detail):
    for attr, value in patches.items():
        monkeypatch.setattr(audio_inputs, attr, value)
        monkeypatch.setattr(limits, attr, value, raising = False)
    response = _upload(client, make(), "clip.mkv")
    assert response.status_code == status and response.json()["detail"] == detail
    assert not list(audio_inputs.inputs_dir().glob("*.wav"))


def test_the_body_middleware_passes_the_upload_through_at_its_own_cap():
    import main

    assert limits.AUDIO_INPUT_MAX_BYTES == 200 * 1024 * 1024
    for path in (INPUTS, f"{INPUTS}/"):
        assert main._get_upload_passthrough_request_max_bytes(path) == limits.AUDIO_INPUT_MAX_BYTES
    assert f"{INPUTS}/x/transcribe" not in main._BODY_UPLOAD_PASSTHROUGH_EXACT_PATHS


def test_the_sweep_drops_the_oldest_past_the_byte_cap_then_all_past_the_ttl(client, monkeypatch):
    clock = [1_000_000.0]
    monkeypatch.setattr(audio_inputs, "_now", lambda: clock[0])
    ids = []
    for freq in (300.0, 500.0, 700.0):
        ids.append(_save(encode("wav", "pcm_s16le", 16000, "mono", 0.5, freq), "a.wav")["id"])
        clock[0] += 10
    one = (audio_inputs.inputs_dir() / f"{ids[0]}.wav").stat().st_size
    assert audio_inputs.sweep(byte_cap = int(one * 2.5)) == 1
    assert [audio_inputs.input_path(i) is not None for i in ids] == [False, True, True]
    assert audio_inputs.prepare_reference({"input_id": ids[1]})[1].is_file()
    clock[0] += audio_inputs.TTL_SECONDS + 1
    assert client.get(f"{INPUTS}/{ids[2]}/file").status_code == 404
    assert audio_inputs.sweep() == 2
    assert not list(audio_inputs.inputs_dir().iterdir())


def test_expired_inputs_are_swept_without_waiting_for_another_upload(monkeypatch, tmp_path):
    from auth import storage

    clock = [1_000_000.0]
    monkeypatch.setattr(audio_inputs, "_now", lambda: clock[0])
    input_id = _save(wav_bytes(0.5), "a.wav")["id"]
    clock[0] += audio_inputs.TTL_SECONDS + 1
    # An account that never uploaded gets no inputs folder from the sweep.
    other = {"account_id": "acct-b", "username": "bob", "role": "user"}
    monkeypatch.setattr(storage, "list_accounts", lambda: [other])
    monkeypatch.setattr(audio_inputs, "account_path", lambda rel: tmp_path / "acct-b" / rel)
    assert audio_inputs.sweep_all_accounts() == 1
    assert audio_inputs.input_path(input_id) is None
    assert not (tmp_path / "acct-b").exists()


def test_a_prepared_reference_is_24k_mono_cut_to_thirty_seconds_and_cached():
    record = _save(wav_bytes(seconds = 31.0, rate = 8000), "long.wav")
    source, path = audio_inputs.prepare_reference({"input_id": record["id"]})
    assert (source.kind, source.name) == ("input", "long.wav")
    assert _frames(path) == (24000, 1, 30 * 24000)
    assert path.parent == audio_inputs.inputs_dir() and path.name.startswith(record["id"])
    mtime = path.stat().st_mtime_ns
    assert audio_inputs.prepare_reference({"input_id": record["id"]})[1].stat().st_mtime_ns == mtime
    assert _frames(audio_inputs.prepared_path(source, 24000, max_seconds = 1.0))[2] == 24000


def test_preparing_a_clip_copy_sweeps_expired_copies_without_an_upload(tmp_path):
    # Cloning from history never uploads, so the sweep has to run from here too.
    directory = audio_inputs.inputs_dir()
    stale = directory / "c-old.24000.mono.m30.wav"
    stale.write_bytes(wav_bytes(0.2))
    old = time.time() - audio_inputs.TTL_SECONDS - 60
    os.utime(stale, (old, old))
    clip = tmp_path / "clip.wav"
    clip.write_bytes(wav_bytes(0.5))
    source = audio_inputs.Source(kind = "clip", id = "fresh", path = clip, name = "clip.wav")
    fresh = audio_inputs.prepared_path(source, 24000, max_seconds = 30.0)
    assert fresh.is_file() and not stale.exists()


@pytest.mark.parametrize("bad", ["../escape", "a/b", "..", "x" * 129, "name.wav", ""])
def test_unsafe_ids_resolve_to_nothing(client, bad):
    assert audio_inputs.input_path(bad) is None
    assert audio_inputs.delete(bad) is False
    with pytest.raises(audio_inputs.AudioInputError) as refused:
        audio_inputs.resolve_source({"input_id": bad or None})
    assert refused.value.status in (400, 404)
    if bad:
        assert client.get(f"{INPUTS}/{quote(bad, safe = '')}/file").status_code == 404
        assert client.delete(f"{INPUTS}/{quote(bad, safe = '')}").status_code == 404


def test_transcribe_sends_a_16k_mono_copy_and_saves_nothing(client, monkeypatch):
    from routes import inference

    calls, model = [], "audio-cpp/audio.cpp-gguf/Qwen3-ASR-0.6B-GGUF"

    async def fake_transcribe(raw, model, language, fast, engine, request, device):
        with wave.open(io.BytesIO(raw)) as w:
            calls.append((w.getframerate(), w.getnchannels(), model, language, engine, device))
        return {"text": " Okay, I'm Cemo. ", "language": "English"}

    monkeypatch.setattr(inference, "_transcribe_audio_result", fake_transcribe)
    record = _upload(client, encode("webm", "libopus", 48000, "stereo", 1.0), "r.webm").json()
    url = f"{INPUTS}/{record['id']}/transcribe"
    response = client.post(url, json = {"model": model, "device": "cpu"})
    assert response.json() == {"text": "Okay, I'm Cemo.", "language": "English", "model": model}
    assert calls == [(16000, 1, model, None, None, "cpu")]
    source, clip_q = f"{INPUTS}/source/transcribe", {"clip_id": _clip()["id"]}
    assert client.post(source, params = clip_q, json = {"model": "m"}).status_code == 200
    # Two sources, or none, are refused; a path is not a field.
    assert client.post(url, params = clip_q, json = {"model": "m"}).status_code == 400
    assert client.post(source, json = {"model": "m"}).status_code == 400
    assert client.post(url, json = {"model": "m", "path": "/etc/passwd"}).status_code == 422
    assert [r["id"] for r in gallery.list_audio()] == [clip_q["clip_id"]]


def test_a_convert_source_is_transcribed_past_the_clone_reference_cap(client, monkeypatch):
    from routes import inference

    seconds = []

    async def fake_transcribe(raw, *args, **kwargs):
        with wave.open(io.BytesIO(raw)) as w:
            seconds.append(round(w.getnframes() / w.getframerate()))
        return {"text": "words"}

    monkeypatch.setattr(inference, "_transcribe_audio_result", fake_transcribe)
    record = _upload(client, encode("wav", "pcm_s16le", 16000, "mono", 40.0), "long.wav").json()
    url = f"{INPUTS}/{record['id']}/transcribe"
    for purpose in ("reference", "convert"):
        response = client.post(url, json = {"model": "m", "purpose": purpose})
        assert response.status_code == 200, response.text
    assert seconds == [30, 40]


def test_the_byte_cap_counts_only_kept_inputs(client, monkeypatch):
    clock = [1_000_000.0]
    monkeypatch.setattr(audio_inputs, "_now", lambda: clock[0])
    small = _save(encode("wav", "pcm_s16le", 16000, "mono", 0.5, 300.0), "s.wav")["id"]
    clock[0] += 10
    big = _save(encode("wav", "pcm_s16le", 16000, "mono", 2.0, 500.0), "b.wav")["id"]
    clock[0] += 10
    new = _save(encode("wav", "pcm_s16le", 16000, "mono", 0.5, 700.0), "n.wav")["id"]
    size = lambda i: (audio_inputs.inputs_dir() / f"{i}.wav").stat().st_size
    # Newest + small fit; the big one in the middle does not, and evicting it frees its bytes.
    cap = size(new) + size(small) + 10
    assert audio_inputs.sweep(byte_cap = cap) == 1
    assert [audio_inputs.input_path(i) is not None for i in (small, big, new)] == [
        True,
        False,
        True,
    ]


def test_an_upload_prepared_for_audio_cpp_is_mono_at_the_rate_and_leaves_no_copy():
    from routes.inference import _prepared_upload

    path = _prepared_upload(encode("mp3", "libmp3lame", 44100, "stereo", 1.0), 24000)
    assert path.parent == audio_inputs.inputs_dir()
    rate, channels, frames = _frames(path)
    assert (rate, channels) == (24000, 1) and abs(frames - 24000) < 2400
    assert [p.name for p in path.parent.iterdir()] == [path.name]


def test_an_upload_too_long_for_audio_cpp_is_refused_not_cut(monkeypatch):
    from routes.inference import _prepared_upload

    monkeypatch.setattr(audio_inputs, "MAX_SECONDS", 0.5)
    with pytest.raises(audio_inputs.AudioInputError) as refused:
        _prepared_upload(wav_bytes(1.0), 16000)
    assert refused.value.status == 413
    assert list(audio_inputs.inputs_dir().iterdir()) == []


def test_the_sweep_reaps_a_prepared_upload_a_crash_left_behind():
    from routes.inference import _prepared_upload

    stale = _prepared_upload(wav_bytes(0.2), 16000)
    waiting = _prepared_upload(wav_bytes(0.2), 16000)
    for path, age in ((stale, audio_inputs.TTL_SECONDS + 60), (waiting, 2 * 60 * 60)):
        os.utime(path, (time.time() - age, time.time() - age))
    audio_inputs.sweep()
    assert not stale.exists() and waiting.exists()
