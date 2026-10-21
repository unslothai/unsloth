# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every WAV in a /v1/tasks/run reply, counted once (S3: top-level audio repeats audio_0)."""

from __future__ import annotations

import base64
import io
import json
import wave

import pytest

from core.inference import audio_cpp_backend
from core.inference.audio_task_outputs import task_outputs, wav_header


def _wav(
    rate = 44100,
    frames = 441,
    channels = 2,
    fill = 0,
) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes([fill, 0]) * frames * channels)
    return buf.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _json(payload) -> tuple[str, bytes]:
    return "application/json", json.dumps(payload).encode()


def test_a_batch_reply_gives_each_variation_once():
    takes = [_wav(fill = i + 1) for i in range(3)]
    reply = {
        "audio": _b64(takes[0]),
        "sample_rate": 44100,
        "named_audio_outputs": [
            {"id": f"audio_{i}", "audio": _b64(take), "sample_rate": 44100}
            for i, take in enumerate(takes)
        ],
    }
    outputs = task_outputs(*_json(reply))
    assert [o[0] for o in outputs] == ["audio_0", "audio_1", "audio_2"]
    assert [o[1] for o in outputs] == takes


def test_audio_only_named_only_and_raw():
    one = _wav()
    assert task_outputs(*_json({"audio": _b64(one)})) == [("audio", one)]
    stems = {"named_audio_outputs": [{"id": n, "audio": _b64(one)} for n in ("drums", "bass")]}
    assert [o[0] for o in task_outputs(*_json(stems))] == ["drums", "bass"]
    assert task_outputs("audio/wav", one) == [("audio", one)]
    assert task_outputs("application/octet-stream", one) == [("audio", one)]


@pytest.mark.parametrize(
    "reply",
    [
        ("application/json", b"not json"),
        ("application/json", b'{"text": "x"}'),
        ("application/json", b"[1, 2]"),
        _json({"audio": _b64(b"not a wav at all")}),
        _json({"named_audio_outputs": []}),
    ],
)
def test_a_reply_without_audio_is_an_error(reply):
    with pytest.raises(RuntimeError, match = "no audio|unreadable"):
        task_outputs(*reply)


def test_the_sample_rate_comes_from_the_wav_header():
    data = _wav(rate = 48000, frames = 24000)
    ((_id, wav),) = task_outputs(*_json({"audio": _b64(data), "sample_rate": 16000}))
    assert wav_header(wav) == (48000, 0.5)


def test_the_legacy_single_audio_reader_keeps_its_answer():
    takes = [_wav(fill = 1), _wav(fill = 2)]
    reply = {
        "audio": _b64(takes[0]),
        "named_audio_outputs": [
            {"id": f"audio_{i}", "audio": _b64(t)} for i, t in enumerate(takes)
        ],
    }
    assert audio_cpp_backend._audio_from_task_response(*_json(reply)) == takes[0]
    nested = {"result": {"wav": _b64(takes[1])}}
    assert audio_cpp_backend._audio_from_task_response(*_json(nested)) == takes[1]
