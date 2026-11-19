# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import socket
import threading

import pytest

from core.inference import stt_live
from core.inference.stt_live import LiveTranscript, LiveTranscriptError


def _runtime(reply_after_audio):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    seen = {"audio": b""}

    def serve():
        conn, _ = listener.accept()
        reader = conn.makefile("rb")
        seen["request"] = reader.readline().decode()
        while reader.readline().strip():
            pass
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
        while True:
            size = int(reader.readline().strip() or b"0", 16)
            if size == 0:
                break
            seen["audio"] += reader.read(size)
            reader.readline()
            for line in reply_after_audio(seen["audio"], False):
                conn.sendall(line)
        for line in reply_after_audio(seen["audio"], True):
            conn.sendall(line)
        conn.close()

    threading.Thread(target = serve, daemon = True).start()
    return listener.getsockname()[1], seen


def test_words_arrive_while_audio_is_still_being_sent():
    def reply(audio, ended):
        if ended:
            return [
                b'data: {"type":"transcript.text.done","text":"Hello there."}\n\n',
                b"data: [DONE]\n\n",
            ]
        return [b'data: {"type":"transcript.text.delta","delta":"Hello"}\n\n'] if len(audio) == 4 else []

    port, seen = _runtime(reply)
    session = LiveTranscript(port, "studio-1", 16000, "en")
    events = session.events()
    session.feed(b"\x01\x00\x02\x00")
    assert next(events) == {"type": "delta", "text": "Hello"}
    session.feed(b"\x03\x00")
    session.finish()
    assert list(events) == [{"type": "done", "text": "Hello there."}]
    assert seen["audio"] == b"\x01\x00\x02\x00\x03\x00"
    assert "model=studio-1" in seen["request"] and "sample_rate=16000" in seen["request"]
    assert "language=en" in seen["request"] and "sample_format=s16le" in seen["request"]
    with pytest.raises(LiveTranscriptError):
        session.feed(b"\x00\x00")
    session.close()


def test_a_session_the_runtime_gives_up_on_is_an_error_not_an_empty_transcript():
    port, _ = _runtime(
        lambda audio, ended: [b'data: {"type":"error","error":{"message":"needs 16 kHz"}}\n\n']
        if ended
        else []
    )
    session = LiveTranscript(port, "studio-1", 48000)
    session.finish()
    with pytest.raises(LiveTranscriptError, match = "needs 16 kHz"):
        list(session.events())
    session.close()


def test_the_runtime_s_own_spec_says_which_models_transcribe_live(monkeypatch):
    specs = {
        "nemotron_asr": {"tasks": ["asr"], "modes": ["offline", "streaming"]},
        "canary_asr": {"tasks": ["asr"], "modes": ["offline"]},
        "voxcpm2": {"tasks": ["tts"], "modes": ["offline", "streaming"]},
    }
    monkeypatch.setattr("core.inference.audio_cpp_models.runtime_spec", specs.get)
    assert stt_live.family_streams("nemotron_asr")
    assert not stt_live.family_streams("canary_asr")
    assert not stt_live.family_streams("voxcpm2")
    assert not stt_live.family_streams("unknown")
