# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AudioCppServer against a real child process: a small fake ``audiocpp_server`` speaking the HTTP contract."""

import io
import json
import sys
import textwrap
import threading
import time
import wave

import pytest

from core.inference import audio_cpp_server as srv
from core.inference.audio_cpp_models import AudioCppModel, AudioCppVariant, RepoFile


def _model(folder: str, family: str, task: str) -> AudioCppModel:
    main = RepoFile(f"{folder}/{folder.lower()}-q8_0.gguf", 1)
    variant = AudioCppVariant("Q8_0", (main,), main.path)
    return AudioCppModel(
        id = f"audio-cpp/audio.cpp-gguf/{folder}",
        repo_id = "audio-cpp/audio.cpp-gguf",
        folder = folder,
        display_name = folder,
        family = family,
        task = task,
        server_task = task,
        variant = variant,
        variants = (variant,),
        default_variant = "Q8_0",
    )


CANARY = _model("Canary-180M-Flash-GGUF", "canary_asr", "asr")
KOKORO = _model("Kokoro-82M-GGUF", "kokoro_tts", "tts")

FAKE_SERVER = textwrap.dedent(
    r"""
    import email, io, json, sys, time, wave
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    cfg = json.load(open(sys.argv[sys.argv.index("--config") + 1], encoding="utf-8"))
    model_id = cfg["models"][0]["id"]
    listed = "wrong-id" if cfg["models"][0]["path"].endswith("wrong-id.gguf") else model_id

    def wav():
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b"\0\0" * 1600)
        return buf.getvalue()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def reply(self, code, body, ctype="application/json"):
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            if self.path == "/health":
                return self.reply(200, json.dumps({"status": "ok", "backend": cfg["backend"]}).encode())
            if self.path == "/v1/models":
                return self.reply(200, json.dumps({"data": [{"id": listed}]}).encode())
            self.reply(404, b"{}")
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.path == "/v1/audio/speech":
                req = json.loads(body)
                if req.get("input") == "slow":
                    time.sleep(30)
                return self.reply(200, wav(), "audio/wav")
            if self.path == "/v1/audio/transcriptions":
                msg = email.message_from_bytes(b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + body)
                fields = {}
                for part in msg.get_payload():
                    name = part.get_param("name", header="content-disposition")
                    fields[name] = part.get_filename() or part.get_payload(decode=True).decode()
                if fields.get("language") == "slow":
                    time.sleep(30)
                return self.reply(200, json.dumps({"text": json.dumps(fields, sort_keys=True)}).encode())
            self.reply(404, b"{}")

    ThreadingHTTPServer(("127.0.0.1", cfg["port"]), H).serve_forever()
    """
)


@pytest.fixture
def fake_binary(tmp_path, monkeypatch):
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER, encoding = "utf-8")
    binary = tmp_path / srv.BINARY_NAME
    binary.write_bytes(b"")
    real_popen = srv.subprocess.Popen

    def launch(command, *args, **kwargs):
        # The fake runs as the direct child (no shell wrapper), so stop() terminates the server itself.
        if command and command[0] == str(binary):
            command = [sys.executable, str(script), *command[1:]]
        return real_popen(command, *args, **kwargs)

    monkeypatch.setattr(srv.subprocess, "Popen", launch)
    monkeypatch.setattr(srv, "ensure_binary", lambda: str(binary))
    monkeypatch.setattr(srv, "model_runtime_problem", lambda model, binary = None: None)
    monkeypatch.setattr(srv, "select_backend", lambda binary, force_cpu: "cpu")
    return binary


def test_ready_only_when_our_model_id_is_listed(fake_binary, tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "_SERVER_START_TIMEOUT_SECONDS", 4.0)
    model = CANARY
    server = srv.AudioCppServer.start(model, str(tmp_path / "m.gguf"))
    try:
        assert server.alive() and server._probe()
    finally:
        server.stop()
    assert not server.alive()
    assert not server._config_dir.exists()
    # A responder that does not list this launch's id is not our child: never treated as ready.
    with pytest.raises(srv.AudioCppUnavailableError, match = "did not start in time"):
        srv.AudioCppServer.start(model, str(tmp_path / "wrong-id.gguf"))


def test_readiness_ignores_an_ambient_http_proxy(fake_binary, tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "_SERVER_START_TIMEOUT_SECONDS", 6.0)
    for name in ("HTTP_PROXY", "http_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising = False)
    server = srv.AudioCppServer.start(KOKORO, str(tmp_path / "m.gguf"))
    try:
        assert server._probe()
    finally:
        server.stop()


def test_cancel_closes_the_socket_mid_request(fake_binary, tmp_path):
    server = srv.AudioCppServer.start(KOKORO, str(tmp_path / "m.gguf"))
    try:
        cancel = threading.Event()
        threading.Timer(0.3, cancel.set).start()
        started = time.monotonic()
        with pytest.raises(srv.AudioCppRequestCancelledError):
            server.post_json(
                "/v1/audio/speech",
                {"model": server.model_id, "input": "slow"},
                timeout = 20,
                cancel_event = cancel,
            )
        # Returns on the cancel, not on the server's reply or the timeout (Windows never woke the recv).
        assert time.monotonic() - started < 5
        # The server is untouched by the client-side cancel and still serves.
        _ctype, data = server.post_json(
            "/v1/audio/speech", {"model": server.model_id, "input": "hi"}
        )
        with wave.open(io.BytesIO(data)) as w:
            assert w.getframerate() == 16000
    finally:
        server.stop()


def test_transcription_multipart_carries_model_language_and_file(fake_binary, tmp_path):
    from core.inference.stt_audiocpp_sidecar import AudioCppSttSidecar

    model = CANARY
    server = srv.AudioCppServer.start(model, str(tmp_path / "m.gguf"))
    try:
        side = AudioCppSttSidecar()
        side._server = server
        fields = json.loads(side._post_transcription(b"RIFF....WAVE", "en", None))
        assert fields == {"file": "dictation.wav", "language": "en", "model": server.model_id}
    finally:
        server.stop()


def test_cancelled_transcription_stops_the_busy_server(fake_binary, tmp_path):
    from core.inference.stt_audiocpp_sidecar import (
        AudioCppSttSidecar,
        SttTranscriptionCancelledError,
    )
    server = srv.AudioCppServer.start(CANARY, str(tmp_path / "m.gguf"))
    try:
        side = AudioCppSttSidecar()
        side._server = server
        cancel = threading.Event()
        threading.Timer(0.3, cancel.set).start()
        with side._lock, pytest.raises(SttTranscriptionCancelledError):
            side._post_transcription(b"RIFF....WAVE", "slow", cancel)
        # The child still decoding the abandoned clip is gone, so the next load starts a fresh one.
        assert side._server is None and not server.alive()
    finally:
        server.stop()


def test_threads_budget_reaches_the_command_line(fake_binary, tmp_path, monkeypatch):
    seen = {}
    launch = srv.subprocess.Popen  # the fixture's launcher

    def spy(command, *args, **kwargs):
        seen["command"] = command
        return launch(command, *args, **kwargs)

    monkeypatch.setattr(srv.subprocess, "Popen", spy)
    monkeypatch.setenv("UNSLOTH_CPU_THREADS", "3")
    server = srv.AudioCppServer.start(KOKORO, str(tmp_path / "m.gguf"))
    server.stop()
    assert seen["command"][-2:] == ["--threads", "3"]
    monkeypatch.setenv("UNSLOTH_CPU_THREADS", "zero")
    assert srv._cpu_threads() is None
