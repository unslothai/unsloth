# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The exact request audio.cpp receives for a separation, against a recording fake server."""

from __future__ import annotations

import base64
import io
import json
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from core.inference import audio_cpp_backend as acb
from core.inference import audio_cpp_files
from core.inference import audio_cpp_models as acm
from core.inference import audio_cpp_server as srv
from core.inference.audio_cpp_models import AUDIO_CPP_REPO, AudioCppModel, AudioCppVariant, RepoFile
from core.inference.audio_errors import AudioGenerationCancelledError, AudioRuntimeError

SOURCE = "/srv/accounts/a/audio/inputs/0123.44100.stereo.wav"
_REAL_START = srv.AudioCppServer.start


def _wav(frames = 441) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(b"\x01\x00" * 2 * frames)
    return buf.getvalue()


def _answer(ids) -> bytes:
    stems = [
        {
            "id": stem,
            "audio": base64.b64encode(_wav()).decode(),
            "sample_rate": 44100,
            "channels": 2,
        }
        for stem in ids
    ]
    return json.dumps({"named_audio_outputs": stems, "timing": {"wall_ms": 1.0}}).encode()


def _model(family, display = None) -> AudioCppModel:
    policy = acm.family_policy(family)
    variant = AudioCppVariant("Q8_0", (RepoFile("m-q8_0.gguf", 1),), "m-q8_0.gguf")
    return AudioCppModel(
        id = f"{AUDIO_CPP_REPO}/{display or family}",
        repo_id = AUDIO_CPP_REPO,
        folder = display or family,
        display_name = display or family,
        family = family,
        task = policy.task,
        server_task = policy.default_server_task,
        variant = variant,
        variants = (variant,),
        default_variant = "Q8_0",
        model_options = dict(policy.model_options),
        separation = policy.separation,
    )


class _Recorder:
    model_id = "studio-test"
    backend = "cpu"

    def __init__(
        self,
        model,
        answer = None,
        error = None,
    ):
        self.model = model
        self.answer = answer
        self.error = error
        self.calls: list[tuple[str, dict]] = []
        self.stopped = False

    def alive(self):
        return not self.stopped

    def stop(self):
        self.stopped = True

    def post_json(self, *_args, **_kwargs):
        raise AssertionError("a separation streams its answer to disk")

    def post_json_to_file(self, path, payload, dest, **_kwargs):
        self.calls.append((path, json.loads(json.dumps(payload))))
        Path(dest).write_bytes(b'{"named_audio_outputs": [')
        if self.error is not None:
            raise self.error
        Path(dest).write_bytes(self.answer)
        return "application/json"


@pytest.fixture
def started(monkeypatch, tmp_path):
    """Every server start, as ``(served model, recorder)``; the answer comes from the family."""
    starts: list[tuple[AudioCppModel, _Recorder]] = []
    answers = {
        "htdemucs": ["drums", "bass", "other", "vocals"],
        "bs_roformer": ["vocals", "instrumental"],
    }

    def start(served, path, **_kwargs):
        server = _Recorder(served, _answer(answers[served.family]))
        starts.append((served, server))
        return server

    monkeypatch.setattr(audio_cpp_files, "materialize", lambda m: str(tmp_path / "m.gguf"))
    monkeypatch.setattr(acb.AudioCppServer, "start", start)
    return starts


def _backend(model):
    backend = acb.AudioCppBackend()
    backend._model = model
    backend.models = {model.id: {"is_audio": True}}
    backend.active_model_name = model.id
    return backend


def _out(tmp_path) -> Path:
    out = tmp_path / ".separate-x"
    out.mkdir(exist_ok = True)
    return out


def test_the_body_is_exactly_the_model_and_the_track_and_htdemucs_never_restarts(started, tmp_path):
    backend = _backend(_model("htdemucs", "HTDemucs-GGUF"))
    out = _out(tmp_path)
    outputs = backend.separate_audio(SOURCE, str(out), {"num_overlap": 2})
    ((_served, server),) = started
    ((path, body),) = server.calls
    assert path == "/v1/tasks/run"
    assert body == {"model": "studio-test", "audio": SOURCE}
    assert [o["id"] for o in outputs] == ["drums", "bass", "other", "vocals"]
    for output in outputs:
        assert Path(output["path"]).parent == out and Path(output["path"]).is_file()
        assert (output["sample_rate"], output["channels"], output["duration_s"]) == (44100, 2, 0.01)
    assert not (out / ".response.json").exists()
    backend.separate_audio(SOURCE, str(_out(tmp_path)))
    backend.separate_audio(SOURCE, str(_out(tmp_path)), {"num_overlap": 1})
    assert len(started) == 1
    assert started[0][0].model_options.get("session_options") in (None, {})


def test_roformer_overlap_is_a_session_option_that_restarts_only_on_change(started, tmp_path):
    backend = _backend(_model("bs_roformer", "BS-RoFormer-ep368-GGUF"))
    backend._start_server(backend._model)
    assert len(started) == 1
    backend.separate_audio(SOURCE, str(_out(tmp_path)))
    assert len(started) == 1
    backend.separate_audio(SOURCE, str(_out(tmp_path)), {"num_overlap": 1})
    assert len(started) == 2
    assert started[1][0].model_options["session_options"] == {"num_overlap": "1"}
    backend.separate_audio(SOURCE, str(_out(tmp_path)), {"num_overlap": 1})
    assert len(started) == 2
    backend.separate_audio(SOURCE, str(_out(tmp_path)))
    assert len(started) == 3
    assert started[2][0].model_options.get("session_options") == {}
    for _served, server in started[1:]:
        for _path, body in server.calls:
            assert set(body) == {"model", "audio"}


def test_the_server_entry_runs_the_sep_task_with_the_overlap(monkeypatch, tmp_path):
    seen = {}
    binary = tmp_path / srv.BINARY_NAME
    binary.write_bytes(b"")

    def spy(command, *_args, **_kwargs):
        with open(command[command.index("--config") + 1], encoding = "utf-8") as f:
            seen.update(json.load(f))
        raise OSError("not launched in tests")

    monkeypatch.setattr(srv, "ensure_binary", lambda: str(binary))
    monkeypatch.setattr(srv, "model_runtime_problem", lambda model, binary = None: None)
    monkeypatch.setattr(srv, "select_backend", lambda binary, force_cpu: "cpu")
    monkeypatch.setattr(srv.subprocess, "Popen", spy)
    from dataclasses import replace

    model = replace(
        _model("mel_band_roformer", "Mel-Band-RoFormer-GGUF"),
        model_options = {"session_options": {"num_overlap": "1"}},
    )
    with pytest.raises(srv.AudioCppUnavailableError):
        _REAL_START(model, "/models/m-q8_0.gguf")
    (entry,) = seen["models"]
    assert (entry["family"], entry["task"], entry["mode"]) == (
        "mel_band_roformer",
        "sep",
        "offline",
    )
    assert entry["session_options"] == {"num_overlap": "1"}


def test_a_cancel_stops_the_server_and_removes_the_answer(started, tmp_path, monkeypatch):
    backend = _backend(_model("htdemucs", "HTDemucs-GGUF"))
    backend._start_server(backend._model)
    server = started[0][1]
    cancel = threading.Event()

    def cancelled(path, payload, dest, **_kwargs):
        Path(dest).write_bytes(b"{")
        cancel.set()
        raise srv.AudioCppRequestCancelledError("Request cancelled.")

    server.post_json_to_file = cancelled
    out = _out(tmp_path)
    with pytest.raises(AudioGenerationCancelledError):
        backend.separate_audio(SOURCE, str(out), cancel_event = cancel)
    assert server.stopped and backend._server is None
    assert list(out.iterdir()) == []


def test_a_runtime_refusal_keeps_its_reason(started, tmp_path):
    backend = _backend(_model("htdemucs", "HTDemucs-GGUF"))
    backend._start_server(backend._model)
    started[0][1].error = srv.AudioCppRequestError(
        400, "HTDemucs prepare() sample rate mismatch: expected 44100, got 48000"
    )
    with pytest.raises(AudioRuntimeError) as caught:
        backend.separate_audio(SOURCE, str(_out(tmp_path)))
    assert caught.value.status == 400
    assert caught.value.detail == (
        "The audio runtime could not separate the track: "
        "HTDemucs prepare() sample rate mismatch: expected 44100, got 48000"
    )


def test_an_answer_without_stems_is_an_empty_list(started, tmp_path):
    backend = _backend(_model("htdemucs", "HTDemucs-GGUF"))
    backend._start_server(backend._model)
    started[0][1].answer = json.dumps({"audio": base64.b64encode(b"nope").decode()}).encode()
    out = _out(tmp_path)
    assert backend.separate_audio(SOURCE, str(out)) == []
    assert list(out.iterdir()) == []


class _Handler(BaseHTTPRequestHandler):
    body = b""
    status = 200

    def log_message(self, *_args):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)


@pytest.fixture
def http_server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target = httpd.serve_forever, daemon = True)
    thread.start()

    class _Process:
        pid = 0

        def poll(self):
            return None

    server = srv.AudioCppServer(
        _Process(), httpd.server_address[1], None, "m", "cpu", Path("/nonexistent")
    )
    yield server
    httpd.shutdown()


def test_post_json_to_file_streams_a_large_answer(http_server, tmp_path):
    _Handler.status, _Handler.body = 200, b'{"named_audio_outputs":[]}' + b" " * (3 << 20)
    dest = tmp_path / "answer.json"
    ctype = http_server.post_json_to_file(
        "/v1/tasks/run", {"model": "m"}, dest, cancel_event = threading.Event()
    )
    assert ctype == "application/json"
    assert dest.read_bytes() == _Handler.body


def test_post_json_to_file_raises_the_runtime_error(http_server, tmp_path):
    _Handler.status = 400
    _Handler.body = json.dumps(
        {"error": {"message": "unknown htdemucs request option: seed"}}
    ).encode()
    with pytest.raises(srv.AudioCppRequestError) as caught:
        http_server.post_json_to_file("/v1/tasks/run", {"model": "m"}, tmp_path / "a.json")
    assert caught.value.status == 400
    assert caught.value.detail == "unknown htdemucs request option: seed"
