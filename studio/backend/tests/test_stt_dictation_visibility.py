# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""In-app dictation shows on the API page and its llama-server leaves a log (#19387)."""

import os
import subprocess
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.inference import stt_mtmd_sidecar as mtmd
from core.inference.api_monitor import api_monitor
from core.inference.stt_sidecar import SttModelBusyError
from routes import inference as ri
from utils import debug_log_sources


def _raw_client(monkeypatch, transcribe):
    sidecar = SimpleNamespace(loaded_model = None, device = None, transcribe = transcribe)
    monkeypatch.setattr(ri.account_access, "managed_account", lambda: False)
    monkeypatch.setattr(ri, "_resolve_serving_stt_engine", lambda engine: engine)
    monkeypatch.setattr(ri, "_prepare_runtime_fallback_checkpoint", lambda *args: None)
    monkeypatch.setattr(ri, "_stt_sidecar_for", lambda engine: sidecar)
    monkeypatch.setattr(ri, "_stt_lifecycle", lambda: (lambda *a, **k: None, lambda *a: []))
    app = FastAPI()
    app.add_api_route(
        "/api/inference/audio/transcribe/raw", ri.transcribe_audio_raw, methods = ["POST"]
    )
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app)


def test_local_dictation_opens_a_monitor_row(monkeypatch):
    def transcribe(*args, **kwargs):
        return {"text": "hello sloth", "model": "/home/me/models/qwen3-asr-1.7b", "language": "en"}

    client = _raw_client(monkeypatch, transcribe)
    api_monitor.clear()
    response = client.post(
        "/api/inference/audio/transcribe/raw", params = {"engine": "mtmd"}, content = b"a"
    )
    assert response.status_code == 200, response.text
    assert response.json()["text"] == "hello sloth"
    rows = api_monitor.snapshot(include_details = False, subject = "test-user")
    assert len(rows) == 1
    assert rows[0]["endpoint"] == "/api/inference/audio/transcribe/raw"
    assert rows[0]["status"] == "completed"
    assert rows[0]["prompt_preview"] == "Recording"
    assert rows[0]["reply_preview"] == "hello sloth"
    assert rows[0]["model"] == "qwen3-asr-1.7b"


def test_local_dictation_failure_records_an_error_row(monkeypatch):
    def transcribe(*args, **kwargs):
        raise SttModelBusyError("Model is busy.")

    client = _raw_client(monkeypatch, transcribe)
    api_monitor.clear()
    response = client.post(
        "/api/inference/audio/transcribe/raw", params = {"engine": "mtmd"}, content = b"a"
    )
    assert response.status_code == 409
    rows = api_monitor.snapshot(include_details = False, subject = "test-user")
    assert [row["status"] for row in rows] == ["error"]


class _Process:
    pid = 91000
    returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout = None):
        return self.returncode


@pytest.fixture
def launches(monkeypatch):
    from core.inference import llama_cpp

    seen = []

    def spawn(command, **kwargs):
        stdout = kwargs["stdout"]
        if hasattr(stdout, "write"):
            stdout.write("main: server is listening\n")
        seen.append((list(command), kwargs))
        return _Process()

    def context(self, *args, **kwargs):
        command = seen[-1][0]
        return int(command[command.index("--ctx-size") + 1])

    monkeypatch.setattr(mtmd.subprocess, "Popen", spawn)
    monkeypatch.setattr(mtmd, "adopt_pid", lambda _pid: None)
    monkeypatch.setattr(mtmd, "forget_pid", lambda _pid: None)
    monkeypatch.setattr(mtmd, "ensure_engine_available", lambda: "/fake/llama-server")
    monkeypatch.setattr(mtmd, "_llama_server_child_env", lambda _binary: {})
    monkeypatch.setattr(mtmd, "_training_active", lambda: False)
    monkeypatch.setattr(llama_cpp, "_llama_server_api_key_enabled", lambda: False)
    monkeypatch.setattr(
        mtmd.MtmdSttSidecar, "_ensure_model_downloaded", lambda *_args: ("m.gguf", "p.gguf")
    )
    monkeypatch.setattr(
        mtmd.MtmdSttSidecar, "_wait_for_server", staticmethod(lambda *_args, **_kwargs: True)
    )
    monkeypatch.setattr(mtmd.MtmdSttSidecar, "_read_server_context", context)
    sidecar = mtmd.MtmdSttSidecar(keep_alive_seconds = 0)
    yield SimpleNamespace(sidecar = sidecar, seen = seen)
    sidecar.unload(wait = False)


def test_dictation_server_output_goes_to_a_listed_log_file(launches):
    launches.sidecar.load("qwen3-asr-0.6b", context_length = 4096)
    _command, kwargs = launches.seen[-1]
    assert kwargs["stderr"] is subprocess.STDOUT
    log_path = kwargs["stdout"].name
    assert kwargs["stdout"].closed, "the parent must not keep the child's log handle open"
    home = os.environ["UNSLOTH_STUDIO_HOME"]
    assert os.path.dirname(log_path) == os.path.join(home, "logs", "llama-server")
    with open(log_path, encoding = "utf-8") as handle:
        assert "server is listening" in handle.read()
    listed = [source.realpath for source in debug_log_sources.list_sources()]
    assert any(os.path.samefile(path, log_path) for path in listed)


def test_a_log_that_cannot_open_still_loads(launches, monkeypatch):
    monkeypatch.setattr(mtmd, "_open_server_log", lambda _port: None)
    launches.sidecar.load("qwen3-asr-0.6b", context_length = 4096)
    _command, kwargs = launches.seen[-1]
    assert kwargs["stdout"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.DEVNULL


def test_a_relaunch_in_the_same_second_keeps_the_earlier_log(monkeypatch):
    monkeypatch.setattr(mtmd.time, "time", lambda: 1_700_000_000)
    first = mtmd._open_server_log(4000)
    first.write("crash trace\n")
    first.close()
    second = mtmd._open_server_log(4000)
    second.close()
    assert first.name == second.name
    with open(second.name, encoding = "utf-8") as handle:
        assert "crash trace" in handle.read()


def test_a_real_child_writes_both_streams_to_the_log():
    import sys

    handle = mtmd._open_server_log(4001)
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", "import sys; print('out'); print('err', file = sys.stderr)"],
            stdout = handle,
            stderr = subprocess.STDOUT,
            stdin = subprocess.DEVNULL,
        )
    finally:
        handle.close()
    assert process.wait(timeout = 60) == 0
    with open(handle.name, encoding = "utf-8") as log:
        lines = log.read().split()
    assert "out" in lines and "err" in lines
