# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An audio runtime that dies or goes silent mid-request is a 500 with code runtime_stopped, from
the backend through the worker and orchestrator to the route, instead of "Could not reach an
upstream service"."""

from __future__ import annotations

import asyncio
import queue
import sys
import threading
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from core.inference import audio_cpp_failures as af
from core.inference.audio_cpp_server import AudioCppUnavailableError
from core.inference.audio_errors import AUDIO_RUNTIME_ERROR_CODE, AudioRuntimeError
from core.inference.audio_cpp_backend import _runtime_stopped

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_audio_cpp_music_requests as music  # noqa: E402
import test_audio_cpp_separation as sep  # noqa: E402
from test_account_media_isolation import ALICE, isolated  # noqa: E402,F401
from test_audio_run_cancel import KOKORO, SPEAK_INFO  # noqa: E402
from test_audio_run_extras import Backend, app_for, use_backend  # noqa: E402,F401
from test_audio_runtime_errors import _Nullcontext  # noqa: E402

_DEAD = (
    "The audio runtime stopped while serving the request. Last output: abort at "
    "/home/alice/.unsloth/models/m.gguf token hf_AbCdEfGhIjKlMnOpQrStUvWxYz123456"
)
_PREFIX = (
    "The audio runtime stopped: it quit while serving the request. Last output: abort at m.gguf token "
    "hf_<redacted>"
)
_SUFFIX = ". Try again; if it fails again, load the model again."


def _raise(error):
    def post(*_args, **_kwargs):
        raise error

    return post


def _generate(error):
    b = music._backend(music._model("stable_audio"))
    b._server.post_json = _raise(error)
    return lambda: music._run(b, duration_s = 5)


def _speak(error):
    from test_audio_cpp_models import _backend_with, _speech_model

    b, server = _backend_with(_speech_model("moss_tts_nano"), None)
    server.post_json = _raise(error)
    return lambda: b.generate_audio_response("hello")


def _separate(error, tmp_path):
    model = sep._model("htdemucs", "HTDemucs-GGUF")
    b = sep._backend(model)
    b._server = sep._Recorder(model, error = error)
    b._served_session = dict((model.model_options or {}).get("session_options") or {})
    return lambda: b.separate_audio(sep.SOURCE, str(sep._out(tmp_path)))


@pytest.fixture(params = ["speak", "music", "separate"])
def call(request, tmp_path):
    if request.param == "separate":
        return lambda error: _separate(error, tmp_path)
    return {"speak": _speak, "music": _generate}[request.param]


def test_a_dead_runtime_is_runtime_stopped_without_paths_or_tokens(call):
    with pytest.raises(AudioRuntimeError) as raised:
        call(AudioCppUnavailableError(_DEAD))()
    error = raised.value
    failure = af.failure_of(error)
    assert error.status == 500 and failure is not None
    assert (failure.code, failure.status) == ("runtime_stopped", 500)
    assert failure.message == str(error) == _PREFIX + _SUFFIX
    assert "alice" not in str(error) and "hf_AbCd" not in str(error)


def test_a_silent_runtime_says_so():
    error = _runtime_stopped(
        AudioCppUnavailableError("The audio runtime did not answer: timed out")
    )
    assert str(error) == "The audio runtime stopped: did not answer: timed out" + _SUFFIX
    assert str(_runtime_stopped(AudioCppUnavailableError(""))).startswith(
        "The audio runtime stopped: no reason given."
    )


def test_a_crash_already_named_keeps_its_code(call):
    oom = af.classify_audio_cpp_failure("cudaMalloc failed: out of memory", backend = "cuda")
    crash = af.attach_failure(AudioCppUnavailableError(_DEAD), oom)
    with pytest.raises(AudioCppUnavailableError) as raised:
        call(crash)()
    assert raised.value is crash and af.failure_of(raised.value).code == "out_of_memory"


def test_the_worker_sends_the_code_and_the_orchestrator_rebuilds_it(monkeypatch):
    from core.inference.orchestrator import InferenceOrchestrator
    from core.inference.worker import _handle_generate_audio

    class _Dead:
        def generate_audio_response(self, **_kwargs):
            raise _runtime_stopped(AudioCppUnavailableError(_DEAD))

    responses: queue.Queue = queue.Queue()
    _handle_generate_audio(
        _Dead(), {"request_id": "r1", "text": "hi"}, responses, threading.Event()
    )
    sent = responses.get_nowait()
    assert (sent["code"], sent["status"]) == (AUDIO_RUNTIME_ERROR_CODE, 500)
    assert sent["failure"]["code"] == "runtime_stopped" and sent["failure"]["status"] == 500

    def rebuilt(payload):
        orchestrator = InferenceOrchestrator.__new__(InferenceOrchestrator)
        cmds = []
        reader = lambda *, timeout: {**payload, "request_id": cmds[0]["request_id"]}
        for name, value in {
            "_ensure_subprocess_alive": lambda: True,
            "_send_cmd": cmds.append,
            "_direct_reader": lambda _rid, _cancel = None: (reader, lambda **_k: None, lambda: None),
            "_reserve_worker": lambda _why: _Nullcontext(),
            "_wait_worker_idle": lambda **_k: True,
            "_claim_worker": lambda _cancel: None,
            "_release_worker": lambda *_a, **_k: None,
        }.items():
            monkeypatch.setattr(orchestrator, name, value, raising = False)
        orchestrator.active_model_name = "m"
        orchestrator.models = {"m": {}}
        orchestrator._gen_lock = threading.Lock()
        orchestrator._send_order_lock = threading.Lock()
        orchestrator._unload_pending = False
        with pytest.raises(AudioRuntimeError) as raised:
            orchestrator.generate_audio_response("hello")
        return raised.value

    error = rebuilt(sent)
    assert error.status == 500 and af.failure_of(error).code == "runtime_stopped"
    assert af.failure_of(rebuilt({**sent, "failure": {**sent["failure"], "code": "bogus"}})) is None
    # Without the code in the known set the parent drops it (the commit this test pins).
    monkeypatch.setattr(af, "_CODES", af._CODES - {"runtime_stopped"})
    assert af.failure_of(rebuilt(sent)) is None


def test_the_route_answers_500_with_the_code_and_no_retry_header(use_backend):
    import routes.inference as ri
    from models.inference import ChatCompletionRequest
    from utils.api_errors import install_api_error_handlers

    def dead(_kwargs):
        raise _runtime_stopped(AudioCppUnavailableError(_DEAD))

    use_backend(Backend(KOKORO, SPEAK_INFO, run = dead))
    payload = ChatCompletionRequest(messages = [{"role": "user", "content": "hello"}])
    with pytest.raises(HTTPException) as raised:
        asyncio.run(ri._generate_tts_wav("hello", payload, request = None, current_subject = "t"))
    assert raised.value.status_code == 500 and raised.value.headers is None
    assert raised.value.detail == {"code": "runtime_stopped", "message": _PREFIX + _SUFFIX}

    app = app_for(ALICE)
    install_api_error_handlers(app)
    with TestClient(app) as client:
        response = client.post("/v1/audio/run", json = {"workflow": "speak", "text": "hello"})
    assert response.status_code == 500 and "retry-after" not in response.headers
    assert response.json()["error"]["code"] == "runtime_stopped"
