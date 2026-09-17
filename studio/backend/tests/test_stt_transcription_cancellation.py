# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import socket
import threading

import pytest
from fastapi import HTTPException

import routes.inference as inference_route
from core.inference import stt_ggml_sidecar as ggml_module
from core.inference import stt_mtmd_sidecar as mtmd_module
from core.inference import stt_sidecar as stt_module
from core.inference.stt_ggml_sidecar import GgmlSttSidecar
from core.inference.stt_mtmd_sidecar import MtmdSttSidecar
from core.inference.stt_sidecar import (
    SttLoadCancelledError,
    SttTranscriptionCancelledError,
    WhisperSttSidecar,
)


def test_transformers_load_inherits_disconnect_before_registration(monkeypatch):
    # The load reaches `import torch` before it registers, so a runner without torch
    # cannot exercise this at all (the cross-platform jobs have no torch wheel).
    pytest.importorskip("torch")
    owner = threading.Event()
    sidecar = WhisperSttSidecar()
    resident = object()
    sidecar._engine = resident
    sidecar._model_id = "tiny"
    sidecar._device = "cpu"
    monkeypatch.setattr(
        stt_module,
        "ensure_stt_available",
        lambda: sidecar.cancel_transcription(owner),
    )
    monkeypatch.setattr(
        sidecar,
        "_ensure_model_downloaded",
        lambda _model: pytest.fail("cancelled load reached its cached checkpoint"),
    )

    with pytest.raises(SttLoadCancelledError):
        sidecar.load("small", owner)
    assert sidecar.is_loading() is False
    assert sidecar._engine is resident
    assert sidecar.loaded_model == "tiny"


def test_ggml_load_inherits_disconnect_before_registration(monkeypatch):
    class _Reservation:
        def close(self):
            pass

    owner = threading.Event()
    sidecar = GgmlSttSidecar()
    sidecar._model_id = "base"

    def disconnect_during_engine_probe():
        sidecar.cancel_transcription(owner)
        return "whisper-server"

    monkeypatch.setattr(ggml_module, "ensure_engine_available", disconnect_during_engine_probe)
    monkeypatch.setattr(sidecar, "_process_alive", lambda: True)
    monkeypatch.setattr(sidecar, "_ensure_model_downloaded", lambda _model: "model.gguf")
    monkeypatch.setattr(
        sidecar,
        "_release_locked",
        lambda: pytest.fail("cancelled load evicted the resident GGUF model"),
    )
    monkeypatch.setattr(sidecar, "_reserve_free_port", lambda: (_Reservation(), 12345))
    monkeypatch.setattr(ggml_module, "_whisper_install_marker", lambda _binary: None)
    monkeypatch.setattr(
        ggml_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("cancelled load spawned whisper-server"),
    )

    with pytest.raises(SttLoadCancelledError):
        sidecar.load("tiny", owner)
    assert sidecar.is_loading() is False


def test_mtmd_load_inherits_disconnect_before_registration(monkeypatch):
    owner = threading.Event()
    sidecar = MtmdSttSidecar()

    def disconnect_during_training_check():
        sidecar.cancel_transcription(owner)
        return False

    monkeypatch.setattr(mtmd_module, "ensure_engine_available", lambda: "llama-server")
    monkeypatch.setattr(mtmd_module, "_training_active", disconnect_during_training_check)
    monkeypatch.setattr(sidecar, "_process_alive", lambda: False)
    monkeypatch.setattr(
        sidecar,
        "_ensure_model_downloaded",
        lambda _model: pytest.fail("cancelled load reached its cached checkpoint"),
    )

    with pytest.raises(SttLoadCancelledError):
        sidecar.load("qwen3-asr-0.6b", owner)
    assert sidecar.is_loading() is False


def test_disconnected_load_cancels_only_its_request_event(monkeypatch):
    class _DisconnectedRequest:
        async def is_disconnected(self):
            return True

    sidecar = WhisperSttSidecar()
    sibling_owner = threading.Event()
    sibling_load_cancel = threading.Event()
    with sidecar._load_state_lock:
        sidecar._loading = True
        sidecar._load_cancel_event = sibling_load_cancel
        sidecar._load_owner_cancel_event = sibling_owner
    cancelled_request = None

    def load(
        _model,
        _engine,
        request_cancel_event,
        device = None,
    ):
        nonlocal cancelled_request
        cancelled_request = request_cancel_event
        assert request_cancel_event.wait(1), "disconnect must cancel this load request"
        assert not sibling_load_cancel.is_set(), "another caller's startup was cancelled"
        raise SttLoadCancelledError("STT load cancelled.")

    monkeypatch.setattr(inference_route, "_stt_sidecar_for", lambda _engine: sidecar)
    monkeypatch.setattr(inference_route, "_stt_lifecycle", lambda: (load, None))
    monkeypatch.setattr(
        inference_route,
        "_resolve_serving_stt_engine",
        lambda _engine: "transformers",
    )

    with pytest.raises(HTTPException) as raised:
        asyncio.run(
            inference_route.stt_load(
                inference_route.SttLoadRequest(model = "small", engine = "transformers"),
                _DisconnectedRequest(),
                "test-subject",
            )
        )

    assert raised.value.status_code == 409
    assert cancelled_request is not None and cancelled_request.is_set()
    assert not sibling_load_cancel.is_set()


def test_disconnected_raw_transcription_cancels_its_sidecar(monkeypatch):
    class _DisconnectedRequest:
        async def is_disconnected(self):
            return True

    class _Sidecar:
        cancelled = False

        def cancel_transcription(self, cancel_event):
            self.cancelled = True
            cancel_event.set()
            return True

        def transcribe(self, _raw, _model, _language, _fast, cancel_event):
            assert cancel_event.wait(1), "disconnect must reach the sidecar"
            raise SttTranscriptionCancelledError("Transcription cancelled.")

    loaded = []

    def load(
        model,
        _engine,
        _request_cancel_event,
        device = None,
    ):
        # Stubbed for the same reason the sibling test above stubs it: the real
        # implicit load goes through the registry, which refuses with
        # SttModelNotDownloadedError (409) unless a snapshot happens to be on disk.
        # Without this the disconnect below is never reached and the assert reads
        # `409 == 499`, which is what this test does on a machine that has never
        # downloaded an STT model. Cancellation, not download state, is the subject.
        loaded.append(model)
        return None

    sidecar = _Sidecar()
    monkeypatch.setattr(inference_route, "_stt_sidecar_for", lambda _engine: sidecar)
    monkeypatch.setattr(inference_route, "_stt_lifecycle", lambda: (load, None))
    monkeypatch.setattr(
        inference_route,
        "_resolve_serving_stt_engine",
        lambda _engine: "transformers",
    )

    with pytest.raises(HTTPException) as raised:
        asyncio.run(
            inference_route._transcribe_audio_result(
                b"RIFFfake",
                "small",
                None,
                True,
                "transformers",
                _DisconnectedRequest(),
            )
        )

    assert raised.value.status_code == 499
    assert sidecar.cancelled is True
    # The stub has to stay load-bearing. If the route stops loading through the
    # registry, this records nothing and the stub quietly becomes dead code that
    # keeps the test green for a path it no longer covers.
    assert loaded == ["small"], f"the transcribe path did not load through the registry: {loaded}"


def test_server_cancel_watcher_abandons_the_request_and_keeps_the_server():
    """A cancel used to SIGTERM whisper-server, so the next dictation paid a full relaunch."""
    from core.inference.stt_sidecar import _close_connection_on_cancel

    class _Socket:
        shutdown_with = None

        def shutdown(self, how):
            self.shutdown_with = how

    class _Connection:
        closed = False

        def __init__(self):
            self.sock = _Socket()

        def close(self):
            self.closed = True

    connection = _Connection()
    sock = connection.sock
    cancelled = threading.Event()
    cancelled.set()

    _close_connection_on_cancel(connection, cancelled, threading.Event())

    assert sock.shutdown_with == socket.SHUT_RDWR
    assert connection.closed is True


def test_the_gguf_sidecar_cancels_through_its_connection():
    """Both HTTP sidecars share one watcher, so neither can drift back to a process kill."""
    import inspect

    from core.inference import stt_ggml_sidecar

    source = inspect.getsource(stt_ggml_sidecar)
    assert "_close_connection_on_cancel" in source
    assert "_terminate_process_on_cancel" not in source
    assert "http.client.HTTPConnection" in source


def test_transformers_disconnect_cancels_only_its_owned_startup():
    sidecar = WhisperSttSidecar()
    owner = threading.Event()
    sibling = threading.Event()
    load_cancel = threading.Event()
    with sidecar._load_state_lock:
        sidecar._loading = True
        sidecar._load_cancel_event = load_cancel
        sidecar._load_owner_cancel_event = owner

    assert sidecar.cancel_transcription(sibling) is True
    assert sibling.is_set() and not load_cancel.is_set()

    assert sidecar.cancel_transcription(owner) is True
    assert owner.is_set() and load_cancel.is_set()


def test_ggml_disconnect_cancels_only_its_owned_startup():
    class _StartingProcess:
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

    sidecar = GgmlSttSidecar()
    owner = threading.Event()
    sibling = threading.Event()
    load_cancel = threading.Event()
    process = _StartingProcess()
    with sidecar._load_state_lock:
        sidecar._loading = True
        sidecar._load_cancel_event = load_cancel
        sidecar._load_owner_cancel_event = owner
        sidecar._starting_process = process

    assert sidecar.cancel_transcription(sibling) is True
    assert sibling.is_set() and not load_cancel.is_set()
    assert process.terminated is False

    assert sidecar.cancel_transcription(owner) is True
    assert owner.is_set() and load_cancel.is_set()
    assert process.terminated is True
