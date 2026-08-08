# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import threading

import pytest
from fastapi import HTTPException

import routes.inference as inference_route
from core.inference.stt_ggml_sidecar import GgmlSttSidecar
from core.inference.stt_sidecar import SttTranscriptionCancelledError, WhisperSttSidecar


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

    sidecar = _Sidecar()
    monkeypatch.setattr(inference_route, "_stt_sidecar_for", lambda _engine: sidecar)
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


def test_server_cancel_watcher_terminates_blocked_runtime():
    from core.inference.stt_sidecar import _terminate_process_on_cancel

    class _Process:
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

    process = _Process()
    cancelled = threading.Event()
    cancelled.set()

    _terminate_process_on_cancel(process, cancelled, threading.Event())

    assert process.terminated is True


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
