# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""POST /audio/cancel stops the one /audio/run sent with that cancel_id, and nothing else."""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path

import httpx

from core.inference.audio_errors import AudioGenerationCancelledError

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_account_media_isolation import ALICE, isolated  # noqa: E402,F401
from test_audio_run_extras import Backend, app_for, use_backend  # noqa: E402,F401

KOKORO = "audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF"
SPEAK_INFO = {
    "is_audio": True,
    "audio_type": "audiocpp_tts",
    "audio_workflows": ["speak"],
    "audio_reference_text": None,
    "audio_required_inputs": [],
}
RUN_A, RUN_B, EARLY = "run-aaaaaaaa", "run-bbbbbbbb", "run-cccccccc"


def _blocking_backend():
    """Each run waits for its cancel event or ``release``; ``started`` names the runs under way."""
    started = {RUN_A: threading.Event(), RUN_B: threading.Event(), EARLY: threading.Event()}
    release = threading.Event()

    def run(kwargs):
        started[kwargs["text"]].set()
        cancel = kwargs["cancel_event"]
        while not release.is_set():
            if cancel.wait(0.02):
                raise AudioGenerationCancelledError("Audio generation cancelled")

    return Backend(KOKORO, SPEAK_INFO, run = run), started, release


def test_cancel_by_id_stops_only_that_run(use_backend):
    backend, started, release = _blocking_backend()
    use_backend(backend)

    async def scenario():
        transport = httpx.ASGITransport(app = app_for(ALICE))
        async with httpx.AsyncClient(transport = transport, base_url = "http://t") as client:

            def run(cancel_id):
                body = {"workflow": "speak", "text": cancel_id, "cancel_id": cancel_id}
                return asyncio.create_task(client.post("/api/inference/audio/run", json = body))

            a, b = run(RUN_A), run(RUN_B)
            for name in (RUN_A, RUN_B):
                assert await asyncio.to_thread(started[name].wait, 10)
            # A chat cancel_id lives in another namespace.
            chat = await client.post("/api/inference/cancel", json = {"cancel_id": RUN_A})
            stopped = await client.post("/v1/audio/cancel", json = {"cancel_id": RUN_A})
            try:
                done_a = await asyncio.wait_for(asyncio.shield(a), 10)
            finally:
                release.set()
            return chat.json(), stopped.json(), done_a, await b

    chat, stopped, done_a, done_b = asyncio.run(scenario())
    assert chat == {"cancelled": 0}
    assert stopped == {"cancelled": 1}
    assert done_a.status_code == 499
    assert done_b.status_code == 200 and len(done_b.json()["clips"]) == 1


def test_an_early_cancel_is_kept_for_the_run_that_follows(use_backend):
    backend, started, _release = _blocking_backend()
    use_backend(backend)

    async def scenario():
        transport = httpx.ASGITransport(app = app_for(ALICE))
        async with httpx.AsyncClient(transport = transport, base_url = "http://t") as client:
            early = await client.post("/api/inference/audio/cancel", json = {"cancel_id": EARLY})
            body = {"workflow": "speak", "text": EARLY, "cancel_id": EARLY}
            return early.json(), await client.post("/api/inference/audio/run", json = body)

    early, response = asyncio.run(scenario())
    assert early == {"cancelled": 0}
    assert response.status_code == 499


def test_cancel_ids_are_checked():
    from fastapi.testclient import TestClient
    with TestClient(app_for(ALICE)) as client:
        for bad in ("short", "has space!", "x" * 65):
            response = client.post("/api/inference/audio/cancel", json = {"cancel_id": bad})
            assert response.status_code == 422
        run = {"workflow": "speak", "text": "hi", "cancel_id": "no"}
        assert client.post("/api/inference/audio/run", json = run).status_code == 422
