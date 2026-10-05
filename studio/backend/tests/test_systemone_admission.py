# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Route admission must bound preprocessing, even after caller cancellation."""

import asyncio
import base64
import contextvars
import io
import threading

import pytest
from fastapi import HTTPException
from PIL import Image

from core.systemone import catalog, media
from routes import systemone


@pytest.mark.parametrize("model", ["clef-flash", "clef"])
@pytest.mark.parametrize("finish", ["success", "invalid", "cancel"])
def test_admission_precedes_decode_and_survives_caller_cancellation(monkeypatch, model, finish):
    entered, proceed, released = threading.Event(), threading.Event(), threading.Event()
    calls = []
    caller = contextvars.ContextVar("caller", default = None)
    caller.set("managed-account")
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format = "PNG")
    images = ["data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()]
    original = media.prepare

    class Slots(threading.BoundedSemaphore):
        def release(self):
            super().release()
            released.set()

    slots = Slots(1)

    def prepare(*args, **kwargs):
        assert caller.get() == "managed-account"
        first = not calls
        calls.append(True)
        entered.set()
        assert proceed.wait(5), "decoder was not released"
        if first and finish == "invalid":
            raise media.InvalidMedia("planned invalid image")
        return original(*args, **kwargs)

    monkeypatch.setattr(systemone, "_media_admission", slots, raising = False)
    monkeypatch.setattr(media, "prepare", prepare)
    monkeypatch.setattr(systemone.decision_runtime, "accepts_images", lambda _: True)
    monkeypatch.setattr(systemone.decision_runtime, "decide", lambda *_: {"answers": {}})

    async def scenario():
        def request():
            return systemone._decide(
                catalog.CHECKPOINTS[model],
                "Inspect",
                {"q": systemone.QuestionIn(type = "noul")},
                images,
            )

        first = asyncio.create_task(request())
        second = None
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            if finish == "cancel":
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
            second = asyncio.create_task(request())
            await asyncio.wait({second}, timeout = 0.5)
            assert calls == [True], "excess request entered decoding before admission"
            assert second.done() and not released.is_set()
        finally:
            proceed.set()
            outcomes = await asyncio.gather(
                first, *([second] if second else []), return_exceptions = True
            )
        busy = outcomes[1]
        assert isinstance(busy, HTTPException) and busy.status_code == 529
        assert busy.detail["error_type"] == "overloaded" and busy.headers["Retry-After"] == "1"
        if finish == "cancel":
            assert isinstance(outcomes[0], asyncio.CancelledError)
        elif finish == "invalid":
            assert isinstance(outcomes[0], HTTPException) and outcomes[0].status_code == 422
        else:
            assert outcomes[0] == {"answers": {}}
        assert await asyncio.to_thread(released.wait, 5)
        assert slots.acquire(blocking = False), "completed decoder leaked its admission slot"
        slots.release()

    asyncio.run(scenario())


def test_cancelled_queued_decoder_retains_its_slot(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    unblock = threading.Event()
    monkeypatch.setattr(systemone.decision_runtime, "accepts_images", lambda _: True)

    async def scenario():
        loop = asyncio.get_running_loop()
        entered, released = asyncio.Event(), asyncio.Event()

        class Slots(threading.BoundedSemaphore):
            def release(self):
                super().release()
                loop.call_soon_threadsafe(released.set)

        slots = Slots(1)
        monkeypatch.setattr(systemone, "_media_admission", slots)
        loop.set_default_executor(ThreadPoolExecutor(max_workers = 1))

        def occupy():
            loop.call_soon_threadsafe(entered.set)
            assert unblock.wait(5)

        blocker = loop.run_in_executor(None, occupy)
        await asyncio.wait_for(entered.wait(), 5)
        first = asyncio.create_task(systemone._prepare_media(None, "state", None))
        try:
            await asyncio.sleep(0)  # Let the caller enqueue behind the occupied executor.
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            with pytest.raises(HTTPException) as busy:
                await systemone._prepare_media(None, "state", None)
            assert busy.value.status_code == 529 and not slots.acquire(blocking = False)
        finally:
            unblock.set()
            await blocker
            await asyncio.wait_for(released.wait(), 5)
        assert slots.acquire(blocking = False)
        slots.release()

    asyncio.run(scenario())


def test_executor_submission_failure_releases_admission(monkeypatch):
    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(systemone, "_media_admission", slots, raising = False)
    monkeypatch.setattr(systemone.decision_runtime, "accepts_images", lambda _: True)

    async def scenario():
        def fail(*args):
            raise RuntimeError("executor unavailable")

        monkeypatch.setattr(asyncio.get_running_loop(), "run_in_executor", fail)
        with pytest.raises(RuntimeError, match = "executor unavailable"):
            await systemone._decide(
                catalog.CHECKPOINTS["clef-flash"],
                "Inspect",
                {"q": systemone.QuestionIn(type = "noul")},
            )
        assert slots.acquire(blocking = False)
        slots.release()

    asyncio.run(scenario())
