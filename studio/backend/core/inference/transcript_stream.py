# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Progress and durable results for the Studio transcription screen."""

import asyncio
import contextlib
import json
import time

from fastapi import HTTPException

from core.inference import transcript_gallery
from loggers import get_logger

logger = get_logger(__name__)


async def stream_transcript(transcribe, title: str):
    loop = asyncio.get_running_loop()
    updates = asyncio.Queue(maxsize=1)
    closed = False
    last_update = 0.0

    def publish(update):
        if closed:
            return
        if updates.full():
            updates.get_nowait()
        updates.put_nowait({"type": "progress", **update})

    def on_progress(update):
        nonlocal last_update
        if closed:
            return
        now = time.monotonic()
        if now - last_update >= 0.25:
            last_update = now
            loop.call_soon_threadsafe(publish, update)

    async def run():
        result = await transcribe(on_progress)
        record = None
        if result.get("text", "").strip():
            try:
                record = await asyncio.to_thread(transcript_gallery.save, result, title)
            except Exception as exc:
                logger.warning("Could not save transcript to the gallery (%s)", type(exc).__name__)
        return {"type": "complete", **result, "record": record}

    task = asyncio.create_task(run())
    pending = asyncio.create_task(updates.get())
    try:
        yield json.dumps({"type": "progress", "text": ""}) + "\n"
        while not task.done():
            done, _ = await asyncio.wait(
                {task, pending}, timeout=5, return_when=asyncio.FIRST_COMPLETED
            )
            if pending in done:
                yield json.dumps(pending.result()) + "\n"
                pending = asyncio.create_task(updates.get())
            elif not done:
                yield json.dumps({"type": "heartbeat"}) + "\n"
        yield json.dumps(task.result()) + "\n"
    except HTTPException as exc:
        yield json.dumps({"type": "error", "message": exc.detail}) + "\n"
    except Exception as exc:
        logger.warning("Transcription stream failed (%s)", type(exc).__name__)
        yield json.dumps({"type": "error", "message": "Transcription failed. Try again."}) + "\n"
    finally:
        closed = True
        pending.cancel()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pending
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
