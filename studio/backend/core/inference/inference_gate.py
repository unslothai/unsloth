# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Mark a function as inference so background archive embedding never runs alongside it."""

from __future__ import annotations

import functools
import inspect
import weakref

import httpx


def holds_inference(fn):
    """The archive's gap worker pauses (waiting out its batch in flight) for ``fn``'s whole run; a generator
    holds it until exhausted or closed. Imported lazily: this module must stay free of the RAG stack."""
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def coroutine(*args, **kwargs):
            import asyncio

            from core.rag.conversation_archive import _pause_prewarm, hold_inference

            release = hold_inference()
            try:
                # The join waits out the batch in flight, so off the event loop.
                await asyncio.to_thread(_pause_prewarm)
                return await fn(*args, **kwargs)
            finally:
                release()

        return coroutine

    if inspect.isgeneratorfunction(fn):

        @functools.wraps(fn)
        def generator(*args, **kwargs):
            from core.rag.conversation_archive import inference_running
            with inference_running():
                yield from fn(*args, **kwargs)

        return generator

    @functools.wraps(fn)
    def call(*args, **kwargs):
        from core.rag.conversation_archive import inference_running
        with inference_running():
            return fn(*args, **kwargs)

    return call


def pauses_background_embedding(fn):
    """For starts of long jobs (training): stop the gap worker and wait out its batch before admitting the job;
    the worker's own busy probe keeps it idle for the job's lifetime."""

    @functools.wraps(fn)
    def call(*args, **kwargs):
        from core.rag.conversation_archive import _busy

        # Held across the start, so a stream ending meanwhile cannot restart the worker before the job publishes
        # the state the busy probe reads.
        with _busy(restart = False):
            return fn(*args, **kwargs)

    return call


def gated_async_client(*args, **kwargs):
    """A per-request ``httpx.AsyncClient`` to a local engine (API passthroughs): background embedding stays paused
    until it is ``aclose``d, or collected if a path never closes it. Built through ``httpx.AsyncClient`` at call
    time, so a patched client class is honoured."""
    from core.rag.conversation_archive import hold_inference

    client = httpx.AsyncClient(*args, **kwargs)
    release = weakref.finalize(client, hold_inference())
    close = client.aclose
    send = getattr(client, "send", None)

    async def gated_send(*send_args, **send_kwargs):
        # The first request waits out the batch in flight (off the event loop); httpx routes every verb and
        # stream() through send.
        from core.rag.conversation_archive import _pause_prewarm, prewarm_running
        if prewarm_running():
            import asyncio
            await asyncio.to_thread(_pause_prewarm)
        return await send(*send_args, **send_kwargs)

    async def aclose():
        try:
            return await close()
        finally:
            release()

    client.aclose = aclose
    if inspect.iscoroutinefunction(send):
        client.send = gated_send
    return client
