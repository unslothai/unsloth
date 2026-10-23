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
        from core.rag.conversation_archive import _pause_prewarm
        _pause_prewarm()
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

    async def aclose():
        try:
            return await close()
        finally:
            release()

    client.aclose = aclose
    return client
