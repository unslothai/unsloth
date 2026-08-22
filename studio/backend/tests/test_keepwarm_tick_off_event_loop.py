# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep idle-unload settings reads off the event loop."""

from __future__ import annotations

import asyncio
import threading

import core.inference.llama_keepwarm as llama_keepwarm
import core.inference.media_keepwarm as media_keepwarm
import utils.openai_auto_switch_settings as auto_switch_settings


def _record_thread(threads: list[int], result):
    def _stub(*_args, **_kwargs):
        threads.append(threading.get_ident())
        return result

    return _stub


def test_the_media_half_reads_off_the_event_loop_thread(monkeypatch):
    threads: list[int] = []
    monkeypatch.setattr(media_keepwarm, "_effective_ttl", _record_thread(threads, 0.0))

    async def _drive():
        await media_keepwarm.idle_unload_step()
        return threading.get_ident()

    loop_thread = asyncio.run(_drive())

    assert threads, "the tick never read the TTL"
    assert threads[0] != loop_thread, "the media TTL read ran on the event loop thread"


def test_the_chat_half_reads_off_the_event_loop_thread(monkeypatch):
    """Exercise the settings read through the real polling loop."""
    threads: list[int] = []
    monkeypatch.setattr(
        auto_switch_settings, "get_auto_unload_idle_seconds", _record_thread(threads, 0.0)
    )

    async def _no_media_half() -> None:
        return None

    monkeypatch.setattr(media_keepwarm, "idle_unload_step", _no_media_half)

    async def _drive():
        tick = asyncio.create_task(llama_keepwarm.idle_unload_loop(poll_seconds = 0.001))
        for _ in range(2000):
            if threads:
                break
            await asyncio.sleep(0.001)
        tick.cancel()
        try:
            await tick
        except asyncio.CancelledError:
            pass
        return threading.get_ident()

    loop_thread = asyncio.run(_drive())

    assert threads, "the tick never read the TTL"
    assert threads[0] != loop_thread, "the idle-unload TTL read ran on the event loop thread"
