# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Restyle uploads are read whole into memory: only a few may be held at once."""

import asyncio

import pytest
from fastapi import HTTPException

from routes import video_restyle as route


def test_concurrent_restyle_uploads_are_bounded(monkeypatch):
    entered = 0
    peak = 0
    release = None

    async def stalled_read(request):
        nonlocal entered, peak
        entered += 1
        peak = max(peak, entered)
        try:
            await release.wait()
            return {}, []
        finally:
            entered -= 1

    async def frames(files):
        return [], None

    monkeypatch.setattr(route, "_read_upload", stalled_read)
    monkeypatch.setattr(route, "_frames_from_upload", frames)
    monkeypatch.setattr(route.video_restyle, "preview", lambda frames, seconds: {"ok": True})

    async def main():
        nonlocal release
        release = asyncio.Event()
        tasks = [asyncio.create_task(route.preview_restyle(None, "owner")) for _ in range(12)]
        await asyncio.sleep(0.05)
        release.set()
        results = await asyncio.gather(*tasks, return_exceptions = True)
        busy = [r for r in results if isinstance(r, HTTPException) and r.status_code == 429]
        done = [r for r in results if r == {"ok": True}]
        # Slots come back once the held uploads finish.
        again = await route.preview_restyle(None, "owner")
        return busy, done, again

    busy, done, again = asyncio.run(main())
    assert peak <= 4, f"{peak} restyle uploads were held in memory at once"
    assert busy and done and len(busy) + len(done) == 12
    assert again == {"ok": True}
