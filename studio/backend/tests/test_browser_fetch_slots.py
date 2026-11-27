# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Page fetches are bounded like page requests: queued fetches keep their request bodies."""

from __future__ import annotations

import asyncio
import threading

import pytest
from fastapi import HTTPException

import routes.browser as browser_mod


class _Client:
    def __init__(self, disconnected = False):
        self.disconnected = disconnected

    async def is_disconnected(self):
        return self.disconnected


def test_page_fetches_hold_a_slot_until_their_worker_ends(monkeypatch):
    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(browser_mod, "_FETCH_SLOTS", slots, raising = False)
    release, finished = threading.Event(), threading.Event()
    # Never hang the suite: the stalled worker gives up after a few seconds either way.
    threading.Timer(5, release.set).start()

    def stalled(
        request,
        cancel_event,
        subject = None,
    ):
        release.wait(10)
        finished.set()
        return {}

    monkeypatch.setattr(browser_mod, "_fetch_and_build", stalled)
    request = browser_mod.BrowserFetchRequest(
        url = "https://site.test/", method = "POST", body = "A" * 1024
    )

    def call(client):
        return asyncio.run(browser_mod.browser_fetch(request, client, current_subject = "user"))

    with pytest.raises(HTTPException) as caught:
        call(_Client(disconnected = True))
    assert caught.value.status_code == 499
    # The first fetch's worker is still stuck on the site: a second one is refused, not queued.
    with pytest.raises(HTTPException) as caught:
        call(_Client())
    assert caught.value.status_code == 503
    release.set()
    assert finished.wait(10)
    for _ in range(100):
        if slots.acquire(blocking = False):
            break
        threading.Event().wait(0.05)
    else:
        pytest.fail("the slot was never given back")
