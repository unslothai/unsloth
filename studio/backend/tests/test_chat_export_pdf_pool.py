# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Queued PDF exports must not hold the shared default executor that other routes use."""

import asyncio
import subprocess
import threading

import routes.chat_export as chat_export


def test_queued_exports_leave_the_default_executor_free(monkeypatch):
    release = threading.Event()

    def slow_render(*_args, **_kwargs):
        release.wait(30)
        return subprocess.CompletedProcess([], 0, stdout = b"%PDF-1.7 ok", stderr = b"")

    monkeypatch.setattr(chat_export.subprocess, "run", slow_render)
    # More queued exports than the default executor has threads; the waiting cap is tested on its own.
    monkeypatch.setattr(chat_export, "MAX_PENDING_EXPORTS", 1000, raising = False)
    request = chat_export.ChatPdfRequest(title = "t", html = "<p>x</p>")

    async def scenario():
        exports = [
            asyncio.create_task(chat_export.export_chat_pdf(request, current_subject = "u"))
            for _ in range(64)
        ]
        await asyncio.sleep(0.5)
        try:
            # Any other route doing asyncio.to_thread (training status, history, ...).
            other = await asyncio.wait_for(asyncio.to_thread(lambda: "ok"), timeout = 5)
        finally:
            release.set()
            responses = await asyncio.gather(*exports)
        return other, responses

    other, responses = asyncio.run(scenario())
    assert other == "ok"
    assert len(responses) == 64
    assert all(response.body.startswith(b"%PDF") for response in responses)
