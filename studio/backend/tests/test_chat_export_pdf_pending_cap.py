# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Each queued PDF export holds its whole HTML body; past a small number waiting, new ones are
turned away with 429 instead of piling up in the shared backend's memory."""

import asyncio
import subprocess
import threading

from fastapi import HTTPException

import routes.chat_export as chat_export

QUEUED = 40


def test_waiting_exports_are_bounded(monkeypatch):
    release = threading.Event()
    rendered = []

    def slow_render(*_args, **_kwargs):
        release.wait(30)
        rendered.append(1)
        return subprocess.CompletedProcess([], 0, stdout = b"%PDF-1.7 ok", stderr = b"")

    monkeypatch.setattr(chat_export.subprocess, "run", slow_render)
    request = chat_export.ChatPdfRequest(title = "t", html = "<p>x</p>")

    async def scenario():
        exports = [
            asyncio.create_task(chat_export.export_chat_pdf(request, current_subject = "u"))
            for _ in range(QUEUED)
        ]
        await asyncio.sleep(0.5)
        release.set()
        return await asyncio.gather(*exports, return_exceptions = True)

    results = asyncio.run(scenario())
    busy = [r for r in results if isinstance(r, HTTPException)]
    done = [r for r in results if not isinstance(r, BaseException)]
    assert busy, "every queued export was accepted and kept its body"
    assert all(r.status_code == 429 for r in busy)
    assert len(done) <= 16 and all(r.body.startswith(b"%PDF") for r in done)
    assert len(rendered) == len(done)

    release.set()
    response = asyncio.run(chat_export.export_chat_pdf(request, current_subject = "u"))
    assert response.body.startswith(b"%PDF")
