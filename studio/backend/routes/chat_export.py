# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject
from core.chat_pdf import TOO_LONG_EXIT_CODE
from loggers import get_logger
from utils.utils import log_and_http_error

router = APIRouter()

logger = get_logger(__name__)

MAX_EXPORT_HTML_CHARS = 64 * 1024 * 1024
RENDER_TIMEOUT_S = 300
_BACKEND_DIR = Path(__file__).resolve().parent.parent
# A thread semaphore, taken in the worker thread: an asyncio one binds to a single event loop.
_render_slots = threading.BoundedSemaphore(2)
# Own pool: queued exports wait here instead of holding the shared default executor other routes use.
_render_pool = ThreadPoolExecutor(max_workers = 2, thread_name_prefix = "chat-pdf")
# Each queued export holds its HTML (up to MAX_EXPORT_HTML_CHARS); a person has one or two pending.
MAX_PENDING_EXPORTS = 16
_pending_exports = 0


class ChatPdfTooLong(Exception):
    pass


class ChatPdfRequest(BaseModel):
    model: Optional[str] = Field(None, max_length = 2000)
    title: str = Field("", max_length = 2000)
    html: str = Field(..., min_length = 1, max_length = MAX_EXPORT_HTML_CHARS)


def render_chat_pdf_in_child(
    title: str,
    body_html: str,
    model: str | None = None,
) -> bytes:
    """PyMuPDF must not run on server threads (it can crash the process); a crash here only loses this export."""
    with _render_slots:
        result = subprocess.run(
            [sys.executable, "-m", "core.chat_pdf"],
            input = json.dumps({"title": title, "html": body_html, "model": model}).encode("utf-8"),
            capture_output = True,
            cwd = _BACKEND_DIR,
            timeout = RENDER_TIMEOUT_S,
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    if result.returncode == TOO_LONG_EXIT_CODE:
        raise ChatPdfTooLong(result.stderr.decode("utf-8", "replace").strip())
    if result.returncode != 0 or not result.stdout.startswith(b"%PDF"):
        tail = result.stderr.decode("utf-8", "replace").strip()[-2000:]
        raise RuntimeError(f"PDF renderer exited with {result.returncode}: {tail}")
    return result.stdout


@router.post("/pdf")
async def export_chat_pdf(
    request: ChatPdfRequest, current_subject: str = Depends(get_current_subject)
) -> Response:
    global _pending_exports
    if _pending_exports >= MAX_PENDING_EXPORTS:
        raise HTTPException(
            status_code = 429, detail = "PDF export is busy. Please try again in a moment."
        )
    # Counted on the event loop thread only, so no lock is needed.
    _pending_exports += 1
    try:
        content = await asyncio.get_running_loop().run_in_executor(
            _render_pool,
            partial(render_chat_pdf_in_child, request.title, request.html, request.model),
        )
    except ChatPdfTooLong as error:
        raise log_and_http_error(
            error, 413, "This export is too long for one PDF.", log = logger
        ) from error
    except Exception as error:
        raise log_and_http_error(error, 500, "Could not render the PDF.", log = logger) from error
    finally:
        _pending_exports -= 1
    return Response(content = content, media_type = "application/pdf")
