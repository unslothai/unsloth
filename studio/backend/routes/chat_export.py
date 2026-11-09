# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import subprocess
import sys
import threading
from pathlib import Path

from fastapi import APIRouter, Depends
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


class ChatPdfTooLong(Exception):
    pass


class ChatPdfRequest(BaseModel):
    title: str = Field("", max_length = 2000)
    html: str = Field(..., min_length = 1, max_length = MAX_EXPORT_HTML_CHARS)


def render_chat_pdf_in_child(title: str, body_html: str) -> bytes:
    """PyMuPDF must not run on server threads (it can crash the process); a crash here only loses this export."""
    with _render_slots:
        result = subprocess.run(
            [sys.executable, "-m", "core.chat_pdf"],
            input = json.dumps({"title": title, "html": body_html}).encode("utf-8"),
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
    try:
        content = await asyncio.to_thread(render_chat_pdf_in_child, request.title, request.html)
    except ChatPdfTooLong as error:
        raise log_and_http_error(
            error, 413, "This export is too long for one PDF.", log = logger
        ) from error
    except Exception as error:
        raise log_and_http_error(error, 500, "Could not render the PDF.", log = logger) from error
    return Response(content = content, media_type = "application/pdf")
