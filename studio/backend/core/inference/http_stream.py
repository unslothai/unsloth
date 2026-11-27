# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""HTTP stream cleanup shared by local and external OpenAI transports."""

import re

# SSE lines end only at CRLF, LF or CR. str.splitlines() (and httpx's aiter_lines) also cuts at
# U+2028, U+2029 and U+0085, which JSON leaves unescaped inside a delta.
SSE_LINE_END = re.compile(r"\r\n|\r|\n")


async def aiter_sse_lines(response):
    """``response.aiter_lines()`` that splits only at SSE line ends."""
    if not hasattr(response, "aiter_text"):
        # Not an httpx response: its own lines are already whole.
        lines = response.aiter_lines()
        try:
            async for line in lines:
                yield line
        finally:
            await lines.aclose()
        return
    pending = ""
    # Closed here, not by the finalizer in another task (see routes.inference passthrough).
    chunks = response.aiter_text()
    try:
        async for text in chunks:
            held_cr = pending.endswith("\r")
            pending += text
            if not held_cr and "\n" not in text and "\r" not in text:
                continue
            # A trailing CR may be the first half of a CRLF.
            cut = len(pending) - pending.endswith("\r")
            *lines, rest = SSE_LINE_END.split(pending[:cut])
            pending = rest + pending[cut:]
            for line in lines:
                yield line
    finally:
        await chunks.aclose()
    if pending:
        lines = SSE_LINE_END.split(pending)
        if lines[-1] == "":
            lines.pop()
        for line in lines:
            yield line


async def closing_response_lines(response):
    """Close the HTTP response before the line iterator, including cancellation."""
    lines = aiter_sse_lines(response)
    try:
        async for line in lines:
            yield line
    finally:
        await response.aclose()
        await lines.aclose()
