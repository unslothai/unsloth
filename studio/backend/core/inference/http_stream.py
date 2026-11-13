# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""HTTP stream cleanup shared by local and external OpenAI transports."""


async def closing_response_lines(response):
    """Close the HTTP response before the line iterator, including cancellation."""
    lines = response.aiter_lines().__aiter__()
    try:
        async for line in lines:
            yield line
    finally:
        await response.aclose()
        await lines.aclose()
