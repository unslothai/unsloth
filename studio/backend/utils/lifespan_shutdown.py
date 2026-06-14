# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resilient FastAPI lifespan shutdown cleanup.

On an abrupt shutdown (closing the Windows console window, or interpreter
teardown racing uvicorn's graceful stop) the event loop's default thread-pool
executor can already be shut down by the time the lifespan shutdown runs, so
``asyncio.to_thread()`` raises ``RuntimeError: cannot schedule new futures after
shutdown``. If that raise escapes, it propagates up through every nested
``merged_lifespan`` ``__aexit__`` and aborts the rest of the shutdown chain,
which surfaces as "Application shutdown failed. Exiting." and skips the later
cleanup steps.

Kept dependency-injected (callables + the hardware module passed in) and free of
the heavy backend import graph so it can be unit-tested in isolation.
"""

import asyncio
import contextvars
from typing import Callable

import structlog

logger = structlog.get_logger(__name__)


async def run_lifespan_shutdown(
    terminate_downloads: Callable[[], None], clear_compiled_cache: Callable[[], None], hw_module
) -> None:
    """Run shutdown cleanup defensively; never raise.

    Each step is guarded independently so a failure in one can't skip the
    others (the original bug: an unguarded ``to_thread`` failure dropped the
    later cleanup and the whole nested-lifespan shutdown).
    """
    # Schedule and await separately (rather than asyncio.to_thread) so the two
    # failure modes stay distinct: a dead default executor makes run_in_executor
    # raise synchronously at submit time, whereas an exception from
    # terminate_downloads itself only surfaces when the future is awaited. That
    # way we only retry inline when the work never got scheduled, never when the
    # body ran and raised (which would double-execute it).
    loop = asyncio.get_running_loop()
    # Copy the current context so terminate_downloads runs with the same
    # contextvars asyncio.to_thread would have given it (exact parity with the
    # previous implementation). The only intended behaviour change is the inline
    # fallback below, taken when scheduling onto a dead executor fails.
    ctx = contextvars.copy_context()
    try:
        future = loop.run_in_executor(None, ctx.run, terminate_downloads)
    except RuntimeError:
        # Default executor already gone (teardown race). terminate_downloads is
        # itself best-effort and quick, so run it inline on the loop thread.
        try:
            ctx.run(terminate_downloads)
        except Exception as exc:
            logger.warning("terminate_downloads (inline) failed at shutdown: %s", exc)
    else:
        try:
            await future
        except Exception as exc:
            logger.warning("terminate_downloads failed at shutdown: %s", exc)

    try:
        hw_module.DEVICE = None
    except Exception as exc:
        logger.warning("clearing hardware DEVICE failed at shutdown: %s", exc)

    try:
        clear_compiled_cache()
    except Exception as exc:
        logger.warning("clear_compiled_cache failed at shutdown: %s", exc)
