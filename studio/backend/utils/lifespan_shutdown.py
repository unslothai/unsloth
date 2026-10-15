# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resilient FastAPI lifespan shutdown cleanup.

On an abrupt shutdown (Windows console-close, interpreter teardown racing
uvicorn) the loop's default executor may already be dead, so an unguarded
``asyncio.to_thread`` raise here would abort the nested-lifespan unwind and
surface as "Application shutdown failed". Dependency-injected so it can be
unit-tested without the heavy backend import graph.
"""

import asyncio
import contextvars
import types
from typing import Callable

import structlog

logger = structlog.get_logger(__name__)


async def run_lifespan_shutdown(
    terminate_downloads: Callable[[], None],
    clear_compiled_cache: Callable[[], None],
    hw_module: types.ModuleType,
) -> None:
    """Run each shutdown step guarded so one failure can't skip the others; never raise."""
    loop = asyncio.get_running_loop()
    # Schedule and await separately: a dead executor runs inline, a body error is only logged.
    ctx = contextvars.copy_context()
    try:
        future = loop.run_in_executor(None, ctx.run, terminate_downloads)
    except RuntimeError:
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
        # Retire in-flight detection so it cannot publish over the reset.
        invalidate = getattr(hw_module, "invalidate_detection", None)
        if invalidate is not None:
            invalidate()
        hw_module.DEVICE = None
        # Health reads a set event as DEVICE authoritative. getattr: tests inject a stub.
        detection_complete = getattr(hw_module, "DETECTION_COMPLETE", None)
        if detection_complete is not None:
            detection_complete.clear()
        # Health falls back to CHAT_ONLY while clear; hide Train/Export until detection.
        hw_module.CHAT_ONLY = True
        hw_module.CHAT_ONLY_REASON = None
        hw_module.IS_ROCM = False
    except Exception as exc:
        logger.warning("clearing hardware detection state failed at shutdown: %s", exc)

    try:
        clear_compiled_cache()
    except Exception as exc:
        logger.warning("clear_compiled_cache failed at shutdown: %s", exc)
