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
    # Copy context for parity with asyncio.to_thread. Schedule and await
    # separately so a dead executor (raises at submit) runs inline, while a
    # body exception (raised at await) is logged, not re-run.
    ctx = contextvars.copy_context()
    try:
        future = loop.run_in_executor(None, ctx.run, terminate_downloads)
    except RuntimeError:
        # Executor gone: run inline on the loop thread.
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
