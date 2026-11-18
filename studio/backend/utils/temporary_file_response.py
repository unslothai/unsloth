# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import logging
from pathlib import Path

from anyio import CancelScope
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse
from starlette.types import Receive, Scope, Send

logger = logging.getLogger(__name__)


class TemporaryFileResponse(FileResponse):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with CancelScope(shield = True):
                try:
                    await run_in_threadpool(Path(self.path).unlink, missing_ok = True)
                except OSError:
                    logger.warning(
                        "Could not remove temporary download %s", self.path, exc_info = True
                    )
