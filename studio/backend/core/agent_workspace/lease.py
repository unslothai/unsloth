# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Full-request project workspace mutation fence."""

from __future__ import annotations

import asyncio
import threading
from typing import Optional

from .guidance import project_id_from_session


class ProjectWorkspaceRequestLease:
    def __init__(self, context) -> None:
        self._context = context
        self._released = False
        self._entered = False
        self._lock = threading.Lock()

    def _enter(self) -> None:
        self._context.__enter__()
        with self._lock:
            abandoned = self._released
            self._entered = not abandoned
        if abandoned:
            self._context.__exit__(None, None, None)

    @classmethod
    async def acquire(cls, session_id: Optional[str]):
        if not isinstance(session_id, str) or not session_id.startswith("project-"):
            return None
        project_id = session_id[len("project-") :]
        if not project_id:
            return None
        from core.inference.tools import project_workspace_in_flight

        context = project_workspace_in_flight(project_id)
        lease = cls(context)
        try:
            await asyncio.to_thread(lease._enter)
            if await asyncio.to_thread(project_id_from_session, session_id) != project_id:
                await lease.release()
                return None
        except BaseException:
            await lease.release()
            raise
        return lease

    async def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
            entered = self._entered
        if entered:
            await asyncio.to_thread(self._context.__exit__, None, None, None)


__all__ = ["ProjectWorkspaceRequestLease"]
