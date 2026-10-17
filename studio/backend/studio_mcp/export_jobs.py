# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Export jobs started over MCP. Loading a checkpoint and exporting it can take many minutes, longer than an agent waits on one tool call, so export_model runs them as a background task in the job store and get_job reports on it. Jobs are keyed by account, and an id from another account reads as unknown."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from studio_mcp import job_store
from studio_mcp.job_store import finish, mark_cancelled, settle  # noqa: F401  (public names)
from studio_mcp.tools import leaf_name

CAPACITY = 64


@dataclass
class ExportJob(job_store.Job):
    kind: str = "export"
    format: str = ""
    # The op id sent with the step in flight; cancel stops the worker only while that op runs.
    op_id: Optional[str] = None
    output: Optional[str] = None


STORE = job_store.JobStore(
    capacity = CAPACITY, unknown = "No such export job", failure = "The export stopped unexpectedly."
)
_jobs = STORE.jobs


def _evict() -> None:
    STORE.evict()


def output_name(path: Any) -> Optional[str]:
    """An export's output as the agent may see it: relative to Unsloth Studio's exports folder, or just its name."""
    if not isinstance(path, str) or not path:
        return None
    if not (path.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[\\/]", path)):
        return path
    try:
        from utils.paths import exports_root
        return Path(path).resolve().relative_to(Path(exports_root()).resolve()).as_posix()
    except Exception:
        return f"{leaf_name(path)} (outside exports folder)"


def start(account_id: str, format: str, run: Callable[[ExportJob], Awaitable[None]]) -> ExportJob:
    job = ExportJob(job_id = job_store.new_id(), account_id = account_id, format = format)
    return STORE.start(job, run)


def lookup(account_id: str, job_id: str) -> ExportJob:
    return STORE.lookup(account_id, job_id)


def any_running() -> bool:
    return bool(STORE.running())


def jobs_of(account_id: str) -> list[ExportJob]:
    return STORE.jobs_of(account_id)


def _reset() -> None:
    """Test hook."""
    STORE.reset()
