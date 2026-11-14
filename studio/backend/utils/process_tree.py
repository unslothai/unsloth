# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Descendants of one process from /proc/<pid>/task/<tid>/children, so /api/system polls skip
psutil's recursive children(), which reads every process on the host (~180 ms at ~10k processes)."""

import os
import sys
from typing import Optional

_PROC = "/proc"


def linux_descendant_pids(pid: int) -> Optional[list[int]]:
    """Every descendant of ``pid``, or None (not Linux, no CONFIG_PROC_CHILDREN, root gone, any error)
    so the caller asks psutil. Exited descendants are skipped, as psutil skips them."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        root_tasks = os.listdir(f"{_PROC}/{pid}/task")
        # Unsupported kernel: the file is absent for a live task.
        if not root_tasks or not os.path.exists(f"{_PROC}/{pid}/task/{root_tasks[0]}/children"):
            return None
        found: list[int] = []
        seen = {pid}
        pending = [pid]
        while pending:
            parent = pending.pop()
            try:
                tasks = root_tasks if parent == pid else os.listdir(f"{_PROC}/{parent}/task")
            except (FileNotFoundError, ProcessLookupError):
                continue
            for tid in tasks:
                try:
                    with open(f"{_PROC}/{parent}/task/{tid}/children", "rb") as handle:
                        listed = handle.read().split()
                except (FileNotFoundError, ProcessLookupError):
                    # A thread that exits mid-walk reparents its live children to a sibling task whose
                    # file may already be read, so this walk could miss them: psutil answers instead.
                    return None
                for raw in listed:
                    child = int(raw)
                    if child not in seen:
                        seen.add(child)
                        found.append(child)
                        pending.append(child)
        return found
    except Exception:  # noqa: BLE001 -- psutil answers instead
        return None
