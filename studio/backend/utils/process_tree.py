# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Descendants via /proc/<pid>/task/<tid>/children, not psutil's recursive children() scan."""

import os
import sys
from typing import Optional

_PROC = "/proc"


def linux_descendant_pids(pid: int) -> Optional[list[int]]:
    """Live descendant pids, or None if unsupported or failing; the caller then asks psutil."""
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
                    # Exiting threads reparent children; a walk may miss them, so use psutil.
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
