# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""PID record line 4 is linux:<boot_id>:<start_ticks>; keep lines 1-3 compatible, Rust reader aligned."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable, Sequence, Union

RecordedStart = Union[float, str, None]


def linux_process_identity(pid: int) -> str | None:
    if sys.platform != "linux":
        return None
    try:
        # comm may contain spaces and parentheses. Field 22 follows the last ')'.
        stat = Path(f"/proc/{pid}/stat").read_text(encoding = "utf-8")
        ticks = int(stat[stat.rindex(")") + 1 :].split()[19])
        boot = Path("/proc/sys/kernel/random/boot_id").read_text(encoding = "ascii").strip()
        return f"linux:{boot}:{ticks}" if boot else None
    except (OSError, ValueError, IndexError, UnicodeError):
        return None


def recorded_start(lines: Sequence[str]) -> RecordedStart:
    if len(lines) > 3:
        parts = lines[3].strip().split(":")
        if len(parts) == 3 and parts[0] == "linux" and parts[1]:
            try:
                ticks = int(parts[2])
                if 0 <= ticks < 2**64:
                    return f"linux:{parts[1]}:{ticks}"
            except ValueError:
                pass
    try:
        epoch = float(lines[1])
        return epoch if math.isfinite(epoch) else None
    except (IndexError, ValueError):
        return None


def process_create_time(pid: int) -> float | None:
    try:
        import psutil
        return psutil.Process(pid).create_time()
    except Exception:
        return None


def pid_start_matches(
    pid: int,
    starts: Sequence[RecordedStart],
    create_time: Callable[[int], float | None] = process_create_time,
) -> bool:
    known = [start for start in starts if start is not None]
    if not known:
        return True
    identity = linux_process_identity(pid) if any(isinstance(s, str) for s in known) else None
    epoch = create_time(pid) if any(not isinstance(s, str) for s in known) else None
    compared = False
    for start in known:
        if isinstance(start, str):
            if identity is not None:
                compared = True
                if identity == start:
                    return True
        elif epoch is not None:
            compared = True
            if abs(epoch - start) <= 1.0:
                return True
    return not compared


def pid_reuse_proven(
    pid: int,
    starts: Sequence[RecordedStart],
    create_time: Callable[[int], float | None] = process_create_time,
) -> bool:
    """A mismatch proves reuse only with stable starts: Linux epoch times move with boot-time readjustments."""
    known = [start for start in starts if start is not None]
    if not known or pid_start_matches(pid, known, create_time):
        return False
    if sys.platform != "linux":
        return True
    return (
        all(isinstance(start, str) for start in known) and linux_process_identity(pid) is not None
    )
