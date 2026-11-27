# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""generate_audio runs in flight, so a cancel can stop a run between its takes."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class _Run:
    cancelled: bool = False
    in_flight: bool = False


# generate_audio runs by (account, cancel_id), and ids cancelled lately; the route keeps an early
# cancel as long, so a reused id would be stopped on arrival.
_RUNS: dict[tuple[str, str], _Run] = {}
_CANCELLED: dict[tuple[str, str], float] = {}
CANCEL_MEMORY_S = 30.0


def _forget_old_cancels() -> None:
    cutoff = time.monotonic() - CANCEL_MEMORY_S
    for key in [key for key, at in _CANCELLED.items() if at < cutoff]:
        del _CANCELLED[key]


def stop_run(account: str, cancel_id: str) -> Optional[bool]:
    """Mark run ``cancel_id`` cancelled: None when no generate_audio run here has that id, True when it stops between takes, False when a take is running and the route must stop it."""
    _forget_old_cancels()
    _CANCELLED[(account, cancel_id)] = time.monotonic()
    run = _RUNS.get((account, cancel_id))
    if run is None:
        return None
    run.cancelled = True
    return not run.in_flight
