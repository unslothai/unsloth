# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whether a local-model scan actually saw everything it walked.

The scanners are deliberately forgiving: a child that cannot be read is skipped so one bad
entry does not cost the whole listing. For a LISTING that is right, but a caller memoizing a
MISS as a confirmed absence needs to know the difference between "not there" and "could not
look", and a suppressed per-child error is invisible from the outside -- the pass returns a
shorter list and no exception.

So a scanner notes it here, and only a caller that opened a collector pays any attention.
Scoped with a ContextVar rather than a module global because the same scanners serve the
models route concurrently: the pass that opened the collector is the only one whose
incidents it sees, and any other caller's note is a no-op.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

_collector: ContextVar[Optional[list[str]]] = ContextVar("scan_incidents", default = None)


def note_scan_incident(reason: str) -> None:
    """Record that this scan could not read something it walked past."""
    incidents = _collector.get()
    if incidents is not None:
        incidents.append(reason)


@contextmanager
def collecting_scan_incidents() -> Iterator[list[str]]:
    """Collect the incidents noted by the scans run inside this block.

    The list is live: read it after the block and it holds what happened. Reset on exit, so
    a nested or later pass starts clean.
    """
    incidents: list[str] = []
    token = _collector.set(incidents)
    try:
        yield incidents
    finally:
        _collector.reset(token)
