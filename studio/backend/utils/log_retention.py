# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep-newest-N retention for the per-session log directories.

The server session log has capped itself since it was added; the llama-server and
diffusion-server subprocess logs beside it never did, so they accumulated one file per
model load for the life of the install (319 files going back two months on the machine
this was found on). One helper so a fourth log directory cannot quietly opt out again.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_KEEP = 20


def prune_log_dir(log_dir: Path, pattern: str, keep: int = DEFAULT_KEEP) -> None:
    """Delete all but the ``keep`` most recently modified ``pattern`` files in ``log_dir``.

    Best effort in every direction: retention must never take down the thing it is
    logging for, and losing the race to a concurrent writer just leaves the file for the
    next call.
    """
    if keep < 0:
        return
    try:
        entries = sorted(log_dir.glob(pattern), key = lambda p: p.stat().st_mtime)
    except OSError:
        return
    if keep:
        entries = entries[:-keep]
    for old in entries:
        try:
            old.unlink(missing_ok = True)
        except OSError:
            pass
