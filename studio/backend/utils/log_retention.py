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
from typing import Optional

DEFAULT_KEEP = 20


def prune_log_dir(
    log_dir: Path,
    pattern: str,
    keep: int = DEFAULT_KEEP,
    protect: Optional[Path] = None,
) -> None:
    """The protected log counts toward keep and is never deleted; call after opening it, not before."""
    if keep < 0:
        return

    protected = None
    if protect is not None:
        try:
            protected = Path(protect).resolve()
        except OSError:
            protected = Path(protect)

    entries = []
    saw_protected = False
    try:
        candidates = list(log_dir.glob(pattern))
    except OSError:
        return
    for path in candidates:
        try:
            stat = path.stat()
            if not path.is_file():
                continue
            if protected is not None and path.resolve() == protected:
                saw_protected = True
                continue
        except OSError:
            continue
        entries.append((stat.st_mtime, path))

    # The protected file occupies one slot.
    room = keep - 1 if saw_protected else keep
    entries.sort(key = lambda item: item[0])
    if room > 0:
        entries = entries[:-room]
    for _mtime, old in entries:
        try:
            old.unlink(missing_ok = True)
        except OSError:
            pass
