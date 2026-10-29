# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep-newest-N pruning for all per-session log directories, so a new one cannot quietly opt out."""

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
    """Protected log counts toward keep and is never deleted; open it before pruning. Only regular files count; one unreadable entry skips itself, not the directory."""
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
            # Dangling symlink, vanished file, or an unreadable directory.
            continue
        entries.append((stat.st_mtime, path))

    # The protected file occupies one of the slots, so the total stays at `keep`.
    room = keep - 1 if saw_protected else keep
    entries.sort(key = lambda item: item[0])
    if room > 0:
        entries = entries[:-room]
    for _mtime, old in entries:
        try:
            old.unlink(missing_ok = True)
        except OSError:
            pass
