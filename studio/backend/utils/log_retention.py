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
    """Delete all but the ``keep`` most recently modified ``pattern`` files in ``log_dir``.

    ``protect`` is the log the caller is writing to. It is never deleted and counts as one
    of the ``keep``, so the directory settles at ``keep``, not ``keep + 1``. Call this
    *after* opening it: pruning first leaves an extra file every time, and the new file
    only sorts newest until two loads share a second or a clock steps back.

    Only regular files count. A directory matching the glob is not a log, and a symlink is
    stat'd through but unlinked by name, so counting either shrinks how many real logs are
    kept.

    Best effort throughout: retention must never take down the thing it logs for, and
    losing a race to a concurrent writer just leaves the file for the next call. One
    unreadable entry skips that entry, not the directory -- a single dangling symlink used
    to abort the sort and disable retention entirely.
    """
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
            # Dangling symlink, vanished file, or an unreadable directory. Skip the entry
            # only: one bad name must not disable retention for the whole directory.
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
