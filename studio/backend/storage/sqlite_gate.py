# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One process-wide gate around every Studio SQLite connection open and close.

SQLite 3.51.0-3.51.1 deadlocks when one thread's WAL close (unixIsSharingShmNode) races another
thread's open or close of the same file: they take the VFS and inode mutexes in opposite order
(fixed in 3.51.2, #10022). The VFS mutex is process-wide, so one wedged file hangs every database
(#9008): studio.db, auth.db and rag.db all open through here. Gating only close() still deadlocks.
"""

import os
import sqlite3
import threading
from typing import Any

# Reentrant: a GC finalizer can close a pooled connection inside a gated connect on this thread.
_CONNECTION_GATE = threading.RLock()


def _reset_connection_gate_after_fork() -> None:
    global _CONNECTION_GATE
    _CONNECTION_GATE = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child = _reset_connection_gate_after_fork)


class GatedConnection(sqlite3.Connection):
    # A connection dropped without close() is finalized natively and bypasses this gate.
    def close(self) -> None:
        with _CONNECTION_GATE:
            super().close()


def connect(database: str | os.PathLike[str], **kwargs: Any) -> sqlite3.Connection:
    """sqlite3.connect with opens and closes serialized process-wide."""
    with _CONNECTION_GATE:
        return sqlite3.connect(str(database), factory = GatedConnection, **kwargs)
