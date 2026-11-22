# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""join() raises on a thread enumerated before it starts, so retry the join until the deadline."""

from __future__ import annotations

import threading
import time


def join_when_started(thread: threading.Thread, timeout: float = 5.0) -> bool:
    """True only for a thread that actually finished; one still unstarted at the deadline is False."""
    # Joining yourself never becomes possible, so do not retry it.
    if thread is threading.current_thread():
        raise RuntimeError("cannot join current thread")
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        try:
            thread.join(timeout = max(remaining, 0.0))
        except RuntimeError:
            if remaining <= 0:
                # Not started by the deadline means undrained; is_alive() is False before start too.
                return False
            time.sleep(0.005)
            continue
        return not thread.is_alive()
