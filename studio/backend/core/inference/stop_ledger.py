# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What the parent tells the worker without sending it."""

from typing import Any, Optional

_ID_BYTES = 36

_SLOTS = 256


class StopLedger:
    """The request ids the parent has stopped, and whether the worker holding this reads."""

    def __init__(self, ctx: Any):
        self._lock = ctx.Lock()
        self._slots = ctx.Array("c", _SLOTS * _ID_BYTES, lock = False)
        self._read_by_worker = ctx.Value("b", 0, lock = False)
        self._written = ctx.Value("l", 0, lock = False)

    def worker_reads_this(self) -> None:
        """Said once by the worker, as it enters the loop that reads names from here."""
        self._read_by_worker.value = 1

    def read_by_worker(self) -> bool:
        """Whether the worker's loop takes names from here. Not that it has read this one."""
        return bool(self._read_by_worker.value)

    def stop(self, request_id: str) -> bool:
        """Record a stop. False for an id no slot could hold, which is never recorded."""
        entry = _entry(request_id)
        if entry is None:
            return False
        with self._lock:
            if not self._holds(entry):
                start = (self._written.value % _SLOTS) * _ID_BYTES
                self._slots[start : start + _ID_BYTES] = entry
                self._written.value += 1
        return True

    def snapshot(self, since: int = -1) -> tuple[int, Optional[set]]:
        """The recorded ids, and the number of writes they were read at."""
        written = self._written.value
        if written == since:
            return written, None
        with self._lock:
            written = self._written.value
            raw = bytes(self._slots)
        return written, {
            raw[start : start + _ID_BYTES].rstrip(b"\0").decode("utf-8", "replace")
            for start in range(0, min(written, _SLOTS) * _ID_BYTES, _ID_BYTES)
        }

    def _holds(self, entry: bytes) -> bool:
        """Whether this entry is one of the ones still recorded. Callers hold the lock."""
        for slot in range(min(self._written.value, _SLOTS)):
            start = slot * _ID_BYTES
            if bytes(self._slots[start : start + _ID_BYTES]) == entry:
                return True
        return False


def _entry(request_id: Optional[str]) -> Optional[bytes]:
    """One slot's worth of bytes, or None for an id no slot could hold."""
    if not request_id:
        return None
    entry = str(request_id).encode("utf-8", "replace")
    if len(entry) > _ID_BYTES or b"\0" in entry:
        return None
    return entry.ljust(_ID_BYTES, b"\0")


class PendingTeardowns:
    """How many commands that end everything are on their way to the worker."""

    def __init__(self, ctx: Any):
        self._count = ctx.Value("l", 0)

    def sending(self) -> None:
        with self._count.get_lock():
            self._count.value += 1

    def unsent(self) -> None:
        """Count back a command that could not be sent after all, which nothing else will"""
        self._counted_off()

    def taken(self) -> None:
        self._counted_off()

    def _counted_off(self) -> None:
        """One fewer on its way. Clamped, because the two ends are different processes:"""
        with self._count.get_lock():
            self._count.value = max(0, self._count.value - 1)

    def any_in_flight(self) -> bool:
        return self._count.value > 0
