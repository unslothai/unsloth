# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional


def new_path() -> str:
    return os.path.join(tempfile.gettempdir(), f"unsloth-dl-heartbeat-{uuid.uuid4().hex}")


def read(path: Optional[str]) -> Optional[int]:
    if not path:
        return None
    try:
        text = Path(path).read_text(encoding = "utf-8").strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def age(path: Optional[str]) -> Optional[float]:
    if not path:
        return None
    try:
        return max(0.0, time.time() - Path(path).stat().st_mtime)
    except OSError:
        return None


def remove(path: Optional[str]) -> None:
    if not path:
        return
    for candidate in (path, f"{path}.tmp"):
        try:
            Path(candidate).unlink(missing_ok = True)
        except OSError:
            pass


class HeartbeatWriter:
    def __init__(
        self,
        path: str,
        interval: float = 1.0,
    ) -> None:
        self._path = path
        self._interval = interval
        self._total = 0
        self._flushed_at = 0.0
        self._lock = threading.Lock()

    def add(self, n: int) -> None:
        with self._lock:
            self._total += n
            now = time.monotonic()
            if now - self._flushed_at < self._interval:
                return
            self._flushed_at = now
            tmp = f"{self._path}.tmp"
            try:
                with open(tmp, "w", encoding = "utf-8") as handle:
                    handle.write(str(self._total))
                os.replace(tmp, self._path)
            except OSError:
                pass
