# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Best-effort host-side file for one managed engine process launch."""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path
from typing import TextIO

from loggers import get_logger
from utils.debug_log_sources import MANAGED_ENGINE_LOG_FAMILIES
from utils.log_retention import prune_log_dir


_LOG_LIMIT_BYTES = 16 * 1024 * 1024


class ManagedEngineLog:
    def __init__(self, path: Path, stream: TextIO):
        self.path = str(path)
        self._stream = stream
        self._lock = threading.Lock()
        self._bytes = 0

    @classmethod
    def open(cls, engine: str) -> ManagedEngineLog | None:
        from utils.paths import studio_root

        subdirectory, pattern = MANAGED_ENGINE_LOG_FAMILIES[engine]
        try:
            directory = studio_root() / subdirectory
            directory.mkdir(mode = 0o700, parents = True, exist_ok = True)
            stream = tempfile.NamedTemporaryFile(
                mode = "w",
                encoding = "utf-8",
                newline = "\n",
                buffering = 1,
                prefix = f"{engine}-{time.time_ns()}-",
                suffix = ".log",
                dir = directory,
                delete = False,
            )
        except (OSError, ValueError, RuntimeError):
            get_logger(__name__).warning("Could not open the managed %s log", engine)
            return None

        path = Path(stream.name)
        log = cls(path, stream)
        if not log.write_redacted("Starting managed engine."):
            try:
                path.unlink(missing_ok = True)
            except OSError:
                pass
            return None
        prune_log_dir(directory, pattern, protect = path)
        return log

    def write_redacted(self, line: str) -> bool:
        """The caller supplies sanitized output. Internal lifecycle messages contain no user data."""
        with self._lock:
            if self._stream.closed or self._bytes >= _LOG_LIMIT_BYTES:
                return False
            try:
                line = line.rstrip("\r\n") + "\n"
                size = len(line.encode("utf-8"))
                if self._bytes + size + 64 > _LOG_LIMIT_BYTES:
                    self._stream.write("Managed engine log size limit reached.\n")
                    self._bytes = _LOG_LIMIT_BYTES
                else:
                    self._stream.write(line)
                    self._bytes += size
                return True
            except (OSError, ValueError):
                self._close()
                get_logger(__name__).warning("Could not write the managed engine log")
                return False

    def close(self) -> None:
        with self._lock:
            self._close()

    def _close(self) -> None:
        try:
            self._stream.close()
        except OSError:
            pass
