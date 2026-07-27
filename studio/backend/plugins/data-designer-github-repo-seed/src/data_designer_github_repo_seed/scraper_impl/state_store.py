# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Checkpoint state management for resumable scraping."""

from __future__ import annotations

import json
import locale
import os
import threading
from pathlib import Path
from typing import Any, Dict


class StateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents = True, exist_ok = True)
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {}
        if self.path.exists():
            try:
                # errors="replace" so a file we chose not to migrate still yields
                # every key; a mangled key is one possible duplicate, not a loss.
                with self.path.open(encoding = "utf-8", errors = "replace") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._flush()

    def update(self, key: str, **kwargs) -> None:
        with self._lock:
            sub = dict(self._data.get(key, {}))
            sub.update(kwargs)
            self._data[key] = sub
            self._flush()

    def all(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._data)

    def _flush(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding = "utf-8") as f:
            json.dump(self._data, f, indent = 2, default = str)
        os.replace(tmp, self.path)


class JsonlWriter:
    """Append-only JSONL writer, thread-safe, with line buffering."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents = True, exist_ok = True)
        self._lock = threading.Lock()
        self._migrate_legacy_encoding()
        self._fh = self.path.open("a", buffering = 1, encoding = "utf-8")
        self._count_seen_keys: set[str] = set()
        # Preload seen keys for dedup across resumes
        if self.path.exists() and self.path.stat().st_size > 0:
            try:
                # errors="replace" so a file we chose not to migrate still yields
                # every key; a mangled key is one possible duplicate, not a loss.
                with self.path.open(encoding = "utf-8", errors = "replace") as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            k = self._key(obj)
                            if k is not None:
                                self._count_seen_keys.add(k)
                        except Exception:
                            pass
            except Exception:
                pass

    def _migrate_legacy_encoding(self) -> None:
        """Rewrite a file an older build wrote in the operator's locale as utf-8.

        Appending utf-8 to locale-encoded lines would leave one file in two
        encodings, and the dedup preload would stop at the first byte it cannot
        decode and re-append every record after it.
        """
        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        data = self.path.read_bytes()
        try:
            data.decode("utf-8")
            return
        except UnicodeDecodeError:
            pass
        legacy = locale.getpreferredencoding(False)
        try:
            text = data.decode(legacy)
        except (LookupError, UnicodeDecodeError):
            return  # cannot name the original encoding, so leave the file alone
        if text.encode(legacy) != data:
            return  # the guess does not round-trip, so rewriting would lose bytes
        tmp = self.path.with_suffix(self.path.suffix + ".mig")
        tmp.write_text(text, encoding = "utf-8")
        os.replace(tmp, self.path)

    def _key(self, obj: dict) -> str | None:
        for k in ("id", "node_id", "number", "sha", "url"):
            if k in obj:
                return f"{k}:{obj[k]}"
        return None

    def has(self, key: str) -> bool:
        return key in self._count_seen_keys

    def write(self, obj: dict) -> bool:
        """Return True if newly written, False if already present."""
        k = self._key(obj)
        with self._lock:
            if k is not None and k in self._count_seen_keys:
                return False
            if k is not None:
                self._count_seen_keys.add(k)
            self._fh.write(json.dumps(obj, default = str, ensure_ascii = False))
            self._fh.write("\n")
            self._fh.flush()
        return True

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
