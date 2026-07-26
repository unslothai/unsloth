# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Checkpoint state management for resumable scraping."""

from __future__ import annotations

import json
import locale
import os
import threading
from pathlib import Path
from typing import Any, Dict, Tuple


def _decode_checkpoint(raw: bytes) -> Tuple[str, bool]:
    """Decode a checkpoint file, returning (text, was_utf8).

    Releases before UTF-8 was made explicit wrote these in the locale codepage,
    so a resume on Windows-de finds cp1252 bytes here. latin-1 never raises, so
    a resume degrades to replacement characters instead of losing the file.
    """
    encodings = ["utf-8"]
    try:
        encodings.append(locale.getencoding())
    except AttributeError:  # Python < 3.11
        encodings.append(locale.getpreferredencoding(False))
    for index, encoding in enumerate(encodings):
        try:
            return raw.decode(encoding), index == 0
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("latin-1", errors = "replace"), False


class StateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents = True, exist_ok = True)
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {}
        if self.path.exists():
            try:
                self._data = json.loads(_decode_checkpoint(self.path.read_bytes())[0])
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
        self._count_seen_keys: set[str] = set()
        # Preload seen keys for dedup across resumes. Must run before the append
        # handle opens: a legacy file is rewritten as UTF-8 first, and Windows
        # cannot replace a file it still holds open.
        if self.path.exists() and self.path.stat().st_size > 0:
            self._preload_seen_keys()
        self._fh = self.path.open("a", buffering = 1, encoding = "utf-8")

    def _preload_seen_keys(self) -> None:
        try:
            raw = self.path.read_bytes()
        except OSError:
            return
        text, was_utf8 = _decode_checkpoint(raw)
        for line in text.splitlines():
            try:
                k = self._key(json.loads(line))
            except Exception:
                continue
            if k is not None:
                self._count_seen_keys.add(k)
        if not was_utf8:
            self._rewrite_as_utf8(text)

    def _rewrite_as_utf8(self, text: str) -> None:
        """Convert a locale-encoded file so appended lines match what precedes them."""
        tmp = self.path.with_suffix(self.path.suffix + ".utf8.tmp")
        try:
            tmp.write_text(text, encoding = "utf-8")
            os.replace(tmp, self.path)
        except OSError:
            tmp.unlink(missing_ok = True)

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
