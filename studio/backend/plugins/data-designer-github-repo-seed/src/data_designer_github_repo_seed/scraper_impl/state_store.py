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


def _legacy_encoding() -> str:
    """The codepage a pre-UTF-8 release would have written these files in."""
    try:
        return locale.getencoding()
    except AttributeError:  # Python < 3.11
        return locale.getpreferredencoding(False)


def _decode_json_line(raw: bytes) -> Tuple[Any, bool]:
    """Parse one JSON line, returning (object, was_legacy).

    A line only counts as legacy if the locale codepage both decodes it and
    yields valid JSON. That separates a genuine cp1252 record from a half-written
    UTF-8 one: retrying a torn multibyte character as cp1252 "succeeds" but gives
    mojibake, so requiring it to parse keeps one bad byte from relabelling the
    file. Anything that parses under neither is damaged and is left untouched.
    """
    try:
        return json.loads(raw.decode("utf-8")), False
    except (UnicodeDecodeError, ValueError):
        pass
    try:
        return json.loads(raw.decode(_legacy_encoding())), True
    except (UnicodeDecodeError, LookupError, ValueError):
        raise ValueError("undecodable line")


class StateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents = True, exist_ok = True)
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {}
        # Small single-document file, so a whole read is fine here. Older
        # releases wrote it in the locale codepage; _flush() rewrites the whole
        # file as UTF-8, so it can never end up half in one encoding.
        if self.path.exists():
            try:
                self._data = _decode_json_line(self.path.read_bytes())[0]
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
        encoding = "utf-8"
        if self.path.exists() and self.path.stat().st_size > 0:
            if self._preload_seen_keys() and not self._rewrite_as_utf8():
                # Could not convert it, so keep matching what is already there
                # rather than appending UTF-8 into a legacy file.
                encoding = _legacy_encoding()
        self._fh = self.path.open("a", buffering = 1, encoding = encoding, errors = "replace")

    def _preload_seen_keys(self) -> bool:
        """Collect seen keys, returning True if the file needs migrating.

        Reads line by line: these shards reach gigabytes on a large scrape, so
        neither the bytes nor the decoded text are held whole.
        """
        legacy = False
        try:
            with self.path.open("rb") as f:
                for raw in f:
                    try:
                        obj, was_legacy = _decode_json_line(raw.strip())
                    except ValueError:
                        continue
                    legacy = legacy or was_legacy
                    k = self._key(obj) if isinstance(obj, dict) else None
                    if k is not None:
                        self._count_seen_keys.add(k)
        except OSError:
            return False
        return legacy

    def _rewrite_as_utf8(self) -> bool:
        """Convert legacy lines so appends match what precedes them.

        Streams through a temp file. Lines that are already UTF-8, and damaged
        lines that parse under no encoding, are copied through byte for byte.
        """
        tmp = self.path.with_suffix(self.path.suffix + ".utf8.tmp")
        legacy_encoding = _legacy_encoding()
        try:
            with self.path.open("rb") as src, tmp.open("wb") as dst:
                for raw in src:
                    try:
                        _decode_json_line(raw.strip())
                    except ValueError:  # damaged line, preserve verbatim
                        dst.write(raw)
                        continue
                    try:
                        raw.decode("utf-8")
                    except UnicodeDecodeError:
                        raw = raw.decode(legacy_encoding).encode("utf-8")
                    dst.write(raw)
            os.replace(tmp, self.path)
            return True
        except (OSError, UnicodeError):
            tmp.unlink(missing_ok = True)
            return False

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
