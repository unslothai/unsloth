# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Checkpoint state management for resumable scraping."""

from __future__ import annotations

import json
import locale
import os
import threading
from pathlib import Path
from typing import Any, Dict, NamedTuple


def _locale_encoding() -> str:
    """The codepage a pre-UTF-8 release here would have written, or "". Empty on a UTF-8 host, where there is no codepage to attribute the file to."""
    try:
        # novermin -- 3.11, and the except below IS the guard. vermin reads names
        # rather than control flow, so it cannot see that this is already handled.
        preferred = locale.getencoding()  # novermin
    except AttributeError:  # Python < 3.11
        preferred = locale.getpreferredencoding(False)
    if preferred.lower().replace("-", "").replace("_", "") == "utf8":
        return ""
    return preferred


# Trail bytes can land on JSON punctuation, so single-byte fallback misreads these.
_DOUBLE_BYTE_ENCODINGS = ("cp932", "cp936", "cp949", "cp950")


def _parse(raw: bytes, encoding: str) -> Any:
    """Parse one JSON document under *encoding*, or None if it does not. RecursionError is a RuntimeError, so nesting json.loads will not descend is the one parse failure the other three miss. Both callers run this outside any further handler, so it has to answer None here or a single damaged record aborts the scraper at startup instead of being skipped."""
    try:
        return json.loads(raw.decode(encoding))
    except (UnicodeDecodeError, LookupError, ValueError, RecursionError):
        return None


class _Reading(NamedTuple):
    as_utf8: Any
    as_legacy: Any


def _read_line(raw: bytes, codepage: str) -> _Reading:
    """Codepage reading is never authoritative, it only recovers ASCII dedup keys; first valid parse
    wins."""
    as_utf8 = _parse(raw, "utf-8")
    if isinstance(as_utf8, dict):
        return _Reading(as_utf8, None)
    for encoding in (codepage, "latin-1", *_DOUBLE_BYTE_ENCODINGS):
        if not encoding:
            continue
        as_legacy = _parse(raw, encoding)
        if as_legacy is not None:
            return _Reading(as_utf8, as_legacy)
    return _Reading(as_utf8, None)


class _Scan(NamedTuple):
    """What a pass over an existing shard established about it."""

    legacy: bool
    readable: bool
    saw_non_ascii: bool
    utf8_keys: set
    legacy_keys: set


class StateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents = True, exist_ok = True)
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {}
        # UTF-8 only: a mojibaked cursor would end the stream; a dropped checkpoint just re-scrapes.
        if self.path.exists():
            try:
                raw = self.path.read_bytes()
            except OSError:
                raw = b""
            data = _parse(raw, "utf-8")
            self._data = data if isinstance(data, dict) else {}

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
        self._codepage = _locale_encoding()
        self._ensure_ascii = False
        encoding = "utf-8"
        if self.path.exists() and self.path.stat().st_size > 0:
            scan = self._scan_existing()
            self._count_seen_keys = scan.utf8_keys
            if scan.legacy:
                self._count_seen_keys |= scan.legacy_keys
            if scan.saw_non_ascii or not scan.readable:
                # Never convert: the writing encoding is unrecoverable; ASCII appends read alike.
                encoding = "ascii"
                self._ensure_ascii = True
        self._fh = self.path.open("a", buffering = 1, encoding = encoding, errors = "strict")

    def _scan_existing(self) -> _Scan:
        """Vote per non-ASCII line; codepage-only parses count as legacy, and more than one vote is
        needed."""
        legacy_votes = 0
        utf8_votes = 0
        saw_non_ascii = False
        utf8_keys: set[str] = set()
        legacy_keys: set[str] = set()
        try:
            with self.path.open("rb") as handle:
                for raw in handle:
                    line = raw.strip()
                    reading = _read_line(line, self._codepage)
                    if not line.isascii():
                        saw_non_ascii = True
                        if reading.as_utf8 is None and reading.as_legacy is not None:
                            legacy_votes += 1
                        elif reading.as_utf8 is not None:
                            utf8_votes += 1
                    if isinstance(reading.as_utf8, dict):
                        key = self._key(reading.as_utf8)
                        if key is not None:
                            utf8_keys.add(key)
                    elif isinstance(reading.as_legacy, dict):
                        key = self._key(reading.as_legacy)
                        if key is not None:
                            legacy_keys.add(key)
        except OSError:
            return _Scan(False, False, False, utf8_keys, legacy_keys)
        return _Scan(
            legacy_votes > 1 and legacy_votes > utf8_votes,
            True,
            saw_non_ascii,
            utf8_keys,
            legacy_keys,
        )

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
            self._fh.write(json.dumps(obj, default = str, ensure_ascii = self._ensure_ascii))
            self._fh.write("\n")
            self._fh.flush()
        return True

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
