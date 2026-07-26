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


def _legacy_encodings() -> Tuple[str, ...]:
    """Codepages a pre-UTF-8 release might have written these files in.

    The locale codepage first, since that is usually the machine that wrote the
    file. latin-1 then decodes any byte at all, so a shard carried to a UTF-8
    machine still has a reading instead of none.
    """
    try:
        preferred = locale.getencoding()
    except AttributeError:  # Python < 3.11
        preferred = locale.getpreferredencoding(False)
    seen = []
    for candidate in (preferred, "latin-1"):
        if candidate.lower().replace("-", "").replace("_", "") == "utf8":
            continue
        if candidate not in seen:
            seen.append(candidate)
    return tuple(seen)


def _parse(raw: bytes, encoding: str) -> Any:
    """Parse one JSON document under *encoding*, or None if it does not."""
    try:
        return json.loads(raw.decode(encoding))
    except (UnicodeDecodeError, LookupError, ValueError):
        return None


def _parse_both(raw: bytes) -> Tuple[Any, Any, str]:
    """Read a line both ways, returning (as_utf8, as_legacy, legacy_encoding).

    Requiring valid JSON, not merely a successful decode, is what separates a
    genuine legacy record from a half-written UTF-8 one: a torn multibyte
    character decodes under cp1252 but leaves the JSON unterminated, so one bad
    byte cannot relabel the file. Some byte strings parse both ways, e.g. cp1251
    ``Р°`` is ``D0 B0``, which is also UTF-8 ``а``, so the caller decides from
    the whole file rather than the line alone.
    """
    as_utf8 = _parse(raw, "utf-8")
    for encoding in _legacy_encodings():
        as_legacy = _parse(raw, encoding)
        if as_legacy is not None:
            return as_utf8, as_legacy, encoding
    return as_utf8, None, ""


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
                raw = self.path.read_bytes()
            except OSError:
                raw = b""
            as_utf8, as_legacy, _ = _parse_both(raw)
            data = as_utf8 if as_utf8 is not None else as_legacy
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
        # Encoding to fall back to if the file turns out to be legacy and the
        # migration cannot be written.
        self._append_encoding = "utf-8"
        # Preload seen keys for dedup across resumes. Must run before the append
        # handle opens: a legacy file is rewritten as UTF-8 first, and Windows
        # cannot replace a file it still holds open.
        encoding = "utf-8"
        if self.path.exists() and self.path.stat().st_size > 0:
            if self._preload_seen_keys() and not self._rewrite_as_utf8():
                # Could not convert it, so keep matching what is already there
                # rather than appending UTF-8 into a legacy file.
                encoding = self._append_encoding
        self._fh = self.path.open("a", buffering = 1, encoding = encoding, errors = "replace")

    def _preload_seen_keys(self) -> bool:
        """Collect seen keys, returning True if the file needs migrating.

        Reads line by line: these shards reach gigabytes on a large scrape, so
        neither the bytes nor the decoded text are held whole. A file counts as
        legacy on the evidence of any one line that parses under the codepage
        and not as UTF-8. Lines that parse as both are decided by that verdict,
        since a shard of Cyrillic or Japanese is only ambiguous line by line.
        """
        legacy = False
        try:
            with self.path.open("rb") as f:
                for raw in f:
                    as_utf8, as_legacy, encoding = _parse_both(raw.strip())
                    if as_utf8 is None and as_legacy is not None:
                        legacy = True
                        self._append_encoding = encoding
                    obj = as_utf8 if as_utf8 is not None else as_legacy
                    key = self._key(obj) if isinstance(obj, dict) else None
                    if key is not None:
                        self._count_seen_keys.add(key)
        except OSError:
            return False
        return legacy

    def _rewrite_as_utf8(self) -> bool:
        """Convert legacy lines so appends match what precedes them.

        Streams through a temp file. Lines with no legacy reading, damaged ones
        included, are copied through byte for byte.
        """
        tmp = self.path.with_suffix(self.path.suffix + ".utf8.tmp")
        try:
            with self.path.open("rb") as src, tmp.open("wb") as dst:
                for raw in src:
                    _, as_legacy, encoding = _parse_both(raw.strip())
                    if as_legacy is not None:
                        # The file is legacy, so this is the authoritative
                        # reading even where UTF-8 also parsed.
                        key = self._key(as_legacy) if isinstance(as_legacy, dict) else None
                        if key is not None:
                            self._count_seen_keys.add(key)
                        try:
                            raw = raw.decode(encoding).encode("utf-8")
                        except UnicodeDecodeError:
                            pass
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
