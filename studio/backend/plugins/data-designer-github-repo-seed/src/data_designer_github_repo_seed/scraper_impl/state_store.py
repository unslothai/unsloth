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
    """The codepage a pre-UTF-8 release here would have written, or "".

    Empty on a UTF-8 host, where there is no codepage to attribute the file to.
    """
    try:
        preferred = locale.getencoding()
    except AttributeError:  # Python < 3.11
        preferred = locale.getpreferredencoding(False)
    if preferred.lower().replace("-", "").replace("_", "") == "utf8":
        return ""
    return preferred


def _parse(raw: bytes, encoding: str) -> Any:
    """Parse one JSON document under *encoding*, or None if it does not."""
    try:
        return json.loads(raw.decode(encoding))
    except (UnicodeDecodeError, LookupError, ValueError):
        return None


class _Reading(NamedTuple):
    as_utf8: Any
    as_legacy: Any
    # False when the legacy reading came from the latin-1 guess rather than a
    # codepage we can actually attribute the file to.
    trusted: bool


def _read_line(raw: bytes, codepage: str) -> _Reading:
    """Read one line as UTF-8 and as the codepage it may have been written in.

    Requiring valid JSON, not merely a successful decode, is what separates a
    genuine legacy record from a half-written UTF-8 one: a torn multibyte
    character decodes under cp1252 but leaves the JSON unterminated, so one bad
    byte cannot relabel the file. Some byte strings parse both ways, e.g. cp1251
    ``Р°`` is ``D0 B0``, which is also UTF-8 ``а``, so the caller decides from
    the whole file rather than the line alone.

    Off the writing machine there is no codepage to try. latin-1 still gives a
    reading, which is enough to recover the ASCII dedup keys, but it is only the
    right text for cp1252 data: cp1251 ``Привет`` reads back as ``Ïðèâåò``. Such
    a reading is marked untrusted and never rewritten into the file.
    """
    as_utf8 = _parse(raw, "utf-8")
    if codepage:
        as_legacy = _parse(raw, codepage)
        if as_legacy is not None:
            return _Reading(as_utf8, as_legacy, True)
    return _Reading(as_utf8, _parse(raw, "latin-1"), False)


class _Scan(NamedTuple):
    """What a pass over an existing shard established about it."""

    legacy: bool  # the codepage reading won the vote
    trusted: bool  # that reading came from the real codepage
    readable: bool  # the file could be read at all
    saw_non_utf8: bool  # some line is bytes UTF-8 cannot decode
    utf8_keys: set  # keys from lines UTF-8 could read
    legacy_keys: set  # keys only the codepage reading yields


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
            reading = _read_line(raw, _locale_encoding())
            data = reading.as_utf8 if reading.as_utf8 is not None else reading.as_legacy
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
        # Must run before the append handle opens: a legacy file is rewritten
        # first, and Windows cannot replace a file it still holds open.
        encoding = "utf-8"
        if self.path.exists() and self.path.stat().st_size > 0:
            scan = self._scan_existing()
            migrated = scan.legacy and scan.trusted and self._rewrite_as_utf8()
            self._count_seen_keys = scan.utf8_keys
            if scan.legacy or migrated:
                self._count_seen_keys |= scan.legacy_keys
            if not migrated and (scan.saw_non_utf8 or not scan.readable):
                # Bytes UTF-8 cannot read are in the file, or we could not look.
                # Either way appending UTF-8 risks a second encoding, so write
                # pure ASCII instead: \uXXXX escapes are stored identically by
                # every ASCII-compatible codepage and json.loads gives the exact
                # characters back, so the file keeps decoding as it did and no
                # record is lost.
                encoding = self._codepage or "latin-1"
                self._ensure_ascii = True
        self._fh = self.path.open("a", buffering = 1, encoding = encoding, errors = "strict")

    def _scan_existing(self) -> _Scan:
        """Read the shard once to recover dedup keys and judge its encoding.

        Line by line: these shards reach gigabytes on a large scrape, so neither
        the bytes nor the decoded text are held whole.

        The verdict weighs the whole file. Each line with non-ASCII bytes votes:
        one that parses only under the codepage is evidence of a legacy shard,
        one that parses as UTF-8 is evidence against, since arbitrary codepage
        text almost never forms valid multibyte UTF-8. A single corrupt byte in
        a healthy shard therefore cannot outvote the records around it, and a
        genuinely legacy shard has a legacy vote on every line that carries an
        umlaut.

        A tie leaves the file alone. Rewriting a healthy shard would mojibake
        every good record in it, which is far worse than declining to convert
        one, and the caller keeps the file readable either way.
        """
        legacy_votes = 0
        utf8_votes = 0
        saw_non_utf8 = False
        trusted = False
        utf8_keys: set[str] = set()
        legacy_keys: set[str] = set()
        try:
            with self.path.open("rb") as handle:
                for raw in handle:
                    line = raw.strip()
                    reading = _read_line(line, self._codepage)
                    if reading.as_utf8 is None:
                        saw_non_utf8 = True
                    if line.isascii():
                        pass  # identical either way, so it casts no vote
                    elif reading.as_utf8 is None and reading.as_legacy is not None:
                        legacy_votes += 1
                        trusted = trusted or reading.trusted
                    elif reading.as_utf8 is not None:
                        utf8_votes += 1
                    # Keys are kept apart so a damaged line in a healthy shard
                    # does not mark its record seen and block the retry that
                    # would replace it.
                    if isinstance(reading.as_utf8, dict):
                        key = self._key(reading.as_utf8)
                        if key is not None:
                            utf8_keys.add(key)
                    elif isinstance(reading.as_legacy, dict):
                        key = self._key(reading.as_legacy)
                        if key is not None:
                            legacy_keys.add(key)
        except OSError:
            return _Scan(False, False, False, False, utf8_keys, legacy_keys)
        return _Scan(
            legacy_votes > utf8_votes,
            trusted,
            True,
            saw_non_utf8,
            utf8_keys,
            legacy_keys,
        )

    def _rewrite_as_utf8(self) -> bool:
        """Convert codepage lines so appends match what precedes them.

        Only ever called with the codepage that wrote the file, never the
        latin-1 guess. Streams through a temp file; lines the codepage cannot
        read, damaged ones included, are copied through byte for byte.
        """
        tmp = self.path.with_suffix(self.path.suffix + ".utf8.tmp")
        try:
            with self.path.open("rb") as src, tmp.open("wb") as dst:
                for raw in src:
                    as_legacy = _parse(raw.strip(), self._codepage)
                    if as_legacy is not None:
                        # The file is legacy, so this is the authoritative
                        # reading even where UTF-8 also parsed.
                        key = self._key(as_legacy) if isinstance(as_legacy, dict) else None
                        if key is not None:
                            self._count_seen_keys.add(key)
                        try:
                            raw = raw.decode(self._codepage).encode("utf-8")
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
            self._fh.write(json.dumps(obj, default = str, ensure_ascii = self._ensure_ascii))
            self._fh.write("\n")
            self._fh.flush()
        return True

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
