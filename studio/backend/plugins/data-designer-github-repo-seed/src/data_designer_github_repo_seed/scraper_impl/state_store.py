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


# Legacy Windows codepages whose trail bytes can land on JSON punctuation, so a
# single-byte fallback cannot read them. Tried only to recover dedup keys.
_DOUBLE_BYTE_ENCODINGS = ("cp932", "cp936", "cp949", "cp950")


def _parse(raw: bytes, encoding: str) -> Any:
    """Parse one JSON document under *encoding*, or None if it does not."""
    try:
        return json.loads(raw.decode(encoding))
    except (UnicodeDecodeError, LookupError, ValueError):
        return None


class _Reading(NamedTuple):
    as_utf8: Any
    as_legacy: Any


def _read_line(raw: bytes, codepage: str) -> _Reading:
    """Read one line as UTF-8 and as a codepage, for dedup keys only.

    Requiring valid JSON, not merely a successful decode, is what separates a
    genuine legacy record from a half-written UTF-8 one: a torn multibyte
    character decodes under cp1252 but leaves the JSON unterminated. Some byte
    strings parse both ways, e.g. cp1251 ``Р°`` is ``D0 B0``, which is also
    UTF-8 ``а``.

    The codepage reading is never authoritative, because the file's own encoding
    cannot be recovered from its bytes. Reading a cp1251 shard on a cp1252
    machine turns ``Привет`` into ``Ïðèâåò`` and every byte of it decodes
    cleanly, so a successful decode proves nothing about who wrote it. It is
    used only to recover the dedup keys, which are ASCII ids and come back the
    same under any of these, so the first reading that parses will do.

    That is also why several are tried. latin-1 alone mangles the double-byte
    codepages: cp932 ``表`` is ``95 5C``, and latin-1 turns the trail byte into
    a JSON backslash, so the record fails to parse and its id is forgotten.
    """
    as_utf8 = _parse(raw, "utf-8")
    for encoding in (codepage, "latin-1", *_DOUBLE_BYTE_ENCODINGS):
        if not encoding:
            continue
        as_legacy = _parse(raw, encoding)
        if as_legacy is not None:
            return _Reading(as_utf8, as_legacy)
    return _Reading(as_utf8, None)


class _Scan(NamedTuple):
    """What a pass over an existing shard established about it."""

    legacy: bool  # enough evidence to trust keys from the codepage reading
    readable: bool  # the file could be read at all
    saw_non_ascii: bool  # some line carries bytes whose reading depends on encoding
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
        encoding = "utf-8"
        if self.path.exists() and self.path.stat().st_size > 0:
            scan = self._scan_existing()
            self._count_seen_keys = scan.utf8_keys
            if scan.legacy:
                self._count_seen_keys |= scan.legacy_keys
            if scan.saw_non_ascii or not scan.readable:
                # The file holds bytes whose meaning depends on the encoding, or
                # we could not look.
                # It is never converted: the encoding that wrote it cannot be
                # recovered from its bytes, and guessing mojibakes the records.
                # Appending UTF-8 would add a second encoding, so append pure
                # ASCII, which every codepage stores identically. json.loads
                # turns the \uXXXX escapes back into the exact characters, so
                # the file keeps decoding as it did and nothing is lost.
                encoding = "ascii"
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

        More than one such line is required, because a single one is genuinely
        undecidable: a legacy record holding one accented character and an ASCII
        record holding one stray byte are the same shape. Reading it as damage
        risks a duplicate; reading it as legacy marks an unreadable record seen
        and blocks the retry that would replace it, losing it for good. Only one
        of those is recoverable.

        The verdict only picks which reading supplies the dedup keys. The file
        itself is never rewritten either way, so a wrong answer costs at most a
        duplicate, never a corrupted record.
        """
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
                    # An ASCII line reads the same under every encoding, so it
                    # neither votes nor constrains what may be appended.
                    if not line.isascii():
                        saw_non_ascii = True
                        if reading.as_utf8 is None and reading.as_legacy is not None:
                            legacy_votes += 1
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
