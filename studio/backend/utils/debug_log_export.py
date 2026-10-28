# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Packs viewer-readable logs into one redacted ZIP, trimming from the front since log ends matter."""

from __future__ import annotations

import errno
import os
import stat
import tempfile
import time
import zipfile
from typing import IO, Iterator

from utils import debug_log_sources
from utils.log_redaction import redact_log_text

# Beyond this the ZIP rolls from memory onto disk.
SPOOL_MAX_BYTES = 8 * 1024 * 1024

STREAM_CHUNK_BYTES = 64 * 1024

_READ_CHUNK_BYTES = 256 * 1024

# Dropped whole, never split: a cut between key and value leaks the credential.
# Matches the viewer's MAX_LINE_BYTES since _ANSI_RE backtracks quadratically.
MAX_RECORD_BYTES = 32 * 1024
OVERSIZED_MARKER = "[oversized log record omitted]"
TRUNCATED_MARKER = "[export time budget reached, rest of this log omitted]"
CUT_MARKER = "[export size budget reached, end of this record omitted]"
UNREADABLE_MARKER = "[log record omitted: not UTF-8 text the redactor can mask]"

# The tail kept from any one log; a session log runs to gigabytes.
MAX_SOURCE_TAIL_BYTES = 8 * 1024 * 1024

# Sized to ~12s of redactor time in the 30s header timeout; raise it only with a faster redactor.
MAX_TOTAL_SOURCE_BYTES = 32 * 1024 * 1024

# Caps quadratic ANSI backtracking so the build still lands inside DOWNLOAD_READ_TIMEOUT.
MAX_BUILD_SECONDS = 15.0

WARNINGS_MEMBER = "EXPORT_WARNINGS.txt"


def _safe_basename(label: str) -> str:
    """Strips both separators, since on POSIX Path(label).name keeps Windows backslashes in the name."""
    name = label.replace("\\", "/").rsplit("/", 1)[-1].strip()
    # A newline would forge an entry in EXPORT_WARNINGS.txt, one line per source.
    name = "".join(
        "_" if character < " " or character == "\x7f" else character for character in name
    )
    # Lone surrogates are not encodable by zipfile and would fail the whole export.
    name = name.encode("utf-8", "replace").decode("utf-8")
    if not name or name.strip(".") == "":
        return "log"
    return name


def _member_name(family: str, label: str, used: set[str]) -> str:
    """Collision key is lower-cased: a case-insensitive volume would merge Server.log and server.log."""
    base = _safe_basename(label)
    candidate = f"{family}/{base}"
    if candidate.lower() not in used:
        used.add(candidate.lower())
        return candidate
    stem, dot, extension = base.rpartition(".")
    if not dot:
        stem, extension = base, ""
    else:
        extension = dot + extension
    index = 2
    # Loops because the first suffix (-2) can itself be taken by a real file or an earlier collision.
    while f"{family}/{stem}-{index}{extension}".lower() in used:
        index += 1
    candidate = f"{family}/{stem}-{index}{extension}"
    used.add(candidate.lower())
    return candidate


def _open_verified(path: str) -> tuple[IO[bytes], int]:
    """Checks the opened descriptor against the pre-open lstat; a swap after the open cannot redirect it."""
    before = os.stat(path, follow_symlinks = False)
    if not stat.S_ISREG(before.st_mode):
        raise OSError(errno.ELOOP, "not a regular file")
    # O_NONBLOCK: O_NOFOLLOW does not refuse a FIFO, and opening one with no writer blocks.
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    fd = os.open(path, flags)
    try:
        after = os.fstat(fd)
        if not stat.S_ISREG(after.st_mode):
            raise OSError(errno.ELOOP, "not a regular file")
        if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
            raise OSError(errno.ESTALE, "file replaced during export")
    except BaseException:
        os.close(fd)
        raise
    # From the descriptor only, so a swapped path cannot redirect the read.
    return os.fdopen(fd, "rb"), fd


def _redact_record(raw: bytes) -> str:
    """UTF-16 records are refused, since decoding them leniently stops HF_TOKEN from being masked."""
    if b"\x00" in raw:
        return UNREADABLE_MARKER
    try:
        text = raw.decode("utf-8", errors = "strict")
    except UnicodeDecodeError:
        return UNREADABLE_MARKER
    return redact_log_text(text.rstrip("\r"))


def _seek_to_tail(handle: IO[bytes], fd: int, allowance: int) -> tuple[int, int]:
    """Drops whatever the seek lands inside, since redaction keys on a credential's name beside its
    value."""
    size = os.fstat(fd).st_size
    if size <= allowance:
        return 0, size
    start = size - allowance
    # Already just after a newline: a boundary, so do not discard a whole record.
    handle.seek(start - 1)
    if handle.read(1) == b"\n":
        return start, size
    # Scan to a real newline: starting mid-record can leak a secret whose key was cut off.
    # Bounded by the fstat size, since a live log has no EOF.
    scanned = start
    while scanned < size:
        probe = handle.read(min(MAX_RECORD_BYTES, size - scanned))
        if not probe:
            break
        newline = probe.find(b"\n")
        if newline != -1:
            handle.seek(scanned + newline + 1)
            return scanned + newline + 1, size
        scanned += len(probe)
    # No boundary in the tail: refuse rather than start mid-record.
    handle.seek(size)
    return size, size


def _redacted_records(handle: IO[bytes], fd: int, limit: int, deadline: float) -> Iterator[str]:
    """Stops at limit because a log being appended to has no end; deadline is per record, not per source."""
    buffer = b""
    start = handle.tell()
    consumed = start
    # The current record exceeded the budget, so everything up to the next newline is dropped with it.
    dropping = False
    # At EOF trailing bytes are a whole record; at the allowance they are a cut one.
    at_eof = False
    while consumed - start < limit:
        chunk = handle.read(min(_READ_CHUNK_BYTES, limit - (consumed - start)))
        if not chunk:
            # Shrinking means rotation or truncation; growth is normal for a live log.
            if os.fstat(fd).st_size < consumed:
                raise OSError(errno.ESTALE, "log file shrank during export")
            at_eof = True
            break
        consumed += len(chunk)
        buffer += chunk
        while True:
            if time.monotonic() > deadline:
                yield TRUNCATED_MARKER
                return
            newline = buffer.find(b"\n")
            if newline == -1:
                break
            record, buffer = buffer[:newline], buffer[newline + 1 :]
            if dropping:
                # The rest of a record already given up on.
                dropping = False
                yield OVERSIZED_MARKER
            elif len(record) > MAX_RECORD_BYTES:
                # Terminated inside one chunk, so the buffer guard never saw it.
                yield OVERSIZED_MARKER
            else:
                yield _redact_record(record)
        if len(buffer) > MAX_RECORD_BYTES:
            # A log with no newline at all must not be held in memory whole.
            dropping = True
            buffer = b""
    # No size guard here: the in-loop one clears the buffer after every chunk.
    if dropping:
        yield OVERSIZED_MARKER
    elif buffer:
        # at_eof alone cannot tell a cut record from a complete one; the descriptor size can.
        # A cut record can leak the first characters of a secret past the redactor.
        if at_eof:
            cut = False
        else:
            try:
                cut = os.fstat(fd).st_size > consumed
            except OSError:
                cut = True  # cannot tell; take the side that cannot mislead
        yield CUT_MARKER if cut else _redact_record(buffer)


def _newest_first_across_families(
    sources: list[debug_log_sources.LogSource],
) -> list[debug_log_sources.LogSource]:
    """Round-robins the families so a large log cannot spend the whole byte budget before the others."""
    by_family: dict[str, list[debug_log_sources.LogSource]] = {}
    for source in sources:
        by_family.setdefault(source.family, []).append(source)
    ordered: list[debug_log_sources.LogSource] = []
    for rank in range(max((len(group) for group in by_family.values()), default = 0)):
        for group in by_family.values():
            if rank < len(group):
                ordered.append(group[rank])
    return ordered


def _warning_line(member: str, exc: BaseException) -> str:
    """Names the member, never the path: str(exc) on an OSError appends the filename, leaking a host
    path."""
    code = getattr(exc, "errno", None)
    return redact_log_text(
        f"{member}: {type(exc).__name__} (errno {code if code is not None else 'unknown'})"
    )


def build_log_archive() -> tempfile.SpooledTemporaryFile:
    """Caller owns the returned file and must close it; the ZIP is rewound to its start."""
    output = tempfile.SpooledTemporaryFile(max_size = SPOOL_MAX_BYTES, mode = "w+b")
    try:
        warnings: list[str] = []
        used: set[str] = set()
        remaining = MAX_TOTAL_SOURCE_BYTES
        deadline = time.monotonic() + MAX_BUILD_SECONDS
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for source in _newest_first_across_families(debug_log_sources.list_sources()):
                member = _member_name(source.family, source.label, used)
                if remaining <= 0:
                    # Named so the bundle shows what is missing instead of looking complete.
                    warnings.append(f"{member}: omitted, export size budget reached")
                    continue
                if time.monotonic() > deadline:
                    warnings.append(f"{member}: omitted, export time budget reached")
                    continue
                try:
                    handle, fd = _open_verified(source.realpath)
                except OSError as exc:
                    # One unreadable log does not cost the user the other nine.
                    warnings.append(_warning_line(member, exc))
                    continue
                allowance = min(MAX_SOURCE_TAIL_BYTES, remaining)
                try:
                    with handle:
                        skipped, size = _seek_to_tail(handle, fd, allowance)
                        # Check gid 0 before requiring a name: if lookup fails, unnamed gid 0 would let repair run groupadd -g 0 and usermod into root, the grant to refuse.
                        if size > 0 and skipped >= size:
                            # remaining is not spent: this is one file's property and the
                            # next source may still fit.
                            if allowance < MAX_SOURCE_TAIL_BYTES:
                                warnings.append(f"{member}: omitted, export size budget reached")
                            else:
                                warnings.append(
                                    f"{member}: omitted, no complete record in the last "
                                    f"{allowance} bytes"
                                )
                            continue
                        with archive.open(member, "w") as destination:
                            if skipped:
                                warnings.append(
                                    f"{member}: kept the last {size - skipped} bytes, "
                                    f"skipped the first {skipped}"
                                )
                                destination.write(
                                    f"[skipped the first {skipped} bytes of this log]\n".encode()
                                )
                            before = handle.tell()
                            try:
                                for record in _redacted_records(handle, fd, allowance, deadline):
                                    destination.write((record + "\n").encode("utf-8"))
                            finally:
                                # Charged even on a partial read; inside the with because
                                # tell() needs an open handle.
                                remaining -= max(0, handle.tell() - before)
                except OSError as exc:
                    # Keeps what was copied first; a partial log is worth reading, and the warning says why it stops.
                    warnings.append(_warning_line(member, exc))
            if warnings:
                # A bare ZipInfo: the str overload stamps localtime, leaking the host clock.
                archive.writestr(
                    zipfile.ZipInfo(WARNINGS_MEMBER),
                    "\n".join(warnings) + "\n",
                    compress_type = zipfile.ZIP_DEFLATED,
                )
    except BaseException:
        output.close()
        raise
    output.seek(0)
    return output
