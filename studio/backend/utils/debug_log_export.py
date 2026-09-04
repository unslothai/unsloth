# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Build a bounded-memory, credential-masked archive of Studio log sources."""

from __future__ import annotations

import errno
import os
import re
import stat
import tempfile
import zipfile
from pathlib import Path
from typing import BinaryIO, Iterable

from utils.debug_log_sources import LogSource
from utils.log_redaction import StreamingLogRedactor, redact_log_text


# One log record is normally a few hundred bytes. A limit keeps a malformed
# newline-free record from becoming one unbounded allocation. Oversized records
# are omitted whole rather than split: splitting could put a credential prefix
# in one chunk and its value in the next, outside the redactor's context.
EXPORT_READ_BYTES = 1024 * 1024
EXPORT_CHUNK_BYTES = 64 * 1024
OMITTED_CONTEXT_BYTES = 4096
ARCHIVE_MEMORY_BYTES = 8 * 1024 * 1024
_RECORD_END_RE = re.compile(rb"\r\n|[\r\n]")


def _safe_text(value: str) -> str:
    """Make surrogate-bearing filesystem text safe for UTF-8 and zipfile."""
    return value.encode("utf-8", errors = "backslashreplace").decode("utf-8")


def _unique_archive_name(source: LogSource, used: set[str]) -> str:
    """A relative, collision-free member name without exposing the host path."""
    base = _safe_text(Path(source.label).name) or "log.txt"
    family = _safe_text(source.family)
    candidate = f"{family}/{base}"
    if candidate in used:
        digest = _safe_text(source.id.partition(":")[2])
        stem, suffix = os.path.splitext(base)
        candidate = f"{family}/{stem}-{digest}{suffix}"
    used.add(candidate)
    return candidate


def _copy_redacted(source: BinaryIO, destination: BinaryIO, max_bytes: int) -> None:
    redactor = StreamingLogRedactor()
    remaining = max(0, max_bytes)
    record = bytearray()
    held_cr = b""
    omitted = 0
    scan_tail = b""
    sensitive_context = False
    continuation_kind: str | None = None
    omitted_private_key_block = False
    omitted_quote: str | None = None
    omitted_quote_escaped = False

    def write_piece(piece: bytes, *, terminated: bool) -> None:
        nonlocal omitted, scan_tail, sensitive_context, continuation_kind
        nonlocal omitted_private_key_block, omitted_quote, omitted_quote_escaped
        if omitted:
            omitted += len(piece)
            scan = (scan_tail + piece).decode("utf-8", errors = "replace")
            sensitive_context |= redactor.omitted_record_chunk_has_sensitive_context(scan)
            continuation_kind = redactor.omitted_record_continuation_kind(
                scan,
                continuation_kind,
                sensitive_context,
            )
            omitted_private_key_block = redactor.omitted_record_private_key_state(
                scan,
                omitted_private_key_block,
            )
            quote_scan = piece.decode("utf-8", errors = "replace") if omitted_quote else scan
            omitted_quote, omitted_quote_escaped = redactor.omitted_record_quote_state(
                quote_scan,
                omitted_quote,
                omitted_quote_escaped,
            )
            scan_tail = (scan_tail + piece)[-OMITTED_CONTEXT_BYTES:]
        else:
            record.extend(piece)
            if len(record) > EXPORT_READ_BYTES:
                omitted = len(record)
                scan = bytes(record).decode("utf-8", errors = "replace")
                sensitive_context = redactor.omitted_record_chunk_has_sensitive_context(scan)
                continuation_kind = redactor.omitted_record_continuation_kind(
                    scan,
                    sensitive_context = sensitive_context,
                )
                omitted_private_key_block = redactor.omitted_record_private_key_state(scan)
                omitted_quote, omitted_quote_escaped = redactor.omitted_record_quote_state(
                    scan,
                    None,
                )
                scan_tail = bytes(record[-OMITTED_CONTEXT_BYTES:])
                record.clear()

        if not terminated:
            return
        if omitted:
            if continuation_kind is not None or omitted_quote is not None:
                redactor.mark_omitted_sensitive_record(
                    omitted_quote,
                    continuation_kind,
                )
            if omitted_private_key_block:
                redactor.mark_omitted_private_key_block()
            destination.write(f"[oversized log record omitted: {omitted} bytes]\n".encode("ascii"))
        elif record:
            text = bytes(record).decode("utf-8", errors = "replace")
            destination.write(redactor.redact_record(text).encode("utf-8"))
        record.clear()
        omitted = 0
        scan_tail = b""
        sensitive_context = False
        continuation_kind = None
        omitted_private_key_block = False
        omitted_quote = None
        omitted_quote_escaped = False

    while remaining:
        chunk = source.read(min(EXPORT_CHUNK_BYTES, remaining))
        if not chunk:
            break
        remaining -= len(chunk)
        data = held_cr + chunk
        held_cr = b""
        if data.endswith(b"\r") and remaining:
            data, held_cr = data[:-1], b"\r"
        start = 0
        for match in _RECORD_END_RE.finditer(data):
            write_piece(data[start : match.end()], terminated = True)
            start = match.end()
        write_piece(data[start:], terminated = False)

    if held_cr:
        write_piece(held_cr, terminated = True)
    if record or omitted:
        write_piece(b"", terminated = True)
    if remaining:
        raise OSError(
            getattr(errno, "ESTALE", errno.EIO),
            "Log source was truncated during export",
        )


def _safe_error_summary(exc: Exception) -> str:
    """Describe a failed source without echoing its host path or message."""
    errno = getattr(exc, "errno", None)
    if isinstance(errno, int):
        return f"{type(exc).__name__} (errno {errno})"
    return type(exc).__name__


def _open_regular_source(
    path: str, expected_device_id: int | None, expected_inode: int | None
) -> BinaryIO:
    """Open the enumerated file without accepting a later symlink substitution."""
    before = os.stat(path, follow_symlinks = False)
    if not stat.S_ISREG(before.st_mode):
        raise OSError(errno.ELOOP, "Log source is not a regular file")
    expected_identity = (
        (expected_device_id, expected_inode)
        if expected_device_id is not None and expected_inode is not None
        else (before.st_dev, before.st_ino)
    )
    if (before.st_dev, before.st_ino) != expected_identity:
        raise OSError(getattr(errno, "ESTALE", errno.EIO), "Log source changed after enumeration")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != expected_identity:
            raise OSError(
                getattr(errno, "ESTALE", errno.EIO),
                "Log source changed while it was opened",
            )
        return os.fdopen(descriptor, "rb")
    except Exception:
        os.close(descriptor)
        raise


def build_debug_log_archive(sources: Iterable[LogSource]) -> BinaryIO:
    """Return a seeked temporary ZIP; the caller owns and must close it.

    Only paths already produced by ``debug_log_sources.list_sources`` reach
    this function. Files are opened one at a time and the archive spills to a
    temporary file, so exporting a large session log does not pin it in RAM.
    A file that rotates between enumeration and opening is recorded in the
    archive instead of failing the entire support bundle.
    """
    output = tempfile.SpooledTemporaryFile(max_size = ARCHIVE_MEMORY_BYTES, mode = "w+b")
    used: set[str] = set()
    failures: list[str] = []
    try:
        with zipfile.ZipFile(
            output,
            mode = "w",
            compression = zipfile.ZIP_DEFLATED,
            compresslevel = 6,
            allowZip64 = True,
        ) as archive:
            for source in sources:
                member = _unique_archive_name(source, used)
                try:
                    with _open_regular_source(
                        source.realpath,
                        source.device_id,
                        source.inode,
                    ) as log_file:
                        with archive.open(member, "w", force_zip64 = True) as archived:
                            _copy_redacted(log_file, archived, source.size_bytes)
                except (OSError, ValueError) as exc:
                    failures.append(
                        f"{_safe_text(source.family)}/{_safe_text(Path(source.label).name)}: "
                        f"{_safe_error_summary(exc)}"
                    )

            if failures:
                archive.writestr(
                    "EXPORT_WARNINGS.txt",
                    "Some log files changed or could not be read while the archive was created:\n"
                    + "\n".join(redact_log_text(failure) for failure in failures)
                    + "\n",
                )
        output.seek(0)
        return output
    except Exception:
        output.close()
        raise
