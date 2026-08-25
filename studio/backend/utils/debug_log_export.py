# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Build a bounded-memory, credential-masked archive of Studio log sources."""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path
from typing import BinaryIO, Iterable

from utils.debug_log_sources import LogSource
from utils.log_redaction import redact_log_text


# One log record is normally a few hundred bytes. A limit keeps a malformed
# newline-free record from becoming one unbounded allocation. Oversized records
# are omitted whole rather than split: splitting could put a credential prefix
# in one chunk and its value in the next, outside the redactor's context.
EXPORT_READ_BYTES = 1024 * 1024
ARCHIVE_MEMORY_BYTES = 8 * 1024 * 1024


def _unique_archive_name(source: LogSource, used: set[str]) -> str:
    """A relative, collision-free member name without exposing the host path."""
    base = Path(source.label).name or "log.txt"
    candidate = f"{source.family}/{base}"
    if candidate in used:
        digest = source.id.partition(":")[2]
        stem, suffix = os.path.splitext(base)
        candidate = f"{source.family}/{stem}-{digest}{suffix}"
    used.add(candidate)
    return candidate


def _copy_redacted(source: BinaryIO, destination: BinaryIO) -> None:
    while True:
        record = source.readline(EXPORT_READ_BYTES + 1)
        if not record:
            return
        if len(record) > EXPORT_READ_BYTES and not record.endswith(b"\n"):
            omitted = len(record)
            while record and not record.endswith(b"\n"):
                record = source.readline(EXPORT_READ_BYTES + 1)
                omitted += len(record)
            destination.write(f"[oversized log record omitted: {omitted} bytes]\n".encode("ascii"))
            continue
        text = record.decode("utf-8", errors = "replace")
        destination.write(redact_log_text(text).encode("utf-8"))


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
                    with open(source.realpath, "rb") as log_file:
                        with archive.open(member, "w", force_zip64 = True) as archived:
                            _copy_redacted(log_file, archived)
                except (OSError, ValueError) as exc:
                    failures.append(f"{source.family}/{Path(source.label).name}: {exc}")

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
