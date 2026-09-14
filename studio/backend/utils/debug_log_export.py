# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pack every log the viewer may read into one redacted ZIP.

Same allowlist as the picker (`debug_log_sources.list_sources`) through the same
`redact_log_text`, so the bundle can hold no file, and no credential, the tab
would not have shown.

The per-family cap upstream bounds FILES, not bytes, and the session log is
never rotated. Hence two more bounds here, a tail per file and a budget across
all of them, both cutting from the FRONT: the end of a log explains the problem.
"""

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

# Dropped WHOLE, never split: the redactor is anchored on a key next to its
# value, so a cut between the two hands out the credential. Matches the viewer's
# MAX_LINE_BYTES because `_ANSI_RE` backtracks quadratically, so cost grows with
# the SQUARE of what one call is handed.
MAX_RECORD_BYTES = 32 * 1024
OVERSIZED_MARKER = "[oversized log record omitted]"
TRUNCATED_MARKER = "[export time budget reached, rest of this log omitted]"
CUT_MARKER = "[export size budget reached, end of this record omitted]"
UNREADABLE_MARKER = "[log record omitted: not UTF-8 text the redactor can mask]"

# The tail kept from any one log; a session log runs to gigabytes.
MAX_SOURCE_TAIL_BYTES = 8 * 1024 * 1024

# Across every log together. Sized against the redactor's ~4.2 MB/s: the route
# builds before it answers and DOWNLOAD_READ_TIMEOUT (native_file_dialogs.rs)
# allows 30s for headers, so 32 MB is ~12s of that. Raising it without making
# the redactor faster, or streaming the ZIP as it builds, times the caller out.
# Also bounds the browser path, which holds the response in a Blob.
MAX_TOTAL_SOURCE_BYTES = 32 * 1024 * 1024

# The byte budget assumes a throughput the redactor does not guarantee: ANSI
# rules backtrack quadratically, so 12 MB of unterminated C1 introducers takes
# ~16 minutes without this and ~22s with it. Checked BEFORE each record, so the
# build overshoots by one record (~2.5s worst case) and still lands inside
# DOWNLOAD_READ_TIMEOUT, returning a short archive rather than a timeout.
MAX_BUILD_SECONDS = 15.0

WARNINGS_MEMBER = "EXPORT_WARNINGS.txt"


def _safe_basename(label: str) -> str:
    """The filename out of a label, with no way to reach a directory.

    Both separators, not `Path(label).name`: a Windows-shaped label read on
    POSIX keeps its backslashes, which some extractors treat as a path.
    """
    name = label.replace("\\", "/").rsplit("/", 1)[-1].strip()
    # A newline would forge an entry in EXPORT_WARNINGS.txt, one line per source.
    name = "".join(
        "_" if character < " " or character == "\x7f" else character for character in name
    )
    # Lone surrogates from `surrogateescape` are not encodable by zipfile, and
    # the raise lands outside both OSError handlers: a 500 for the whole export.
    name = name.encode("utf-8", "replace").decode("utf-8")
    if not name or name.strip(".") == "":
        return "log"
    return name


def _member_name(family: str, label: str, used: set[str]) -> str:
    """`family/basename`, made unique against everything already in the ZIP.

    A duplicated member extracts as whichever entry the tool reaches last,
    silently losing the other. The key is case-folded because the volume the
    archive is EXTRACTED on decides what collides: Windows and default APFS
    treat `Server.log` and `server.log` as one name. `.lower()` not
    `.casefold()`, which over-folds for filenames (Turkish dotless i, sharp s);
    only the KEY is folded, the member keeps its real name.
    """
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
    # Loops: the first suffix can itself be taken, by a real file or an earlier
    # collision.
    while f"{family}/{stem}-{index}{extension}".lower() in used:
        index += 1
    candidate = f"{family}/{stem}-{index}{extension}"
    used.add(candidate.lower())
    return candidate


def _open_verified(path: str) -> tuple[IO[bytes], int]:
    """Open a source without ever following a link, and prove what we opened.

    The walk in `debug_log_sources` already refuses a symlink whose target
    escapes the log directory, but that check ran at enumeration time; anything
    can replace the entry before this open. So: lstat first and refuse a
    non-regular file, open with O_NOFOLLOW, then fstat the DESCRIPTOR and
    require it to be the same regular file. The fstat-versus-lstat compare is
    the part that matters -- it is taken on the handle we are about to read, so
    a swap after it cannot redirect us. `LogSource` carries no device or inode,
    so the pre-open lstat here is the newest identity we have; that narrows the
    window between enumeration and open without closing it entirely.

    Returns the handle and its raw descriptor.

    One residual, on Windows only, that this cannot close. `O_NOFOLLOW` is
    POSIX-only, so the `getattr` above collapses to 0 there and the open follows
    whatever the entry is. A true symlink is still refused -- CPython sets
    `S_IFLNK` for a reparse point whose tag is `IO_REPARSE_TAG_SYMLINK`, and a
    junction resolves to a directory and fails `S_ISREG`. But for any OTHER file
    reparse tag (AppExecLink, a OneDrive placeholder, dedup), `win32_xstat` falls
    back to traversing, so `before` describes the TARGET, the flagless open
    reaches the same target, and the device/inode compare matches. Closing that
    needs `CreateFileW` with `FILE_FLAG_OPEN_REPARSE_POINT`, which `os.open`
    cannot express. It is bounded rather than open: `debug_log_sources` resolves
    every entry with `realpath` and refuses a target outside the log directory,
    so only the enumeration-to-open window is exposed -- and anyone who can write
    to that directory can already hard-link a file into it, which no platform
    here refuses.
    """
    before = os.stat(path, follow_symlinks = False)
    if not stat.S_ISREG(before.st_mode):
        raise OSError(errno.ELOOP, "not a regular file")
    # O_NONBLOCK because O_NOFOLLOW refuses a symlink but NOT a FIFO, and
    # opening a FIFO with no writer blocks forever -- before the fstat below
    # ever gets a turn to reject it. On a regular file it does nothing; on a
    # FIFO it returns immediately and the S_ISREG check then refuses it.
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
    # From the descriptor only. Re-opening by name would hand the read back to
    # whatever the path points at now.
    return os.fdopen(fd, "rb"), fd


def _redact_record(raw: bytes) -> str:
    """One record, masked, or refused if the redactor cannot read it.

    `errors="replace"` is unsafe: a UTF-16 log decoded as UTF-8 keeps a NUL
    between every character, so `HF_TOKEN=hf_...` stops matching every masking
    rule and is copied through in the clear. PowerShell redirection writes
    UTF-16LE by default. A record that cannot be masked must not ship.
    """
    if b"\x00" in raw:
        return UNREADABLE_MARKER
    try:
        text = raw.decode("utf-8", errors = "strict")
    except UnicodeDecodeError:
        return UNREADABLE_MARKER
    return redact_log_text(text.rstrip("\r"))


def _seek_to_tail(handle: IO[bytes], fd: int, allowance: int) -> tuple[int, int]:
    """Position at the last `allowance` bytes, on a record boundary.

    Returns the bytes skipped (0 if the whole file fits) and the size, read from
    the descriptor about to be used rather than re-stat'd later.

    The boundary is not cosmetic: `redact_log_text` is anchored on a key beside
    its value, so a read starting mid-record can emit a credential whose key was
    in the skipped part. Whatever the seek lands inside is dropped.
    """
    size = os.fstat(fd).st_size
    if size <= allowance:
        return 0, size
    start = size - allowance
    # Already just after a newline: a boundary, so do not discard a whole record.
    handle.seek(start - 1)
    if handle.read(1) == b"\n":
        return start, size
    # Scan FORWARD to a real newline, however far. Giving up after one probe
    # starts mid-record, which leaks: an `aws_secret_access_key="..."` whose key
    # fell before that offset arrives masked of nothing, and an AWS secret has
    # no prefix for a shape rule to catch. Bounded by the size taken from the
    # descriptor, not by EOF -- a live log has none, so scanning to it follows
    # the writer indefinitely.
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
    # No boundary in the tail: reading anyway is the same mid-record start, so
    # refuse the file. The caller turns this into a warning, not a member.
    handle.seek(size)
    return size, size


def _redacted_records(handle: IO[bytes], fd: int, limit: int, deadline: float) -> Iterator[str]:
    """Every line of one log, masked, in bounded chunks, stopping after `limit`.

    Never `handle.read()`: a runner log can be gigabytes.

    `limit` is a different bound from the seek that positioned this read. The
    seek says where to start; a log being APPENDED to has no end, so without a
    ceiling the loop follows the writer -- on the session log, which is live by
    definition while someone is exporting it.

    `deadline` is per RECORD. Per source overruns because one pathological file
    blows the whole build (the redactor's cost is quadratic in what it is
    handed); per chunk overruns because one 256 KiB chunk holds thousands of
    records.
    """
    buffer = b""
    start = handle.tell()
    consumed = start
    # The current record blew the budget; everything to the next newline goes
    # with it.
    dropping = False
    # Whether the read stopped at the FILE's end or the ALLOWANCE's. At EOF the
    # trailing bytes are a real last record with no newline; at the allowance
    # they are the front of one whose remainder was never read.
    at_eof = False
    while consumed - start < limit:
        chunk = handle.read(min(_READ_CHUNK_BYTES, limit - (consumed - start)))
        if not chunk:
            # Shorter than what was read means it rotated or was truncated under
            # us, so the tail does not line up. Growth is ordinary: it is live.
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
        # Whole record, or the front of one? `at_eof` alone cannot say: the loop
        # stops as soon as the allowance is spent, so a file whose last byte
        # lands exactly there never gets the read that reports EOF, and its
        # final newline-less record is complete. The descriptor separates them.
        #
        # It matters because a cut record reads as a complete line, and
        # `redact_log_text` needs several characters of value before it masks,
        # so a cut just past a key ships the first few in the clear. Reachable:
        # `_seek_to_tail`'s forward scan starts later than `size - allowance`,
        # so on a log being appended to the read ends on the allowance, not EOF.
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
    """Round-robin the families instead of draining them one at a time.

    Consumed in `list_sources` order, a large session log spends the whole byte
    budget on the server family and the bundle arrives with no runner logs --
    usually the half that explains the problem. Round-robin means the budget
    runs out on the oldest attempts rather than on a whole category.
    """
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
    """One failed source, named by its member, never by its path.

    Not `str(exc)`: `OSError.__str__` appends the filename, which would put a
    host path into the archive -- the one thing member names avoid.
    """
    code = getattr(exc, "errno", None)
    return redact_log_text(
        f"{member}: {type(exc).__name__} (errno {code if code is not None else 'unknown'})"
    )


def build_log_archive() -> tempfile.SpooledTemporaryFile:
    """Every allowlisted log, redacted, as a ZIP rewound to its start.

    The caller owns the returned file and must close it.
    """
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
                    # Named, so the bundle says what is missing rather than
                    # looking complete.
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
                        # `size > 0` first: an empty log returns (0, 0), and
                        # calling that "no complete record" blames the file for
                        # being empty. An empty member is the honest answer.
                        if size > 0 and skipped >= size:
                            # No boundary in the tail, so nothing whole to keep
                            # and reading anyway starts mid-record. `remaining`
                            # is deliberately NOT spent: this is one file's
                            # property and the next source may still fit.
                            #
                            # Which cap produced the window decides the wording.
                            # Blaming the file when the budget shrank it would
                            # then repeat for every later source, since
                            # `remaining` is untouched here.
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
                                # Charged even on a partial read: those bytes were
                                # still redacted and written. Inside the `with`,
                                # because tell() needs an open handle.
                                remaining -= max(0, handle.tell() - before)
                except OSError as exc:
                    # Keeps whatever was copied first: a partial log is still
                    # worth reading, and the warning says why it stops.
                    warnings.append(_warning_line(member, exc))
            if warnings:
                # A bare ZipInfo, not the str overload: that one stamps
                # time.localtime(), leaking the host's clock and UTC offset.
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
