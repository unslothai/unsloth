# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pack every log the viewer may read into one redacted ZIP.

Sharing a problem means collecting the server log and each runner's log by
hand, and the desktop app has no way to reach the log folder at all. This is
the same allowlist the picker shows -- `debug_log_sources.list_sources` -- read
through the same `redact_log_text` that already guards the live viewer, so the
bundle can never contain a file, or a credential, the tab would not have shown.

The 10-per-family cap in `debug_log_sources` is not lifted here, but it caps
FILES, not bytes: the active session log is never rotated (see
`debug_log_reader`) and is routinely many GB on its own. So the archive is
bounded twice more -- a tail per file and a budget across all of them. Both cut
from the FRONT, because the end of a log is the part that explains a problem.
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

# Beyond this the ZIP rolls from memory onto disk. An export of ten files per
# family compresses well below it in the ordinary case, and a busy host pays a
# temp file rather than the resident memory of one.
SPOOL_MAX_BYTES = 8 * 1024 * 1024

# What the route hands to StreamingResponse.
STREAM_CHUNK_BYTES = 64 * 1024

_READ_CHUNK_BYTES = 256 * 1024

# A record longer than this is dropped WHOLE rather than split. Splitting is
# what debug_log_reader does for the viewer, and it is wrong here: the redactor
# is anchored on a key name next to its value, so a cut between the two hands
# out the credential in the clear.
#
# Deliberately the same 32 KiB the viewer caps a line at (MAX_LINE_BYTES in
# debug_log_reader). `_ANSI_RE` in log_redaction backtracks quadratically on an
# unterminated OSC introducer -- 40k such characters take ~16s against ~0.005s
# for the same length of ordinary text -- so the cost of a call grows with the
# SQUARE of what is passed to it. Handing the redactor a 1 MiB record instead of
# a 32 KiB one multiplies the cost of the same bytes by ~1000. Matching the
# viewer keeps the export's worst case no worse than the tab's.
MAX_RECORD_BYTES = 32 * 1024
OVERSIZED_MARKER = "[oversized log record omitted]"
TRUNCATED_MARKER = "[export time budget reached, rest of this log omitted]"
UNREADABLE_MARKER = "[log record omitted: not UTF-8 text the redactor can mask]"

# The tail kept from any one log. A session log runs to gigabytes and the whole
# point of the bundle is to attach it to an issue, so the head is both the least
# useful part and the part that makes the archive unusable.
MAX_SOURCE_TAIL_BYTES = 8 * 1024 * 1024

# And across every log together, because eighty tails still add up.
#
# Sized against the redactor rather than against taste. `redact_log_text` runs
# at ~4.2 MB/s (measured on this tree, and flat regardless of whether a line
# holds a credential), the route builds the archive before it answers, and
# DOWNLOAD_READ_TIMEOUT in native_file_dialogs.rs gives the response 30s to
# produce headers. 32 MB is ~12s of that, which still leaves margin on a host
# slower than this one. Raising it without either making the redactor faster or
# streaming the ZIP as it is built turns a big export into a timed-out one.
#
# It also bounds the browser path, which lands the whole response in a Blob in
# the tab's memory.
MAX_TOTAL_SOURCE_BYTES = 32 * 1024 * 1024

# A ceiling on the whole build, in seconds.
#
# The byte budget assumes a throughput, and that assumption is not safe: the
# redactor's ANSI rules backtrack quadratically, so a log carrying unterminated
# C1 introducers costs thousands of times what the same bytes of ordinary text
# do. A size cap cannot express "and do not take an hour"; this can. Measured:
# 12 MB of such records takes ~16 minutes without this and ~22s with it.
#
# The deadline is tested BEFORE each record, so the build can overshoot by the
# cost of the one record already in flight -- ~2.5s for a 32 KiB pathological
# record on this host. 15s leaves that overshoot, and a slower host's, inside
# DOWNLOAD_READ_TIMEOUT (30s) in native_file_dialogs.rs, so the caller gets a
# short archive that says it is short rather than a timeout with nothing in it.
MAX_BUILD_SECONDS = 15.0

WARNINGS_MEMBER = "EXPORT_WARNINGS.txt"


def _safe_basename(label: str) -> str:
    """The filename out of a label, with no way to reach a directory.

    Both separators, not `Path(label).name`: a Windows-shaped label read on
    POSIX keeps its backslashes, and the result is a member name that some
    extractors treat as a path.
    """
    name = label.replace("\\", "/").rsplit("/", 1)[-1].strip()
    # A newline would forge an entry in EXPORT_WARNINGS.txt, which is one line
    # per source; the rest are simply not filenames.
    name = "".join("_" if character < " " or character == "\x7f" else character
                   for character in name)
    # POSIX filenames are bytes, so `Path.name` can hand back lone surrogates
    # from `surrogateescape`. zipfile cannot encode those, and the raise lands
    # outside both OSError handlers and takes the whole export down with a 500.
    name = name.encode("utf-8", "replace").decode("utf-8")
    if not name or name.strip(".") == "":
        return "log"
    return name


def _member_name(family: str, label: str, used: set[str]) -> str:
    """`family/basename`, made unique against everything already in the ZIP.

    Two studio homes can hold a same-named file in the same family, and a ZIP
    with a duplicated member extracts as whichever entry the tool happens to
    reach last, silently losing the other one.
    """
    base = _safe_basename(label)
    candidate = f"{family}/{base}"
    if candidate not in used:
        used.add(candidate)
        return candidate
    stem, dot, extension = base.rpartition(".")
    if not dot:
        stem, extension = base, ""
    else:
        extension = dot + extension
    index = 2
    # Loops: the first suffix can itself already be taken, by a real file named
    # that way or by an earlier collision.
    while f"{family}/{stem}-{index}{extension}" in used:
        index += 1
    candidate = f"{family}/{stem}-{index}{extension}"
    used.add(candidate)
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
    """One record, masked -- or refused if the redactor cannot read it.

    `errors="replace"` is not safe here. A UTF-16 log decoded as UTF-8 keeps a
    NUL between every character, so `HF_TOKEN=hf_...` becomes
    `H\\x00F\\x00_\\x00T...`: every masking rule stops matching, the record is
    copied through verbatim, and the credential is still perfectly readable to
    anyone who opens the archive -- or runs `strings` on it. Windows PowerShell
    redirection writes UTF-16LE by default, so this is not a contrived shape.

    The export cannot re-encode its way out of that (the redactor is shared with
    the live viewer and is not being changed here), so it refuses instead. A
    record it cannot mask is a record it must not ship.
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

    Returns the bytes skipped (0 if the whole file fits) and the file's size,
    read from the descriptor we are about to use rather than re-stat'd later.

    Landing on a boundary is not cosmetic. `redact_log_text` is anchored on a
    key name sitting next to its value, so a read that begins in the middle of a
    record can hand out the value of a credential whose key was left behind in
    the part we skipped. Whatever the seek lands in the middle of is therefore
    dropped, and reading starts after the next newline.
    """
    size = os.fstat(fd).st_size
    if size <= allowance:
        return 0, size
    start = size - allowance
    # If `start` already sits just after a newline it IS a record boundary, and
    # scanning forward would throw away a complete record for nothing.
    handle.seek(start - 1)
    if handle.read(1) == b"\n":
        return start, size
    # Otherwise the handle is at `start`, which is where the scan below begins.
    # Scan FORWARD to a real newline, however far that is. Stopping after one
    # probe and reading from wherever it ended would start mid-record at an
    # arbitrary offset -- which is the whole thing this function exists to
    # prevent, and it leaks: a record holding `aws_secret_access_key="..."`
    # whose key falls before that offset arrives redacted of nothing, because
    # the redactor is anchored on the key and the key is in the part we threw
    # away. An AWS secret has no prefix for a shape rule to catch, so the
    # anchor was the only defence.
    # Bounded by the size already taken from the descriptor, not by EOF. A live
    # log has no EOF: scanning to it follows the writer for as long as it keeps
    # up, which is the same unbounded read `_redacted_records` is careful to
    # avoid, reintroduced one function earlier.
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
    # No boundary anywhere in the tail. Reading from `scanned` would be the same
    # mid-record start, so refuse the file; the caller turns this into a
    # warning, not a member.
    handle.seek(size)
    return size, size


def _redacted_records(
    handle: IO[bytes], fd: int, limit: int, deadline: float
) -> Iterator[str]:
    """Every line of one log, masked, in bounded chunks, stopping after `limit`.

    Never `handle.read()`: a runner log can be gigabytes, and the export must
    not size its memory on the largest file a host happens to hold.

    `limit` is not the same bound as the seek that positioned this read. The
    seek says where to start; a log that is being APPENDED to while the export
    runs has no end, so without a ceiling here the loop follows the writer for
    as long as it keeps up -- on the session log, which is the file both caps
    exist for, and which is by definition live when someone is exporting it.

    `deadline` is checked per RECORD, not per source and not per chunk, because
    both coarser choices overrun: one file is enough to blow the whole build,
    since the redactor's cost is quadratic in the length handed to it and a log
    of unterminated C1 introducers takes minutes on its own; and one 256 KiB
    chunk holds thousands of records, so checking between chunks lets the budget
    run over by the cost of all of them. Stopping mid-file truncates that file;
    letting it run times the caller out and produces nothing.
    """
    buffer = b""
    start = handle.tell()
    consumed = start
    # The current record already blew the budget; everything up to the next
    # newline belongs to it and is dropped with it.
    dropping = False
    while consumed - start < limit:
        chunk = handle.read(min(_READ_CHUNK_BYTES, limit - (consumed - start)))
        if not chunk:
            # EOF on a file that is now shorter than what we read means it
            # rotated or was truncated under us, so the tail we produced does
            # not line up with the head. Growth is ordinary -- it is a live log.
            if os.fstat(fd).st_size < consumed:
                raise OSError(errno.ESTALE, "log file shrank during export")
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
                # Over budget but terminated inside one chunk, so the buffer
                # guard below never saw it. Same rule, or a record just past
                # the limit would come through whole while a longer one did not.
                yield OVERSIZED_MARKER
            else:
                yield _redact_record(record)
        if len(buffer) > MAX_RECORD_BYTES:
            # Bounds the buffer as well as the record: a log with no newline in
            # it at all must not be held in memory whole.
            dropping = True
            buffer = b""
    # No `len(buffer) > MAX_RECORD_BYTES` here: the in-loop guard clears the
    # buffer after every chunk, so it cannot be over budget by this point.
    if dropping:
        yield OVERSIZED_MARKER
    elif buffer:
        yield _redact_record(buffer)


def _newest_first_across_families(
    sources: list[debug_log_sources.LogSource],
) -> list[debug_log_sources.LogSource]:
    """Round-robin the families instead of draining them one at a time.

    `list_sources` groups by family, newest first within each. Consumed in that
    order, a host with a large session log spends the whole byte budget on the
    server family and the bundle arrives with no runner logs at all -- which is
    usually the half that explains the problem. Taking the newest of every
    family, then the second newest of every family, means the budget runs out on
    the oldest attempts rather than on an entire category.
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

    `str(exc)` is what you would reach for and it is exactly wrong here:
    `OSError.__str__` appends the filename, which would put a host path -- the
    one thing member names are careful not to carry -- into the archive.
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
                    # Named rather than dropped silently, so the bundle says what
                    # it is missing instead of looking complete.
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
                        # reporting that as "no complete record" blames the
                        # export for a file that simply has nothing in it yet.
                        # An empty member is the honest answer there.
                        if size > 0 and skipped >= size:
                            # No record boundary anywhere in the tail, so there
                            # is no whole record to keep and reading anyway
                            # would start mid-record. An empty member reads as
                            # "this log was empty", which is a different and
                            # wrong story. `remaining` is deliberately NOT
                            # spent: this is a property of one file, and the
                            # next source may still fit.
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
                                # Charged even when the read failed partway: those
                                # bytes were still redacted and written, so a log
                                # that dies mid-export is not budget-free. Inside
                                # the `with`, because tell() needs an open handle.
                                remaining -= max(0, handle.tell() - before)
                except OSError as exc:
                    # The entry keeps whatever was copied before the failure:
                    # a partial log is still worth reading, and the warning
                    # says why it stops where it does.
                    warnings.append(_warning_line(member, exc))
            if warnings:
                # Through a bare ZipInfo, not the str overload: that one stamps
                # time.localtime(), which hands out the exporting host's clock
                # and, against the response Date header, its UTC offset. Every
                # log member is already at zipfile's 1980 default.
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
