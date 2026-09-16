# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep a worker's stderr where the parent can read it back after the worker is gone.

Studio's workers are ``multiprocessing`` spawn children that inherit the server's stderr.
When such a child dies from an unhandled exception, the traceback ``multiprocessing``
prints travels down that inherited handle and never through the response queue, so the
parent is left holding an exit status and nothing else: "pid=6145, exitcode=1" with no
cause (#7843).

Two halves:

* Parent side. :class:`WorkerStderrCapture` owns a file, hands its path to the child, and
  :meth:`WorkerStderrCapture.tail` reads the end of it back once the child has exited.
* Child side. :func:`install_worker_stderr_mirror` tees file descriptor 2 into that file
  while still writing everything through to the inherited stderr, so the server log keeps
  exactly what it had before.

Everything here is best effort by design. A mirror that cannot be installed, a sink that
cannot be written and a tail that cannot be read all degrade to the previous behaviour
rather than turning a diagnosable crash into a failure to start.
"""

from __future__ import annotations

import atexit
import logging
import os
import sys
import tempfile
import threading
import time

__all__ = [
    "LOG_RECORD_CONTINUATION_PREFIX",
    "STDERR_MIRROR_KWARG",
    "WorkerStderrCapture",
    "decode_worker_stderr",
    "install_worker_stderr_mirror",
    "mark_log_record_continuations",
    "stderr_tail_from_bytes",
]

# What a log record's SECOND and later lines carry, so the parent can tell them apart from
# the lines a crashing runtime writes.
#
# The parent's filter has to answer, per line, "did the logging stack write this". A record
# is prefixed on its first line only, so every line after it is bare content at column 0 --
# and `logger.error(..., exc_info = True)`, which `worker.py` uses for a request failure it
# RECOVERS from, writes a whole `Traceback (most recent call last):` that way. No amount of
# pattern matching separates that from the traceback of a process that actually died: the
# bytes are the same. Guessing in one direction hands another account's exception text to
# the next caller as their crash; guessing in the other drops the diagnosis this capture
# exists to deliver.
#
# So the writer says which it is instead. Every continuation line of every record is marked
# here, at the one handler that writes them, and anything left unmarked at column 0 came
# from something that was not the logging stack.
LOG_RECORD_CONTINUATION_PREFIX = "    | "

# The reserved keyword argument the shared child entrypoint intercepts. Passed as a kwarg
# rather than an environment variable on purpose: several workers can be spawned at once,
# and a process-wide variable would hand one worker's sink to another worker's child.
STDERR_MIRROR_KWARG = "unsloth_stderr_mirror_path"

# A traceback plus the couple of lines that preceded it. Enough to name the exception and
# where it came from, short enough to put in a chat error bubble.
DEFAULT_TAIL_LINES = 20
DEFAULT_TAIL_CHARS = 4000

# What the mirror file is allowed to hold. A worker's stderr is not small: progress bars,
# transformers warnings and llama.cpp chatter all land in it over a long session. The pump
# compacts the file down to this many trailing bytes once it has grown past twice it, so
# the tail survives without the file growing without bound.
MIRROR_FILE_CAP_BYTES = 256 * 1024

# How much of the file the parent reads to build a tail. Bounded separately from the cap so
# a sink written by some other producer cannot make the read expensive.
TAIL_READ_BYTES = 64 * 1024

_PUMP_JOIN_TIMEOUT_S = 2.0


def _without_partial_utf8_edges(data: bytes) -> bytes:
    """Drop a UTF-8 sequence that a byte window cut in half at either end.

    Both byte windows in this module open at an arbitrary offset: the parent reads the last
    ``TAIL_READ_BYTES`` of the sink, and the pump compacts the sink down to its last
    ``cap_bytes``. Neither can land on a character boundary on purpose. Without this, one
    severed multi-byte character made strict UTF-8 fail for the WHOLE window, the probe fell
    through to cp1252, and every non-ASCII character in the traceback came back as the two
    or three cp1252 characters its UTF-8 bytes spell. Trimming at most three bytes off each
    end costs nothing and keeps the rest readable.
    """
    start = 0
    # A window that opens mid-character starts with continuation bytes (0b10xxxxxx).
    while start < len(data) and start < 3 and 0x80 <= data[start] < 0xC0:
        start += 1
    end = len(data)
    # A window that closes mid-character ends with a lead byte and too few continuations.
    for back in range(1, min(4, end - start) + 1):
        byte = data[end - back]
        if byte < 0x80:
            break
        if byte >= 0xC0:
            width = 2 if byte < 0xE0 else 3 if byte < 0xF0 else 4
            if back < width:
                end -= back
            break
    return data[start:end]


def decode_worker_stderr(data: bytes) -> str:
    """Decode worker stderr bytes and normalise their line endings.

    Encoding is probed, not assumed. A worker on Linux or macOS writes UTF-8, while on
    Windows ``sys.stderr`` uses the ANSI code page unless UTF-8 mode is on, and native
    libraries write whatever the console page is; cp1252 is the common case there. UTF-8 is
    tried first because it is the stricter of the two, so text that is valid UTF-8 is never
    mis-read as cp1252. That claim only holds once a character severed by the byte window is
    trimmed first, which is what ``_without_partial_utf8_edges`` is for; cp1252 has a meaning
    for almost every byte, so without the trim a single severed character silently converted
    the entire tail to mojibake. Anything that is neither encoding is decoded with
    replacement rather than discarded: a mangled traceback still names the exception.

    CRLF and lone CR both become LF, so a Windows traceback does not arrive with a trailing
    carriage return on every line, and a progress bar that redraws itself with CR becomes
    separate lines that the tail can then drop.
    """
    text: str | None = None
    for candidate in (data, _without_partial_utf8_edges(data)):
        try:
            text = candidate.decode("utf-8")
        except UnicodeDecodeError:
            continue
        break
    if text is None:
        try:
            text = data.decode("cp1252")
        except (UnicodeDecodeError, LookupError):
            text = data.decode("utf-8", errors = "replace")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def stderr_tail_from_bytes(
    data: bytes,
    max_lines: int = DEFAULT_TAIL_LINES,
    max_chars: int = DEFAULT_TAIL_CHARS,
) -> str:
    """Return the last few meaningful lines of *data* as text.

    Blank lines are dropped: they carry nothing and, after the CR normalisation above, a
    redrawn progress bar produces a great many of them. The character cap is applied after
    the line cap and trims from the front, so the exception line at the end is the last
    thing to go; a partial first line is dropped rather than shown cut in half.
    """
    text = decode_worker_stderr(data)
    lines = [line.rstrip() for line in text.split("\n")]
    lines = [line for line in lines if line.strip()]
    if not lines:
        return ""
    if max_lines > 0:
        lines = lines[-max_lines:]
    joined = "\n".join(lines)
    if max_chars > 0 and len(joined) > max_chars:
        joined = joined[-max_chars:]
        first_break = joined.find("\n")
        if first_break != -1:
            joined = joined[first_break + 1 :]
    return joined


# Sinks this process opened and has not retired yet. Its own paths only, never a pattern and
# never a sweep of the directory: several Studios can share one temporary directory (separate
# UNSLOTH_STUDIO_HOME values, or two accounts on one machine), and a sweep by prefix would
# delete a sink belonging to another Studio's live worker.
_OPEN_SINKS: "set[str]" = set()
_ATEXIT_REGISTERED = False
# Which process the set above belongs to. A fork inherits both the set and the atexit
# registration, and a forked child that exits normally would then run the handler and delete
# a sink its parent is still filling. Spawn children are unaffected, since they re-import
# this module clean, but the backend does fork elsewhere (dataset preprocessing pools), and
# "the exact paths this process opened" has to mean this process.
_SINKS_OWNER_PID: "int | None" = None


def _unlink_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _remove_open_sinks() -> None:
    """Retire whatever is still open at interpreter exit, bounding the residue to a hard kill."""
    if os.getpid() != _SINKS_OWNER_PID:
        # An inherited handler in a forked child. These paths are the parent's and the
        # parent is still writing to one of them.
        return
    for path in list(_OPEN_SINKS):
        _unlink_quietly(path)
    _OPEN_SINKS.clear()


class WorkerStderrCapture:
    """A file a spawned worker mirrors its stderr into, readable after the worker exits."""

    def __init__(
        self,
        directory: "str | None" = None,
        prefix: str = "unsloth-worker-",
    ) -> None:
        global _ATEXIT_REGISTERED, _SINKS_OWNER_PID
        handle, self._path = tempfile.mkstemp(
            prefix = prefix,
            suffix = ".stderr",
            dir = directory,
        )
        os.close(handle)
        # A sink is retired when the next worker is spawned, so at most one is live at a time.
        # The server exiting is the case with no next worker: without this the last sink
        # outlives the process that made it and waits for the temporary directory to be
        # reclaimed, which on a long-lived desktop can be never.
        _OPEN_SINKS.add(self._path)
        _SINKS_OWNER_PID = os.getpid()
        if not _ATEXIT_REGISTERED:
            atexit.register(_remove_open_sinks)
            _ATEXIT_REGISTERED = True

    @property
    def path(self) -> str:
        return self._path

    def tail(
        self,
        max_lines: int = DEFAULT_TAIL_LINES,
        max_chars: int = DEFAULT_TAIL_CHARS,
    ) -> str:
        """The end of what the worker wrote, or an empty string when there is nothing."""
        try:
            # O_NOFOLLOW for the same reason the child uses it: between this process
            # unlinking a sink and reading one, the path is a name in a shared temporary
            # directory, and a tail is never worth following a symlink for.
            fd = os.open(self._path, os.O_RDONLY | _O_NOFOLLOW | _O_BINARY)
            with os.fdopen(fd, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - TAIL_READ_BYTES))
                data = handle.read()
        except OSError:
            return ""
        return stderr_tail_from_bytes(data, max_lines = max_lines, max_chars = max_chars)

    def close(self) -> None:
        """Remove the sink. Tolerates a child still holding it open, which is the norm on
        Windows: the file stays behind for the temporary directory to reclaim rather than
        the caller taking a PermissionError during shutdown."""
        _OPEN_SINKS.discard(self._path)
        _unlink_quietly(self._path)


# O_NOFOLLOW is POSIX only and O_BINARY is Windows only; both are absent-means-zero here.
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_BINARY = getattr(os, "O_BINARY", 0)


def _open_existing_sink(path: str):
    """Open the sink the parent already created, and never create one here.

    Deliberately not ``open(path, "wb")``. The sink lives in the system temporary directory,
    which on a shared machine is world writable, and the only way the child finds the file
    missing is that the parent retired it. Creating it back would do two things this must not
    do: it would create with ``0666 & ~umask``, so another account on the machine could read
    a worker's stderr (measured at 0664 under the default umask), and it would follow a
    symlink planted at that path and truncate whatever the Studio user can write. The sink is
    a diagnostic, so there is nothing to gain by re-creating it: no sink is the old behaviour.
    """
    # Read-write, because the pump compacts the file in place once it passes the cap.
    flags = os.O_RDWR | _O_NOFOLLOW | _O_BINARY
    try:
        handle = os.open(path, flags)
    except OSError:
        return None
    try:
        return os.fdopen(handle, "r+b", buffering = 0)
    except OSError:
        try:
            os.close(handle)
        except OSError:
            pass
        return None


# How long the tail thread waits before asking the sink for more. The worker's stderr is a
# diagnostic, not a stream anyone is watching character by character, and a poll this cheap
# costs nothing next to a load.
_TAIL_POLL_S = 0.05


def _open_sink_for_append(path: str):
    """The sink opened O_APPEND, as a raw fd, for fd 2 to become.

    Never creates, and never follows a symlink, for the reasons in `_open_existing_sink`.
    """
    flags = os.O_WRONLY | os.O_APPEND | _O_NOFOLLOW | _O_BINARY
    try:
        return os.open(path, flags)
    except OSError:
        return None


def _open_sink_for_reading(path: str):
    """A second, independent handle on the sink for the tail thread."""
    flags = os.O_RDONLY | _O_NOFOLLOW | _O_BINARY
    try:
        handle = os.open(path, flags)
    except OSError:
        return None
    try:
        return os.fdopen(handle, "rb", buffering = 0)
    except OSError:
        try:
            os.close(handle)
        except OSError:
            pass
        return None


# How many times the compactor re-reads the end of the sink before it truncates. The
# writer is fd 2 in this same process and each round only has to catch up with what was
# appended during the previous one, so this converges immediately in practice; the bound is
# there so a worker flooding stderr cannot hold the compactor in the loop for ever.
_COMPACT_CATCH_UP_ROUNDS = 8


def _compact_sink(
    sink,
    cap_bytes: int,
    reader = None,
    inherited_fd: "int | None" = None,
) -> int:
    """Rewrite *sink* so it holds roughly its last *cap_bytes* bytes. Returns the new size.

    The subtlety is that fd 2 is open ``O_APPEND`` on this same file and the worker can
    write at any point during this function. A plain read-rewrite-truncate deletes anything
    that lands after the read: it is beyond the offset the compactor is about to truncate
    to, and the reader has not forwarded it either, so a fatal diagnostic written in that
    window disappears from the capture AND from the inherited stderr. Two things narrow it:

    * *reader* and *inherited_fd*, when given, are drained to the console FIRST, so nothing
      still unforwarded can be deleted without having been seen.
    * The file is read past its accounted-for end twice, once after the tail read and again
      after the rewrite, and whatever arrived in the meantime is kept instead of being
      truncated away. Those two reads cover the wide parts of the window: reading and then
      writing a quarter of a megabyte.

    What remains is the gap between the last read that came back empty and the ``truncate``
    itself, a single syscall apart. It cannot be closed portably: an ``O_APPEND`` write and
    a truncate cannot be ordered against each other without cooperation from the writer,
    and rotating the sink instead would need a rename over a file that fd 2 still holds
    open, which Windows refuses.
    """
    if reader is not None and inherited_fd is not None:
        # Forward the unread tail before touching the file, so the bytes at risk below are
        # at least already on the server's own stderr.
        while True:
            try:
                pending = reader.read(65536)
            except (OSError, ValueError):
                break
            if not pending:
                break
            try:
                os.write(inherited_fd, pending)
            except OSError:
                pass
    size = sink.seek(0, os.SEEK_END)
    keep = min(size, cap_bytes)
    sink.seek(size - keep)
    data = sink.read(keep)
    # How many bytes of the file have been accounted for. Everything past it is an append
    # that arrived after this function started looking.
    total = size
    for _ in range(_COMPACT_CATCH_UP_ROUNDS):
        # Continues from the old end of the file, so this is exactly what was appended
        # while the read above was in flight, with nothing counted twice.
        appended = sink.read()
        if not appended:
            break
        if inherited_fd is not None:
            try:
                os.write(inherited_fd, appended)
            except OSError:
                pass
        data += appended
        total += len(appended)
    sink.seek(0)
    sink.write(data)
    end = len(data)
    # The rewrite above is the other wide part of the window: writing a quarter of a
    # megabyte takes long enough for a native thread to emit a fatal diagnostic into the
    # region that is about to be truncated away. Read past the accounted-for end again and
    # keep whatever arrived, rather than letting `truncate` delete it.
    for _ in range(_COMPACT_CATCH_UP_ROUNDS):
        sink.seek(total)
        late = sink.read()
        if not late:
            break
        if inherited_fd is not None:
            try:
                os.write(inherited_fd, late)
            except OSError:
                pass
        total += len(late)
        sink.seek(end)
        sink.write(late)
        end += len(late)
    sink.truncate(end)
    return end


def _tail_sink_to_stderr(
    reader, inherited_fd: int, sink, cap_bytes: int, stop: "threading.Event"
) -> None:
    """Copy what fd 2 has written into the sink onward to the inherited stderr.

    The direction is the point. fd 2 is the SINK, so every byte is in the file the moment
    the kernel returns from the write, whatever happens to this process next; this thread
    only forwards them to the server's own stderr so the console keeps behaving as it did.
    An earlier shape put a pipe on fd 2 and had this thread write BOTH ends, which meant
    anything still in the pipe buffer when a fatal signal arrived -- `SIGSEGV`, `SIGABRT`,
    the `terminate called after throwing an instance of 'c10::Error'` that precedes an
    abort -- died with the writer, and it never reached the inherited stderr either. The
    capture would then have made the crash LESS visible than it was before it existed.

    Compaction rewrites the file from the front; fd 2 is opened ``O_APPEND``, so the
    writer's next write still lands at the (new) end rather than at a stale offset. The
    reader is handed to the compactor so that everything it has not forwarded yet goes to
    the console before any of it can be rewritten away: see `_compact_sink`.
    """
    forwarded = 0
    while True:
        try:
            chunk = reader.read(65536)
        except (OSError, ValueError):
            break
        if not chunk:
            if stop.is_set():
                break
            time.sleep(_TAIL_POLL_S)
            continue
        try:
            os.write(inherited_fd, chunk)
        except OSError:
            # The server's own stderr going away must not stop the capture.
            pass
        forwarded += len(chunk)
        if cap_bytes > 0 and forwarded > 2 * cap_bytes:
            try:
                _compact_sink(sink, cap_bytes, reader = reader, inherited_fd = inherited_fd)
                # Everything up to here has already been forwarded, and the file has just
                # been rewritten from the front, so the reader's old offset points into the
                # middle of retained text. The end is the only meaningful place to resume.
                reader.seek(0, os.SEEK_END)
            except (OSError, ValueError):
                pass
            forwarded = 0
    try:
        sink.close()
    except OSError:
        pass
    try:
        reader.close()
    except (OSError, ValueError):
        pass


def _stop_mirror(
    inherited_fd: int,
    pump: threading.Thread,
    stop = None,
) -> None:
    """Restore the inherited stderr and let the pump drain what is still in the pipe.

    Registered with :mod:`atexit`, which is what makes this worth having: the pump has to be
    a daemon thread or ``threading._shutdown`` inside ``BaseProcess._bootstrap`` would wait
    on it for ever, and a daemon thread is killed at interpreter shutdown without draining.
    ``multiprocessing`` prints the traceback and flushes the standard streams before that
    point, so restoring fd 2 here closes the pipe's only writer, the pump reads EOF, and the
    last bytes reach the sink before the process is gone.
    """
    try:
        sys.stderr.flush()
    except Exception:
        pass
    try:
        os.dup2(inherited_fd, 2)
    except OSError:
        pass
    if stop is not None:
        # fd 2 no longer points at the sink, so nothing more will be appended: tell the
        # tail thread that the next empty read is the end rather than a pause.
        stop.set()
    pump.join(timeout = _PUMP_JOIN_TIMEOUT_S)
    if pump.is_alive():
        # The join timed out, which means something else in this process still holds a
        # descriptor onto the pipe (a logging handler or a native library that dup'd fd 2),
        # so the pump has not seen EOF and may be inside os.write(inherited_fd, ...) right
        # now. Closing the descriptor here would free its number for the next open() in any
        # thread, and the pump would then write a worker's stderr into an unrelated file.
        # The interpreter is on its way out, so leaking one descriptor is free; corrupting
        # somebody else's file is not.
        return
    try:
        os.close(inherited_fd)
    except OSError:
        pass


def install_worker_stderr_mirror(
    path: "str | None", cap_bytes: int = MIRROR_FILE_CAP_BYTES
) -> bool:
    """Tee this process's stderr into *path*. Returns True when the mirror is installed.

    Installed at the file-descriptor level rather than by replacing ``sys.stderr``: a worker
    that dies inside a C extension, or that is killed after ``faulthandler`` has written its
    report, writes to fd 2 directly and would bypass a Python-level wrapper.
    """
    if not path:
        return False
    sink = _open_existing_sink(path)
    if sink is None:
        return False
    try:
        inherited = os.dup(2)
    except OSError:
        sink.close()
        return False
    # The WRITE side of the sink, opened O_APPEND and put straight on fd 2. Append matters
    # twice: the compaction below rewrites the file from the front, and two handles are on
    # the same file, so a fixed offset would either overwrite or leave a hole.
    writer_fd = _open_sink_for_append(path)
    if writer_fd is None:
        os.close(inherited)
        sink.close()
        return False
    reader = _open_sink_for_reading(path)
    if reader is None:
        os.close(writer_fd)
        os.close(inherited)
        sink.close()
        return False
    try:
        sys.stderr.flush()
    except Exception:
        pass
    try:
        reader.seek(0, os.SEEK_END)
    except (OSError, ValueError):
        pass
    try:
        os.dup2(writer_fd, 2)
    except OSError:
        os.close(writer_fd)
        reader.close()
        os.close(inherited)
        sink.close()
        return False
    os.close(writer_fd)
    stop = threading.Event()
    pump = threading.Thread(
        target = _tail_sink_to_stderr,
        args = (reader, inherited, sink, cap_bytes, stop),
        name = "unsloth-worker-stderr-mirror",
        daemon = True,
    )
    pump.start()
    atexit.register(_stop_mirror, inherited, pump, stop)
    return True


class _EveryLineCarriesThePrefix(logging.Formatter):
    """A formatter that marks a record's continuation lines and delegates everything else.

    Wrapping rather than replacing: the handler's own formatter decides what a record looks
    like, including structlog's, and this only touches what happens after the first
    newline. A single-line record is returned byte for byte.
    """

    def __init__(self, inner: "logging.Formatter") -> None:
        super().__init__()
        self._inner = inner

    def format(self, record: "logging.LogRecord") -> str:
        text = self._inner.format(record)
        first, newline, rest = text.partition("\n")
        if not newline:
            return text
        return (
            first
            + "\n"
            + "\n".join(LOG_RECORD_CONTINUATION_PREFIX + line for line in rest.split("\n"))
        )

    def __getattr__(self, name: str):
        # Anything else a handler asks of its formatter belongs to the real one.
        return getattr(self._inner, name)


# The unpatched methods, kept so the hook below is installed exactly once and can be lifted
# again by a test. None until `mark_log_record_continuations` installs it.
_UNHOOKED_SET_FORMATTER = None
_UNHOOKED_ADD_HANDLER = None


def _mark_handler(handler) -> bool:
    """Wrap one handler's formatter, whatever it is or is not. True when this call did it."""
    formatter = getattr(handler, "formatter", None)
    if isinstance(formatter, _EveryLineCarriesThePrefix):
        return False
    setter = _UNHOOKED_SET_FORMATTER or type(handler).setFormatter
    # A handler with no formatter of its own still writes multi-line records: logging falls
    # back to its module-level default, which marks nothing.
    setter(handler, _EveryLineCarriesThePrefix(formatter or logging.Formatter()))
    return True


def _install_continuation_hook() -> bool:
    """Make the marking outlive the moment it was asked for. True when installed here.

    The worker configures logging and then imports the ML stack, and libraries add their own
    handlers on import -- and a later `setFormatter` on an already-marked handler would drop
    the wrapper again. Both are patched on the CLASS, so they cover every logger in the
    process and not just the one that existed at startup.
    """
    global _UNHOOKED_SET_FORMATTER, _UNHOOKED_ADD_HANDLER
    if _UNHOOKED_SET_FORMATTER is not None:
        return False
    unhooked_set = logging.Handler.setFormatter
    unhooked_add = logging.Logger.addHandler

    def setFormatter(self, fmt):  # noqa: N802 -- matches logging's own spelling
        if fmt is not None and not isinstance(fmt, _EveryLineCarriesThePrefix):
            fmt = _EveryLineCarriesThePrefix(fmt)
        unhooked_set(self, fmt)

    def addHandler(self, hdlr):  # noqa: N802 -- matches logging's own spelling
        unhooked_add(self, hdlr)
        try:
            _mark_handler(hdlr)
        except Exception:  # noqa: BLE001 -- marking must never break someone's logging
            pass

    _UNHOOKED_SET_FORMATTER = unhooked_set
    _UNHOOKED_ADD_HANDLER = unhooked_add
    logging.Handler.setFormatter = setFormatter
    logging.Logger.addHandler = addHandler
    return True


def mark_log_record_continuations(logger_object = None, *, cover_later_handlers = True) -> int:
    """Mark every continuation line written by the handlers on *logger_object*.

    Called by the worker right after its logging is configured, so the parent reading its
    stderr can tell a logged traceback from a crash. Returns how many handlers were wrapped,
    which is what a test can assert on; wrapping twice is a no-op, so calling it again after
    a reconfiguration is safe.

    The handlers present at the instant of the call are not the whole of it, and a miss here
    is not cosmetic: an unmarked traceback belonging to a request this worker RECOVERED from
    looks exactly like the traceback of a process that died, and on a shared worker the tail
    is handed to the next caller as their crash. So this also covers

    * ``logging.lastResort``, which is what stdlib logging writes through when a record
      reaches a logger with no handler at all -- the worker's own structlog setup adds no
      root handler, so ordinary library logging lands there;
    * every handler added AFTER this call, and every later ``setFormatter`` on one already
      marked, through a class-level hook (the worker imports the whole ML stack after
      configuring its logging).

    Both of those are process-wide rather than about one logger, so ``cover_later_handlers``
    turns them off together for a caller that wants only the handlers it named.
    """
    root = logger_object if logger_object is not None else logging.getLogger()
    wrapped = 0
    for handler in list(getattr(root, "handlers", ())):
        if _mark_handler(handler):
            wrapped += 1
    if cover_later_handlers:
        # Process-wide, so it is one flag: a caller that only wants THIS logger's handlers
        # marked (a test, mainly) gets exactly that and leaves the interpreter as it found it.
        last_resort = getattr(logging, "lastResort", None)
        if last_resort is not None and _mark_handler(last_resort):
            wrapped += 1
        _install_continuation_hook()
    return wrapped
