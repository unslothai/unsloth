# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""#7843: a spawn child's traceback goes down the inherited stderr, not the response queue. Best effort throughout."""

from __future__ import annotations

import atexit
import collections
import logging
import os
import sys
import tempfile
import threading

__all__ = [
    "LOG_RECORD_CONTINUATION_PREFIX",
    "LOG_RECORD_START_MARK",
    "STDERR_MIRROR_KWARG",
    "WorkerStderrCapture",
    "decode_worker_stderr",
    "install_worker_stderr_mirror",
    "mark_log_record_continuations",
    "stderr_tail_from_bytes",
]

# Marks a record's continuation lines: a recovered request's `exc_info` traceback is byte-identical to a dying process's.
LOG_RECORD_CONTINUATION_PREFIX = "    | "

# And its first line. UNIT SEPARATOR so it stays invisible in the operator's mirrored copy.
LOG_RECORD_START_MARK = "\x1f"

# A kwarg, not an environment variable: a process-wide value would cross workers spawning at once.
STDERR_MIRROR_KWARG = "unsloth_stderr_mirror_path"

DEFAULT_TAIL_LINES = 20
DEFAULT_TAIL_CHARS = 4000

MIRROR_FILE_CAP_BYTES = 256 * 1024

TAIL_READ_BYTES = 64 * 1024

_PUMP_JOIN_TIMEOUT_S = 2.0


def _utf8_edge_bounds(data: bytes, *, trim_end: bool = True) -> "tuple[int, int]":
    start = 0
    while start < len(data) and start < 3 and 0x80 <= data[start] < 0xC0:
        start += 1
    end = len(data)
    if not trim_end:
        return start, end
    for back in range(1, min(4, end - start) + 1):
        byte = data[end - back]
        if byte < 0x80:
            break
        if byte >= 0xC0:
            width = 2 if byte < 0xE0 else 3 if byte < 0xF0 else 4
            if back < width:
                end -= back
            break
    return start, end


def _without_partial_utf8_edges(data: bytes, *, trim_end: bool = True) -> bytes:
    """``trim_end`` is off at EOF, where a trailing 0xC0+ byte is a complete cp1252 character, not a severed one."""
    start, end = _utf8_edge_bounds(data, trim_end = trim_end)
    return data[start:end]


def decode_worker_stderr(data: bytes, *, ends_at_eof: bool = True) -> str:
    """UTF-8 before cp1252: cp1252 has a meaning for almost every byte, so the stricter encoding goes first."""
    text: str | None = None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = None
    if text is None:
        start, end = _utf8_edge_bounds(data)
        try:
            text = data[start:end].decode("utf-8")
        except UnicodeDecodeError:
            text = None
        else:
            # cp1252 costs at worst one spurious character; discarding loses crash detail.
            tail_bytes = data[end:] if ends_at_eof else b""
            if tail_bytes:
                try:
                    text += tail_bytes.decode("cp1252")
                except (UnicodeDecodeError, LookupError):
                    pass
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
    *,
    ends_at_eof: bool = True,
) -> str:
    text = decode_worker_stderr(data, ends_at_eof = ends_at_eof)
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


# Exact paths, never a pattern: Studios share one temporary directory.
_OPEN_SINKS: "set[str]" = set()
_ATEXIT_REGISTERED = False
# A fork inherits the set and the atexit registration, so the handler must check these paths are its own.
_SINKS_OWNER_PID: "int | None" = None


def _unlink_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _remove_open_sinks() -> None:
    if os.getpid() != _SINKS_OWNER_PID:
        # An inherited handler in a forked child: these paths are the parent's.
        return
    for path in list(_OPEN_SINKS):
        _unlink_quietly(path)
    _OPEN_SINKS.clear()


class WorkerStderrCapture:
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
        try:
            # O_NOFOLLOW: a tail is never worth following a symlink in a shared tmpdir for.
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
        """Tolerates a child still holding the sink open, which is the norm on Windows."""
        _OPEN_SINKS.discard(self._path)
        _unlink_quietly(self._path)


# O_NOFOLLOW is POSIX only and O_BINARY is Windows only; both are absent-means-zero here.
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_BINARY = getattr(os, "O_BINARY", 0)


def _open_existing_sink(path: str):
    """``open(path, "wb")`` is WRONG here: world-readable under the default umask, and follows a planted symlink."""
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


_TAIL_POLL_S = 0.05

# 64 x 64 KiB: a stalled operator stderr costs 4 MiB of buffer before the mirror starts dropping.
_MIRROR_RELAY_CHUNKS = 64


def _open_sink_for_append(path: str):
    flags = os.O_WRONLY | os.O_APPEND | _O_NOFOLLOW | _O_BINARY
    try:
        return os.open(path, flags)
    except OSError:
        return None


def _open_sink_for_reading(path: str):
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


_COMPACT_CATCH_UP_ROUNDS = 8


def _compact_sink(
    sink,
    cap_bytes: int,
    reader = None,
    emit = None,
) -> int:
    """Rewrite *sink* to roughly its last *cap_bytes* bytes.

    fd 2 is ``O_APPEND`` on it, so a plain read-rewrite-truncate loses concurrent appends.
    """
    if reader is not None and emit is not None:
        while True:
            try:
                pending = reader.read(65536)
            except (OSError, ValueError):
                break
            if not pending:
                break
            emit(pending)
    size = sink.seek(0, os.SEEK_END)
    keep = min(size, cap_bytes)
    sink.seek(size - keep)
    data = sink.read(keep)
    # Everything past `total` is an append that arrived after this function started looking.
    total = size
    for _ in range(_COMPACT_CATCH_UP_ROUNDS):
        appended = sink.read()
        if not appended:
            break
        if emit is not None:
            emit(appended)
        data += appended
        total += len(appended)
    sink.seek(0)
    sink.write(data)
    end = len(data)
    # A native thread can write into the region about to be truncated during the rewrite.
    for _ in range(_COMPACT_CATCH_UP_ROUNDS):
        sink.seek(total)
        late = sink.read()
        if not late:
            break
        if emit is not None:
            emit(late)
        total += len(late)
        sink.seek(end)
        sink.write(late)
        end += len(late)
    sink.truncate(end)
    return end


class _MirrorRelay:
    """Owns the only blocking write, so the thread that bounds the sink never parks in one.

    The operator's stderr is a pipe in the packaged app, and a reader that stops draining it
    would otherwise stall the pump inside ``os.write`` and leave the sink growing without limit.
    The mirrored copy is best effort, so a full buffer drops its oldest chunk; the sink keeps
    the tail, which is the crash record the parent reports.
    """

    def __init__(self, inherited_fd: int, max_chunks: int) -> None:
        self._fd = inherited_fd
        self._chunks = collections.deque(maxlen = max_chunks)
        self._wake = threading.Event()
        self._closed = False
        self._thread = threading.Thread(
            target = self._run,
            name = "unsloth-worker-stderr-relay",
            daemon = True,
        )
        self._thread.start()

    def emit(self, data: bytes) -> None:
        if not data:
            return
        self._chunks.append(data)
        self._wake.set()

    def close(self) -> None:
        self._closed = True
        self._wake.set()

    def join(self, timeout: "float | None" = None) -> None:
        self._thread.join(timeout = timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        while True:
            try:
                data = self._chunks.popleft()
            except IndexError:
                if self._closed:
                    return
                self._wake.wait(_TAIL_POLL_S)
                self._wake.clear()
                continue
            try:
                os.write(self._fd, data)
            except OSError:
                pass


def _sink_size(sink) -> int:
    try:
        return os.fstat(sink.fileno()).st_size
    except (OSError, ValueError):
        return 0


def _tail_sink_to_stderr(
    reader, relay: "_MirrorRelay", sink, cap_bytes: int, stop: "threading.Event"
) -> None:
    """fd 2 must be the SINK, not a pipe: a fatal signal kills anything buffered in a pipe with its writer."""
    while True:
        try:
            chunk = reader.read(65536)
        except (OSError, ValueError):
            break
        if not chunk:
            if stop.is_set():
                break
            stop.wait(_TAIL_POLL_S)
            continue
        relay.emit(chunk)
        # Bytes ON DISK, not bytes relayed: the relay can fall arbitrarily far behind.
        if cap_bytes > 0 and _sink_size(sink) > 2 * cap_bytes:
            try:
                _compact_sink(sink, cap_bytes, reader = reader, emit = relay.emit)
                # The file was rewritten from the front; the old offset is now meaningless.
                reader.seek(0, os.SEEK_END)
            except (OSError, ValueError):
                pass
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
    relay = None,
) -> None:
    """The pump must be a daemon or ``BaseProcess._bootstrap`` waits on it for ever.

    ``multiprocessing`` prints the traceback before atexit, so draining here still catches it.
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
        # fd 2 no longer points at the sink, so the next empty read is the end, not a pause.
        stop.set()
    pump.join(timeout = _PUMP_JOIN_TIMEOUT_S)
    if relay is not None:
        relay.close()
        relay.join(timeout = _PUMP_JOIN_TIMEOUT_S)
    if pump.is_alive() or (relay is not None and relay.is_alive()):
        # A thread may be inside os.write(inherited_fd, ...); closing frees the number for any thread's next open().
        return
    try:
        os.close(inherited_fd)
    except OSError:
        pass


def install_worker_stderr_mirror(
    path: "str | None", cap_bytes: int = MIRROR_FILE_CAP_BYTES
) -> bool:
    """At the descriptor level: a worker dying inside a C extension writes fd 2 directly and bypasses a Python wrapper."""
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
    # O_APPEND: compaction rewrites from the front and two handles share the file.
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
    relay = _MirrorRelay(inherited, _MIRROR_RELAY_CHUNKS)
    pump = threading.Thread(
        target = _tail_sink_to_stderr,
        args = (reader, relay, sink, cap_bytes, stop),
        name = "unsloth-worker-stderr-mirror",
        daemon = True,
    )
    pump.start()
    atexit.register(_stop_mirror, inherited, pump, stop, relay)
    return True


class _EveryLineCarriesThePrefix(logging.Formatter):
    def __init__(self, inner: "logging.Formatter") -> None:
        super().__init__()
        self._inner = inner

    def format(self, record: "logging.LogRecord") -> str:
        text = self._inner.format(record)
        first, newline, rest = text.partition("\n")
        # The first line too: a default-formatted single-line record has no shape to spot.
        marked_first = (
            first if first.startswith(LOG_RECORD_START_MARK) else LOG_RECORD_START_MARK + first
        )
        if not newline:
            return marked_first
        return (
            marked_first
            + "\n"
            + "\n".join(LOG_RECORD_CONTINUATION_PREFIX + line for line in rest.split("\n"))
        )

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


_UNHOOKED_SET_FORMATTER = None
_UNHOOKED_ADD_HANDLER = None


def _mark_handler(handler) -> bool:
    formatter = getattr(handler, "formatter", None)
    if isinstance(formatter, _EveryLineCarriesThePrefix):
        return False
    setter = _UNHOOKED_SET_FORMATTER or type(handler).setFormatter
    setter(handler, _EveryLineCarriesThePrefix(formatter or logging.Formatter()))
    return True


def _install_continuation_hook() -> bool:
    """Patched on the CLASS: the ML stack is imported after logging is configured and adds its own handlers."""
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
    """Covers ``logging.lastResort`` and later handlers too: a miss hands a RECOVERED
    request's traceback to the next caller as their crash."""
    root = logger_object if logger_object is not None else logging.getLogger()
    wrapped = 0
    for handler in list(getattr(root, "handlers", ())):
        if _mark_handler(handler):
            wrapped += 1
    if cover_later_handlers:
        last_resort = getattr(logging, "lastResort", None)
        if last_resort is not None and _mark_handler(last_resort):
            wrapped += 1
        _install_continuation_hook()
    return wrapped
