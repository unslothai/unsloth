# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Masked terminal password prompt for the first-exposure password change.

Mirror of ``studio/backend/auth/terminal_prompt.py`` -- keep the two in sync.
The CLI parent cannot import the backend package outside the studio venv, so the
reader is duplicated here (like the auth mirroring in ``commands/studio.py``).

Input echoes one ``*`` per character (unlike ``getpass``). All output goes to
stderr so redirected stdout stays clean.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, TextIO

# Keep in sync with studio/backend/models/auth.py ChangePasswordRequest and auth/storage.py.
# The bound is `new_password` min_length.
MIN_PASSWORD_LENGTH = 8

# Mirror of studio/backend/auth/terminal_prompt.py; keep the name in sync.
SUPPLIED_PASSWORD_ENV = "UNSLOTH_STUDIO_PASSWORD"

_BACKSPACE_CHARS = ("\x7f", "\x08")
_SUBMIT_CHARS = ("\r", "\n")


class PromptUnattended(Exception):
    """A terminal is attached but nobody answered before the deadline.

    A pty is not a person: ``tmux new -d``, ``screen -dmS`` and ``docker run -dt``
    allocate a real, foreground pty nobody reads, so isatty() is True on both
    streams and the read never returns. Callers that must not block a launch pass
    a deadline and treat this as a refusal. Mirror of
    studio/backend/auth/terminal_prompt.py -- keep the two in sync.
    """


def _wait_for_first_key(timeout: float) -> bool:
    """Whether a keystroke arrived within ``timeout`` seconds.

    Must be called with the terminal already in cbreak mode: in canonical mode
    the driver holds input until a newline, so the fd would not become readable
    on the first character. True on any doubt, which is the blocking behaviour.
    """
    if os.name == "nt":
        import time

        try:
            import msvcrt
        except ImportError:
            return True
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if msvcrt.kbhit():
                return True
            time.sleep(0.05)
        return False
    import select

    try:
        fd = sys.stdin.fileno()
        ready, _, _ = select.select([fd], [], [], timeout)
    except (AttributeError, OSError, ValueError):
        return True
    return bool(ready)


class _RestoreTtyOnSignals:
    """Restore terminal attrs if SIGTERM/SIGHUP kills the prompt mid-read.

    A finally block can't run when a signal terminates the process, leaving the
    shared terminal in cbreak/no-echo. Best-effort: no-op off the main thread or
    where the signals are absent.
    """

    def __init__(self, fd: int, old_attrs) -> None:
        self._fd = fd
        self._old_attrs = old_attrs
        self._previous: list = []

    def __enter__(self) -> "_RestoreTtyOnSignals":
        import signal
        import termios

        def _restore_and_reraise(signum, frame):
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
            signal.signal(signum, signal.SIG_DFL)
            signal.raise_signal(signum)

        for name in ("SIGTERM", "SIGHUP"):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                self._previous.append((sig, signal.signal(sig, _restore_and_reraise)))
            except (ValueError, OSError):
                pass
        return self

    def __exit__(self, *exc) -> None:
        import signal
        for sig, previous in self._previous:
            try:
                signal.signal(sig, previous)
            except (ValueError, OSError):
                pass


def _read_masked_posix(prompt: str, out: TextIO, first_key_timeout: "float | None" = None) -> str:
    import codecs
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    out.write(prompt)
    out.flush()
    chars: list[str] = []
    try:
        with _RestoreTtyOnSignals(fd, old_attrs):
            # cbreak + ISIG off (mirrors terminal_prompt.py): with ISIG on, Ctrl-Z suspends mid-read and leaves
            # the shell no-echo.
            tty.setcbreak(fd)
            new_attrs = termios.tcgetattr(fd)
            new_attrs[3] &= ~termios.ISIG
            termios.tcsetattr(fd, termios.TCSADRAIN, new_attrs)
            # Deadline the FIRST keystroke only: a detached pty is a terminal that
            # nobody will ever type into, and blocking there never binds the socket.
            if first_key_timeout is not None and not _wait_for_first_key(first_key_timeout):
                raise PromptUnattended
            # os.read + incremental decoder with errors="replace": text-mode read(1) can raise or yield a lone
            # surrogate that later crashes pbkdf2.
            # It raises UnicodeDecodeError.
            decoder = codecs.getincrementaldecoder(sys.stdin.encoding or "utf-8")("replace")
            submitted = False
            while not submitted:
                raw = os.read(fd, 1)
                if not raw:  # stream ended mid-line: abort, don't submit
                    raise EOFError
                for ch in decoder.decode(raw):
                    if ch in _SUBMIT_CHARS:
                        submitted = True
                        break
                    if ch == "\x03":  # Ctrl-C (ISIG off: surfaces as a char)
                        raise KeyboardInterrupt
                    if ch in ("\x04", "\x1a"):
                        if not chars:
                            raise EOFError
                        continue
                    if ch in _BACKSPACE_CHARS:
                        if chars:
                            chars.pop()
                            out.write("\b \b")
                            out.flush()
                        continue
                    if ch < " ":
                        continue
                    chars.append(ch)
                    out.write("*")
                    out.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        out.write("\n")
        out.flush()
    return "".join(chars)


def _read_masked_windows(prompt: str, out: TextIO, first_key_timeout: "float | None" = None) -> str:
    import msvcrt

    out.write(prompt)
    out.flush()
    chars: list[str] = []
    try:
        if first_key_timeout is not None and not _wait_for_first_key(first_key_timeout):
            raise PromptUnattended
        while True:
            ch = msvcrt.getwch()
            if ch in _SUBMIT_CHARS:
                break
            if ch == "\x03":  # Ctrl-C: getwch swallows the signal, re-raise
                raise KeyboardInterrupt
            if ch in ("\x04", "\x1a"):
                if not chars:
                    raise EOFError
                continue
            if ch in ("\x00", "\xe0"):
                msvcrt.getwch()
                continue
            if ch in _BACKSPACE_CHARS:
                if chars:
                    chars.pop()
                    out.write("\b \b")
                    out.flush()
                continue
            if ch < " ":
                continue
            chars.append(ch)
            out.write("*")
            out.flush()
    finally:
        out.write("\n")
        out.flush()
    return "".join(chars)


def read_masked(
    prompt: str, out: TextIO | None = None, *, first_key_timeout: "float | None" = None
) -> str:
    """Read one line with ``*`` echo. Raises KeyboardInterrupt on Ctrl-C and
    EOFError on Ctrl-D/Ctrl-Z at an empty prompt, and PromptUnattended when
    ``first_key_timeout`` passes with no keystroke at all."""
    if out is None:
        out = sys.stderr
    if os.name == "nt":
        return _read_masked_windows(prompt, out, first_key_timeout)
    return _read_masked_posix(prompt, out, first_key_timeout)


def prompt_new_password(
    verify_current: Callable[[str], bool],
    out: TextIO | None = None,
    *,
    first_key_timeout: "float | None" = None,
) -> str:
    """Prompt for a new admin password until a valid, confirmed one is given.

    ``verify_current`` returns True when the candidate equals the current stored
    password; such candidates are rejected. KeyboardInterrupt/EOFError propagate
    so the caller can abort the launch.

    ``first_key_timeout`` bounds the wait for the FIRST keystroke and raises
    PromptUnattended if it never comes; once someone types there is no deadline.
    Only a caller that must not block a launch passes it -- a detached pty
    (``tmux new -d``, ``docker run -dt``) is a terminal nobody will answer.
    """
    if out is None:
        out = sys.stderr
    pending_timeout = first_key_timeout
    while True:
        password = read_masked("New password: ", out, first_key_timeout = pending_timeout)
        pending_timeout = None
        if len(password) < MIN_PASSWORD_LENGTH:
            out.write(f"Password must be at least {MIN_PASSWORD_LENGTH} characters. Try again.\n")
            out.flush()
            continue
        if any(ch.isspace() for ch in password):
            out.write("Password cannot contain spaces. Try again.\n")
            out.flush()
            continue
        if verify_current(password):
            out.write("New password must differ from the current password. Try again.\n")
            out.flush()
            continue
        confirmation = read_masked("Confirm new password: ", out)
        if confirmation != password:
            out.write("Passwords do not match. Try again.\n")
            out.flush()
            continue
        return password


def resolve_supplied_password(cli_value: "str | None", out: TextIO | None = None) -> "str | None":
    """Resolve a non-interactive initial admin password, or None if unset.

    Precedence: an explicit ``--password`` (literal ``-`` reads a line from
    stdin), then the ``UNSLOTH_STUDIO_PASSWORD`` env var; empty/omitted means off.
    A literal argv value is visible in the process list, so a note points at the
    env var or stdin instead. Mirror of the backend helper -- keep the two in sync.
    """
    if out is None:
        out = sys.stderr
    if cli_value == "-":
        line = sys.stdin.readline()
        if not line:
            return None
        return line.rstrip("\r\n") or None
    if cli_value:
        out.write(
            "Note: --password is visible in the process list and shell history; "
            f"prefer {SUPPLIED_PASSWORD_ENV} or --password - (stdin).\n"
        )
        out.flush()
        return cli_value
    return os.environ.get(SUPPLIED_PASSWORD_ENV) or None


def validate_new_password(candidate: str, verify_current: Callable[[str], bool]) -> "str | None":
    """Error message if ``candidate`` is unacceptable (too short or equal to the
    current password), else None. Same policy as the interactive loop."""
    if len(candidate) < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
    if any(ch.isspace() for ch in candidate):
        return "Password cannot contain spaces."
    if verify_current(candidate):
        return "New password must differ from the current password."
    return None
