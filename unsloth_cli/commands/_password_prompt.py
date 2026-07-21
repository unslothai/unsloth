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

# Keep in sync with studio/backend/models/auth.py ChangePasswordRequest
# (new_password min_length) and studio/backend/auth/storage.py.
MIN_PASSWORD_LENGTH = 8

# Env var that supplies the initial admin password non-interactively (mirror in
# studio/backend/auth/terminal_prompt.py). Keep the name in sync.
SUPPLIED_PASSWORD_ENV = "UNSLOTH_STUDIO_PASSWORD"

_BACKSPACE_CHARS = ("\x7f", "\x08")
_SUBMIT_CHARS = ("\r", "\n")


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
            except (ValueError, OSError):  # non-main thread / unsupported
                pass
        return self

    def __exit__(self, *exc) -> None:
        import signal
        for sig, previous in self._previous:
            try:
                signal.signal(sig, previous)
            except (ValueError, OSError):
                pass


def _read_masked_posix(prompt: str, out: TextIO) -> str:
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
            # cbreak + ISIG off (mirrors terminal_prompt.py): with ISIG on,
            # Ctrl-Z would suspend mid-read and leave the shell no-echo before
            # the finally restores it. Ctrl-C/Ctrl-Z arrive as \x03/\x1a here.
            tty.setcbreak(fd)
            new_attrs = termios.tcgetattr(fd)
            new_attrs[3] &= ~termios.ISIG
            termios.tcsetattr(fd, termios.TCSADRAIN, new_attrs)
            # Decode byte-at-a-time with errors="replace" (mirrors
            # terminal_prompt.py): text-mode read(1) can raise UnicodeDecodeError
            # on a pasted non-UTF-8 password or yield a lone surrogate that later
            # crashes pbkdf2. os.read + incremental decoder maps bad bytes to
            # U+FFFD and continues.
            decoder = codecs.getincrementaldecoder(sys.stdin.encoding or "utf-8")(
                "replace"
            )
            submitted = False
            while not submitted:
                raw = os.read(fd, 1)
                if not raw:  # stream ended mid-line: abort, don't submit
                    raise EOFError
                # One byte can complete >1 char, so iterate over the decoder's output.
                for ch in decoder.decode(raw):
                    if ch in _SUBMIT_CHARS:
                        submitted = True
                        break
                    if ch == "\x03":  # Ctrl-C (ISIG off: surfaces as a char)
                        raise KeyboardInterrupt
                    if ch in ("\x04", "\x1a"):  # Ctrl-D / Ctrl-Z
                        if not chars:
                            raise EOFError
                        continue
                    if ch in _BACKSPACE_CHARS:
                        if chars:
                            chars.pop()
                            out.write("\b \b")
                            out.flush()
                        continue
                    if ch < " ":  # other control characters
                        continue
                    chars.append(ch)
                    out.write("*")
                    out.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        out.write("\n")
        out.flush()
    return "".join(chars)


def _read_masked_windows(prompt: str, out: TextIO) -> str:
    import msvcrt

    out.write(prompt)
    out.flush()
    chars: list[str] = []
    try:
        while True:
            ch = msvcrt.getwch()
            if ch in _SUBMIT_CHARS:
                break
            if ch == "\x03":  # Ctrl-C: getwch swallows the signal, re-raise
                raise KeyboardInterrupt
            if ch in ("\x04", "\x1a"):  # Ctrl-D / Ctrl-Z
                if not chars:
                    raise EOFError
                continue
            if ch in ("\x00", "\xe0"):  # function/arrow key: swallow the code
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


def read_masked(prompt: str, out: TextIO | None = None) -> str:
    """Read one line with ``*`` echo. Raises KeyboardInterrupt on Ctrl-C and
    EOFError on Ctrl-D/Ctrl-Z at an empty prompt."""
    if out is None:
        out = sys.stderr
    if os.name == "nt":
        return _read_masked_windows(prompt, out)
    return _read_masked_posix(prompt, out)


def prompt_new_password(
    verify_current: Callable[[str], bool], out: TextIO | None = None
) -> str:
    """Prompt for a new admin password until a valid, confirmed one is given.

    ``verify_current`` returns True when the candidate equals the current stored
    password; such candidates are rejected. KeyboardInterrupt/EOFError propagate
    so the caller can abort the launch.
    """
    if out is None:
        out = sys.stderr
    while True:
        password = read_masked("New password: ", out)
        if len(password) < MIN_PASSWORD_LENGTH:
            out.write(
                f"Password must be at least {MIN_PASSWORD_LENGTH} characters. Try again.\n"
            )
            out.flush()
            continue
        if verify_current(password):
            out.write(
                "New password must differ from the current password. Try again.\n"
            )
            out.flush()
            continue
        confirmation = read_masked("Confirm new password: ", out)
        if confirmation != password:
            out.write("Passwords do not match. Try again.\n")
            out.flush()
            continue
        return password


def resolve_supplied_password(
    cli_value: "str | None", out: TextIO | None = None
) -> "str | None":
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


def validate_new_password(
    candidate: str, verify_current: Callable[[str], bool]
) -> "str | None":
    """Error message if ``candidate`` is unacceptable (too short or equal to the
    current password), else None. Same policy as the interactive loop."""
    if len(candidate) < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
    if verify_current(candidate):
        return "New password must differ from the current password."
    return None
