# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Interactive terminal prompt that forces a bootstrap password change before
Studio is exposed on a public Cloudflare URL (``--secure`` / ``--cloudflare``).

Masked input echoes one ``*`` per keystroke (unlike ``getpass``, which hides
input entirely and routinely reads as "the prompt is frozen"). Works on
Windows (``msvcrt``) and Linux/macOS (``termios``). All prompt output goes to
stderr so redirected stdout (``--silent`` consumers, log pipes) never swallows
a security prompt.

Mirrored for the CLI parent process at ``unsloth_cli/commands/_password_prompt.py``
(the CLI cannot import the Studio backend package); keep the two in sync.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, TextIO

_CTRL_C = "\x03"
_CTRL_D = "\x04"
_CTRL_Z = "\x1a"
_BACKSPACES = ("\x7f", "\x08")
_SUBMITS = ("\r", "\n")


def _getch_windows() -> str:  # pragma: no cover - exercised via fake on Linux CI
    import msvcrt

    ch = msvcrt.getwch()
    # Function/arrow keys arrive as a two-wchar sequence starting with \x00 or
    # \xe0; consume the second half and report a no-op control char.
    if ch in ("\x00", "\xe0"):
        msvcrt.getwch()
        return "\x00"
    return ch


class _RestoreTtyOnSignals:
    """Restore terminal attrs if SIGTERM/SIGHUP kills the prompt mid-read.

    A finally block cannot run when a default-disposition signal terminates
    the process, which would leave the shared terminal in cbreak/no-echo mode.
    Best-effort: silently a no-op off the main thread or on platforms without
    the signals.
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


def _getch_posix() -> str:  # pragma: no cover - needs a real tty
    import codecs
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        with _RestoreTtyOnSignals(fd, old_attrs):
            # cbreak (not raw) keeps output post-processing while disabling echo
            # and line buffering. cbreak leaves ISIG on, so clear it and surface
            # Ctrl-C as \x03 to the caller loop, which restores the tty first.
            tty.setcbreak(fd, termios.TCSADRAIN)
            new_attrs = termios.tcgetattr(fd)
            new_attrs[3] &= ~termios.ISIG
            termios.tcsetattr(fd, termios.TCSADRAIN, new_attrs)
            # Byte-at-a-time incremental decode so a multi-byte UTF-8 char whose
            # bytes straddle a read boundary isn't dropped (a fixed os.read(fd, 4)
            # with errors="ignore" silently ate characters during fast pastes).
            decoder = codecs.getincrementaldecoder(sys.stdin.encoding or "utf-8")("replace")
            while True:
                b = os.read(fd, 1)
                if not b:
                    return ""  # stream EOF; caller raises EOFError
                ch = decoder.decode(b)
                if ch:
                    return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


_getch: Callable[[], str] = _getch_windows if os.name == "nt" else _getch_posix


def _read_password(prompt: str, *, out: "TextIO | None" = None) -> str:
    """Read one masked line: echo ``*`` per char, support backspace editing.

    Raises KeyboardInterrupt on Ctrl-C and EOFError on Ctrl-D/Ctrl-Z with an
    empty buffer, after restoring the terminal (the per-keystroke getch never
    leaves the tty in raw mode between reads).
    """
    if out is None:
        out = sys.stderr
    out.write(prompt)
    out.flush()
    chars: list[str] = []
    while True:
        key = _getch()
        if key == "":  # stream ended mid-line: abort, don't submit a partial
            out.write("\n")
            out.flush()
            raise EOFError
        for ch in key:  # a paste can deliver several chars per read
            if ch in _SUBMITS:
                out.write("\n")
                out.flush()
                return "".join(chars)
            if ch == _CTRL_C:
                out.write("\n")
                out.flush()
                raise KeyboardInterrupt
            if ch in (_CTRL_D, _CTRL_Z):
                if not chars:
                    out.write("\n")
                    out.flush()
                    raise EOFError
                continue  # ignore mid-input
            if ch in _BACKSPACES:
                if chars:
                    chars.pop()
                    out.write("\b \b")
                    out.flush()
                continue
            if ch < " ":  # other control characters (tab, escape, ...)
                continue
            chars.append(ch)
            out.write("*")
            out.flush()


def should_prompt_password_change(
    *, tunnel_will_start: bool, requires_change: bool, stdin_isatty: bool, stderr_isatty: bool
) -> bool:
    """Whether to block startup on an interactive terminal password change.

    Only when the public Cloudflare tunnel is actually about to start (a
    loopback ``--cloudflare`` no-op or a raw wildcard bind never prompts), the
    admin account still has the seeded bootstrap password, and both stdin and
    stderr are real terminals (headless launches keep the bootstrap-timeout
    protection instead of hanging on a prompt nobody can answer).
    """
    return tunnel_will_start and requires_change and stdin_isatty and stderr_isatty


def prompt_for_password_change(
    *,
    min_length: int,
    is_current_password: Callable[[str], bool],
    apply_change: Callable[[str], None],
    username: str = "unsloth",
    out: "TextIO | None" = None,
) -> bool:
    """Force a new admin password before public exposure; True on success.

    Loops until a valid, confirmed password is committed via ``apply_change``.
    Ctrl-C / EOF aborts and returns False; the caller must then abort the
    launch (interactive refusal is not the headless fallback case).
    """
    if out is None:
        out = sys.stderr
    out.write(
        "\n"
        "Studio is about to be exposed on a public Cloudflare URL, but the\n"
        f"admin account ('{username}') still has its auto-generated bootstrap\n"
        "password. Set a new password now (input shows * per keystroke).\n"
        f"Minimum length: {min_length} characters. Press Ctrl-C to abort.\n\n"
    )
    out.flush()
    try:
        while True:
            new_password = _read_password("New password: ", out = out)
            if len(new_password) < min_length:
                out.write(f"Password must be at least {min_length} characters; try again.\n")
                out.flush()
                continue
            if is_current_password(new_password):
                out.write(
                    "New password must differ from the current bootstrap password; try again.\n"
                )
                out.flush()
                continue
            confirmation = _read_password("Confirm new password: ", out = out)
            if confirmation != new_password:
                out.write("Passwords do not match; try again.\n")
                out.flush()
                continue
            apply_change(new_password)
            out.write(f"Password updated for '{username}'.\n")
            out.flush()
            return True
    except (KeyboardInterrupt, EOFError):
        out.write("Password change aborted; not exposing Studio.\n")
        out.flush()
        return False
