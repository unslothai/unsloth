# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Interactive terminal prompt that forces a bootstrap password change before
Unsloth is exposed on a public Cloudflare URL (``--secure`` / ``--cloudflare``).

Masked input echoes one ``*`` per keystroke (unlike ``getpass``). Works on
Windows (``msvcrt``) and Linux/macOS (``termios``). All output goes to stderr so
redirected stdout never swallows the prompt.

Mirrored for the CLI at ``unsloth_cli/commands/_password_prompt.py`` (the CLI
cannot import the Unsloth backend package); keep the two in sync.
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

# Env var that supplies the initial admin password non-interactively (mirror in
# unsloth_cli/commands/_password_prompt.py). Keep the name in sync.
SUPPLIED_PASSWORD_ENV = "UNSLOTH_STUDIO_PASSWORD"


def _getch_windows() -> str:  # pragma: no cover - exercised via fake on Linux CI
    import msvcrt

    ch = msvcrt.getwch()
    # Function/arrow keys arrive as a two-wchar \x00/\xe0 sequence; consume the
    # second half and report a no-op control char.
    if ch in ("\x00", "\xe0"):
        msvcrt.getwch()
        return "\x00"
    return ch


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


class _prompt_raw_mode:
    """Hold cbreak + cleared ISIG (no echo) on stdin for the WHOLE prompt line,
    restoring when the line finishes (and on SIGTERM/SIGHUP).

    Echo must never re-enable mid-line: cbreak echoes on receipt, so a keystroke
    arriving while echo is on would appear in cleartext. One cbreak block for the
    whole line closes that window. No-op when stdin is not a real terminal, so
    the _getch seam can be faked in tests.
    """

    def __enter__(self) -> "_prompt_raw_mode":
        self._fd = None
        self._old_attrs = None
        self._signals = None
        try:
            import termios
            import tty
        except ImportError:  # non-POSIX (Windows uses msvcrt, no mode to hold)
            return self
        try:
            fd = sys.stdin.fileno()
            old_attrs = termios.tcgetattr(fd)
        except (AttributeError, ValueError, OSError, termios.error):
            return self  # redirected / captured stdin (tests): nothing to hold
        self._fd = fd
        self._old_attrs = old_attrs
        self._signals = _RestoreTtyOnSignals(fd, old_attrs)
        self._signals.__enter__()
        # cbreak (not raw) keeps output post-processing while disabling echo/line
        # buffering. It leaves ISIG on, so clear it and surface Ctrl-C as \x03 to
        # the caller loop, which restores the tty itself.
        tty.setcbreak(fd, termios.TCSADRAIN)
        new_attrs = termios.tcgetattr(fd)
        new_attrs[3] &= ~termios.ISIG
        termios.tcsetattr(fd, termios.TCSADRAIN, new_attrs)
        return self

    def __exit__(self, *exc) -> None:
        if self._old_attrs is None:
            return
        import termios
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
        finally:
            if self._signals is not None:
                self._signals.__exit__(*exc)


def _getch_posix() -> str:  # pragma: no cover - needs a real tty
    # Terminal already in cbreak+no-echo for the whole line (_prompt_raw_mode),
    # so just read. Byte-at-a-time incremental decode so a multi-byte UTF-8 char
    # straddling a read boundary isn't dropped.
    import codecs

    fd = sys.stdin.fileno()
    decoder = codecs.getincrementaldecoder(sys.stdin.encoding or "utf-8")("replace")
    while True:
        b = os.read(fd, 1)
        if not b:
            return ""  # stream EOF; caller raises EOFError
        ch = decoder.decode(b)
        if ch:
            return ch


_getch: Callable[[], str] = _getch_windows if os.name == "nt" else _getch_posix


def _read_password(prompt: str, *, out: "TextIO | None" = None) -> str:
    """Read one masked line: echo ``*`` per char, support backspace editing.

    Raises KeyboardInterrupt on Ctrl-C and EOFError on Ctrl-D/Ctrl-Z with an
    empty buffer; the terminal is restored on every exit path.
    """
    if out is None:
        out = sys.stderr
    out.write(prompt)
    out.flush()
    chars: list[str] = []
    with _prompt_raw_mode():
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

    True only when the tunnel is actually about to start, the admin still has
    the seeded password, and both stdin and stderr are real terminals (headless
    launches keep the bootstrap-timeout protection instead of hanging).
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
    Ctrl-C / EOF returns False; the caller must then abort the launch.
    """
    if out is None:
        out = sys.stderr
    out.write(
        "\n"
        "Unsloth Studio will be exposed on the public internet, so set a\n"
        "password now. Ctrl+C to abort.\n\n"
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
        out.write("Password change aborted; not exposing Unsloth.\n")
        out.flush()
        return False


def resolve_supplied_password(cli_value: "str | None", out: "TextIO | None" = None) -> "str | None":
    """Resolve a non-interactive initial admin password, or None if unset.

    Precedence: an explicit ``--password`` (literal ``-`` reads a line from
    stdin), then the ``UNSLOTH_STUDIO_PASSWORD`` env var; empty/omitted means off.
    A literal argv value is visible in the process list, so a note points at the
    env var or stdin instead. Mirror of the CLI helper -- keep the two in sync.
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
