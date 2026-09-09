# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Interactive terminal prompt that forces a bootstrap password change before
Unsloth becomes reachable: a public Cloudflare URL (``--secure`` / ``--cloudflare``)
or a raw non-loopback bind such as ``-H 0.0.0.0``.

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

# Mirror in unsloth_cli/commands/_password_prompt.py; keep the name in sync.
SUPPLIED_PASSWORD_ENV = "UNSLOTH_STUDIO_PASSWORD"


def _getch_windows() -> str:  # pragma: no cover - exercised via fake on Linux CI
    import msvcrt

    ch = msvcrt.getwch()
    # Function/arrow keys arrive as a two-wchar \x00/\xe0 sequence; consume the second half.
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
        except ImportError:
            return self
        try:
            fd = sys.stdin.fileno()
            old_attrs = termios.tcgetattr(fd)
        except (AttributeError, ValueError, OSError, termios.error):
            return self
        self._fd = fd
        self._old_attrs = old_attrs
        self._signals = _RestoreTtyOnSignals(fd, old_attrs)
        self._signals.__enter__()
        # cbreak leaves ISIG on, so clear it and surface Ctrl-C as \x03 for the caller loop to restore the tty.
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
    # Byte-at-a-time decode so a multi-byte UTF-8 char straddling a read boundary is not dropped.
    import codecs

    fd = sys.stdin.fileno()
    decoder = codecs.getincrementaldecoder(sys.stdin.encoding or "utf-8")("replace")
    while True:
        b = os.read(fd, 1)
        if not b:
            return ""
        ch = decoder.decode(b)
        if ch:
            return ch


_getch: Callable[[], str] = _getch_windows if os.name == "nt" else _getch_posix


class PromptUnattended(Exception):
    """A terminal is attached but nobody answered before the deadline.

    A pty is not a person: ``tmux new -d`` / ``docker run -dt`` allocate a real
    foreground pty nobody reads, so isatty() is True on both streams and the read
    never returns. Callers that must not block a launch treat this as a refusal.
    """


def _wait_for_first_key(timeout: float) -> bool:
    """Whether a keystroke arrived within ``timeout`` seconds.

    Requires cbreak mode already set: canonical mode holds input until a newline,
    so the fd would not become readable on the first character. True on any doubt
    (the blocking behaviour).
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


def _read_password(
    prompt: str,
    *,
    out: "TextIO | None" = None,
    first_key_timeout: "float | None" = None,
) -> str:
    """Read one masked line: echo ``*`` per char, support backspace editing.

    Raises KeyboardInterrupt on Ctrl-C and EOFError on Ctrl-D/Ctrl-Z with an
    empty buffer; the terminal is restored on every exit path. With
    ``first_key_timeout``, raises PromptUnattended when the FIRST keystroke never
    arrives; once someone starts typing there is no deadline.
    """
    if out is None:
        out = sys.stderr
    out.write(prompt)
    out.flush()
    chars: list[str] = []
    with _prompt_raw_mode():
        if first_key_timeout is not None and not _wait_for_first_key(first_key_timeout):
            out.write("\n")
            out.flush()
            raise PromptUnattended
        while True:
            key = _getch()
            if key == "":
                out.write("\n")
                out.flush()
                raise EOFError
            for ch in key:
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
                    continue
                if ch in _BACKSPACES:
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


def should_prompt_password_change(
    *,
    tunnel_will_start: bool,
    requires_change: bool,
    stdin_isatty: bool,
    stderr_isatty: bool,
    bind_is_exposed: bool = False,
) -> bool:
    """Whether to block startup on an interactive terminal password change.

    True when the launch puts the web UI where others can reach it, the admin
    still has the seeded password, and both stdin and stderr are real terminals
    (headless launches keep the bootstrap-timeout protection instead of hanging).

    Two ways to be reachable: ``tunnel_will_start`` (public Cloudflare URL) and
    ``bind_is_exposed`` (a raw non-loopback bind like ``-H 0.0.0.0``, reachable by
    the whole network yet starting no tunnel). The second used to get no prompt at
    all, so the seeded password stayed live and was served to anyone who loaded
    the page.
    """
    if not (tunnel_will_start or bind_is_exposed):
        return False
    return requires_change and stdin_isatty and stderr_isatty


def prompt_for_password_change(
    *,
    min_length: int,
    is_current_password: Callable[[str], bool],
    apply_change: Callable[[str], None],
    username: str = "unsloth",
    out: "TextIO | None" = None,
    exposure: str = "on the public internet",
    first_key_timeout: "float | None" = None,
    refusal_aborts: bool = True,
) -> bool:
    """Force a new admin password before exposure; True on success.

    Loops until a valid, confirmed password is committed via ``apply_change``.
    Ctrl-C / EOF returns False; ``refusal_aborts`` tells the banner what the
    caller does with that, so it never promises an abort that will not happen:
    True for a tunnel (caller aborts), False for a raw bind (launch proceeds,
    because it worked before this prompt existed).

    ``exposure`` names where this launch is reachable: a tunnel really is the
    public internet, a raw bind is every interface (LAN behind NAT, or the
    internet on a cloud box). Claiming the wrong one trains people to ignore it.

    ``first_key_timeout`` bounds the wait for the FIRST keystroke, returning
    False if it never comes. Only a caller that must not block a launch passes
    it: a detached pty (``tmux new -d``, ``docker run -dt``) looks exactly like
    an attended terminal, so undeadlined it waits forever and never binds its
    socket. Unset (the tunnel) blocks indefinitely.
    """
    if out is None:
        out = sys.stderr
    refusal = (
        "Ctrl+C to abort."
        if refusal_aborts
        else "Ctrl+C to skip, and Unsloth starts with the auto-generated password."
    )
    out.write(
        "\n"
        f"Unsloth Studio will be reachable {exposure}, so set a\n"
        f"password now. {refusal}\n\n"
    )
    out.flush()
    # Only the first read is deadlined; a key arriving proves someone is there.
    pending_timeout = first_key_timeout
    try:
        while True:
            new_password = _read_password(
                "New password: ", out = out, first_key_timeout = pending_timeout
            )
            pending_timeout = None
            if len(new_password) < min_length:
                out.write(f"Password must be at least {min_length} characters; try again.\n")
                out.flush()
                continue
            if any(ch.isspace() for ch in new_password):
                out.write("Password cannot contain spaces; try again.\n")
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
    except PromptUnattended:
        out.write(
            "No response at the terminal; leaving the auto-generated admin password in place.\n"
        )
        out.flush()
        return False
    except (KeyboardInterrupt, EOFError):
        out.write(
            "Password change aborted; not exposing Unsloth.\n"
            if refusal_aborts
            else "Password change skipped; leaving the auto-generated admin password in place.\n"
        )
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
