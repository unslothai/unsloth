# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Interactive terminal prompt that forces a bootstrap password change before
Unsloth becomes reachable: a public Cloudflare URL (``--secure`` / ``--cloudflare``)
or a raw non-loopback bind such as ``-H 0.0.0.0``.

Masked input echoes one ``*`` per keystroke (unlike ``getpass``). Works on Windows (``msvcrt``) and Linux/macOS
(``termios``). All output goes to stderr so redirected stdout never swallows the prompt. Mirrored for the CLI at
``unsloth_cli/commands/_password_prompt.py`` (the CLI cannot import the Unsloth backend package); keep the two
in sync.
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

# Keep in sync with unsloth_cli/commands/_password_prompt.py.
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
    """Restore terminal attrs if SIGTERM/SIGHUP kills the prompt mid-read. A finally block can't run when a
    signal terminates the process, leaving the shared terminal in cbreak/no-echo. Best-effort: no-op off the
    main thread or where the signals are absent.
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
        # cbreak leaves ISIG on; clear it and surface Ctrl-C as \x03 so the caller restores the tty.
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
    # Byte-at-a-time decode so a UTF-8 char split across reads is not dropped.
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
    """Needs cbreak mode already set; canonical mode holds input until a newline. True on any doubt."""
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
    """With ``first_key_timeout``, a silent terminal raises PromptUnattended; typing has no deadline."""
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
    """A raw non-loopback bind (-H 0.0.0.0) is reachable without a tunnel, so it prompts too."""
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
) -> "bool | None":
    """Only ``first_key_timeout`` callers pass it: a detached pty looks attended and would wait forever."""
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
    # Only the first read is deadlined.
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
        return None
    except (KeyboardInterrupt, EOFError):
        out.write(
            "Password change aborted; not exposing Unsloth.\n"
            if refusal_aborts
            else "Password change skipped; leaving the auto-generated admin password in place.\n"
        )
        out.flush()
        return False


def resolve_supplied_password(cli_value: "str | None", out: "TextIO | None" = None) -> "str | None":
    """Mirrors the CLI helper, so keep both in sync; a literal --password is visible in the process list."""
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
