# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Set the Studio admin password before the server is exposed to the network.

On a fresh install the default admin (``unsloth``) is seeded with a random
"bootstrap" password and ``must_change_password=1``; the web UI embeds that
plaintext password into ``index.html`` for first-run convenience. That is fine
on a loopback bind, but a ``--secure`` Cloudflare tunnel or a ``0.0.0.0`` bind
would publish the page (and the password) to remote visitors during the
bootstrap window. To avoid ever generating a public-facing bootstrap secret, we
prompt the operator for a real password in the terminal BEFORE the listening
socket binds, and persist it via ``storage.update_password`` (which clears the
bootstrap state). Non-interactive launches can supply it via the
``UNSLOTH_STUDIO_ADMIN_PASSWORD`` environment variable.

The masked reader echoes ``*`` per character (rather than the blank-input
``getpass`` style) and works on Linux, macOS, and Windows (PowerShell/cmd).
"""

import os
import sys

# Mirrors the frontend rule in auth-form.tsx (currentPassword.length < 8).
MIN_PASSWORD_LENGTH = 8

ADMIN_PASSWORD_ENV_VAR = "UNSLOTH_STUDIO_ADMIN_PASSWORD"


def _read_masked_posix(prompt: str) -> str:
    """Masked line reader for Linux/macOS using termios cbreak mode.

    ``cbreak`` disables canonical mode and echo but keeps ``ISIG`` on, so Ctrl-C
    still raises ``KeyboardInterrupt`` instead of being captured as a byte.
    """
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    sys.stdout.write(prompt)
    sys.stdout.flush()
    chars: list[str] = []
    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch in ("\n", "\r", ""):
                break
            if ch == "\x03":  # Ctrl-C
                raise KeyboardInterrupt
            if ch == "\x04":  # Ctrl-D (EOF)
                break
            if ch in ("\x7f", "\b"):  # DEL / Backspace
                if chars:
                    chars.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue
            if ch < " ":  # ignore other control chars (arrows, etc.)
                continue
            chars.append(ch)
            sys.stdout.write("*")
            sys.stdout.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        sys.stdout.write("\n")
        sys.stdout.flush()
    return "".join(chars)


def _read_masked_windows(prompt: str) -> str:
    """Masked line reader for Windows (works under PowerShell and cmd)."""
    import msvcrt

    sys.stdout.write(prompt)
    sys.stdout.flush()
    chars: list[str] = []
    while True:
        ch = msvcrt.getwch()
        if ch in ("\r", "\n"):
            break
        if ch == "\x03":  # Ctrl-C
            raise KeyboardInterrupt
        if ch in ("\x00", "\xe0"):  # function/arrow-key prefix: drop the next code
            msvcrt.getwch()
            continue
        if ch == "\b":  # Backspace
            if chars:
                chars.pop()
                sys.stdout.write("\b \b")
                sys.stdout.flush()
            continue
        chars.append(ch)
        sys.stdout.write("*")
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    return "".join(chars)


def read_masked_password(prompt: str) -> str:
    """Read a password from the terminal, echoing ``*`` per typed character.

    Falls back to :func:`getpass.getpass` (no echo at all) when there is no
    usable TTY or the platform primitive is unavailable (e.g. some MSYS ptys).
    """
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        import getpass
        return getpass.getpass(prompt)
    try:
        if sys.platform == "win32":
            return _read_masked_windows(prompt)
        return _read_masked_posix(prompt)
    except (ImportError, OSError):
        import getpass
        return getpass.getpass(prompt)


def prompt_new_admin_password(
    *,
    min_length: int = MIN_PASSWORD_LENGTH,
    attempts: int = 3,
    reader = read_masked_password,
) -> str:
    """Prompt for a new admin password twice and return it once both match.

    ``reader`` is injectable so the prompt/confirm/retry logic can be unit
    tested without a real terminal.
    """
    for _ in range(attempts):
        pw = reader(f"Set a Studio admin password (min {min_length} chars): ")
        if len(pw) < min_length:
            print(f"Password must be at least {min_length} characters.", file = sys.stderr)
            continue
        confirm = reader("Confirm password: ")
        if pw != confirm:
            print("Passwords do not match. Try again.", file = sys.stderr)
            continue
        return pw
    raise SystemExit(
        "Could not set a Studio admin password after several attempts. Aborting "
        "before exposing the server."
    )


def resolve_admin_password_source(
    *,
    frontend_served: bool,
    exposed: bool,
    is_colab: bool,
    requires_change: bool,
    has_tty: bool,
    env_password,
) -> str:
    """Decide how to obtain the admin password before exposing the web UI.

    Returns one of:
      - ``"skip"``     : nothing to do (not exposed / api-only / Colab / already set).
      - ``"env"``      : take it from ``UNSLOTH_STUDIO_ADMIN_PASSWORD``.
      - ``"prompt"``   : ask interactively on the TTY.
      - ``"backstop"`` : no TTY and no env var; leave the bootstrap state and rely
                         on the local-direct injection gate in main.py.

    Pure decision (no I/O) so it is cheap to unit test exhaustively.
    """
    if not (frontend_served and exposed and not is_colab and requires_change):
        return "skip"
    if env_password:
        return "env"
    if has_tty:
        return "prompt"
    return "backstop"


def _is_exposed_bind(host: str, secure: bool) -> bool:
    """True when this launch puts the web UI on the network (tunnel or non-loopback)."""
    if secure:
        return True
    if host in ("0.0.0.0", "::"):
        return True
    try:
        from utils.host_policy import is_external_host
    except Exception:
        return False
    return bool(is_external_host(host))


def ensure_admin_password_before_exposure(
    *,
    storage,
    host: str,
    secure: bool,
    api_only: bool,
    frontend_served: bool,
    is_colab: bool,
    logger = None,
) -> None:
    """Provision a non-bootstrap admin password before the server is exposed.

    Called from ``run_server`` after the frontend is mounted but before the
    uvicorn socket binds. No-op for loopback binds, ``--api-only`` (no HTML to
    leak), and Colab (owner-auth-gated proxy, never a public tunnel).
    """
    exposed = (not api_only) and _is_exposed_bind(host, secure)
    if not (frontend_served and exposed and not is_colab):
        return

    # Make sure the admin row + schema exist so we can read/flip its state.
    storage.ensure_default_admin()
    if not storage.requires_password_change(storage.DEFAULT_ADMIN_USERNAME):
        return  # operator already chose a real password on a previous run.

    env_password = os.environ.get(ADMIN_PASSWORD_ENV_VAR)
    source = resolve_admin_password_source(
        frontend_served = frontend_served,
        exposed = exposed,
        is_colab = is_colab,
        requires_change = True,
        has_tty = sys.stdin.isatty() and sys.stdout.isatty(),
        env_password = env_password,
    )

    if source == "env":
        if len(env_password) < MIN_PASSWORD_LENGTH:
            raise SystemExit(
                f"{ADMIN_PASSWORD_ENV_VAR} must be at least {MIN_PASSWORD_LENGTH} "
                "characters; refusing to expose the server."
            )
        storage.update_password(storage.DEFAULT_ADMIN_USERNAME, env_password)
        print(
            f"Studio admin password set from {ADMIN_PASSWORD_ENV_VAR} before "
            "exposing the server.",
            flush = True,
        )
        return

    if source == "prompt":
        print(
            "\nThis launch exposes Unsloth Studio on the network. Set the admin "
            "password now (it will not be shown publicly).",
            flush = True,
        )
        password = prompt_new_admin_password()
        storage.update_password(storage.DEFAULT_ADMIN_USERNAME, password)
        print("Admin password set. Starting the server...", flush = True)
        return

    # source == "backstop": cannot prompt and no env var. Do not block startup;
    # the local-direct injection gate keeps the bootstrap password off the wire.
    message = (
        f"Exposing Studio without a chosen admin password and no TTY to prompt. "
        f"Set {ADMIN_PASSWORD_ENV_VAR} to choose one. The first-run password will "
        f"NOT be served to remote visitors; read it from the server console / the "
        f".bootstrap_password file to log in."
    )
    if logger is not None:
        logger.warning(message)
    else:
        print(message, file = sys.stderr, flush = True)
