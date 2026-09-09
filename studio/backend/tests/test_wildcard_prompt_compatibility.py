# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Compatibility surface of the raw-bind password prompt.

The prompt itself is covered by test_password_prompt.py and the gate wiring by
test_password_prompt_backstop.py. This file covers what only breaks somewhere
else: other platforms, old installs, and the no-hardware-coupling claim.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from auth import terminal_prompt  # noqa: E402


# ── platform seams ───────────────────────────────────────────────────


def test_the_windows_getch_path_decodes_a_password(monkeypatch):
    """`_getch` is selected by os.name at import; exercise the Windows half.

    CI is Linux, so the msvcrt branch is otherwise never executed. A stub module
    puts the two-wchar arrow-key sequence and ordinary characters through the
    real handler.
    """
    import types

    keys = list("pw1\r")
    fake = types.ModuleType("msvcrt")
    # An arrow key arrives as \xe0 then a second wchar that must be swallowed.
    queue = ["\xe0", "H"] + keys

    def _getwch():
        return queue.pop(0)

    fake.getwch = _getwch
    monkeypatch.setitem(sys.modules, "msvcrt", fake)

    out = []
    monkeypatch.setattr(terminal_prompt, "_getch", terminal_prompt._getch_windows)
    monkeypatch.setattr(terminal_prompt, "_prompt_raw_mode", lambda: _NullCtx())

    class _Out:
        encoding = "utf-8"

        def write(self, s):
            out.append(s)

        def flush(self):
            pass

    assert terminal_prompt._read_password("pw: ", out = _Out()) == "pw1"
    # The arrow key is swallowed, not masked: one star per character.
    assert "".join(out).count("*") == 3


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_a_stream_with_no_fileno_is_not_interactive():
    """pythonw / a Windows service has no console; isatty must not raise.

    The gate's isatty helpers treat a broken stream as non-interactive, so such a
    launch takes the headless path rather than crashing.
    """

    class _Broken:
        def isatty(self):
            raise ValueError("I/O operation on closed file")

        def fileno(self):
            raise OSError("no console")

    import run as run_mod

    assert run_mod._stream_isatty(_Broken()) is False
    assert run_mod._stream_isatty(None) is False


@pytest.mark.parametrize("stdin_tty,stderr_tty", [(True, False), (False, True), (False, False)])
def test_a_split_terminal_never_prompts(stdin_tty, stderr_tty):
    """Masked echo needs stderr; reading needs stdin. Half a terminal is none."""
    assert (
        terminal_prompt.should_prompt_password_change(
            tunnel_will_start = False,
            bind_is_exposed = True,
            requires_change = True,
            stdin_isatty = stdin_tty,
            stderr_isatty = stderr_tty,
        )
        is False
    )


# ── old installs ─────────────────────────────────────────────────────


def test_an_old_caller_that_omits_bind_is_exposed_behaves_as_before():
    """The new kwarg is optional and defaults off.

    A studio venv can hold an older backend than the CLI that launched it, so a
    caller predating this change must keep tunnel-only semantics.
    """
    assert (
        terminal_prompt.should_prompt_password_change(
            tunnel_will_start = False,
            requires_change = True,
            stdin_isatty = True,
            stderr_isatty = True,
        )
        is False
    )
    assert (
        terminal_prompt.should_prompt_password_change(
            tunnel_will_start = True,
            requires_change = True,
            stdin_isatty = True,
            stderr_isatty = True,
        )
        is True
    )


def test_the_prompt_signature_stays_keyword_compatible():
    """`exposure` must be optional; an older caller passes neither it nor more."""
    import inspect

    params = inspect.signature(terminal_prompt.prompt_for_password_change).parameters
    assert params["exposure"].default == "on the public internet"
    for name, param in params.items():
        assert param.kind is not inspect.Parameter.POSITIONAL_ONLY, name


# ── no hardware coupling ─────────────────────────────────────────────


def test_the_password_gate_imports_no_gpu_or_torch_module():
    """The hardware-independence claim, checked not asserted.

    In a subprocess: importing torch in THIS interpreter would poison the rest of
    the session, and a sys.modules probe after the fact proves nothing about what
    the import graph pulls in.
    """
    probe = (
        "import sys;"
        f"sys.path.insert(0, {str(_BACKEND)!r});"
        "import auth.terminal_prompt, auth.bootstrap_timeout, utils.host_policy;"
        "bad = sorted(m for m in sys.modules"
        " if m.split('.')[0] in ('torch','triton','bitsandbytes','pynvml','amdsmi'));"
        "print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output = True,
        text = True,
        timeout = 300,
        cwd = str(_BACKEND),
        env = {**os.environ, "PYTHONPATH": str(_BACKEND)},
    )
    assert out.returncode == 0, out.stderr
    assert (
        out.stdout.strip() == ""
    ), f"the password gate pulled in hardware modules: {out.stdout.strip()}"
