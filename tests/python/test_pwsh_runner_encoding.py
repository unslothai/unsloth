# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""run_pwsh has to round-trip text, and until now that depended on the host's code pages.

Nobody set the WRITING end. Windows PowerShell 5.1 writes a redirected pipe in the OEM code
page and pwsh 7 writes UTF-8, while `text = True` decodes with the ANSI one, so a non-ASCII
character survived only where those happened to agree. On a cp437/cp1252 box 5.1 gives
U+FFFD, and on a runner whose console is UTF-8 both shells give mojibake. That is what
test_a_non_ascii_marker_survives_the_rollback fails on in parity CI.

Naming `encoding = "utf-8"` at the call site, which that test already does, is the half-fix:
it corrects pwsh 7 and makes 5.1 worse, because 0x84 is not valid UTF-8 and the decode raises
inside subprocess's reader thread, where the exception is swallowed and stdout is left None.

The behavioural tests below are the point: they run a real shell and compare the string that
comes back with the string that went in. The rest pin the shape of the decision, most of
whose branches are about NOT interfering with a caller.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "_shared"))

from unsloth_pwsh_runner import _UTF8_PROLOGUE, _agree_on_utf8, run_pwsh  # noqa: E402

# Both, where both exist: the two disagree about what a redirected pipe is encoded in, which
# is the whole reason this module has anything to fix.
POWERSHELLS = [p for p in (shutil.which("pwsh"), shutil.which("powershell")) if p]

# One character per interesting width, because the failure mode differs by width: U+00E4 is a
# single byte in cp437 and two in UTF-8, U+4E2D has no code-page representation at all, and the
# emoji is a surrogate pair in PowerShell's UTF-16 strings.
SAMPLES = {
    "latin1": "käffee",
    "cjk": "中文",
    "astral": "\U0001f600",
    "mixed": "café 中 \U0001f600",
}


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize("name", sorted(SAMPLES), ids = sorted(SAMPLES))
def test_a_non_ascii_string_survives_the_round_trip(shell: str, name: str):
    """The whole point. Built from code points inside PowerShell so the script itself stays
    ASCII: a script that carried the character would be testing how the -Command argument is
    encoded, which is a different pipe."""
    text = SAMPLES[name]
    literal = " + ".join(f"[char]::ConvertFromUtf32({ord(c)})" for c in text)
    done = run_pwsh(
        [
            shell,
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            f"Write-Output ('<' + {literal} + '>')",
        ],
        check = True,
        capture_output = True,
        text = True,
        timeout = 60,
    )
    out = done.stdout
    assert out[out.index("<") + 1 : out.rindex(">")] == text


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_the_prologue_adds_no_bom(shell: str):
    """[Text.Encoding]::UTF8 carries a preamble; the one we set must not, or every caller's
    first assertion sees a U+FEFF it did not print."""
    done = run_pwsh(
        [shell, "-NoProfile", "-NonInteractive", "-Command", "Write-Output 'plain'"],
        check = True,
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert done.stdout.startswith("plain"), repr(done.stdout)


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_without_the_fix_the_same_string_does_not_survive(shell: str):
    """The control. Same script, run the old way, on this host. It is allowed to pass -- a
    box whose console is already UTF-8 was never broken -- so this asserts the two agree
    when it does, and that the fixed one is right when they do not."""
    script = "Write-Output ('<' + [char]::ConvertFromUtf32(0x00e4) + '>')"
    argv = [shell, "-NoProfile", "-NonInteractive", "-Command", script]
    old = subprocess.run(argv, capture_output = True, text = True, timeout = 60).stdout
    new = run_pwsh(argv, check = True, capture_output = True, text = True, timeout = 60).stdout
    assert new[new.index("<") + 1 : new.rindex(">")] == "ä"
    if "ä" in old:
        assert old.strip() == new.strip()


def test_a_command_script_is_prefixed_once():
    kwargs = {"text": True}
    argv = _agree_on_utf8(["pwsh", "-Command", "Write-Output 1"], kwargs)
    assert argv[-1] == _UTF8_PROLOGUE + "Write-Output 1"
    assert kwargs["encoding"] == "utf-8"


def test_the_call_sites_own_list_is_not_written_to():
    """A caller that reuses its argv, or reports it on failure, must see what it built."""
    original = ["pwsh", "-Command", "Write-Output 1"]
    argv = _agree_on_utf8(original, {"text": True})
    assert original == ["pwsh", "-Command", "Write-Output 1"]
    assert argv is not original


@pytest.mark.parametrize(
    ("kwargs", "why"),
    [
        ({}, "bytes mode: the caller decodes, and a prologue would change the bytes"),
        ({"text": False}, "the same, spelled out"),
        ({"text": True, "encoding": "cp1252"}, "an encoding the caller chose is the answer"),
        ({"text": True, "encoding": "latin-1"}, "the same for one that cannot fail"),
    ],
    ids = ["bytes", "text-false", "explicit-cp1252", "explicit-latin1"],
)
def test_a_caller_that_already_decided_is_left_alone(kwargs, why):
    before = dict(kwargs)
    argv = _agree_on_utf8(["pwsh", "-Command", "Write-Output 1"], kwargs)
    assert argv[-1] == "Write-Output 1", why
    assert kwargs == before, why


@pytest.mark.parametrize("spelling", ["utf-8", "utf8", "UTF-8", "utf_8", "U8", "cp65001"])
def test_a_caller_that_asked_for_utf8_still_gets_the_writing_end(spelling):
    """The half-fix this replaces. Naming the decode alone is what turns 5.1's OEM byte into
    a decode error inside subprocess's reader thread and a silent `stdout is None`, so the
    call sites that already spell it must pick up the prologue rather than be skipped."""
    kwargs = {"text": True, "encoding": spelling}
    argv = _agree_on_utf8(["pwsh", "-Command", "Write-Output 1"], kwargs)
    assert argv[-1] == _UTF8_PROLOGUE + "Write-Output 1"
    assert kwargs["encoding"] == "utf-8"


def test_errors_alone_does_not_count_as_choosing_an_encoding():
    kwargs = {"text": True, "errors": "replace"}
    argv = _agree_on_utf8(["pwsh", "-Command", "Write-Output 1"], kwargs)
    assert argv[-1] == _UTF8_PROLOGUE + "Write-Output 1"
    assert kwargs == {"text": True, "errors": "replace", "encoding": "utf-8"}


@pytest.mark.parametrize(
    "argv",
    [
        ["pwsh", "-NoProfile", "-File", "script.ps1"],
        ["pwsh", "-NoProfile"],
        ["pwsh", "-NoProfile", "-Command"],
    ],
    ids = ["-File", "no-command", "trailing-command"],
)
def test_an_invocation_with_no_script_string_still_gets_the_decode_half(argv):
    """-File names a path, so there is nothing to prepend to. Decoding UTF-8 is still right
    for pwsh 7, and is the only half available here."""
    kwargs = {"text": True}
    assert _agree_on_utf8(list(argv), kwargs) == argv
    assert kwargs["encoding"] == "utf-8"
