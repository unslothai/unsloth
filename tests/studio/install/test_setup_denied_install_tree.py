# SPDX-License-Identifier: AGPL-3.0-only
"""Contract checks for setup.ps1 on an unreadable llama.cpp install tree.

Windows Test-Path raises UnauthorizedAccessException instead of returning
$false when an ACL denies the probe. setup.ps1 runs under "Stop", so the bare
probe of the prebuilt metadata aborted setup with a raw "Test-Path : Access is
denied" and exit code 1, which the desktop app showed as "unsloth studio setup
failed (exit code 1)".

~/.unsloth/llama.cpp lives beside the app rather than inside it, so a reinstall,
even to another drive, reused the unreadable folder and failed on the same line.
The probes now go through Get-PathState / Test-PathQuiet and a denial produces
an actionable [TAURI:ERROR] message.

Behavioural coverage against a real ACL-denied directory lives in
tests/studio/test_path_probe_access_denied.ps1.
"""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SETUP_PS1 = (ROOT / "studio" / "setup.ps1").read_text(encoding = "utf-8")


def test_setup_defines_non_throwing_path_probes():
    for name in ("Test-AccessDeniedError", "Get-PathState", "Test-PathQuiet"):
        assert re.search(rf"^function {re.escape(name)} \{{", SETUP_PS1, re.M), name
    # Get-PathState must keep the three-way answer: collapsing "Denied" into
    # "Absent" would hide the failure again instead of reporting it.
    for state in ('return "Present"', 'return "Absent"', 'return "Denied"'):
        assert state in SETUP_PS1


def test_prebuilt_metadata_probe_cannot_terminate_setup():
    assert "if (Test-Path $existingMetaPath)" not in SETUP_PS1
    assert "$existingMetaState = Get-PathState -Path $existingMetaPath -PathType Leaf" in SETUP_PS1
    assert '$existingMetaState -eq "Denied"' in SETUP_PS1
    assert '$existingMetaState -eq "Present"' in SETUP_PS1


def test_every_denial_route_reports_instead_of_proceeding():
    """An unreadable parent dir, metadata file, .git checkout, or ownership root
    must all stop. Treating any of them as absent lets the caller replace or
    delete a tree it cannot read."""
    assert "$llamaDirState = Get-PathState -Path $LlamaCppDir" in SETUP_PS1
    assert '$llamaDirState -eq "Denied"' in SETUP_PS1
    assert '$llamaGitState = Get-PathState -Path (Join-Path $LlamaCppDir ".git")' in SETUP_PS1
    assert '$llamaGitState -eq "Denied"' in SETUP_PS1
    assert "$pathState = Get-PathState -Path $Path -PathType Container" in SETUP_PS1
    assert '$StudioHomeIsCustom -and $pathState -eq "Denied"' in SETUP_PS1
    assert SETUP_PS1.count("Exit-PathAccessDenied -Path") == 5


def test_denied_install_reports_an_actionable_failure():
    body = SETUP_PS1.split("function Exit-PathAccessDenied", 1)[1].split("\nfunction ", 1)[0]
    assert "cannot be read: access is denied" in body
    # The reporter reinstalled to a different drive and hit the same line; the
    # message has to say why that cannot help.
    assert "reinstalling Unsloth Studio, to any drive, reuses it" in body
    assert "delete or rename $Path" in body
    assert "Controlled folder access" in body
    assert 'Exit-SetupFailure "Access denied reading the existing $Label' in body
    assert "Reinstalling the app does not reset it." in body


def test_recovery_commands_are_separately_runnable():
    """On one line "then" is not a PowerShell separator: takeown would take the
    rest as arguments and icacls would never run."""
    body = SETUP_PS1.split("function Exit-PathAccessDenied", 1)[1].split("\nfunction ", 1)[0]
    command_lines = [
        line for line in body.splitlines() if "takeown /F" in line or "/reset /T" in line
    ]
    assert len(command_lines) == 2, command_lines
    assert not any("takeown" in line and "icacls" in line for line in command_lines), command_lines


def test_failure_reaches_the_desktop_ui():
    """Exit-SetupFailure is the only path that emits [TAURI:ERROR], which the
    desktop app prefers over its generic exit-code message."""
    body = SETUP_PS1.split("function Exit-SetupFailure", 1)[1].split("\n}", 1)[0]
    assert "UNSLOTH_TAURI_MODE" in body
    assert "[TAURI:ERROR] $singleLine" in body


def test_ownership_guard_distinguishes_denied_from_unowned():
    guard = SETUP_PS1.split("function Assert-StudioOwnedOrAbsent", 1)[1].split("\nfunction ", 1)[0]
    assert (
        "$markerState = Get-PathState -Path (Join-Path $Path $StudioOwnedMarker) -PathType Leaf"
        in guard
    )
    assert '$markerState -eq "Denied"' in guard
    # The old wording blamed ownership, which is unknowable while the tree is
    # unreadable; it must stay for the genuinely-unowned case only.
    assert "is not marked as an Unsloth-owned $Label" in guard
    # Both stops stay gated, so default-home installs behave exactly as before.
    assert guard.count("$StudioHomeIsCustom -and") == 3


def test_no_bare_test_path_probes_inside_the_llama_install_tree():
    """Probes that read *inside* a tree whose permissions we do not control are
    the ones that throw; they must all go through the guarded helpers."""
    inside_tree = re.compile(
        r"Test-Path\b[^\n]*(\$existingMetaPath|\$llamaMarker|\$_cand|Join-Path \$LlamaCppDir)"
    )
    offenders = [
        f"{index}: {line.strip()}"
        for index, line in enumerate(SETUP_PS1.splitlines(), start = 1)
        if inside_tree.search(line)
    ]
    assert not offenders, offenders


def test_whisper_phase_stays_non_fatal_on_a_denied_llama_tree():
    """whisper.cpp failures degrade to Transformers dictation by contract, so
    its read of the llama.cpp marker must not be able to terminate setup."""
    assert 'if (Test-PathQuiet $llamaMarker "Leaf")' in SETUP_PS1
