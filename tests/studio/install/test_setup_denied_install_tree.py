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
    # The junction path replaces this destination, so it needs its own stop.
    assert "$destState = Get-PathState -Path $LlamaCppDir" in SETUP_PS1
    assert '$destState -eq "Denied"' in SETUP_PS1
    # Denied counts as surviving removal; collapsing it would junction over it.
    assert '(Get-PathState -Path $LlamaCppDir) -ne "Absent"' in SETUP_PS1
    # Floor, not an exact count: losing a route is the bug, adding one is not.
    # Each route above is pinned by name, so a swap cannot hide under the floor.
    assert SETUP_PS1.count("Exit-PathAccessDenied -Path") >= 9


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


def test_metadata_reads_are_literal_like_the_probes_that_gate_them():
    """A literal probe followed by a globbing read still fails on a path holding
    [ or ], so the probe passes and the read throws into the catch."""
    offenders = [
        f"{index}: {line.strip()}"
        for index, line in enumerate(SETUP_PS1.splitlines(), start = 1)
        if re.search(r"Get-Content\b[^\n]*(\$metadataPath|\$existingMetaPath|\$llamaMarker)", line)
        and "-LiteralPath" not in line
    ]
    assert not offenders, offenders


def test_whisper_phase_stays_non_fatal_on_a_denied_llama_tree():
    """whisper.cpp failures degrade to Transformers dictation by contract, so
    its read of the llama.cpp marker must not be able to terminate setup."""
    assert 'if (Test-PathQuiet $llamaMarker "Leaf")' in SETUP_PS1


def test_reporting_helpers_tolerate_an_empty_path():
    """These run while a failure is being reported. A mandatory [string] rejects
    null and empty, so without these attributes the reporter would throw a
    binding exception instead of printing the actionable message."""
    for name in ("Get-PathDenialDetail", "Exit-PathAccessDenied"):
        body = SETUP_PS1.split(f"function {name}", 1)[1].split("\nfunction ", 1)[0]
        head = body.split("$Path", 1)[0]
        assert "[AllowNull()]" in head, name
        assert "[AllowEmptyString()]" in head, name
    detail = SETUP_PS1.split("function Get-PathDenialDetail", 1)[1].split("\nfunction ", 1)[0]
    assert 'if ([string]::IsNullOrWhiteSpace($Path)) { return "" }' in detail


def test_local_llama_dir_probes_are_three_state():
    """--with-llama-cpp-dir pointed at the canonical location reuses whatever is
    built there so the prebuilt installer cannot replace it. A denied binary read
    as "nothing built" put that replacement back, which is what the branch exists
    to prevent."""
    assert "$localSrcState = Get-PathState -Path $LocalLlamaCppSrc -PathType Container" in SETUP_PS1
    assert '$localSrcState -eq "Denied"' in SETUP_PS1
    local_block = SETUP_PS1.split("$LocalLlamaCppSrc = $env:UNSLOTH_LOCAL_LLAMA_CPP_DIR", 1)[1]
    local_block = local_block.split("if ($LocalLlamaCppLinked) {", 1)[0]
    assert "$candState = Get-PathState -Path $_cand" in local_block
    assert '$candState -eq "Denied"' in local_block
    assert '$candState -eq "Present"' in local_block
    assert "Test-PathQuiet $_cand" not in local_block
    # The disk-space branch keeps Test-PathQuiet on purpose: it only decides
    # whether a preserved binary is usable, and an unreadable one is not.


def test_phase_1b_git_scan_is_guarded_too():
    """The git prerequisite scan probes the same candidate binaries in Phase 1b,
    thousands of lines before the Phase 4 guards, and under "Stop". A denial
    there reproduced the original raw termination."""
    scan = SETUP_PS1.split("$_localLlamaBuilt = $false", 1)[1].split(
        "if (-not $_localLlamaBuilt) {", 1
    )[0]
    assert "$_cState = Get-PathState -Path (Join-Path $_localLlamaDir $_c)" in scan
    assert '$_cState -eq "Denied"' in scan
    assert '$_cState -eq "Present"' in scan
    assert "Test-Path -LiteralPath (Join-Path $_localLlamaDir $_c)" not in scan
    assert "-UserSupplied" in scan


def test_adoption_markers_keep_their_denial():
    """A denied prebuilt marker is not evidence of absence. Collapsing it made the
    ownership guard call an Unsloth tree an unrelated directory and tell the user
    to move it aside, hiding a permissions problem."""
    assert "function Get-StudioAdoptableState" in SETUP_PS1
    state = SETUP_PS1.split("function Get-StudioAdoptableState", 1)[1].split("\nfunction ", 1)[0]
    assert "Get-PathState -Path (Join-Path $Path $marker) -PathType Leaf" in state
    for verdict in ('return "Yes"', 'return "No"', 'return "Denied"'):
        assert verdict in state
    guard = SETUP_PS1.split("function Assert-StudioOwnedOrAbsent", 1)[1].split("\nfunction ", 1)[0]
    assert "$adoptState = Get-StudioAdoptableState -Path $Path" in guard
    assert '$adoptState -eq "Denied"' in guard
    # The denial must be reported before the "not Unsloth-owned" wording.
    assert guard.index('$adoptState -eq "Denied"') < guard.index(
        "is not marked as an Unsloth-owned"
    )


def test_user_supplied_paths_are_never_told_to_delete_themselves():
    """The managed advice ("delete it, Unsloth reinstalls it") is wrong for a tree
    the user pointed us at with UNSLOTH_LOCAL_LLAMA_CPP_DIR."""
    body = SETUP_PS1.split("function Exit-PathAccessDenied", 1)[1].split("\nfunction ", 1)[0]
    assert "[switch]$UserSupplied" in body
    user_branch = body.split("if ($UserSupplied) {", 1)[1].split("} else {", 1)[0]
    assert "delete or rename" not in user_branch
    assert "managed cache" not in user_branch
    assert "UNSLOTH_LOCAL_LLAMA_CPP_DIR at a readable build" in user_branch
    # Every call site that reports a path the user pointed us at must pass the
    # switch, including the canonical location: the override says that tree is
    # the user's build, so "delete it, we reinstall it" is wrong there too.
    for line in SETUP_PS1.splitlines():
        if "Exit-PathAccessDenied" in line and "UNSLOTH_LOCAL_LLAMA_CPP_DIR" in line:
            assert "-UserSupplied" in line, line
    # Keyed on the path being reported, not on position: this block also reports
    # the managed destination ($LlamaCppDir), where "delete it" is the right advice.
    local_block = SETUP_PS1.split("$LocalLlamaCppSrc = $env:UNSLOTH_LOCAL_LLAMA_CPP_DIR", 1)[1]
    local_block = local_block.split("if ($LocalLlamaCppLinked) {", 1)[0]
    for line in local_block.splitlines():
        if "Exit-PathAccessDenied" not in line:
            continue
        if "$ResolvedLocal" in line or "$LocalLlamaCppSrc" in line:
            assert "-UserSupplied" in line, line
        elif "$LlamaCppDir" in line:
            assert "-UserSupplied" not in line, line


def test_the_ownership_guard_never_advises_deleting_an_unverified_tree():
    """The guard stops because it could not read the marker, so it cannot claim
    the tree is ours. It already says "move it aside" when it can prove it."""
    guard = SETUP_PS1.split("function Assert-StudioOwnedOrAbsent", 1)[1].split("\nfunction ", 1)[0]
    calls = [line.strip() for line in guard.splitlines() if "Exit-PathAccessDenied" in line]
    assert len(calls) == 3, calls
    for line in calls:
        assert "-OwnershipUnverified" in line, line
    body = SETUP_PS1.split("function Exit-PathAccessDenied", 1)[1].split("\nfunction ", 1)[0]
    assert "[switch]$OwnershipUnverified" in body
    branch = body.split("} elseif ($OwnershipUnverified) {", 1)[1].split("} else {", 1)[0]
    assert "delete" not in branch.lower(), branch
    assert "managed cache" not in branch, branch
    assert "move the folder aside" in branch, branch


def test_the_reparse_point_unlink_reports_a_denied_delete():
    """A link probes Present, so the destination check below cannot cover it."""
    block = SETUP_PS1.split("$existing = Get-Item -LiteralPath $LlamaCppDir", 1)[1]
    block = block.split("$destState", 1)[0]
    assert "try { $existing.Delete() }" in block, block
    assert "Test-AccessDeniedError" in block, block
