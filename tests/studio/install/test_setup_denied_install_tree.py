# SPDX-License-Identifier: AGPL-3.0-only
"""Test-Path throws on ACL-denied paths, not false, so setup.ps1 probes use Get-PathState."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SETUP_PS1 = (ROOT / "studio" / "setup.ps1").read_text(encoding = "utf-8")


def _denial_reporter() -> str:
    """install.ps1 cannot dot-source this file, so its preflight carries a copy that a test compares."""
    assert "function Write-PathAccessDenied" in SETUP_PS1
    return SETUP_PS1.split("function Write-PathAccessDenied", 1)[1].split("\nfunction ", 1)[0]


def test_the_denial_exit_delegates_to_the_shared_reporter():
    """Exit-PathAccessDenied stays the only stop, with no second copy of the wording."""
    body = SETUP_PS1.split("function Exit-PathAccessDenied", 1)[1].split("\nfunction ", 1)[0]
    assert "Exit-SetupFailure (Write-PathAccessDenied -Path $Path -Label $Label" in body
    assert "-UserSupplied:$UserSupplied -OwnershipUnverified:$OwnershipUnverified)" in body
    assert "takeown" not in body
    assert "substep" not in body


def test_setup_defines_non_throwing_path_probes():
    for name in ("Test-AccessDeniedError", "Get-PathState", "Test-PathQuiet"):
        assert re.search(rf"^function {re.escape(name)} \{{", SETUP_PS1, re.M), name
    # Collapsing 'Denied' into 'Absent' would hide the failure again.
    for state in ('return "Present"', 'return "Absent"', 'return "Denied"'):
        assert state in SETUP_PS1


def test_prebuilt_metadata_probe_cannot_terminate_setup():
    """Bare probe threw; Get-LlamaCppInstallReadState decides denial; this checks presence."""
    assert "if (Test-Path $existingMetaPath)" not in SETUP_PS1
    assert "Test-PathQuiet -Path $existingMetaPath -PathType Leaf" in SETUP_PS1
    assert "$llamaDirState = Get-LlamaCppInstallReadState -Path $LlamaCppDir" in SETUP_PS1
    assert '$llamaDirState -eq "Denied"' in SETUP_PS1
    assert (
        'Get-PathState -Path (Join-Path $Path "UNSLOTH_PREBUILT_INFO.json") -PathType Leaf'
        in SETUP_PS1
    )


def test_every_denial_route_reports_instead_of_proceeding():
    """An unreadable parent dir, metadata file, .git checkout or ownership root must stop setup;
    treating any of them as absent lets the caller replace or delete a tree it cannot read."""
    assert "$llamaDirState = Get-LlamaCppInstallReadState -Path $LlamaCppDir" in SETUP_PS1
    assert '$llamaDirState -eq "Denied"' in SETUP_PS1
    assert '$llamaGitState = Get-PathState -Path (Join-Path $LlamaCppDir ".git")' in SETUP_PS1
    assert '$llamaGitState -eq "Denied"' in SETUP_PS1
    assert "$pathState = Get-PathState -Path $Path -PathType Container" in SETUP_PS1
    # $isCustomRoot is the parameter form of $StudioHomeIsCustom.
    assert '$isCustomRoot -and $pathState -eq "Denied"' in SETUP_PS1
    # The junction path replaces this destination, so it needs its own stop.
    assert "$destState = Get-PathState -Path $LlamaCppDir" in SETUP_PS1
    assert '$destState -eq "Denied"' in SETUP_PS1
    # Denied counts as surviving removal; collapsing it would junction over it.
    assert '(Get-PathState -Path $LlamaCppDir) -ne "Absent"' in SETUP_PS1
    # Floor, not exact: losing a route is the bug; each route is also pinned by name.
    assert SETUP_PS1.count("Exit-PathAccessDenied -Path") >= 9


def test_denied_install_reports_an_actionable_failure():
    body = _denial_reporter()
    assert "cannot be read: access is denied" in body
    # Reinstalling to any drive reuses this tree; the message must say why that cannot help.
    assert "reinstalling Unsloth Studio, to any drive, reuses it" in body
    assert "delete or rename $Path" in body
    assert "Controlled folder access" in body
    assert 'return "Access denied reading the existing $Label' in body
    assert "Reinstalling the app does not reset it." in body


def test_recovery_commands_are_separately_runnable():
    """then" on one line is not a PowerShell separator; takeown takes the rest as arguments."""
    body = _denial_reporter()
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
    assert "UNSLOTH_TAURI_UPDATE" in body
    assert "[TAURI:ERROR] $singleLine" in body


def test_ownership_guard_distinguishes_denied_from_unowned():
    guard = SETUP_PS1.split("function Assert-StudioOwnedOrAbsent", 1)[1].split("\nfunction ", 1)[0]
    assert (
        "$markerState = Get-PathState -Path (Join-Path $Path $StudioOwnedMarker) -PathType Leaf"
        in guard
    )
    assert '$markerState -eq "Denied"' in guard
    # Ownership is unknowable while unreadable; keep that wording for the truly unowned case.
    assert "is not marked as an Unsloth-owned $Label" in guard
    # Both stops stay gated, so default-home installs behave exactly as before.
    assert "$isCustomRoot = $StudioHomeIsCustom" in guard
    assert guard.count("$isCustomRoot -and") >= 3


def test_no_bare_test_path_probes_inside_the_llama_install_tree():
    """Probes inside an uncontrolled tree throw, so they must all use the guarded helpers."""
    inside_tree = re.compile(
        r"Test-Path\b[^\n]*("
        r"\$existingMetaPath|\$llamaMarker|\$_cand|Join-Path \$LlamaCppDir"
        r"|\$LlamaServerBin|\$CmakeCacheFile|\$QuantizeBin|\$altBin"
        r"|Join-Path \$BuildDir)"
    )
    offenders = [
        f"{index}: {line.strip()}"
        for index, line in enumerate(SETUP_PS1.splitlines(), start = 1)
        if inside_tree.search(line.split("#", 1)[0])
    ]
    assert not offenders, offenders


def test_metadata_reads_are_literal_like_the_probes_that_gate_them():
    """A literal probe passes but a globbing read still throws on a path holding [ or ]."""
    offenders = [
        f"{index}: {line.strip()}"
        for index, line in enumerate(SETUP_PS1.splitlines(), start = 1)
        if re.search(r"Get-Content\b[^\n]*(\$metadataPath|\$existingMetaPath|\$llamaMarker)", line)
        and "-LiteralPath" not in line
    ]
    assert not offenders, offenders


def test_whisper_phase_stays_non_fatal_on_a_denied_llama_tree():
    """Its llama.cpp marker read must not terminate setup; whisper failures are non-fatal."""
    assert 'if (Test-PathQuiet $llamaMarker "Leaf")' in SETUP_PS1


def test_reporting_helpers_tolerate_an_empty_path():
    """Mandatory [string] rejects null/empty; [AllowNull()] keeps the reporter from throwing."""
    for name in ("Get-PathDenialDetail", "Exit-PathAccessDenied"):
        body = SETUP_PS1.split(f"function {name}", 1)[1].split("\nfunction ", 1)[0]
        head = body.split("$Path", 1)[0]
        assert "[AllowNull()]" in head, name
        assert "[AllowEmptyString()]" in head, name
    detail = SETUP_PS1.split("function Get-PathDenialDetail", 1)[1].split("\nfunction ", 1)[0]
    assert 'if ([string]::IsNullOrWhiteSpace($Path)) { return "" }' in detail


def test_local_llama_dir_probes_are_three_state():
    """A denied probe must not read as nothing built, or the prebuilt installer replaces a local build."""
    assert "$localSrcState = Get-PathState -Path $LocalLlamaCppSrc -PathType Container" in SETUP_PS1
    assert '$localSrcState -eq "Denied"' in SETUP_PS1
    local_block = SETUP_PS1.split("$LocalLlamaCppSrc = $env:UNSLOTH_LOCAL_LLAMA_CPP_DIR", 1)[1]
    local_block = local_block.split("if ($LocalLlamaCppLinked) {", 1)[0]
    assert "$candState = Get-PathState -Path $_cand" in local_block
    assert '$candState -eq "Denied"' in local_block
    assert '$candState -eq "Present"' in local_block
    assert "Test-PathQuiet $_cand" not in local_block
    # The disk-space branch keeps Test-PathQuiet: an unreadable binary is not usable.


def test_phase_1b_git_scan_is_guarded_too():
    """The Phase 1b git scan runs under Stop, so an unguarded denial there terminates setup."""
    scan = SETUP_PS1.split("$_localLlamaBuilt = $false", 1)[1].split(
        "if (-not $_localLlamaBuilt) {", 1
    )[0]
    assert "$_cState = Get-PathState -Path (Join-Path $_localLlamaDir $_c)" in scan
    assert '$_cState -eq "Denied"' in scan
    assert '$_cState -eq "Present"' in scan
    assert "Test-Path -LiteralPath (Join-Path $_localLlamaDir $_c)" not in scan
    assert "-UserSupplied" in scan


def test_adoption_markers_keep_their_denial():
    """A denied marker is not absence; treating it as absent hides a permissions problem."""
    assert "function Get-StudioAdoptableState" in SETUP_PS1
    state = SETUP_PS1.split("function Get-StudioAdoptableState", 1)[1].split("\nfunction ", 1)[0]
    assert "Get-PathState -Path (Join-Path $Path $marker) -PathType Leaf" in state
    for verdict in ('return "Yes"', 'return "No"', 'return "Denied"'):
        assert verdict in state
    guard = SETUP_PS1.split("function Assert-StudioOwnedOrAbsent", 1)[1].split("\nfunction ", 1)[0]
    assert "$adoptState = Get-StudioAdoptableState -Path $Path" in guard
    assert '$adoptState -eq "Denied"' in guard
    assert guard.index('$adoptState -eq "Denied"') < guard.index(
        "is not marked as an Unsloth-owned"
    )


def test_user_supplied_paths_are_never_told_to_delete_themselves():
    """The managed advice ("delete it, Unsloth reinstalls it") is wrong for a tree
    the user pointed us at with UNSLOTH_LOCAL_LLAMA_CPP_DIR."""
    body = _denial_reporter()
    assert "[switch]$UserSupplied" in body
    user_branch = body.split("if ($UserSupplied) {", 1)[1].split("} else {", 1)[0]
    assert "delete or rename" not in user_branch
    assert "managed cache" not in user_branch
    assert "UNSLOTH_LOCAL_LLAMA_CPP_DIR at a readable build" in user_branch
    # The override says the tree is the user's build, so 'delete it' is wrong there too.
    for line in SETUP_PS1.splitlines():
        if "Exit-PathAccessDenied" in line and "UNSLOTH_LOCAL_LLAMA_CPP_DIR" in line:
            assert "-UserSupplied" in line, line
    # Keyed on the reported path: for $LlamaCppDir 'delete it' is right.
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
    """The guard stops when the marker is unreadable, since it cannot claim the tree is ours."""
    guard = SETUP_PS1.split("function Assert-StudioOwnedOrAbsent", 1)[1].split("\nfunction ", 1)[0]
    calls = [line.strip() for line in guard.splitlines() if "Exit-PathAccessDenied" in line]
    assert len(calls) >= 3, calls
    for line in calls:
        assert "-OwnershipUnverified" in line, line
    body = _denial_reporter()
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


def test_the_temp_dir_swap_checks_both_of_its_destructive_steps():
    """Remove-Item and Move-Item are both non-terminating here, and Move-Item onto a surviving
    directory nests the build instead of replacing it, so both steps must be checked."""
    swap = SETUP_PS1.split("# Swap temp build dir into final location", 1)[1]
    swap = swap.split("} elseif (-not $BuildOk", 1)[0]
    assert "$swapState = Get-PathState -Path $OriginalLlamaCppDir" in swap, swap
    assert '$swapState -eq "Denied"' in swap, swap
    # Stop before the move runs, while the temp build is still whole.
    move = "Move-Item -LiteralPath $LlamaCppDir"
    assert swap.index('$swapState -ne "Absent"') < swap.index(move), swap
    # And catch a move that silently did not happen.
    assert '(Get-PathState -Path $LlamaCppDir) -ne "Absent"' in swap.split(move, 1)[1], swap
    assert "Test-Path -LiteralPath $OriginalLlamaCppDir" not in swap, swap


def test_the_source_build_phase_probes_the_tree_three_state():
    """Forced, pinned or custom builds skip the prebuilt guard; the rebuild check reads first."""
    build = _slice("$llamaBinState = ", "# -- Summary --")
    assert "Get-PathState -Path $LlamaServerBin -PathType Leaf" in build, build
    assert '$llamaBinState -eq "Denied"' in build, build
    assert '$llamaBinState -eq "Present"' in build, build
    assert "Test-Path -LiteralPath $LlamaServerBin" not in build, build


def test_the_source_build_probe_skips_a_linked_local_dir():
    """Probing a linked $LlamaCppDir reads the user's checkout; the block's results go unused."""
    flat = " ".join(SETUP_PS1.split())
    assert '$llamaBinState = if ($LocalLlamaCppLinked) { "Absent" }' in flat


def test_the_source_build_denial_never_advises_deleting_an_unproven_tree():
    """No ownership guard here: under a custom home the tree is unproven, so no delete advice."""
    block = _slice("$llamaBinState = ", "$WillBuildLlamaFromSource")
    denials = [ln.strip() for ln in block.splitlines() if "Exit-PathAccessDenied" in ln]
    assert len(denials) >= 2, denials
    assert all(d.endswith("-OwnershipUnverified:$RuntimeRootIsCustom") for d in denials), denials
    assert all(d.startswith("Exit-PathAccessDenied -Path $LlamaCppDir ") for d in denials), denials


def test_the_cmake_cache_read_is_guarded_not_just_its_probe():
    """Test-PathQuiet only proves the entry is listed; a deny ACE on the file
    itself leaves the probe true and throws on the read below it."""
    block = _slice("$CmakeCacheFile = Join-Path", "$WillBuildLlamaFromSource")
    read = "Select-String -LiteralPath $CmakeCacheFile"
    assert read in block, block
    before, after = block.split(read, 1)
    assert "try {" in before, before
    assert "Test-AccessDeniedError" in after, after
    assert "Exit-PathAccessDenied" in after, after


def _slice(start: str, end: str) -> str:
    """Both bounds are asserted; a missing terminator would widen the slice to end-of-file."""
    assert start in SETUP_PS1, start
    assert end in SETUP_PS1, end
    return SETUP_PS1.split(start, 1)[1].split(end, 1)[0]


def _whisper_phase() -> str:
    """Returns the whisper phase body; both anchors are asserted so a phase renumbering fails as an
    assertion, not an IndexError, and cannot widen to end-of-file."""
    return _slice("Install the whisper.cpp prebuilt", "PHASE 3.5")


def test_the_whisper_phase_survives_an_unreadable_whisper_tree():
    """Phase failure is non-fatal, yet an ownership guard exit would take llama.cpp down."""
    guard = SETUP_PS1.split("function Assert-StudioOwnedOrAbsent", 1)[1].split("\nfunction ", 1)[0]
    assert "[switch]$NonFatal" in guard
    # Ordering, not just presence: below its exit the return is dead code.
    paired = re.findall(
        r'if \(\$NonFatal\) \{ return "Denied" \}\n\s*Exit-PathAccessDenied -Path \$Path', guard
    )
    assert len(paired) == guard.count("Exit-PathAccessDenied -Path $Path"), guard
    # No unpaired return: one above the custom-home gate would call a fresh install unreadable.
    assert len(paired) == guard.count('if ($NonFatal) { return "Denied" }'), guard
    assert len(paired) >= 3, guard
    # Only the denial is handed back; an unowned tree must still stop.
    assert 'Exit-SetupFailure "$Label path is not an Unsloth-owned install' in guard
    whisper = _whisper_phase()
    assert (
        '-Label "whisper.cpp install" -NonFatal -IsCustom $RuntimeRootIsCustom) -eq "Denied"'
        in whisper
    ), whisper
    # Scoped to the new branch: both phrases occur elsewhere in the phase.
    marker = '-NonFatal -IsCustom $RuntimeRootIsCustom) -eq "Denied") {'
    assert marker in whisper, whisper
    denial = whisper.split(marker, 1)[1].split("\n} elseif", 1)[0]
    assert re.search(r'^\s*step "whisper\.cpp" ', denial, re.M), denial
    assert "install directory cannot be read: access is denied" in denial, denial
    assert "browser and Transformers dictation remain available" in denial, denial
    assert "Exit-SetupFailure" not in denial, denial
    assert not re.search(r"\bexit \d", denial), denial
    body = "$whisperArgs = @("
    assert body in whisper, whisper
    assert whisper.index("-NonFatal") < whisper.index(body)


def test_the_whisper_skip_stays_behind_the_installer_gate():
    """Hoisted guard must not make the installer-less tree fatal, which was a no-op before."""
    marker = "-NonFatal -IsCustom $RuntimeRootIsCustom) -eq"
    whisper = _whisper_phase()
    assert marker in whisper, whisper
    head = whisper.split(marker, 1)[0]
    assert "} elseif" in head, head
    branch = head.rsplit("} elseif", 1)[1]
    # A conjunct, not merely present: -or or a negation reopens this case.
    assert re.search(r"-and\s*\([^\n]*\$WhisperInstaller[^\n]*\)\s*-and", branch), branch
    assert "-not (Test-Path" not in branch, branch
    assert " -or " not in branch, branch
