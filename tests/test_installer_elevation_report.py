# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for what a Windows install records about itself.

Two things a support report could not answer, both from the same install:

1. Whether the run was elevated. An install started with "Run as administrator"
   writes %USERPROFILE%\\.unsloth as Administrators, the normal account cannot
   read it back, and the folder outlives an uninstall, so the next install fails
   on a folder nobody remembers creating. The reporter spent the debugging
   session on it; "ran as admin: yes/no" would have ended it immediately.

2. Whether llama.cpp actually landed. In Tauri mode a degraded llama.cpp is
   deliberately not fatal (see tests/sh/test_llama_degraded_tauri_mode.sh), but
   it was announced only through install-progress-detail, which the next
   install-step clears and the install screen discards on close. The install
   then looked clean and the first sign of trouble was a GGUF failing to load.

Both now also go out as [TAURI:DIAG], which install.rs hands to
record_diag_marker and the support report prints under installer_diag_markers.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_PS1 = REPO_ROOT / "install.ps1"
SETUP_PS1 = REPO_ROOT / "studio" / "setup.ps1"
SETUP_SH = REPO_ROOT / "studio" / "setup.sh"
INSTALL_RS = REPO_ROOT / "studio" / "src-tauri" / "src" / "install.rs"
UPDATE_RS = REPO_ROOT / "studio" / "src-tauri" / "src" / "update.rs"
REPORT_RS = REPO_ROOT / "studio" / "src-tauri" / "src" / "diagnostics" / "report.rs"


def _read(path: Path) -> str:
    return path.read_text(encoding = "utf-8")


def _code_only(text: str) -> str:
    """Drop whole-line comments so an assertion about what the code reads is not
    satisfied, or defeated, by prose explaining it."""
    return "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))


def _repair_block(path: Path) -> str:
    """The marker-only block that covers a desktop repair, which runs without
    SKIP_STUDIO_BASE and so never reaches the install-side block."""
    src = _read(path)
    return src[src.index("# A desktop repair runs update.rs") :]


# ── The marker channel these rely on ──


def test_diag_marker_prefix_is_still_parsed_and_reported():
    """Both scripts write [TAURI:DIAG]; if the Rust side stops reading it the
    markers become invisible without either file changing."""
    assert '"[TAURI:DIAG] "' in _read(INSTALL_RS), "install.rs no longer parses [TAURI:DIAG]"
    assert "record_diag_marker" in _read(INSTALL_RS), "the marker is parsed but not recorded"
    assert "installer_diag_markers" in _read(
        REPORT_RS
    ), "the support report no longer prints markers"


# ── Elevation ──


def test_install_ps1_reports_elevation():
    src = _read(INSTALL_PS1)
    assert "function Get-ElevationState" in src
    assert "WindowsBuiltInRole]::Administrator" in src, "elevation must come from the process token"
    assert "[TAURI:DIAG] elevated=$State" in src, "the state must reach the support report"
    # Recorded on every run, not only the bad one: "false" is the answer that
    # rules elevation out, and absence would be ambiguous.
    notice = src[src.index("function Write-ElevationNotice") :]
    diag_idx = notice.index("[TAURI:DIAG] elevated=$State")
    warn_idx = notice.index('if ($State -ne "true") { return }')
    assert diag_idx < warn_idx, "the marker must be written before the warning returns early"


def test_install_ps1_elevation_state_covers_unreadable_tokens():
    src = _read(INSTALL_PS1)
    block = src[
        src.index("function Get-ElevationState") : src.index("function Write-ElevationNotice")
    ]
    assert '"unknown"' in block, "an unreadable token must report unknown, not a confident false"
    for value in ('"true"', '"false"'):
        assert value in block, f"Get-ElevationState must be able to return {value}"


def test_install_ps1_warns_before_anything_is_created():
    """Telling the user to stop is only honest while stopping leaves nothing
    behind. The override resolver creates the custom root and writes a probe
    into it, so an elevated run that warned afterwards had already made the
    admin-owned folder the warning is about."""
    src = _read(INSTALL_PS1)
    notice_idx = src.index("Write-ElevationNotice -State (Get-ElevationState)")
    for marker in (
        "[System.IO.Directory]::CreateDirectory($envOverride)",
        '$probe = Join-Path $StudioHome (".unsloth-write-probe-',
        'step "winget" "available"',
        # `& $script:UvExe`, not a bare `uv`, since #8161 resolved the binary up front.
        "venv $VenvDir --python",
        'step "setup" "running unsloth studio setup..."',
    ):
        assert notice_idx < src.index(
            marker
        ), f"the elevation notice must be printed before {marker!r}"


def test_install_ps1_marker_uses_the_parsed_tauri_flag():
    """install.rs launches this with --tauri and never sets UNSLOTH_TAURI_MODE;
    install.ps1 assigns that variable itself, far below, right before invoking
    setup.ps1. Gating on the env var here would emit nothing for the desktop
    install, and setup.ps1 then skips its own marker under SKIP_STUDIO_BASE=1,
    so the flow this exists for would produce no elevated= line at all."""
    src = _read(INSTALL_PS1)
    notice = _code_only(
        src[src.index("function Write-ElevationNotice") : src.index("$ElevationRoot = if")]
    )
    assert "[TAURI:DIAG] elevated=$State" in notice
    assert (
        "$env:UNSLOTH_TAURI_MODE" not in notice
    ), "the env var is unset at this point; gate on the --tauri flag instead"
    assert "-Tauri:$TauriMode" in src, "the parsed --tauri flag must be what drives the marker"
    # The flag has to be parsed before the call, or it is always false.
    assert src.index('"--tauri"    { $TauriMode = $true }') < src.index("-Tauri:$TauriMode")
    # And the env var must still be unassigned there, or this test proves nothing.
    assert src.index("-Tauri:$TauriMode") < src.index("$env:UNSLOTH_TAURI_MODE = if ($TauriMode)")


def test_install_ps1_warning_names_the_root_actually_written():
    """A custom UNSLOTH_STUDIO_HOME holds every artefact, so naming
    %USERPROFILE%\\.unsloth there points the user at the wrong directory."""
    src = _read(INSTALL_PS1)
    notice = src[src.index("function Write-ElevationNotice") : src.index("-Tauri:$TauriMode")]
    assert "$Root" in notice, "the warning must name the resolved root, not a fixed path"
    assert "outlives an uninstall" in notice, "the warning must say reinstalling does not clear it"
    root = src[src.index("$UnslothRoot = Join-Path") : src.index("-Tauri:$TauriMode")]
    assert '".unsloth"' in root, "a default install must name the parent that also holds llama.cpp"
    # The notice runs before the resolver, so $StudioHome does not exist yet and
    # naming it would render empty; mirror the override precedence instead.
    assert "$StudioHome" not in root, "$StudioHome is not assigned until the resolver below"
    assert (
        "UNSLOTH_STUDIO_HOME" in root and "$env:STUDIO_HOME" in root
    ), "both override names must be honoured, in that order"
    assert root.index("UNSLOTH_STUDIO_HOME") < root.index(
        "$env:STUDIO_HOME"
    ), "UNSLOTH_STUDIO_HOME wins when both are set"


def test_a_legacy_equal_override_names_its_parent():
    """An override that resolves to %USERPROFILE%\\.unsloth\\studio is not a custom
    root downstream: the canonical comparison in setup.ps1 treats it as a default
    install and keeps llama.cpp and node as siblings under ~/.unsloth. Naming only
    the studio directory would send the user past the admin-owned assets."""
    for path, var in ((INSTALL_PS1, "$ElevationRoot"), (SETUP_PS1, "$_elevRoot")):
        src = _read(path)
        idx = src.index(f"{var} = if") - 400
        block = src[idx : idx + 1200]
        assert '"studio"' in block, f"{path.name} does not compare against the legacy root"
        assert "-ieq" in block, f"{path.name} must compare case-insensitively"
        assert (
            block.count("Get-CanonicalRootPath") >= 2
        ), f"{path.name} must canonicalize both sides before comparing"


def test_the_legacy_root_comparison_is_canonical():
    """A raw string compare misses the spellings a user actually types. The env
    override reaches this unresolved: `~/.unsloth/studio`, a trailing separator,
    forward slashes, or a `..` segment all name the legacy root but compare
    unequal, and the notice then sends the user past the admin-owned llama.cpp."""
    for path in (INSTALL_PS1, SETUP_PS1):
        src = _code_only(_read(path))
        idx = src.index("function Get-CanonicalRootPath")
        body = src[idx : src.index("\n    }\n", idx)]
        # GetFullPath alone keeps a literal ~ and resolves it against the cwd.
        assert '-eq "~"' in body and '"~/*"' in body, f"{path.name} must expand a leading ~"
        assert "$env:USERPROFILE" in body, f"{path.name} must expand ~ to the profile"
        assert (
            "[System.IO.Path]::GetFullPath" in body
        ), f"{path.name} must normalize separators and .. segments"
        assert "TrimEnd" in body, f"{path.name} must ignore a trailing separator"
        # A bare "~" leaves an empty child path, which Join-Path rejects on PS 5.1.
        assert "Substring(1)" in body, f"{path.name} must handle a bare ~"
        # An unresolvable path must not take the install down.
        assert "catch { }" in body, f"{path.name} must survive an unresolvable path"


def test_installer_restores_skip_studio_base():
    """`irm ... | iex` runs in the caller's shell. A leaked SKIP_STUDIO_BASE=1
    makes the next direct setup or update look like an installer child, which
    suppresses both the elevation notice and the degraded-repair marker."""
    src = _read(INSTALL_PS1)
    assert "$previousSkipStudioBase = $env:SKIP_STUDIO_BASE" in src
    set_idx = src.index('$env:SKIP_STUDIO_BASE = "1"')
    save_idx = src.index("$previousSkipStudioBase = $env:SKIP_STUDIO_BASE")
    assert save_idx < set_idx, "the previous value must be captured before it is overwritten"
    restore = src[src.index("} finally {") :]
    assert (
        "$env:SKIP_STUDIO_BASE = $previousSkipStudioBase" in restore
    ), "the flag must be restored alongside UNSLOTH_TAURI_MODE"
    assert (
        "Remove-Item Env:SKIP_STUDIO_BASE" in restore
    ), "an unset value must be removed again, not left as an empty string"


def test_the_early_bail_restores_the_environment_too():
    """--with-llama-cpp-dir with a missing path returns before `& $UnslothExe`.
    That return used to sit between the env mutations and the `try`, so the bail
    leaked SKIP_STUDIO_BASE, UNSLOTH_STUDIO_HOME and UNSLOTH_TAURI_MODE into the
    caller's shell. The try has to open before the first mutation."""
    src = _code_only(_read(INSTALL_PS1))
    try_idx = src.index("    try {\n        $env:SKIP_STUDIO_BASE")
    finally_idx = src.index("    } finally {", try_idx)
    body = src[try_idx:finally_idx]
    assert (
        "--with-llama-cpp-dir path does not exist." in body
    ), "the early bail must sit inside the try that the finally covers"
    for var in ("SKIP_STUDIO_BASE", "UNSLOTH_STUDIO_HOME", "UNSLOTH_TAURI_MODE"):
        assert f"$env:{var} =" in body, f"{var} is mutated outside the try"
    # The saves read the previous value, so they must stay outside and before it.
    for save in (
        "$previousSkipStudioBase = $env:SKIP_STUDIO_BASE",
        "$previousUnslothStudioHome = $env:UNSLOTH_STUDIO_HOME",
        "$previousTauriMode = $env:UNSLOTH_TAURI_MODE",
    ):
        assert src.index(save) < try_idx, f"'{save}' must be captured before the try opens"


def test_both_scripts_resolve_the_warning_root_the_same_way():
    """install.ps1 and setup.ps1 each mirror the resolver's precedence for the
    message, so a drift in one would send only half the flows to the right
    directory."""
    install_root = _read(INSTALL_PS1)
    install_root = install_root[
        install_root.index("$UnslothRoot = Join-Path") : install_root.index("-Tauri:$TauriMode")
    ]
    setup_src = _read(SETUP_PS1)
    setup_root = setup_src[
        setup_src.index("$_unslothRoot = Join-Path") : setup_src.index("# Back up User PATH")
    ]
    for fragment in (
        "IsNullOrWhiteSpace($env:UNSLOTH_STUDIO_HOME)",
        "$env:UNSLOTH_STUDIO_HOME.Trim()",
        "IsNullOrWhiteSpace($env:STUDIO_HOME)",
        "$env:STUDIO_HOME.Trim()",
        'Join-Path $env:USERPROFILE ".unsloth"',
    ):
        assert fragment in install_root, f"install.ps1 root resolution lost {fragment!r}"
        assert fragment in setup_root, f"setup.ps1 root resolution lost {fragment!r}"


def test_install_ps1_warning_does_not_assume_a_powershell_window():
    """install.rs spawns this from the desktop app, where there is no console to
    close and no PowerShell prompt to re-run from."""
    src = _read(INSTALL_PS1)
    notice = src[src.index("function Write-ElevationNotice") : src.index("$ElevationRoot = if")]
    for phrase in ("Close this window", "normal PowerShell"):
        assert phrase not in notice, f"{phrase!r} is wrong for the desktop flow"


def test_setup_ps1_reports_elevation_without_duplicating_the_installer():
    """install.ps1 runs setup.ps1 with SKIP_STUDIO_BASE=1 and has already
    reported; this covers direct setup, update and desktop repair."""
    src = _read(SETUP_PS1)
    idx = src.index('if ($env:SKIP_STUDIO_BASE -ne "1") {')
    block = src[idx : src.index("# Back up User PATH")]
    assert "WindowsBuiltInRole]::Administrator" in block
    assert "[TAURI:DIAG] elevated=$ElevationState" in block
    # Desktop repair runs with UNSLOTH_TAURI_UPDATE rather than UNSLOTH_TAURI_MODE.
    assert "UNSLOTH_TAURI_UPDATE" in block, "repair updates must emit the marker too"


def test_setup_ps1_warning_names_the_root_actually_written():
    """$StudioHome is not resolved until much later in setup.ps1, so the message
    mirrors the documented override precedence rather than a fixed path."""
    src = _read(SETUP_PS1)
    idx = src.index('if ($env:SKIP_STUDIO_BASE -ne "1") {')
    block = src[idx : src.index("# Back up User PATH")]
    assert "$_elevRoot" in block, "the warning must name a resolved root"
    assert (
        "UNSLOTH_STUDIO_HOME" in block and "STUDIO_HOME" in block
    ), "both override names must be honoured, in that order"
    assert block.index("UNSLOTH_STUDIO_HOME") < block.index(
        "$env:STUDIO_HOME"
    ), "UNSLOTH_STUDIO_HOME wins when both are set"
    for phrase in ("Close this window", "normal PowerShell"):
        assert phrase not in block, f"{phrase!r} is wrong for the desktop repair flow"
    # Naming $StudioHome directly here would render empty.
    assert idx < src.index(
        "$StudioHome = Join-Path $env:USERPROFILE"
    ), "this test is only meaningful while the notice precedes the resolver"


# ── Degraded llama.cpp ──


def test_degraded_llama_cpp_is_recorded_not_only_flashed():
    """[TAURI:PROGRESS] is transient UI text; the marker is what survives."""
    for path in (SETUP_PS1, SETUP_SH):
        src = _read(path)
        assert "llama_cpp=unavailable" in src, f"{path.name} does not record the degraded verdict"
        progress_idx = src.index("llama.cpp unavailable; GGUF inference is disabled")
        marker_idx = src.index("llama_cpp=unavailable")
        assert (
            marker_idx > progress_idx
        ), f"{path.name} must keep the user-facing progress line and add the marker beside it"


def test_degraded_llama_cpp_still_fails_outside_tauri_mode():
    """The marker must not soften the non-desktop contract: install.sh and
    install.ps1 still need the non-zero exit."""
    assert 'setup_fail 1 "llama.cpp setup did not produce a usable server"' in _read(SETUP_SH)
    assert 'Exit-SetupFailure "llama.cpp setup did not produce a usable server"' in _read(SETUP_PS1)


def test_degraded_llama_cpp_is_recorded_on_a_desktop_repair():
    """update.rs sets UNSLOTH_TAURI_UPDATE alone, neither SKIP_STUDIO_BASE nor
    UNSLOTH_TAURI_MODE, so the install-side block never runs on a repair. It
    parses [TAURI:DIAG] the same way install.rs does."""
    assert 'strip_prefix("[TAURI:DIAG] ")' in _read(UPDATE_RS), "update.rs must parse the marker"
    assert "record_diag_marker" in _read(UPDATE_RS), "the marker is parsed but not recorded"
    for path, guard in (
        (SETUP_SH, '[ "${SKIP_STUDIO_BASE:-0}" != "1" ]'),
        (SETUP_PS1, '$env:SKIP_STUDIO_BASE -ne "1"'),
    ):
        block = _repair_block(path)
        assert guard in block, f"{path.name} must record the repair path the install block skips"
        assert "llama_cpp=unavailable" in block, f"{path.name} repair block emits no marker"
        assert "UNSLOTH_TAURI_UPDATE" in block, f"{path.name} must gate the repair marker on it"


def test_desktop_repair_marker_does_not_change_the_update_contract():
    """A plain 'unsloth studio update' stays silent and successful; only a Tauri
    repair emits, and neither path gains an exit-code change."""
    for path in (SETUP_SH, SETUP_PS1):
        block = _code_only(_repair_block(path))
        for forbidden in ("setup_fail", "Exit-SetupFailure", "TAURI:PROGRESS", "exit "):
            assert (
                forbidden not in block
            ), f"{path.name}: the repair block must be marker-only, found {forbidden!r}"
