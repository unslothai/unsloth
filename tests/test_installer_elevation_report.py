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
REPORT_RS = REPO_ROOT / "studio" / "src-tauri" / "src" / "diagnostics" / "report.rs"


def _read(path: Path) -> str:
    return path.read_text(encoding = "utf-8")


# ── The marker channel these rely on ──


def test_diag_marker_prefix_is_still_parsed_and_reported():
    """Both scripts write [TAURI:DIAG]; if the Rust side stops reading it the
    markers become invisible without either file changing."""
    assert '"[TAURI:DIAG] "' in _read(INSTALL_RS), "install.rs no longer parses [TAURI:DIAG]"
    assert "record_diag_marker" in _read(INSTALL_RS), "the marker is parsed but not recorded"
    assert "installer_diag_markers" in _read(REPORT_RS), "the support report no longer prints markers"


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
    block = src[src.index("function Get-ElevationState") : src.index("function Write-ElevationNotice")]
    assert '"unknown"' in block, "an unreadable token must report unknown, not a confident false"
    for value in ('"true"', '"false"'):
        assert value in block, f"Get-ElevationState must be able to return {value}"


def test_install_ps1_warns_before_any_install_work():
    """The warning is only useful while the user can still close the window."""
    src = _read(INSTALL_PS1)
    notice_idx = src.index("Write-ElevationNotice -State (Get-ElevationState)")
    for marker in (
        'step "winget" "available"',
        "uv venv $VenvDir --python",
        'step "setup" "running unsloth studio setup..."',
    ):
        assert notice_idx < src.index(marker), (
            f"the elevation notice must be printed before {marker!r}"
        )


def test_install_ps1_warning_names_the_folder_that_outlives_uninstall():
    """A bare "you are admin" note does not explain why it matters later."""
    src = _read(INSTALL_PS1)
    notice = src[src.index("function Write-ElevationNotice") :]
    notice = notice[: notice.index("Write-ElevationNotice -State")]
    assert "\\.unsloth" in notice, "the warning must name the folder that becomes unreadable"
    assert "outlives an uninstall" in notice, "the warning must say reinstalling does not clear it"


def test_setup_ps1_reports_elevation_without_duplicating_the_installer():
    """install.ps1 runs setup.ps1 with SKIP_STUDIO_BASE=1 and has already
    reported; this covers direct setup, update and desktop repair."""
    src = _read(SETUP_PS1)
    idx = src.index('if ($env:SKIP_STUDIO_BASE -ne "1") {')
    block = src[idx : idx + 1800]
    assert "WindowsBuiltInRole]::Administrator" in block
    assert "[TAURI:DIAG] elevated=$ElevationState" in block
    # Desktop repair runs with UNSLOTH_TAURI_UPDATE rather than UNSLOTH_TAURI_MODE.
    assert "UNSLOTH_TAURI_UPDATE" in block, "repair updates must emit the marker too"


# ── Degraded llama.cpp ──


def test_degraded_llama_cpp_is_recorded_not_only_flashed():
    """[TAURI:PROGRESS] is transient UI text; the marker is what survives."""
    for path in (SETUP_PS1, SETUP_SH):
        src = _read(path)
        assert "llama_cpp=unavailable" in src, f"{path.name} does not record the degraded verdict"
        progress_idx = src.index("llama.cpp unavailable; GGUF inference is disabled")
        marker_idx = src.index("llama_cpp=unavailable")
        assert marker_idx > progress_idx, (
            f"{path.name} must keep the user-facing progress line and add the marker beside it"
        )


def test_degraded_llama_cpp_still_fails_outside_tauri_mode():
    """The marker must not soften the non-desktop contract: install.sh and
    install.ps1 still need the non-zero exit."""
    assert 'setup_fail 1 "llama.cpp setup did not produce a usable server"' in _read(SETUP_SH)
    assert 'Exit-SetupFailure "llama.cpp setup did not produce a usable server"' in _read(SETUP_PS1)
