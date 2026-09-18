#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# A uv installed but off THIS process's PATH made setup.sh re-download the pinned archive every
# update, 42 of a 53 s Windows no-op. It now searches astral's destinations first, and only a uv
# that runs, and answers as uv at or above install.sh's floor, counts.
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/_harness.sh"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
SETUP_PS1="$SCRIPT_DIR/../../studio/setup.ps1"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT INT TERM

HELPER=$(awk '
    /^_setup_uv_signal_target\(\) \{/ { grab = 1 }
    /^_setup_uv_probe_restore_trap\(\) \{/ { grab = 1 }
    /^_setup_uv_probe_on_signal\(\) \{/ { grab = 1 }
    /^_setup_uv_probe_exec\(\) \{/ { grab = 1 }
    /^_setup_uv_version_at_least\(\) \{/ { grab = 1 }
    /^_setup_find_installed_uv\(\) \{/ { grab = 1 }
    grab { print }
    grab && /^}/ { grab = 0 }
' "$SETUP_SH")
# The floor the finder compares against lives beside the function, not inside it.
HELPER="$(grep '^_SETUP_UV_MIN_VERSION=' "$SETUP_SH")
$HELPER"
for _fn in _setup_uv_signal_target _setup_uv_probe_restore_trap _setup_uv_probe_on_signal _setup_uv_probe_exec \
           _setup_uv_version_at_least _setup_find_installed_uv; do
    printf '%s\n' "$HELPER" | grep -q "^$_fn() {" || {
        echo "FATAL: could not extract $_fn from setup.sh" >&2; exit 1; }
done
printf '%s\n' "$HELPER" | grep -q '^_SETUP_UV_MIN_VERSION=' || {
    echo "FATAL: could not extract _SETUP_UV_MIN_VERSION from setup.sh" >&2; exit 1; }

PROBE="$WORK/probe.sh"
{
    printf '%s\n' "$HELPER"
    cat <<'BODY'
if _setup_find_installed_uv; then printf 'found=%s' "$_SETUP_UV_DIR"; else printf 'none'; fi
BODY
} > "$PROBE"
# The miss diagnostics survive only when the finder runs in the caller's shell, not a substitution.
DIAG="$WORK/diag.sh"
{
    printf '%s\n' "$HELPER"
    cat <<'BODY'
if _setup_find_installed_uv; then printf 'found'; else printf 'looked=%s miss=%s old=%s' "$_SETUP_UV_LOOKED" "$_SETUP_UV_PROBE_MISS" "$_SETUP_UV_TOO_OLD"; fi
BODY
} > "$DIAG"
# setup.ps1's finder, whole, so the checks below do not depend on its line count.
PS_FINDER=$(awk '
    /^function Find-InstalledUv \{/ { grab = 1 }
    grab { print }
    grab && /^\}/ { grab = 0 }
' "$SETUP_PS1")
printf '%s\n' "$PS_FINDER" | grep -q '^function Find-InstalledUv {' || {
    echo "FATAL: could not extract Find-InstalledUv from setup.ps1" >&2; exit 1; }

fake_uv() {  # fake_uv <dir> [exit code] [version]
    # It prints a version because the finder reads one: a silent stand-in would be refused for
    # that reason and every reuse case would go green for the wrong one.
    mkdir -p "$1"
    printf '#!/bin/sh\necho "uv %s (0123456 2026-01-01)"\nexit %s\n' "${3:-0.12.12}" "${2:-0}" > "$1/uv"
    chmod +x "$1/uv"
}

echo "=== test_setup_reuse_installed_uv ==="
for shell in sh bash; do
    command -v "$shell" >/dev/null 2>&1 || continue
    CASE="$WORK/$shell case"
    HOME_DIR="$CASE/home with spaces"
    mkdir -p "$HOME_DIR"
    # A PATH with no uv on it at all, as the miss requires.
    BARE_PATH="/usr/bin:/bin"

    assert_eq "$shell: nothing installed anywhere is a miss" \
        "none" "$(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" "$shell" "$PROBE")"
    case "$(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" "$shell" "$DIAG")" in
        "looked=$HOME_DIR/.local/bin/uv miss= old=") ok "$shell: a miss names the destinations it looked at" ;;
        *) bad "$shell: a miss names the destinations it looked at ($(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" "$shell" "$DIAG"))" ;;
    esac

    fake_uv "$HOME_DIR/.local/bin"
    assert_eq "$shell: astral's default destination is found" \
        "found=$HOME_DIR/.local/bin" "$(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" "$shell" "$PROBE")"

    XDG="$CASE/xdg data"
    fake_uv "$XDG/../bin"
    assert_eq "$shell: XDG_DATA_HOME/../bin outranks the home default" \
        "found=$XDG/../bin" "$(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" XDG_DATA_HOME="$XDG" "$shell" "$PROBE")"

    CUSTOM="$CASE/custom install dir"
    fake_uv "$CUSTOM"
    assert_eq "$shell: UV_INSTALL_DIR outranks every default" \
        "found=$CUSTOM" "$(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" XDG_DATA_HOME="$XDG" UV_INSTALL_DIR="$CUSTOM" "$shell" "$PROBE")"

    # Present but broken (another architecture, a half-written download): never reused.
    BROKEN="$CASE/broken"
    fake_uv "$BROKEN" 1
    assert_eq "$shell: a uv that cannot run is not reused" \
        "found=$HOME_DIR/.local/bin" "$(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" UV_INSTALL_DIR="$BROKEN" "$shell" "$PROBE")"
    rm -f "$HOME_DIR/.local/bin/uv"
    assert_eq "$shell: ...and with nothing else installed that is a miss" \
        "none" "$(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" UV_INSTALL_DIR="$BROKEN" "$shell" "$PROBE")"

    # Runs, but below install.sh's floor: its managed-Python manifest tops out at a CPython that
    # cannot import torch, and a bare "does it run" check reused it over the pinned release.
    OLD="$CASE/old uv"
    fake_uv "$OLD" 0 0.4.0
    assert_eq "$shell: a uv below the minimum version is not reused" \
        "none" "$(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" UV_INSTALL_DIR="$OLD" "$shell" "$PROBE")"
    case "$(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" UV_INSTALL_DIR="$OLD" "$shell" "$DIAG")" in
        *"old=$OLD/uv"*) ok "$shell: ...and the diagnostic names the version it refused" ;;
        *) bad "$shell: ...and the diagnostic names the version it refused ($(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" UV_INSTALL_DIR="$OLD" "$shell" "$DIAG"))" ;;
    esac
    # The floor itself is reused, and a newer one over it.
    fake_uv "$OLD" 0 0.9.3
    assert_eq "$shell: a uv exactly at the minimum version is reused" \
        "found=$OLD" "$(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" UV_INSTALL_DIR="$OLD" "$shell" "$PROBE")"
    # Runs, answers as something else: a miss, and the case a Windows stand-in cannot express.
    NOTUV="$CASE/not uv"
    mkdir -p "$NOTUV"
    printf '#!/bin/sh\necho "curl 8.9.1 (x86_64-pc-linux-gnu)"\nexit 0\n' > "$NOTUV/uv"
    chmod +x "$NOTUV/uv"
    assert_eq "$shell: a binary that runs but does not answer as uv is not reused" \
        "none" "$(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" UV_INSTALL_DIR="$NOTUV" "$shell" "$PROBE")"

    # One directory, however many variables name it: launched once, named once.
    DUP="$CASE/one dir"
    fake_uv "$DUP"
    case "$(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" UV_INSTALL_DIR="$DUP" UV_UNMANAGED_INSTALL="$DUP" XDG_BIN_HOME="$DUP" "$shell" "$DIAG")" in
        found) ok "$shell: duplicate destinations are searched once" ;;
        *) bad "$shell: duplicate destinations are searched once" ;;
    esac
    rm -f "$DUP/uv"
    case "$(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" UV_INSTALL_DIR="$DUP" UV_UNMANAGED_INSTALL="$DUP" XDG_BIN_HOME="$DUP" "$shell" "$DIAG")" in
        "looked=$DUP/uv, $HOME_DIR/.local/bin/uv miss= old=") ok "$shell: ...and named once in the miss diagnostic" ;;
        *) bad "$shell: ...and named once in the miss diagnostic ($(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" UV_INSTALL_DIR="$DUP" UV_UNMANAGED_INSTALL="$DUP" XDG_BIN_HOME="$DUP" "$shell" "$DIAG"))" ;;
    esac

    # A second search in one process reports its own destinations, not the first search's.
    TWICE="$WORK/$shell twice.sh"
    {
        printf '%s\n' "$HELPER"
        cat <<'BODY'
_setup_find_installed_uv || :
_setup_find_installed_uv || :
printf 'looked=%s' "$_SETUP_UV_LOOKED"
BODY
    } > "$TWICE"
    assert_eq "$shell: a second search does not inherit the first one's destinations" \
        "looked=$HOME_DIR/.local/bin/uv" "$(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" "$shell" "$TWICE")"

    # A uv that never answers: the probe is bounded, so a miss is reported, not a hang.
    # NOT gated on `command -v timeout`: it was, and stock macOS has none, so the one platform
    # that takes the watchdog branch skipped the whole block (27 assertions against 43 on Linux).
    if true; then
        HANG="$CASE/hangs"
        mkdir -p "$HANG"
        printf '#!/bin/sh\nsleep 60\n' > "$HANG/uv"
        chmod +x "$HANG/uv"
        HANG_PROBE="$WORK/$shell hang probe.sh"
        # The 20 s ceiling is the helper's; the test only needs it finite.
        { echo '_SETUP_UV_PROBE_SECONDS=2'; cat "$PROBE"; } > "$HANG_PROBE"
        _hang_started=$(date +%s)
        assert_eq "$shell: a uv that never answers is not reused" \
            "none" "$(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" UV_INSTALL_DIR="$HANG" "$shell" "$HANG_PROBE")"
        if [ $(( $(date +%s) - _hang_started )) -lt 30 ]; then
            ok "$shell: ...and the probe returned within its bound"
        else
            bad "$shell: ...and the probe returned within its bound"
        fi
        # Stock macOS has no GNU timeout: a PATH with only the shell and sleep must still return.
        NOTO="$CASE/no timeout bin"
        mkdir -p "$NOTO"
        ln -s "$(command -v sleep)" "$NOTO/sleep"
        ln -s "$(command -v "$shell")" "$NOTO/$shell"
        # Stock macOS ships no GNU timeout and DOES ship ps, which is how the probe learns whether
        # its child got a process group. Leaving ps out made the fixture prove the weaker path.
        [ -n "$(command -v ps)" ] && ln -s "$(command -v ps)" "$NOTO/ps"
        _hang_started=$(date +%s)
        assert_eq "$shell: without GNU timeout a uv that never answers is still not reused" \
            "none" "$(env -i PATH="$NOTO" HOME="$HOME_DIR" UV_INSTALL_DIR="$HANG" "$shell" "$HANG_PROBE")"
        if [ $(( $(date +%s) - _hang_started )) -lt 30 ]; then
            ok "$shell: ...and the fallback bound held"
        else
            bad "$shell: ...and the fallback bound held"
        fi
        # Cancelling setup mid-probe must take the candidate with it. Only the watchdog branch
        # needs this: `timeout` is its own process and enforces the ceiling whatever happens to
        # the shell, while the watchdog's ceiling IS the shell being cancelled.
        CANCEL="$WORK/$shell cancel probe.sh"
        {
            printf '%s\n' "$HELPER"
            printf '_setup_uv_probe_exec "%s/uv"\n' "$HANG"
        } > "$CANCEL"
        env -i PATH="$NOTO" HOME="$HOME_DIR" _SETUP_UV_PROBE_SECONDS=30 \
            "$shell" "$CANCEL" >/dev/null 2>&1 &
        _cancel_sup=$!
        sleep 3
        kill -TERM "$_cancel_sup" 2>/dev/null || :
        # The supervisor dies of the signal, so `wait` reports 143: not a test failure.
        wait "$_cancel_sup" 2>/dev/null || :
        sleep 2
        # Snapshot first and match in the shell: piping into grep puts the pattern on grep's own
        # command line, so `ps | grep` matches itself and the check can never go green.
        _cancel_snap=$(ps -A -o args= 2>/dev/null) || _cancel_snap=""
        case "$_cancel_snap" in
            *"$HANG/uv"*) bad "$shell: cancelling setup kills the probe with it" ;;
            *) ok "$shell: cancelling setup kills the probe with it" ;;
        esac

        # A uv that ignores TERM: both branches must end in KILL.
        DEAF="$CASE/ignores term"
        mkdir -p "$DEAF"
        printf '#!/bin/sh\ntrap "" TERM\nsleep 60\n' > "$DEAF/uv"
        chmod +x "$DEAF/uv"
        for _deaf_path in "$BARE_PATH" "$NOTO"; do
            _hang_started=$(date +%s)
            assert_eq "$shell: a uv that ignores TERM is not reused (PATH=$_deaf_path)" \
                "none" "$(env -i PATH="$_deaf_path" HOME="$HOME_DIR" UV_INSTALL_DIR="$DEAF" "$shell" "$HANG_PROBE")"
            if [ $(( $(date +%s) - _hang_started )) -lt 30 ]; then
                ok "$shell: ...and the KILL escalation held the bound"
            else
                bad "$shell: ...and the KILL escalation held the bound"
            fi
        done
    fi
done

# Source contract: the reuse sits between the PATH probe and the download, in both shells.
_probe_at=$(grep -n '^if command -v uv &>/dev/null; then$' "$SETUP_SH" | head -1 | cut -d: -f1)
_reuse_at=$(grep -n '^elif _setup_find_installed_uv; then$' "$SETUP_SH" | head -1 | cut -d: -f1)
_install_at=$(grep -n 'if _setup_install_uv_pinned; then' "$SETUP_SH" | head -1 | cut -d: -f1)
if [ -n "$_probe_at" ] && [ -n "$_reuse_at" ] && [ -n "$_install_at" ] \
   && [ "$_probe_at" -lt "$_reuse_at" ] && [ "$_reuse_at" -lt "$_install_at" ]; then
    ok "setup.sh reuses an installed uv before it downloads one"
else
    bad "setup.sh reuses an installed uv before it downloads one (probe=$_probe_at reuse=$_reuse_at install=$_install_at)"
fi
_ps_probe_at=$(grep -n '^if (Get-Command uv -ErrorAction SilentlyContinue) {$' "$SETUP_PS1" | head -1 | cut -d: -f1)
_ps_reuse_at=$(grep -n '} elseif ((\$installedUvDir = Find-InstalledUv)) {' "$SETUP_PS1" | head -1 | cut -d: -f1)
_ps_install_at=$(grep -n 'Invoke-SetupCommand { Install-UvFromPinnedRelease }' "$SETUP_PS1" | head -1 | cut -d: -f1)
if [ -n "$_ps_probe_at" ] && [ -n "$_ps_reuse_at" ] && [ -n "$_ps_install_at" ] \
   && [ "$_ps_probe_at" -lt "$_ps_reuse_at" ] && [ "$_ps_reuse_at" -lt "$_ps_install_at" ]; then
    ok "setup.ps1 reuses an installed uv before it downloads one"
else
    bad "setup.ps1 reuses an installed uv before it downloads one (probe=$_ps_probe_at reuse=$_ps_reuse_at install=$_ps_install_at)"
fi
# The same destinations, in the same order, on both sides.
for _name in UV_INSTALL_DIR UV_UNMANAGED_INSTALL XDG_BIN_HOME XDG_DATA_HOME; do
    if printf '%s\n' "$HELPER" | grep -q "$_name" && printf '%s\n' "$PS_FINDER" | grep -q "env:$_name"; then
        ok "both shells consult $_name"
    else
        bad "both shells consult $_name"
    fi
done

# The reused directory goes to the END of PATH in both shells: a python beside uv must not
# step in front of the staged interpreter.
if grep -q '^    export PATH="\$PATH:\$_setup_uv_dir"$' "$SETUP_SH"; then
    ok "setup.sh appends the reused uv directory to PATH"
else
    bad "setup.sh appends the reused uv directory to PATH"
fi
if grep -qF '$env:PATH = "$env:PATH;$installedUvDir"' "$SETUP_PS1"; then
    ok "setup.ps1 appends the reused uv directory to PATH"
else
    bad "setup.ps1 appends the reused uv directory to PATH"
fi

# Only a uv that answered counts on both sides: the bounded probe here, an "ok" verdict in setup.ps1.
if printf '%s\n' "$HELPER" | grep -q '_setup_uv_probe_exec "\$_sfu_dir/uv"'; then
    ok "setup.sh probes the candidate through the bounded helper"
else
    bad "setup.sh probes the candidate through the bounded helper"
fi
if printf '%s\n' "$PS_FINDER" | grep -q -- '-ne "ok") { continue }'; then
    ok "setup.ps1 reuses only a uv with an ok verdict"
else
    bad "setup.ps1 reuses only a uv with an ok verdict"
fi

# ...and on both sides it clears the floor, at the SAME number install.sh keeps: below it uv
# resolves a CPython that cannot import torch, which the installer refuses outright.
_sh_floor=$(grep '^_SETUP_UV_MIN_VERSION="' "$SETUP_SH" | head -1 | sed 's/.*"\(.*\)"/\1/')
_ps_floor=$(grep '^\$SetupUvMinVersion = "' "$SETUP_PS1" | head -1 | sed 's/.*"\(.*\)"/\1/')
_install_floor=$(grep '^UV_MIN_VERSION="' "$SCRIPT_DIR/../../install.sh" | head -1 | sed 's/.*"\(.*\)"/\1/')
if printf '%s\n' "$HELPER" | grep -q '_setup_uv_version_at_least "\$_sfu_ver"'; then
    ok "setup.sh gates the reused uv on a minimum version"
else
    bad "setup.sh gates the reused uv on a minimum version"
fi
if printf '%s\n' "$PS_FINDER" | grep -q 'Test-SetupUvVersionAtLeast'; then
    ok "setup.ps1 gates the reused uv on a minimum version"
else
    bad "setup.ps1 gates the reused uv on a minimum version"
fi
if [ -n "$_sh_floor" ] && [ "$_sh_floor" = "$_ps_floor" ] && [ "$_sh_floor" = "$_install_floor" ]; then
    ok "both shells keep install.sh's floor ($_sh_floor)"
else
    bad "both shells keep install.sh's floor (setup.sh=$_sh_floor setup.ps1=$_ps_floor install.sh=$_install_floor)"
fi

# Get-SetupUvExecutableVerdict returns the verdict only: a Write-Output in it rode along in
# the return value, `-ne "ok"` on that array was true for every probe, and uv was re-downloaded
# on every update. With no cached exit code a printed version is "ok"; a code that is there decides.
_verdict=$(awk '/^function Get-SetupUvExecutableVerdict \{/ { grab = 1 } grab { print } grab && /^\}/ { exit }' "$SETUP_PS1")
if [ -n "$_verdict" ] && ! printf '%s\n' "$_verdict" | grep -q 'Write-Output'; then
    ok "setup.ps1 uv verdict returns only the verdict"
else
    bad "setup.ps1 uv verdict returns only the verdict"
fi
if printf '%s\n' "$_verdict" | grep -q "match '\^uv \\\\d+"; then
    ok "setup.ps1 uv verdict accepts a printed version"
else
    bad "setup.ps1 uv verdict accepts a printed version"
fi

# Candidates use .NET Combine: Join-Path terminates on a missing drive under
# ErrorActionPreference Stop, before the installation branch's try, so one bad XDG_DATA_HOME ended setup.
_finder=$(awk '/^function Find-InstalledUv \{/ { grab = 1 } grab { print } grab && /^\}/ { exit }' "$SETUP_PS1")
if [ -n "$_finder" ] && ! printf '%s\n' "$_finder" | grep -q 'Join-Path' && printf '%s\n' "$_finder" | grep -q 'System.IO.Path\]::Combine'; then
    ok "setup.ps1 builds uv candidates without Join-Path"
else
    bad "setup.ps1 builds uv candidates without Join-Path"
fi

echo ""
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo "FAILED"
    exit 1
fi
echo "ALL PASSED"
