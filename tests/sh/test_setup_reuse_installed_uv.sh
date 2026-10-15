#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# A uv off THIS process's PATH made setup.sh re-download it every update. Astral's
# destinations are searched first; only a uv that runs and is at or above the floor counts.
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/_harness.sh"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
SETUP_PS1="$SCRIPT_DIR/../../studio/setup.ps1"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT INT TERM

HELPER=$(awk '
    /^_setup_probe_signal_target\(\) \{/ { grab = 1 }
    /^_setup_probe_terminate\(\) \{/ { grab = 1 }
    /^_setup_probe_restore_trap\(\) \{/ { grab = 1 }
    /^_setup_probe_on_signal\(\) \{/ { grab = 1 }
    /^_setup_probe_version\(\) \{/ { grab = 1 }
    /^_setup_uv_version_at_least\(\) \{/ { grab = 1 }
    /^_setup_find_installed_uv\(\) \{/ { grab = 1 }
    grab { print }
    grab && /^}/ { grab = 0 }
' "$SETUP_SH")
# The floor lives beside the function, not inside it.
HELPER="$(grep '^_SETUP_UV_MIN_VERSION=' "$SETUP_SH")
$HELPER"
for _fn in _setup_probe_signal_target _setup_probe_terminate _setup_probe_restore_trap _setup_probe_on_signal _setup_probe_version \
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
# Miss diagnostics survive only when the finder runs in the caller's shell.
DIAG="$WORK/diag.sh"
{
    printf '%s\n' "$HELPER"
    cat <<'BODY'
if _setup_find_installed_uv; then printf 'found'; else printf 'looked=%s miss=%s old=%s' "$_SETUP_UV_LOOKED" "$_SETUP_UV_PROBE_MISS" "$_SETUP_UV_TOO_OLD"; fi
BODY
} > "$DIAG"
PS_FINDER=$(awk '
    /^function Find-InstalledUv \{/ { grab = 1 }
    grab { print }
    grab && /^\}/ { grab = 0 }
' "$SETUP_PS1")
printf '%s\n' "$PS_FINDER" | grep -q '^function Find-InstalledUv {' || {
    echo "FATAL: could not extract Find-InstalledUv from setup.ps1" >&2; exit 1; }

fake_uv() {  # fake_uv <dir> [exit code] [version]
    # It prints a version because the finder reads one; a silent stand-in would be refused.
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

    BROKEN="$CASE/broken"
    fake_uv "$BROKEN" 1
    assert_eq "$shell: a uv that cannot run is not reused" \
        "found=$HOME_DIR/.local/bin" "$(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" UV_INSTALL_DIR="$BROKEN" "$shell" "$PROBE")"
    rm -f "$HOME_DIR/.local/bin/uv"
    assert_eq "$shell: ...and with nothing else installed that is a miss" \
        "none" "$(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" UV_INSTALL_DIR="$BROKEN" "$shell" "$PROBE")"

    # Below the floor, uv's managed-Python manifest tops out at a CPython that cannot import torch.
    OLD="$CASE/old uv"
    fake_uv "$OLD" 0 0.4.0
    assert_eq "$shell: a uv below the minimum version is not reused" \
        "none" "$(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" UV_INSTALL_DIR="$OLD" "$shell" "$PROBE")"
    case "$(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" UV_INSTALL_DIR="$OLD" "$shell" "$DIAG")" in
        *"old=$OLD/uv"*) ok "$shell: ...and the diagnostic names the version it refused" ;;
        *) bad "$shell: ...and the diagnostic names the version it refused ($(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" UV_INSTALL_DIR="$OLD" "$shell" "$DIAG"))" ;;
    esac
    fake_uv "$OLD" 0 0.9.3
    assert_eq "$shell: a uv exactly at the minimum version is reused" \
        "found=$OLD" "$(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" UV_INSTALL_DIR="$OLD" "$shell" "$PROBE")"
    # A prerelease of the floor is below it; install.sh's _uv_version_ok refuses it too.
    fake_uv "$OLD" 0 "0.9.3-rc.1"
    assert_eq "$shell: a prerelease of the minimum version is not reused" \
        "none" "$(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" UV_INSTALL_DIR="$OLD" "$shell" "$PROBE")"
    fake_uv "$OLD" 0 "0.9.4-rc1"
    assert_eq "$shell: a prerelease above the minimum version is reused" \
        "found=$OLD" "$(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" UV_INSTALL_DIR="$OLD" "$shell" "$PROBE")"
    # Runs but answers as something else; a Windows stand-in cannot express this.
    NOTUV="$CASE/not uv"
    mkdir -p "$NOTUV"
    printf '#!/bin/sh\necho "curl 8.9.1 (x86_64-pc-linux-gnu)"\nexit 0\n' > "$NOTUV/uv"
    chmod +x "$NOTUV/uv"
    assert_eq "$shell: a binary that runs but does not answer as uv is not reused" \
        "none" "$(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" UV_INSTALL_DIR="$NOTUV" "$shell" "$PROBE")"

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

    # Not gated on `command -v timeout`: stock macOS has none and would skip the watchdog branch.
    if true; then
        HANG="$CASE/hangs"
        mkdir -p "$HANG"
        printf '#!/bin/sh\nsleep 60\n' > "$HANG/uv"
        chmod +x "$HANG/uv"
        HANG_PROBE="$WORK/$shell hang probe.sh"
        # The 20 s ceiling is the helper's; the test only needs it finite.
        { echo '_SETUP_PROBE_SECONDS=2'; cat "$PROBE"; } > "$HANG_PROBE"
        _hang_started=$(date +%s)
        assert_eq "$shell: a uv that never answers is not reused" \
            "none" "$(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" UV_INSTALL_DIR="$HANG" "$shell" "$HANG_PROBE")"
        if [ $(( $(date +%s) - _hang_started )) -lt 30 ]; then
            ok "$shell: ...and the probe returned within its bound"
        else
            bad "$shell: ...and the probe returned within its bound"
        fi
        NOTO="$CASE/no timeout bin"
        mkdir -p "$NOTO"
        ln -s "$(command -v sleep)" "$NOTO/sleep"
        ln -s "$(command -v "$shell")" "$NOTO/$shell"
        # Stock macOS ships ps, which the probe uses to learn whether its child got a process group.
        [ -n "$(command -v ps)" ] && ln -s "$(command -v ps)" "$NOTO/ps"
        _hang_started=$(date +%s)
        assert_eq "$shell: without GNU timeout a uv that never answers is still not reused" \
            "none" "$(env -i PATH="$NOTO" HOME="$HOME_DIR" UV_INSTALL_DIR="$HANG" "$shell" "$HANG_PROBE")"
        if [ $(( $(date +%s) - _hang_started )) -lt 30 ]; then
            ok "$shell: ...and the fallback bound held"
        else
            bad "$shell: ...and the fallback bound held"
        fi
        # Only the watchdog branch needs this: `timeout` enforces its ceiling even if the shell dies.
        CANCEL="$WORK/$shell cancel probe.sh"
        {
            printf '%s\n' "$HELPER"
            printf '_setup_probe_version "%s/uv"\n' "$HANG"
        } > "$CANCEL"
        env -i PATH="$NOTO" HOME="$HOME_DIR" _SETUP_PROBE_SECONDS=30 \
            "$shell" "$CANCEL" >/dev/null 2>&1 &
        _cancel_sup=$!
        sleep 3
        kill -TERM "$_cancel_sup" 2>/dev/null || :
        # The supervisor dies of the signal, so `wait` reports 143.
        wait "$_cancel_sup" 2>/dev/null || :
        sleep 2
        # Snapshot then match in the shell: `ps | grep` matches itself.
        _cancel_snap=$(ps -A -o args= 2>/dev/null) || _cancel_snap=""
        case "$_cancel_snap" in
            *"$HANG/uv"*) bad "$shell: cancelling setup kills the probe with it" ;;
            *) ok "$shell: cancelling setup kills the probe with it" ;;
        esac

        DEAF="$CASE/ignores term"
        mkdir -p "$DEAF"
        # A loop, not one long sleep: TERM ends `sleep`, so the stand-in would not survive TERM.
        printf '#!/bin/sh\ntrap "" TERM\n_i=0\nwhile [ "$_i" -lt 60 ]; do sleep 1; _i=$((_i + 1)); done\n' > "$DEAF/uv"
        chmod +x "$DEAF/uv"
        # Tested on the shared terminate routine (TERM, grace, KILL), not by cancelling setup: orphan
        # lifetime after the shell dies is the host's business.
        TERMINATE="$WORK/$shell terminate.sh"
        DEAF_READY="$CASE/deaf is deaf"
        # It announces itself only after TERM is ignored, so the escalation is really exercised.
        printf '#!/bin/sh\ntrap "" TERM\n: > "%s"\n_i=0\nwhile [ "$_i" -lt 20 ]; do sleep 1; _i=$((_i + 1)); done\n' \
            "$DEAF_READY" > "$DEAF/ready uv"
        chmod +x "$DEAF/ready uv"
        rm -f "$DEAF_READY"
        {
            printf '%s\n' "$HELPER"
            printf '"%s/ready uv" --version >/dev/null 2>&1 </dev/null &\n' "$DEAF"
            printf '_t_ready="%s"\n' "$DEAF_READY"
            cat <<'BODY'
_t_pid=$!
_t_waited=0
while [ ! -f "$_t_ready" ] && [ "$_t_waited" -lt 10 ]; do
    sleep 1
    _t_waited=$((_t_waited + 1))
done
[ -f "$_t_ready" ] || { printf 'never armed'; exit 0; }
_setup_probe_terminate "$_t_pid" "$_t_pid" 1
# How it ended, not whether it is still listed: a killed child sits as a zombie until it is
# waited for, and `kill -0` answers yes for one of those.
wait "$_t_pid" 2>/dev/null
printf '%s' "$?"
BODY
        } > "$TERMINATE"
        # 137 is 128 + SIGKILL.
        assert_eq "$shell: a probe that ignores TERM is escalated to KILL" \
            "137" "$(env -i PATH="$NOTO" HOME="$HOME_DIR" "$shell" "$TERMINATE")"
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
_install_at=$(grep -n 'if _setup_install_uv_pinned[ ;]' "$SETUP_SH" | head -1 | cut -d: -f1)
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
for _name in UV_INSTALL_DIR UV_UNMANAGED_INSTALL XDG_BIN_HOME XDG_DATA_HOME; do
    if printf '%s\n' "$HELPER" | grep -q "$_name" && printf '%s\n' "$PS_FINDER" | grep -q "env:$_name"; then
        ok "both shells consult $_name"
    else
        bad "both shells consult $_name"
    fi
done

# The reused dir goes to the END of PATH so a python beside uv cannot shadow the staged one.
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

if printf '%s\n' "$HELPER" | grep -q '_setup_probe_version "\$_sfu_dir/uv"'; then
    ok "setup.sh probes the candidate through the bounded helper"
else
    bad "setup.sh probes the candidate through the bounded helper"
fi
if printf '%s\n' "$PS_FINDER" | grep -q -- '-ne "ok") { continue }'; then
    ok "setup.ps1 reuses only a uv with an ok verdict"
else
    bad "setup.ps1 reuses only a uv with an ok verdict"
fi

# Each side clears its platform's installer floor; the numbers differ on purpose (install.ps1
# excludes the bad CPython via $PythonSkip and stays at 0.8.16).
_sh_floor=$(grep '^_SETUP_UV_MIN_VERSION="' "$SETUP_SH" | head -1 | sed 's/.*"\(.*\)"/\1/')
_ps_floor=$(grep '^\$SetupUvMinVersion = "' "$SETUP_PS1" | head -1 | sed 's/.*"\(.*\)"/\1/')
_install_floor=$(grep '^UV_MIN_VERSION="' "$SCRIPT_DIR/../../install.sh" | head -1 | sed 's/.*"\(.*\)"/\1/')
_install_ps_floor=$(grep '^    \$UvMinVersion = "' "$SCRIPT_DIR/../../install.ps1" | head -1 | sed 's/.*"\(.*\)"/\1/')
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
if [ -n "$_sh_floor" ] && [ "$_sh_floor" = "$_install_floor" ]; then
    ok "setup.sh keeps install.sh's floor ($_sh_floor)"
else
    bad "setup.sh keeps install.sh's floor (setup.sh=$_sh_floor install.sh=$_install_floor)"
fi
if [ -n "$_ps_floor" ] && [ "$_ps_floor" = "$_install_ps_floor" ]; then
    ok "setup.ps1 keeps install.ps1's floor ($_ps_floor)"
else
    bad "setup.ps1 keeps install.ps1's floor (setup.ps1=$_ps_floor install.ps1=$_install_ps_floor)"
fi

# A Write-Output in the verdict function would ride along in its return value and force a
# re-download every update.
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

# Join-Path terminates on a missing drive under ErrorActionPreference Stop, so use .NET Combine.
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
