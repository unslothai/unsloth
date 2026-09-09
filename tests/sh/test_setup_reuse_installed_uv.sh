#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# studio/setup.sh decides whether it has uv with `command -v uv`. When the uv a previous
# run installed is not on THIS process's PATH -- a desktop shell launched before the
# install, a CI step with a fresh PATH, a login shell whose profile line is unread --
# that miss used to re-download the pinned archive on every update: 19 MB, and 42 of the
# 53 seconds a Windows no-op update took on the staging matrix. It now looks at astral's
# destination priority list first, and only a uv that runs counts.
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/_harness.sh"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
SETUP_PS1="$SCRIPT_DIR/../../studio/setup.ps1"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT INT TERM

HELPER=$(awk '
    /^_setup_find_installed_uv\(\) \{/ { grab = 1 }
    grab { print }
    grab && /^}/ { grab = 0; exit }
' "$SETUP_SH")
printf '%s\n' "$HELPER" | grep -q '^_setup_find_installed_uv() {' || {
    echo "FATAL: could not extract _setup_find_installed_uv from setup.sh" >&2; exit 1; }

PROBE="$WORK/probe.sh"
{
    printf '%s\n' "$HELPER"
    cat <<'BODY'
if _dir=$(_setup_find_installed_uv); then printf 'found=%s' "$_dir"; else printf 'none'; fi
BODY
} > "$PROBE"

fake_uv() {  # fake_uv <dir> [exit code]
    mkdir -p "$1"
    printf '#!/bin/sh\nexit %s\n' "${2:-0}" > "$1/uv"
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

    # Present but broken: a binary for another architecture, a half-written download. It
    # must not be reused, or the rest of setup runs against a uv that cannot answer.
    BROKEN="$CASE/broken"
    fake_uv "$BROKEN" 1
    assert_eq "$shell: a uv that cannot run is not reused" \
        "found=$HOME_DIR/.local/bin" "$(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" UV_INSTALL_DIR="$BROKEN" "$shell" "$PROBE")"
    rm -f "$HOME_DIR/.local/bin/uv"
    assert_eq "$shell: ...and with nothing else installed that is a miss" \
        "none" "$(env -i PATH="$BARE_PATH" HOME="$HOME_DIR" UV_INSTALL_DIR="$BROKEN" "$shell" "$PROBE")"
done

# ── source contract: the reuse sits between the PATH probe and the download, in both shells ──
_probe_at=$(grep -n '^if command -v uv &>/dev/null; then$' "$SETUP_SH" | head -1 | cut -d: -f1)
_reuse_at=$(grep -n '^elif _setup_uv_dir=\$(_setup_find_installed_uv); then$' "$SETUP_SH" | head -1 | cut -d: -f1)
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
    if printf '%s\n' "$HELPER" | grep -q "$_name" && grep -A12 '^function Find-InstalledUv {' "$SETUP_PS1" | grep -q "env:$_name"; then
        ok "both shells consult $_name"
    else
        bad "both shells consult $_name"
    fi
done

echo ""
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo "FAILED"
    exit 1
fi
echo "ALL PASSED"
