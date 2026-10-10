#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Hosts without a pinned uv build fall back to astral's versioned installer script. It now runs
# only when its sha256 matches the pinned script, so a swapped script is refused like a failed
# download, and a host with no sha256 tool keeps the old behaviour.
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
. "$SCRIPT_DIR/_harness.sh"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT INT TERM

HELPER=$(awk '
    /^_setup_uv_sha256\(\) \{/ { grab = 1 }
    /^_setup_uv_fallback_run\(\) \{/ { grab = 1 }
    grab { print }
    grab && /^}/ { grab = 0 }
' "$SETUP_SH")

# A fake installer that leaves a marker when it runs, served by a stubbed downloader.
printf 'touch "%s/ran"\n' "$WORK" > "$WORK/installer.sh"
GOOD_SUM=$(sha256sum "$WORK/installer.sh" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$WORK/installer.sh" | awk '{print $1}')

_run_fallback() {
    # $1 = pinned sha256, $2 = 1 to hide every sha256 tool
    rm -f "$WORK/ran"
    (
        eval "$HELPER"
        _setup_http_get() { cat "$WORK/installer.sh"; }
        _is_verbose() { return 1; }
        _SETUP_UV_PINNED_VERSION="0.0.0"
        _SETUP_UV_INSTALLER_SH_SHA256="$1"
        if [ "$2" = 1 ]; then
            _setup_uv_sha256() { :; }
        fi
        _setup_uv_fallback_run
    ) 2>"$WORK/err"
}

if _run_fallback "$GOOD_SUM" 0 && [ -f "$WORK/ran" ]; then
    ok "the pinned installer script runs"
else
    bad "the pinned installer script runs"
fi

if ! _run_fallback "0000000000000000000000000000000000000000000000000000000000000000" 0 && [ ! -f "$WORK/ran" ]; then
    ok "a script with a different sha256 is refused and never runs"
else
    bad "a script with a different sha256 is refused and never runs"
fi
assert_contains "the refusal says why" "$(cat "$WORK/err")" "failed its sha256 check"

if _run_fallback "0000000000000000000000000000000000000000000000000000000000000000" 1 && [ -f "$WORK/ran" ]; then
    ok "a host with no sha256 tool runs it as before"
else
    bad "a host with no sha256 tool runs it as before"
fi

_setup_pin=$(sed -n 's/^_SETUP_UV_INSTALLER_SH_SHA256="\([0-9a-f]*\)"$/\1/p' "$SETUP_SH")
_install_pin=$(sed -n 's/^UV_INSTALLER_SH_SHA256="\([0-9a-f]*\)"$/\1/p' "$INSTALL_SH")
if [ -n "$_setup_pin" ] && [ "$_setup_pin" = "$_install_pin" ]; then
    ok "install.sh and setup.sh pin the same installer script"
else
    bad "install.sh and setup.sh pin the same installer script"
fi

# install.sh checks the digest before its run_maybe_quiet sh line.
_check=$(grep -n 'UV_INSTALLER_SH_SHA256"' "$INSTALL_SH" | grep -v '^[0-9]*:UV_INSTALLER' | head -1 | cut -d: -f1)
_run=$(grep -n 'run_maybe_quiet sh "\$_uv_tmp"' "$INSTALL_SH" | head -1 | cut -d: -f1)
if [ -n "$_check" ] && [ -n "$_run" ] && [ "$_check" -lt "$_run" ]; then
    ok "install.sh checks the installer digest before running it"
else
    bad "install.sh checks the installer digest before running it"
fi

echo ""
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo "FAILED"
    exit 1
fi
echo "ALL PASSED"
