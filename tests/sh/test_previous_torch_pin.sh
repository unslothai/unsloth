#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit tests for install.sh's _previous_torch_pin, which keeps the previous
# venv's torch release on a re-run (curl | sh over an existing install) instead
# of silently moving the user to a newer release. Helpers are extracted from
# install.sh and sourced.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

# Extract _previous_torch_pin and its dependencies _torch_flavor_tag and
# _torch_release_in_window.
_FUNC_FILE=$(mktemp)
{
    sed -n '/^_torch_flavor_tag()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_torch_release_in_window()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_previous_torch_pin()/,/^}/p' "$INSTALL_SH"
} > "$_FUNC_FILE"
# shellcheck disable=SC1090
. "$_FUNC_FILE"
rm -f "$_FUNC_FILE"

assert_eq() {
    _label="$1"; _expected="$2"; _actual="$3"
    if [ "$_actual" = "$_expected" ]; then
        echo "  PASS: $_label"; PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected '$_expected', got '$_actual')"; FAIL=$((FAIL + 1))
    fi
}

unset UNSLOTH_TORCH_UPGRADE

echo "=== _previous_torch_pin: matching flavor keeps the release ==="
assert_eq "cu126 wheel on cu126 leaf"     "torch==2.10.0" "$(_previous_torch_pin '2.10.0+cu126' 'cu126' 'torch>=2.4,<2.12.0')"
assert_eq "cu130 wheel on cu130 leaf"     "torch==2.10.0" "$(_previous_torch_pin '2.10.0+cu130' 'cu130' 'torch>=2.4,<2.12.0')"
assert_eq "cpu wheel on cpu leaf"         "torch==2.10.0" "$(_previous_torch_pin '2.10.0+cpu' 'cpu' 'torch>=2.4,<2.12.0')"
assert_eq "untagged wheel on cpu leaf"    "torch==2.10.0" "$(_previous_torch_pin '2.10.0' 'cpu' 'torch>=2.4,<2.12.0')"
assert_eq "local suffix stripped"         "torch==2.9.1"  "$(_previous_torch_pin '2.9.1+cu128' 'cu128' 'torch>=2.4,<2.12.0')"

echo "=== _previous_torch_pin: flavor change installs the new build ==="
assert_eq "cu126 wheel on cu130 leaf"     "" "$(_previous_torch_pin '2.10.0+cu126' 'cu130' 'torch>=2.4,<2.12.0')"
assert_eq "cpu wheel on cu126 leaf"       "" "$(_previous_torch_pin '2.10.0+cpu' 'cu126' 'torch>=2.4,<2.12.0')"
assert_eq "cu126 wheel on cpu leaf"       "" "$(_previous_torch_pin '2.10.0+cu126' 'cpu' 'torch>=2.4,<2.12.0')"

echo "=== _previous_torch_pin: rocm and unknown leaves never pin ==="
assert_eq "rocm7.2 leaf keeps its floor"  "" "$(_previous_torch_pin '2.11.0+rocm7.2' 'rocm7.2' 'torch>=2.4,<2.12.0')"
assert_eq "gfx leaf keeps its floor"      "" "$(_previous_torch_pin '2.11.0+rocm7.2' 'gfx120X-all' 'torch>=2.4,<2.12.0')"
assert_eq "unknown mirror leaf"           "" "$(_previous_torch_pin '2.10.0+cu126' 'simple' 'torch>=2.4,<2.12.0')"

echo "=== _previous_torch_pin: probe noise never becomes a pin ==="
assert_eq "empty version"                 "" "$(_previous_torch_pin '' 'cu126' 'torch>=2.4,<2.12.0')"
assert_eq "garbage version"               "" "$(_previous_torch_pin 'not-a-version' 'cpu' 'torch>=2.4,<2.12.0')"
assert_eq "traceback fragment"            "" "$(_previous_torch_pin "ModuleNotFoundError: No module named 'torch'" 'cpu' 'torch>=2.4,<2.12.0')"

echo "=== _previous_torch_pin: out-of-window releases never pin ==="
assert_eq "2.3.x below the cu floor"      "" "$(_previous_torch_pin '2.3.1+cu118' 'cu118' 'torch>=2.4,<2.12.0')"
assert_eq "2.12.x above the cu ceiling"   "" "$(_previous_torch_pin '2.12.0+cu130' 'cu130' 'torch>=2.4,<2.12.0')"
assert_eq "floor boundary 2.4.0 kept"     "torch==2.4.0"  "$(_previous_torch_pin '2.4.0+cu126' 'cu126' 'torch>=2.4,<2.12.0')"
assert_eq "ceiling-adjacent 2.11.x kept"  "torch==2.11.1" "$(_previous_torch_pin '2.11.1+cu130' 'cu130' 'torch>=2.4,<2.12.0')"
assert_eq "cpu window excludes 2.11.x"    "" "$(_previous_torch_pin '2.11.0+cpu' 'cpu' 'torch>=2.4,<2.11.0')"
assert_eq "mac floor excludes 2.5.x"      "" "$(_previous_torch_pin '2.5.1' 'cpu' 'torch>=2.6,<2.11.0')"
assert_eq "malformed window never pins"   "" "$(_previous_torch_pin '2.10.0+cu126' 'cu126' 'torch')"
assert_eq "empty window never pins"       "" "$(_previous_torch_pin '2.10.0+cu126' 'cu126' '')"

echo "=== _torch_release_in_window ==="
assert_eq "in window"            "yes" "$(_torch_release_in_window '2.10.0' 'torch>=2.4,<2.12.0')"
assert_eq "at floor"             "yes" "$(_torch_release_in_window '2.4.0' 'torch>=2.4,<2.12.0')"
assert_eq "below floor"          "no"  "$(_torch_release_in_window '2.3.1' 'torch>=2.4,<2.12.0')"
assert_eq "at ceiling"           "no"  "$(_torch_release_in_window '2.12.0' 'torch>=2.4,<2.12.0')"
assert_eq "next major"           "no"  "$(_torch_release_in_window '3.0.0' 'torch>=2.4,<2.12.0')"
assert_eq "patch-level floor"    "yes" "$(_torch_release_in_window '2.11.5' 'torch>=2.11.0,<2.12.0')"
assert_eq "no ceiling -> no"     "no"  "$(_torch_release_in_window '2.10.0' 'torch>=2.4')"
assert_eq "garbage minor -> no"  "no"  "$(_torch_release_in_window '2.x' 'torch>=2.4,<2.12.0')"

echo "=== _previous_torch_pin: UNSLOTH_TORCH_UPGRADE=1 opts out ==="
assert_eq "upgrade env set"    "" "$(UNSLOTH_TORCH_UPGRADE=1 _previous_torch_pin '2.10.0+cu126' 'cu126' 'torch>=2.4,<2.12.0')"
assert_eq "upgrade env 0"      "torch==2.10.0" "$(UNSLOTH_TORCH_UPGRADE=0 _previous_torch_pin '2.10.0+cu126' 'cu126' 'torch>=2.4,<2.12.0')"

echo "=== install.sh wiring ==="
# The probe must run against the OLD venv, before it is moved aside for rollback.
_probe_line=$(grep -n '_PREV_TORCH_VER=\$(' "$INSTALL_SH" | head -1 | cut -d: -f1)
_move_line=$(grep -n '_start_studio_venv_replacement "\$VENV_DIR"' "$INSTALL_SH" | head -1 | cut -d: -f1)
assert_eq "probe exists"                  "yes" "$([ -n "$_probe_line" ] && echo yes)"
assert_eq "probe before venv replacement" "yes" "$([ -n "$_probe_line" ] && [ -n "$_move_line" ] && [ "$_probe_line" -lt "$_move_line" ] && echo yes)"
# A kept release that vanished from the index must fall back to the supported range.
assert_eq "resolve-failure fallback wired" "yes" "$(grep -q 'TORCH_CONSTRAINT="\$_PREV_FALLBACK_CONSTRAINT"' "$INSTALL_SH" && echo yes)"
assert_eq "pin gated on SKIP_TORCH"        "yes" "$(grep -q 'if \[ "\$SKIP_TORCH" = false \]; then' "$INSTALL_SH" && echo yes)"

echo ""
if [ "$FAIL" -gt 0 ]; then
    echo "$FAIL check(s) FAILED"
    exit 1
fi
echo "All $PASS checks passed"
