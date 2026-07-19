#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit tests for install.sh's _previous_torch_pin, which keeps the previous
# venv's torch RELEASE on a re-run (curl | sh over an existing install) instead
# of silently moving the user to a newer release. The release is preserved
# regardless of the old build's flavor tag (PyPI bare, +cuXXX, +rocm, +cpu):
# the pin installs from the freshly chosen index, so the flavor follows the
# machine while the release follows the user. Per-leaf constraint windows
# still win (rocm7.2 / Strix floors, out-of-window manual installs). Helpers
# are extracted from install.sh and sourced.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

# Extract _previous_torch_pin and its dependency _torch_release_in_window.
_FUNC_FILE=$(mktemp)
{
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

echo "=== _previous_torch_pin: in-window releases are kept, any flavor ==="
assert_eq "cu126 wheel"                  "torch==2.10.0" "$(_previous_torch_pin '2.10.0+cu126' 'torch>=2.4,<2.12.0')"
assert_eq "cu130 wheel"                  "torch==2.10.0" "$(_previous_torch_pin '2.10.0+cu130' 'torch>=2.4,<2.12.0')"
assert_eq "cpu wheel"                    "torch==2.10.0" "$(_previous_torch_pin '2.10.0+cpu' 'torch>=2.4,<2.12.0')"
assert_eq "PyPI bare version (CUDA build on Linux)" "torch==2.10.0" "$(_previous_torch_pin '2.10.0' 'torch>=2.4,<2.12.0')"
assert_eq "rocm wheel"                   "torch==2.10.0" "$(_previous_torch_pin '2.10.0+rocm6.4' 'torch>=2.4,<2.11.0')"
assert_eq "local suffix stripped"        "torch==2.9.1"  "$(_previous_torch_pin '2.9.1+cu128' 'torch>=2.4,<2.12.0')"

echo "=== _previous_torch_pin: raised floors reject older releases ==="
# rocm7.2 / Strix gfx leaves raise TORCH_CONSTRAINT to >=2.11.0 BEFORE the pin
# is evaluated, so an old 2.10 is out of window there and the floor wins.
assert_eq "old 2.10 vs rocm7.2 floor"    "" "$(_previous_torch_pin '2.10.0+rocm7.1' 'torch>=2.11.0,<2.12.0')"
assert_eq "2.11 passes the rocm7.2 floor" "torch==2.11.0" "$(_previous_torch_pin '2.11.0+rocm7.2' 'torch>=2.11.0,<2.12.0')"

echo "=== _previous_torch_pin: probe noise never becomes a pin ==="
assert_eq "empty version"                "" "$(_previous_torch_pin '' 'torch>=2.4,<2.12.0')"
assert_eq "garbage version"              "" "$(_previous_torch_pin 'not-a-version' 'torch>=2.4,<2.12.0')"
assert_eq "traceback fragment"           "" "$(_previous_torch_pin "ModuleNotFoundError: No module named 'torch'" 'torch>=2.4,<2.12.0')"

echo "=== _previous_torch_pin: out-of-window releases never pin ==="
assert_eq "2.3.x below the cu floor"     "" "$(_previous_torch_pin '2.3.1+cu118' 'torch>=2.4,<2.12.0')"
assert_eq "2.12.x above the cu ceiling"  "" "$(_previous_torch_pin '2.12.0+cu130' 'torch>=2.4,<2.12.0')"
assert_eq "floor boundary 2.4.0 kept"    "torch==2.4.0"  "$(_previous_torch_pin '2.4.0+cu126' 'torch>=2.4,<2.12.0')"
assert_eq "ceiling-adjacent 2.11.x kept" "torch==2.11.1" "$(_previous_torch_pin '2.11.1+cu130' 'torch>=2.4,<2.12.0')"
assert_eq "cpu window excludes 2.11.x"   "" "$(_previous_torch_pin '2.11.0+cpu' 'torch>=2.4,<2.11.0')"
assert_eq "mac floor excludes 2.5.x"     "" "$(_previous_torch_pin '2.5.1' 'torch>=2.6,<2.11.0')"
assert_eq "malformed window never pins"  "" "$(_previous_torch_pin '2.10.0+cu126' 'torch')"
assert_eq "empty window never pins"      "" "$(_previous_torch_pin '2.10.0+cu126' '')"

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
assert_eq "upgrade env set"    "" "$(UNSLOTH_TORCH_UPGRADE=1 _previous_torch_pin '2.10.0+cu126' 'torch>=2.4,<2.12.0')"
assert_eq "upgrade env 0"      "torch==2.10.0" "$(UNSLOTH_TORCH_UPGRADE=0 _previous_torch_pin '2.10.0+cu126' 'torch>=2.4,<2.12.0')"

echo "=== install.sh wiring ==="
# The probe must run against the OLD venv, before it is moved aside for rollback.
_probe_line=$(grep -n '_PREV_TORCH_VER=\$(' "$INSTALL_SH" | head -1 | cut -d: -f1)
_move_line=$(grep -n '_start_studio_venv_replacement "\$VENV_DIR"' "$INSTALL_SH" | head -1 | cut -d: -f1)
assert_eq "probe exists"                  "yes" "$([ -n "$_probe_line" ] && echo yes)"
assert_eq "probe before venv replacement" "yes" "$([ -n "$_probe_line" ] && [ -n "$_move_line" ] && [ "$_probe_line" -lt "$_move_line" ] && echo yes)"
# The pin must be evaluated AFTER the last index/constraint decision (the Strix
# reroute raises the floor), so a raised floor rejects an older kept release.
_pin_line=$(grep -n '_prev_pin=\$(_previous_torch_pin' "$INSTALL_SH" | head -1 | cut -d: -f1)
_strix_line=$(grep -n 'Strix Halo / Strix Point: force rocm7.2 wheels' "$INSTALL_SH" | head -1 | cut -d: -f1)
assert_eq "pin evaluated after the Strix reroute" "yes" "$([ -n "$_pin_line" ] && [ -n "$_strix_line" ] && [ "$_pin_line" -gt "$_strix_line" ] && echo yes)"
# A kept release that vanished from the index must fall back to the supported range.
assert_eq "resolve-failure fallback wired" "yes" "$(grep -q 'TORCH_CONSTRAINT="\$_PREV_FALLBACK_CONSTRAINT"' "$INSTALL_SH" && echo yes)"
assert_eq "pin gated on SKIP_TORCH"        "yes" "$(grep -q 'if \[ "\$SKIP_TORCH" = false \]; then' "$INSTALL_SH" && echo yes)"
# Every --default-index torch install path must go through the kept-release
# helper (definition + default path + three ROCm-index fallbacks).
_helper_uses=$(grep -c '_install_torch_default_index' "$INSTALL_SH")
assert_eq "kept-release helper used by all default-index paths" "yes" "$([ "$_helper_uses" -ge 5 ] && echo yes)"
# The Radeon direct-wheel path must also honor the pin: the trio search is
# clamped to the kept release's minor when the repo still offers it, inside
# the versions-match block (i.e. between the initial target computation and
# the downward search loop).
_radeon_clamp_line=$(grep -n '_prev_kept_minor="\${_PREV_TORCH_PIN#torch==}"' "$INSTALL_SH" | head -1 | cut -d: -f1)
_radeon_target_line=$(grep -n '_target_minor=\$_torch_minor' "$INSTALL_SH" | head -1 | cut -d: -f1)
_radeon_loop_line=$(grep -n 'Loop downwards to find the first complete matching trio' "$INSTALL_SH" | head -1 | cut -d: -f1)
assert_eq "Radeon trio search honors the kept release" "yes" "$([ -n "$_radeon_clamp_line" ] && [ -n "$_radeon_target_line" ] && [ -n "$_radeon_loop_line" ] && [ "$_radeon_clamp_line" -gt "$_radeon_target_line" ] && [ "$_radeon_clamp_line" -lt "$_radeon_loop_line" ] && echo yes)"

echo ""
if [ "$FAIL" -gt 0 ]; then
    echo "$FAIL check(s) FAILED"
    exit 1
fi
echo "All $PASS checks passed"
