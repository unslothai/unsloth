#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit test for custom/env-mode stable-diffusion.cpp removal in scripts/uninstall.sh.
#
# A custom Studio (UNSLOTH_STUDIO_HOME=<root>) installs its native diffusion build beside
# the root at <parent>/stable-diffusion.cpp -- find_sd_cpp_binary resolves it from
# UNSLOTH_STUDIO_HOME.parent (sd_cpp_engine.py). Uninstall must remove that sibling too, or
# a stale build lingers and a fresh install's finder can pick it up. Tested hermetically:
# the real custom-root removal loop + its helpers are extracted from uninstall.sh and run
# against per-test fixtures. Follows the extract-via-sed pattern of test_uninstall_shared_icon.sh.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UNINSTALL_SH="$SCRIPT_DIR/../../scripts/uninstall.sh"
PASS=0
FAIL=0

_TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$_TMP_ROOT"' EXIT
# Deterministic deny-list checks: keep $HOME clear of the fixture trees.
HOME="$_TMP_ROOT/home"
mkdir -p "$HOME"

assert_nodir() { _l="$1"; [ -d "$2" ] && { echo "  FAIL: $_l (still present: $2)"; FAIL=$((FAIL+1)); } || { echo "  PASS: $_l"; PASS=$((PASS+1)); }; }
assert_dir()   { _l="$1"; [ -d "$2" ] && { echo "  PASS: $_l"; PASS=$((PASS+1)); } || { echo "  FAIL: $_l (missing dir $2)"; FAIL=$((FAIL+1)); }; }

# Extract the helpers the loop depends on, plus the real custom-root removal loop.
HELPERS_FILE=$(mktemp -p "$_TMP_ROOT")
{
    sed -n '/^_remove_path() {/,/^}/p'      "$UNINSTALL_SH"
    sed -n '/^_is_studio_root() {/,/^}/p'   "$UNINSTALL_SH"
    sed -n '/^_is_unsafe_root() {/,/^}/p'   "$UNINSTALL_SH"
} > "$HELPERS_FILE"
LOOP_FILE=$(mktemp -p "$_TMP_ROOT")
sed -n '/^_custom_studio_roots | while IFS= read -r _custom_root; do/,/^done/p' "$UNINSTALL_SH" > "$LOOP_FILE"
# The real default-mode ~/.unsloth/stable-diffusion.cpp removal block (marker-guarded).
DEFAULT_FILE=$(mktemp -p "$_TMP_ROOT")
sed -n '/^_default_sd_cpp="\$HOME\/\.unsloth\/stable-diffusion\.cpp"/,/^fi/p' "$UNINSTALL_SH" > "$DEFAULT_FILE"

# shellcheck disable=SC1090
. "$HELPERS_FILE"

# make_studio <root> : a valid custom Studio root (share/studio.conf owner marker) plus its
# sibling <parent>/stable-diffusion.cpp build carrying the Studio owner marker, each with a
# file so removal is observable.
make_studio() {
    mkdir -p "$1/share"
    : > "$1/share/studio.conf"
    _sib="$(dirname "$1")/stable-diffusion.cpp"
    mkdir -p "$_sib"
    : > "$_sib/sd-cli"
    : > "$_sib/.unsloth-studio-owned"  # written by install_sd_cpp_prebuilt on a real install
}
run_loop() {
    # shellcheck disable=SC1090
    . "$LOOP_FILE"
}
run_default_removal() {
    # shellcheck disable=SC1090
    . "$DEFAULT_FILE"
}

# 1. Single custom root -> root AND its sibling stable-diffusion.cpp both removed.
p1="$_TMP_ROOT/inst1"
make_studio "$p1/studioA"
: > "$p1/keep.txt"  # unrelated sibling file must be untouched
_custom_studio_roots() { printf '%s\n' "$p1/studioA"; }
run_loop
assert_nodir "single custom root removed"                 "$p1/studioA"
assert_nodir "custom-root sibling stable-diffusion.cpp removed" "$p1/stable-diffusion.cpp"
[ -f "$p1/keep.txt" ] && { echo "  PASS: unrelated sibling file kept"; PASS=$((PASS+1)); } || { echo "  FAIL: unrelated sibling file removed"; FAIL=$((FAIL+1)); }

# 2. Two custom roots sharing a parent share one sd.cpp -> all removed, no error on the
#    second (already-gone) removal.
p2="$_TMP_ROOT/inst2"
make_studio "$p2/studioB"
make_studio "$p2/studioC"  # same parent -> same sibling sd.cpp
_custom_studio_roots() { printf '%s\n%s\n' "$p2/studioB" "$p2/studioC"; }
run_loop
assert_nodir "shared-parent root B removed"               "$p2/studioB"
assert_nodir "shared-parent root C removed"               "$p2/studioC"
assert_nodir "shared sibling stable-diffusion.cpp removed" "$p2/stable-diffusion.cpp"

# 3. A sibling stable-diffusion.cpp WITHOUT the Studio owner marker (a user's own checkout,
#    even a built one, beside a custom root -- or one left when UNSLOTH_SD_CPP_PATH points
#    Studio elsewhere) is KEPT, though the Studio root itself is still removed.
p3="$_TMP_ROOT/inst3"
mkdir -p "$p3/studioD/share"; : > "$p3/studioD/share/studio.conf"
mkdir -p "$p3/stable-diffusion.cpp/build/bin"
: > "$p3/stable-diffusion.cpp/build/bin/sd-cli"  # user's own build, no owner marker
: > "$p3/stable-diffusion.cpp/main.cpp"
_custom_studio_roots() { printf '%s\n' "$p3/studioD"; }
run_loop
assert_nodir "unowned-sibling: custom root still removed" "$p3/studioD"
assert_dir   "unowned sibling stable-diffusion.cpp kept"   "$p3/stable-diffusion.cpp"

# 4. Default-mode sd.cpp (a bare ~/.unsloth/stable-diffusion.cpp with no custom root) is NOT
#    touched by the custom-root loop -- it is removed by the separate default-mode line.
mkdir -p "$HOME/.unsloth/stable-diffusion.cpp"
_custom_studio_roots() { printf '%s\n' "$p1/studioA"; }  # a now-removed root -> guard skips
run_loop
assert_dir "default-mode sd.cpp untouched by custom loop" "$HOME/.unsloth/stable-diffusion.cpp"

# 5. Default-mode ~/.unsloth/stable-diffusion.cpp WITH the Studio owner marker (a real Studio
#    default install) IS removed by the default-mode line.
rm -rf "$HOME/.unsloth/stable-diffusion.cpp"
mkdir -p "$HOME/.unsloth/stable-diffusion.cpp/build/bin"
: > "$HOME/.unsloth/stable-diffusion.cpp/build/bin/sd-cli"
: > "$HOME/.unsloth/stable-diffusion.cpp/.unsloth-studio-owned"  # written by install_sd_cpp_prebuilt
run_default_removal
assert_nodir "default-mode owned sd.cpp removed" "$HOME/.unsloth/stable-diffusion.cpp"

# 6. Default-mode ~/.unsloth/stable-diffusion.cpp WITHOUT the marker -- a user's own checkout at
#    the default path (or a pre-marker Studio build) -- is KEPT, mirroring the custom-root guard,
#    so uninstall never deletes a user file.
rm -rf "$HOME/.unsloth/stable-diffusion.cpp"
mkdir -p "$HOME/.unsloth/stable-diffusion.cpp"
: > "$HOME/.unsloth/stable-diffusion.cpp/main.cpp"  # user's own checkout, no owner marker
run_default_removal
assert_dir "default-mode unowned sd.cpp kept" "$HOME/.unsloth/stable-diffusion.cpp"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
