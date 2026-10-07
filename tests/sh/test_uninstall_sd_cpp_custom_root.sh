#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# A custom root's sd.cpp build sits at <parent>/stable-diffusion.cpp (sd_cpp_engine.py), so
# uninstall must remove it. The real loop and helpers are extracted from uninstall.sh.
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

HELPERS_FILE=$(mktemp -p "$_TMP_ROOT")
{
    sed -n '/^_remove_path() {/,/^}/p'      "$UNINSTALL_SH"
    sed -n '/^_is_owner_marker() {/,/^}/p'  "$UNINSTALL_SH"
    sed -n '/^_is_venv_dir() {/,/^}/p'      "$UNINSTALL_SH"
    sed -n '/^_is_studio_root() {/,/^}/p'   "$UNINSTALL_SH"
    sed -n '/^_is_unsafe_root() {/,/^}/p'   "$UNINSTALL_SH"
    # Without _master_root the block dies on "command not found" and keep-asserts pass vacuously.
    sed -n '/^_master_root() {/,/^}/p'      "$UNINSTALL_SH"
    sed -n '/^_set_marker() {/,/^}/p'              "$UNINSTALL_SH"
    sed -n '/^_restore_owner_marker() {/,/^}/p'     "$UNINSTALL_SH"
    sed -n '/^_remove_root_recording_db() {/,/^}/p' "$UNINSTALL_SH"
    sed -n '/^_owned_sd_cpp_roots() {/,/^}/p'      "$UNINSTALL_SH"
    sed -n '/^_sd_cpp_sibling_bases() {/,/^}/p'    "$UNINSTALL_SH"
} > "$HELPERS_FILE"
grep -q '_owned_sd_cpp_roots' "$HELPERS_FILE" || { echo "FAIL: helpers missing _owned_sd_cpp_roots"; exit 1; }
for _needed in _is_owner_marker _is_venv_dir _restore_owner_marker _master_root; do
    grep -q "^$_needed() {" "$HELPERS_FILE" || { echo "FAIL: helpers missing $_needed"; exit 1; }
done
grep -q '_sd_cpp_sibling_bases() {' "$HELPERS_FILE" || { echo "FAIL: helpers missing _sd_cpp_sibling_bases"; exit 1; }
# Both blocks are indented inside the main function: anchor on optional leading whitespace.
LOOP_FILE=$(mktemp -p "$_TMP_ROOT")
sed -n '/^[[:space:]]*_custom_studio_roots | while IFS= read -r _custom_root; do/,/^[[:space:]]*done/p' "$UNINSTALL_SH" > "$LOOP_FILE"
LEXICAL_FILE=$(mktemp -p "$_TMP_ROOT")
sed -n '/^[[:space:]]*_custom_studio_roots lexical 2>\/dev\/null | while IFS= read -r _lex_root; do/,/^[[:space:]]*done/p' "$UNINSTALL_SH" > "$LEXICAL_FILE"
REAL_ROOTS_FILE=$(mktemp -p "$_TMP_ROOT")
sed -n '/^_custom_studio_roots() {/,/^}/p' "$UNINSTALL_SH" > "$REAL_ROOTS_FILE"
DEFAULT_FILE=$(mktemp -p "$_TMP_ROOT")
sed -n '/^[[:space:]]*_default_sd_cpp="\$HOME\/\.unsloth\/stable-diffusion\.cpp"/,/^[[:space:]]*fi/p' "$UNINSTALL_SH" > "$DEFAULT_FILE"

# A silently empty extraction is what made this suite vacuous, so fail loudly instead.
for _f in "$HELPERS_FILE" "$LOOP_FILE" "$DEFAULT_FILE" "$LEXICAL_FILE" "$REAL_ROOTS_FILE"; do
    [ -s "$_f" ] || { echo "FAIL: extracted an empty fragment from $UNINSTALL_SH"; exit 1; }
done
grep -q '_remove_path' "$LOOP_FILE" || { echo "FAIL: loop fragment missing _remove_path"; exit 1; }
grep -q '_remove_path' "$DEFAULT_FILE" || { echo "FAIL: default fragment missing _remove_path"; exit 1; }

# shellcheck disable=SC1090
. "$HELPERS_FILE"

# make_studio <root>: a custom root plus a marked sibling sd.cpp build.
make_studio() {
    mkdir -p "$1/share"
    : > "$1/share/studio.conf"
    _sib="$(dirname "$1")/stable-diffusion.cpp"
    mkdir -p "$_sib"
    : > "$_sib/sd-cli"
    : > "$_sib/.unsloth-studio-owned"
}
run_loop() {
    # shellcheck disable=SC1090
    . "$LOOP_FILE"
}
run_default_removal() {
    # shellcheck disable=SC1090
    . "$DEFAULT_FILE"
}
run_lexical_removal() {
    # shellcheck disable=SC1090
    . "$LEXICAL_FILE"
}

p1="$_TMP_ROOT/inst1"
make_studio "$p1/studioA"
: > "$p1/keep.txt"
_custom_studio_roots() { printf '%s\n' "$p1/studioA"; }
run_loop
assert_nodir "single custom root removed"                 "$p1/studioA"
assert_nodir "custom-root sibling stable-diffusion.cpp removed" "$p1/stable-diffusion.cpp"
[ -f "$p1/keep.txt" ] && { echo "  PASS: unrelated sibling file kept"; PASS=$((PASS+1)); } || { echo "  FAIL: unrelated sibling file removed"; FAIL=$((FAIL+1)); }

p2="$_TMP_ROOT/inst2"
make_studio "$p2/studioB"
make_studio "$p2/studioC"
_custom_studio_roots() { printf '%s\n%s\n' "$p2/studioB" "$p2/studioC"; }
run_loop
assert_nodir "shared-parent root B removed"               "$p2/studioB"
assert_nodir "shared-parent root C removed"               "$p2/studioC"
assert_nodir "shared sibling stable-diffusion.cpp removed" "$p2/stable-diffusion.cpp"

# 3. An unmarked sibling stable-diffusion.cpp (user's own) is KEPT; the root is still removed.
p3="$_TMP_ROOT/inst3"
mkdir -p "$p3/studioD/share"; : > "$p3/studioD/share/studio.conf"
mkdir -p "$p3/stable-diffusion.cpp/build/bin"
: > "$p3/stable-diffusion.cpp/build/bin/sd-cli"
: > "$p3/stable-diffusion.cpp/main.cpp"
_custom_studio_roots() { printf '%s\n' "$p3/studioD"; }
run_loop
assert_nodir "unowned-sibling: custom root still removed" "$p3/studioD"
assert_dir   "unowned sibling stable-diffusion.cpp kept"   "$p3/stable-diffusion.cpp"

mkdir -p "$HOME/.unsloth/stable-diffusion.cpp"
_custom_studio_roots() { printf '%s\n' "$p1/studioA"; }
run_loop
assert_dir "default-mode sd.cpp untouched by custom loop" "$HOME/.unsloth/stable-diffusion.cpp"

rm -rf "$HOME/.unsloth/stable-diffusion.cpp"
mkdir -p "$HOME/.unsloth/stable-diffusion.cpp/build/bin"
: > "$HOME/.unsloth/stable-diffusion.cpp/build/bin/sd-cli"
: > "$HOME/.unsloth/stable-diffusion.cpp/.unsloth-studio-owned"
run_default_removal
assert_nodir "default-mode owned sd.cpp removed" "$HOME/.unsloth/stable-diffusion.cpp"

# 6. Unmarked default-mode sd.cpp is KEPT, mirroring the custom-root guard.
rm -rf "$HOME/.unsloth/stable-diffusion.cpp"
mkdir -p "$HOME/.unsloth/stable-diffusion.cpp"
: > "$HOME/.unsloth/stable-diffusion.cpp/main.cpp"
run_default_removal
assert_dir "default-mode unowned sd.cpp kept" "$HOME/.unsloth/stable-diffusion.cpp"


# ── _owned_sd_cpp_roots: sd-servers stopped before their tree is deleted ──
# A resident sd-server survives unlinking its binary and keeps holding its port.
assert_lists() {
    _l="$1"; _want="$2"
    if _owned_sd_cpp_roots | grep -qxF "$_want"; then
        echo "  PASS: $_l"; PASS=$((PASS+1))
    else
        echo "  FAIL: $_l (not listed: $_want)"; FAIL=$((FAIL+1))
    fi
}
assert_not_lists() {
    _l="$1"; _want="$2"
    if _owned_sd_cpp_roots | grep -qxF "$_want"; then
        echo "  FAIL: $_l (listed anyway: $_want)"; FAIL=$((FAIL+1))
    else
        echo "  PASS: $_l"; PASS=$((PASS+1))
    fi
}

p7="$_TMP_ROOT/inst7/studioE"
mkdir -p "$p7/stable-diffusion.cpp/build/bin"
: > "$p7/stable-diffusion.cpp/build/bin/sd-server"
: > "$p7/stable-diffusion.cpp/.unsloth-studio-owned"
_custom_studio_roots() { printf '%s\n' "$p7"; }
assert_lists "nested <root>/stable-diffusion.cpp is stopped before removal" "$p7/stable-diffusion.cpp"

p8="$_TMP_ROOT/inst8/studioF"
mkdir -p "$p8/stable-diffusion.cpp/build/bin"
: > "$p8/stable-diffusion.cpp/build/bin/sd-server"
_custom_studio_roots() { printf '%s\n' "$p8"; }
assert_not_lists "unowned nested build under a non-Unsloth root is left running" "$p8/stable-diffusion.cpp"

# 8b. Under a real Unsloth root the unmarked nested build is stopped anyway: the tree goes.
p8b="$_TMP_ROOT/inst8b/studioF2"
mkdir -p "$p8b/share" "$p8b/stable-diffusion.cpp/build/bin"
: > "$p8b/share/studio.conf"
: > "$p8b/stable-diffusion.cpp/build/bin/sd-server"
_custom_studio_roots() { printf '%s\n' "$p8b"; }
assert_lists "unmarked nested build under a doomed Unsloth root is stopped" "$p8b/stable-diffusion.cpp"

p9="$_TMP_ROOT/inst9"
mkdir -p "$p9/studioG" "$p9/stable-diffusion.cpp"
: > "$p9/stable-diffusion.cpp/.unsloth-studio-owned"
_custom_studio_roots() { printf '%s\n' "$p9/studioG"; }
assert_lists "legacy <parent>/stable-diffusion.cpp still stopped" "$p9/stable-diffusion.cpp"

p10="$_TMP_ROOT/inst10"
mkdir -p "$p10/studioH/stable-diffusion.cpp" "$p10/stable-diffusion.cpp"
: > "$p10/studioH/stable-diffusion.cpp/.unsloth-studio-owned"
: > "$p10/stable-diffusion.cpp/.unsloth-studio-owned"
_custom_studio_roots() { printf '%s\n' "$p10/studioH"; }
assert_lists "both locations: nested listed"  "$p10/studioH/stable-diffusion.cpp"
assert_lists "both locations: sibling listed" "$p10/stable-diffusion.cpp"

p11="$_TMP_ROOT/inst11"
make_studio "$p11/studioI"
mkdir -p "$p11/studioI/stable-diffusion.cpp/build/bin"
: > "$p11/studioI/stable-diffusion.cpp/build/bin/sd-cli"
: > "$p11/studioI/stable-diffusion.cpp/.unsloth-studio-owned"
_custom_studio_roots() { printf '%s\n' "$p11/studioI"; }
run_loop
assert_nodir "nested stable-diffusion.cpp removed with its root" "$p11/studioI/stable-diffusion.cpp"
assert_nodir "its custom root removed"                           "$p11/studioI"


# ── A symlinked Unsloth home: the old build put sd.cpp beside the LINK, not its target ──
# shellcheck disable=SC1090
. "$REAL_ROOTS_FILE"
unset STUDIO_HOME

p12="$_TMP_ROOT/inst12"
mkdir -p "$p12/real/studioJ/share" "$p12/stable-diffusion.cpp"
: > "$p12/real/studioJ/share/studio.conf"
ln -s "$p12/real/studioJ" "$p12/link"
: > "$p12/stable-diffusion.cpp/sd-server"
: > "$p12/stable-diffusion.cpp/.unsloth-studio-owned"
UNSLOTH_STUDIO_HOME="$p12/link"
export UNSLOTH_STUDIO_HOME
assert_lists "symlinked home: sd.cpp beside the link is stopped" "$p12/stable-diffusion.cpp"

run_lexical_removal
assert_nodir "symlinked home: sd.cpp beside the link removed" "$p12/stable-diffusion.cpp"

p14="$_TMP_ROOT/inst14"
mkdir -p "$p14/real/studioK/share" "$p14/stable-diffusion.cpp"
: > "$p14/real/studioK/share/studio.conf"
ln -s "$p14/real/studioK" "$p14/link"
: > "$p14/stable-diffusion.cpp/main.cpp"
UNSLOTH_STUDIO_HOME="$p14/link"
run_lexical_removal
assert_dir "symlinked home: unowned sd.cpp beside the link kept" "$p14/stable-diffusion.cpp"
assert_not_lists "symlinked home: unowned sd.cpp beside the link is left running" "$p14/stable-diffusion.cpp"

# 14b. A mistyped home is not a root: the lexical pass must not take a sibling install's sd.cpp.
p14b="$_TMP_ROOT/inst14b"
mkdir -p "$p14b/other/share" "$p14b/stable-diffusion.cpp"
: > "$p14b/other/share/studio.conf"
: > "$p14b/stable-diffusion.cpp/sd-cli"
: > "$p14b/stable-diffusion.cpp/.unsloth-studio-owned"
UNSLOTH_STUDIO_HOME="$p14b/typo"
run_lexical_removal
assert_dir "a mistyped home does not take a neighbour's legacy sd.cpp" "$p14b/stable-diffusion.cpp"

# 15. The deny list is a string match, so the lexical path must be canonicalized first.
p15="$_TMP_ROOT/inst15"
mkdir -p "$p15/sub" "$p15/studioL/share" "$p15/stable-diffusion.cpp"
: > "$p15/studioL/share/studio.conf"
: > "$p15/stable-diffusion.cpp/sd-cli"
: > "$p15/stable-diffusion.cpp/.unsloth-studio-owned"
_HOME_BEFORE="$HOME"
HOME="$p15/stable-diffusion.cpp/home"  # so dirname "$HOME" is the denied path
mkdir -p "$HOME"
UNSLOTH_STUDIO_HOME="$p15/sub/../studioL"
run_lexical_removal
assert_dir "lexical sibling resolving into a denied tree is refused" "$p15/stable-diffusion.cpp"
HOME="$_HOME_BEFORE"
unset UNSLOTH_STUDIO_HOME

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
