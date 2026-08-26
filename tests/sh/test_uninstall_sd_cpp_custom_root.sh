#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit test for custom/env-mode stable-diffusion.cpp removal in scripts/uninstall.sh.
#
# A custom Unsloth (UNSLOTH_STUDIO_HOME=<root>) installs its native diffusion build beside
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
    # The loop removes roots through this wrapper; without it and its marker helper the
    # fragment dies with "command not found" and every assertion below is vacuous.
    sed -n '/^_set_marker() {/,/^}/p'              "$UNINSTALL_SH"
    sed -n '/^_remove_root_recording_db() {/,/^}/p' "$UNINSTALL_SH"
    # The owned-root lister that decides which sd-servers are pkilled before a tree is deleted.
    sed -n '/^_owned_sd_cpp_roots() {/,/^}/p'      "$UNINSTALL_SH"
    # ... and the sibling-base lister it asks for the legacy <parent>/stable-diffusion.cpp paths.
    sed -n '/^_sd_cpp_sibling_bases() {/,/^}/p'    "$UNINSTALL_SH"
} > "$HELPERS_FILE"
grep -q '_owned_sd_cpp_roots' "$HELPERS_FILE" || { echo "FAIL: helpers missing _owned_sd_cpp_roots"; exit 1; }
grep -q '_sd_cpp_sibling_bases() {' "$HELPERS_FILE" || { echo "FAIL: helpers missing _sd_cpp_sibling_bases"; exit 1; }
# Both blocks sit inside the main removal function, so they are indented: anchor on optional
# leading whitespace, never on column 0, or the range matches nothing and every assertion below
# passes or fails vacuously against a fragment that was never extracted.
LOOP_FILE=$(mktemp -p "$_TMP_ROOT")
sed -n '/^[[:space:]]*_custom_studio_roots | while IFS= read -r _custom_root; do/,/^[[:space:]]*done/p' "$UNINSTALL_SH" > "$LOOP_FILE"
# The follow-up block that clears the legacy sibling hanging off the LEXICAL parent.
LEXICAL_FILE=$(mktemp -p "$_TMP_ROOT")
sed -n '/^[[:space:]]*_custom_studio_roots lexical 2>\/dev\/null | while IFS= read -r _lex_root; do/,/^[[:space:]]*done/p' "$UNINSTALL_SH" > "$LEXICAL_FILE"
# The real root resolver, for the symlinked-home cases at the end (the others stub it).
REAL_ROOTS_FILE=$(mktemp -p "$_TMP_ROOT")
sed -n '/^_custom_studio_roots() {/,/^}/p' "$UNINSTALL_SH" > "$REAL_ROOTS_FILE"
# The real default-mode ~/.unsloth/stable-diffusion.cpp removal block (marker-guarded).
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

# make_studio <root> : a valid custom Unsloth root (share/studio.conf owner marker) plus its
# sibling <parent>/stable-diffusion.cpp build carrying the Unsloth owner marker, each with a
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
run_lexical_removal() {
    # shellcheck disable=SC1090
    . "$LEXICAL_FILE"
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

# 3. A sibling stable-diffusion.cpp WITHOUT the Unsloth owner marker (a user's own checkout,
#    even a built one, beside a custom root -- or one left when UNSLOTH_SD_CPP_PATH points
#    Unsloth elsewhere) is KEPT, though the Unsloth root itself is still removed.
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

# 5. Default-mode ~/.unsloth/stable-diffusion.cpp WITH the Unsloth owner marker (a real Unsloth
#    default install) IS removed by the default-mode line.
rm -rf "$HOME/.unsloth/stable-diffusion.cpp"
mkdir -p "$HOME/.unsloth/stable-diffusion.cpp/build/bin"
: > "$HOME/.unsloth/stable-diffusion.cpp/build/bin/sd-cli"
: > "$HOME/.unsloth/stable-diffusion.cpp/.unsloth-studio-owned"  # written by install_sd_cpp_prebuilt
run_default_removal
assert_nodir "default-mode owned sd.cpp removed" "$HOME/.unsloth/stable-diffusion.cpp"

# 6. Default-mode ~/.unsloth/stable-diffusion.cpp WITHOUT the marker -- a user's own checkout at
#    the default path (or a pre-marker Unsloth build) -- is KEPT, mirroring the custom-root guard,
#    so uninstall never deletes a user file.
rm -rf "$HOME/.unsloth/stable-diffusion.cpp"
mkdir -p "$HOME/.unsloth/stable-diffusion.cpp"
: > "$HOME/.unsloth/stable-diffusion.cpp/main.cpp"  # user's own checkout, no owner marker
run_default_removal
assert_dir "default-mode unowned sd.cpp kept" "$HOME/.unsloth/stable-diffusion.cpp"


# ── _owned_sd_cpp_roots: which sd-servers get stopped before their tree is deleted ────────────
#
# The install now lands at <root>/stable-diffusion.cpp, inside the custom root, and the custom
# root is removed wholesale by the loop above. A resident sd-server survives unlinking its
# binary, so if that nested path is not listed here the tree disappears while the server keeps
# running and holding its port. The legacy <parent>/stable-diffusion.cpp sibling an older build
# wrote still has to be listed too, since it is deleted separately.
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

# 7. The nested install under the custom root is listed, so its sd-server is stopped first.
p7="$_TMP_ROOT/inst7/studioE"
mkdir -p "$p7/stable-diffusion.cpp/build/bin"
: > "$p7/stable-diffusion.cpp/build/bin/sd-server"
: > "$p7/stable-diffusion.cpp/.unsloth-studio-owned"
_custom_studio_roots() { printf '%s\n' "$p7"; }
assert_lists "nested <root>/stable-diffusion.cpp is stopped before removal" "$p7/stable-diffusion.cpp"

# 8. A root this run does NOT delete (no Unsloth sentinels) keeps its unowned nested build running.
p8="$_TMP_ROOT/inst8/studioF"
mkdir -p "$p8/stable-diffusion.cpp/build/bin"
: > "$p8/stable-diffusion.cpp/build/bin/sd-server"  # no owner marker: the user's own build
_custom_studio_roots() { printf '%s\n' "$p8"; }
assert_not_lists "unowned nested build under a non-Unsloth root is left running" "$p8/stable-diffusion.cpp"

# 8b. But under a real Unsloth root, which the loop deletes wholesale, the unmarked nested build is
#     stopped anyway: the current-root finder can select it without a marker, and deleting the tree
#     around a live server just leaves it holding its port.
p8b="$_TMP_ROOT/inst8b/studioF2"
mkdir -p "$p8b/share" "$p8b/stable-diffusion.cpp/build/bin"
: > "$p8b/share/studio.conf"
: > "$p8b/stable-diffusion.cpp/build/bin/sd-server"  # no owner marker
_custom_studio_roots() { printf '%s\n' "$p8b"; }
assert_lists "unmarked nested build under a doomed Unsloth root is stopped" "$p8b/stable-diffusion.cpp"

# 9. The legacy sibling an older build installed is still listed (it is still deleted).
p9="$_TMP_ROOT/inst9"
mkdir -p "$p9/studioG" "$p9/stable-diffusion.cpp"
: > "$p9/stable-diffusion.cpp/.unsloth-studio-owned"
_custom_studio_roots() { printf '%s\n' "$p9/studioG"; }
assert_lists "legacy <parent>/stable-diffusion.cpp still stopped" "$p9/stable-diffusion.cpp"

# 10. Both locations at once: neither shadows the other.
p10="$_TMP_ROOT/inst10"
mkdir -p "$p10/studioH/stable-diffusion.cpp" "$p10/stable-diffusion.cpp"
: > "$p10/studioH/stable-diffusion.cpp/.unsloth-studio-owned"
: > "$p10/stable-diffusion.cpp/.unsloth-studio-owned"
_custom_studio_roots() { printf '%s\n' "$p10/studioH"; }
assert_lists "both locations: nested listed"  "$p10/studioH/stable-diffusion.cpp"
assert_lists "both locations: sibling listed" "$p10/stable-diffusion.cpp"

# 11. The nested tree goes away with the root it lives in.
p11="$_TMP_ROOT/inst11"
make_studio "$p11/studioI"
mkdir -p "$p11/studioI/stable-diffusion.cpp/build/bin"
: > "$p11/studioI/stable-diffusion.cpp/build/bin/sd-cli"
: > "$p11/studioI/stable-diffusion.cpp/.unsloth-studio-owned"
_custom_studio_roots() { printf '%s\n' "$p11/studioI"; }
run_loop
assert_nodir "nested stable-diffusion.cpp removed with its root" "$p11/studioI/stable-diffusion.cpp"
assert_nodir "its custom root removed"                           "$p11/studioI"


# ── An Unsloth home that is itself a symlink ────────────────────────────────────────────────────
#
# The old build derived the sd.cpp root with a plain `dirname "$UNSLOTH_STUDIO_HOME"`, so for
# UNSLOTH_STUDIO_HOME=<link> the tree it installed sits beside the LINK. _custom_studio_roots
# canonicalizes the home before this code takes its parent, which points every legacy consumer
# beside the link's TARGET instead: the tree beside the link was never stopped and never deleted.
# The real resolver from here on, since a stub cannot express the link at all.
# shellcheck disable=SC1090
. "$REAL_ROOTS_FILE"
unset STUDIO_HOME

# 12. The legacy sibling beside the link is listed, so its sd-server is stopped before removal.
p12="$_TMP_ROOT/inst12"
mkdir -p "$p12/real/studioJ/share" "$p12/stable-diffusion.cpp"
: > "$p12/real/studioJ/share/studio.conf"
ln -s "$p12/real/studioJ" "$p12/link"
: > "$p12/stable-diffusion.cpp/sd-server"
: > "$p12/stable-diffusion.cpp/.unsloth-studio-owned"
UNSLOTH_STUDIO_HOME="$p12/link"
export UNSLOTH_STUDIO_HOME
assert_lists "symlinked home: sd.cpp beside the link is stopped" "$p12/stable-diffusion.cpp"

# 13. ... and removed.
run_lexical_removal
assert_nodir "symlinked home: sd.cpp beside the link removed" "$p12/stable-diffusion.cpp"

# 14. An unowned checkout beside the link is kept, exactly like the canonical pass keeps one.
p14="$_TMP_ROOT/inst14"
mkdir -p "$p14/real/studioK/share" "$p14/stable-diffusion.cpp"
: > "$p14/real/studioK/share/studio.conf"
ln -s "$p14/real/studioK" "$p14/link"
: > "$p14/stable-diffusion.cpp/main.cpp"  # the user's own checkout, no owner marker
UNSLOTH_STUDIO_HOME="$p14/link"
run_lexical_removal
assert_dir "symlinked home: unowned sd.cpp beside the link kept" "$p14/stable-diffusion.cpp"
assert_not_lists "symlinked home: unowned sd.cpp beside the link is left running" "$p14/stable-diffusion.cpp"

# 14b. A stale or mistyped home is not an Unsloth root, and the lexical pass must apply the same
#      ownership check the canonical loop does: "<parent>/typo" must not take the marked
#      <parent>/stable-diffusion.cpp belonging to a different, valid install.
p14b="$_TMP_ROOT/inst14b"
mkdir -p "$p14b/other/share" "$p14b/stable-diffusion.cpp"
: > "$p14b/other/share/studio.conf"          # somebody else's Unsloth, sharing the parent
: > "$p14b/stable-diffusion.cpp/sd-cli"
: > "$p14b/stable-diffusion.cpp/.unsloth-studio-owned"
UNSLOTH_STUDIO_HOME="$p14b/typo"             # never existed
run_lexical_removal
assert_dir "a mistyped home does not take a neighbour's legacy sd.cpp" "$p14b/stable-diffusion.cpp"

# 15. The deny list is a string match, so the lexical path has to be canonicalized before it is
#     applied: a home carrying ".." (or a symlinked ancestor) otherwise produces a sibling that
#     misses the pattern while resolving straight into a protected tree. $HOME's parent is on that
#     list, so point HOME inside the fixture to get a deterministic denied target.
p15="$_TMP_ROOT/inst15"
mkdir -p "$p15/sub" "$p15/studioL/share" "$p15/stable-diffusion.cpp"
: > "$p15/studioL/share/studio.conf"
: > "$p15/stable-diffusion.cpp/sd-cli"
: > "$p15/stable-diffusion.cpp/.unsloth-studio-owned"
_HOME_BEFORE="$HOME"
HOME="$p15/stable-diffusion.cpp/home"  # so dirname "$HOME" is the denied path
mkdir -p "$HOME"
UNSLOTH_STUDIO_HOME="$p15/sub/../studioL"  # lexical parent "$p15/sub/.." -> misses the raw match
run_lexical_removal
assert_dir "lexical sibling resolving into a denied tree is refused" "$p15/stable-diffusion.cpp"
HOME="$_HOME_BEFORE"
unset UNSLOTH_STUDIO_HOME

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
