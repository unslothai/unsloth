#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit test for UNSLOTH_HOME master-root removal in scripts/uninstall.sh.
#
# setup.sh and setup.ps1 install llama.cpp, node and whisper.cpp as CHILDREN of the master
# root, beside studio/, so removing the Studio root alone strands multi-gigabyte trees. The
# root is user-chosen, so only a tree carrying the Unsloth owner marker may be deleted.
# Hermetic, following test_uninstall_sd_cpp_custom_root.sh: the real helpers and the real
# removal block are extracted from uninstall.sh and run against per-test fixtures.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UNINSTALL_SH="$SCRIPT_DIR/../../scripts/uninstall.sh"
PASS=0
FAIL=0

_TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$_TMP_ROOT"' EXIT
HOME="$_TMP_ROOT/home"
mkdir -p "$HOME"

assert_nodir() { _l="$1"; [ -e "$2" ] && { echo "  FAIL: $_l (still present: $2)"; FAIL=$((FAIL+1)); } || { echo "  PASS: $_l"; PASS=$((PASS+1)); }; }
assert_dir()   { _l="$1"; [ -e "$2" ] && { echo "  PASS: $_l"; PASS=$((PASS+1)); } || { echo "  FAIL: $_l (missing $2)"; FAIL=$((FAIL+1)); }; }
assert_eq()    { _l="$1"; [ "$2" = "$3" ] && { echo "  PASS: $_l"; PASS=$((PASS+1)); } || { echo "  FAIL: $_l (got '$2', want '$3')"; FAIL=$((FAIL+1)); }; }

HELPERS_FILE=$(mktemp -p "$_TMP_ROOT")
{
    sed -n '/^_remove_path() {/,/^}/p'      "$UNINSTALL_SH"
    sed -n '/^_remove_lock_file() {/,/^}/p' "$UNINSTALL_SH"
    sed -n '/^_is_unsafe_root() {/,/^}/p'   "$UNINSTALL_SH"
    sed -n '/^_master_root() {/,/^}/p'      "$UNINSTALL_SH"
    sed -n '/^_set_marker() {/,/^}/p'       "$UNINSTALL_SH"
} > "$HELPERS_FILE"
# Every helper the block calls, by name. A helper that stops being extracted goes "command not
# found" inside a subshell the harness already swallows, which reads as "nothing was removed"
# and passes the keep-assertions while failing the remove-assertions for the wrong reason.
for _h in _remove_path _remove_lock_file _is_unsafe_root _master_root _set_marker; do
    grep -q "^$_h() {" "$HELPERS_FILE" || { echo "FAIL: helpers missing $_h"; exit 1; }
done

# The real block, anchored on its own first line so a rename fails loudly instead of vacuously.
BLOCK_FILE=$(mktemp -p "$_TMP_ROOT")
sed -n '/^[[:space:]]*_mr_root="\$(_master_root)"/,/^[[:space:]]*# end master-root children$/p' "$UNINSTALL_SH" > "$BLOCK_FILE"
[ -s "$BLOCK_FILE" ] || { echo "FAIL: master-root removal block not extracted"; exit 1; }
grep -q 'unsloth-studio-owned' "$BLOCK_FILE" || { echo "FAIL: extracted block lost the marker gate"; exit 1; }

run_block() {
    ( set -e; HOME="$1"; UNSLOTH_HOME="$2"; export HOME UNSLOTH_HOME
      . "$HELPERS_FILE"; . "$BLOCK_FILE" ) || true
}

echo "== an owned master root loses its runtime children =="
MR="$_TMP_ROOT/portable"
mkdir -p "$MR"/{studio,llama.cpp,node,whisper.cpp}
for d in llama.cpp node whisper.cpp; do : > "$MR/$d/.unsloth-studio-owned"; done
: > "$MR/.llama.cpp.install.lock"
: > "$MR/.node.install.lock"
run_block "$HOME" "$MR"
assert_nodir "llama.cpp removed"   "$MR/llama.cpp"
assert_nodir "node removed"        "$MR/node"
assert_nodir "whisper.cpp removed" "$MR/whisper.cpp"
assert_nodir "llama lock removed"  "$MR/.llama.cpp.install.lock"
assert_nodir "node lock removed"   "$MR/.node.install.lock"
assert_dir   "studio root left to the Studio removal" "$MR/studio"

echo "== only a FILE is removed from an install-lock path =="
# prebuilt_core.install_lock creates the lock with os.open(O_CREAT | O_EXCL), so it is always a
# regular file. The names are fixed, so in a user-chosen root a directory here is the user's:
# rm -rf on it would take a tree that never carried the owner marker the children above need.
MRL="$_TMP_ROOT/lockshapes"
mkdir -p "$MRL/studio" "$MRL/.node.install.lock/keep"
: > "$MRL/.node.install.lock/keep/user-file"
: > "$MRL/.llama.cpp.install.lock"
ln -s "$MRL/.llama.cpp.install.lock" "$MRL/.whisper.cpp.install.lock"
: > "$MRL/.sd.cpp.install.lock.stale.4242"
run_block "$HOME" "$MRL"
assert_nodir "a real lock file is removed"          "$MRL/.llama.cpp.install.lock"
assert_nodir "a stale lock file is removed"         "$MRL/.sd.cpp.install.lock.stale.4242"
assert_nodir "a symlinked lock is unlinked"         "$MRL/.whisper.cpp.install.lock"
assert_dir   "a directory at a lock path is kept"   "$MRL/.node.install.lock"
assert_dir   "and so is everything under it"        "$MRL/.node.install.lock/keep/user-file"

echo "== an unmarked tree is somebody else's and is kept =="
MR2="$_TMP_ROOT/mixed"
mkdir -p "$MR2"/{llama.cpp,node}
: > "$MR2/node/.unsloth-studio-owned"
: > "$MR2/llama.cpp/my-own-build"
run_block "$HOME" "$MR2"
assert_dir   "unmarked llama.cpp kept" "$MR2/llama.cpp"
assert_dir   "its contents kept"       "$MR2/llama.cpp/my-own-build"
assert_nodir "marked node removed"     "$MR2/node"

echo "== an emptied master root is pruned, a used one is not =="
MR3="$_TMP_ROOT/empty-after"
mkdir -p "$MR3/node"
: > "$MR3/node/.unsloth-studio-owned"
run_block "$HOME" "$MR3"
assert_nodir "empty master root pruned" "$MR3"
MR4="$_TMP_ROOT/still-used"
mkdir -p "$MR4/node" "$MR4/my-datasets"
: > "$MR4/node/.unsloth-studio-owned"
run_block "$HOME" "$MR4"
assert_dir "master root with user files kept" "$MR4"
assert_dir "the user's files kept"            "$MR4/my-datasets"

echo "== the value is stripped and tilde-expanded, like the installers =="
MR5="$HOME/tilde-root"
mkdir -p "$MR5/node"
: > "$MR5/node/.unsloth-studio-owned"
run_block "$HOME" "  ~/tilde-root  "
assert_nodir "padded tilde value resolved" "$MR5/node"

echo "== a shared .staging keeps whatever is not ours =="
MR6="$_TMP_ROOT/staged"
mkdir -p "$MR6/node" "$MR6/.staging/somebody-elses"
: > "$MR6/node/.unsloth-studio-owned"
run_block "$HOME" "$MR6"
assert_dir "a non-empty .staging is kept whole" "$MR6/.staging/somebody-elses"
MR7="$_TMP_ROOT/staged-empty"
mkdir -p "$MR7/node" "$MR7/.staging"
: > "$MR7/node/.unsloth-studio-owned"
run_block "$HOME" "$MR7"
assert_nodir "an empty .staging is pruned" "$MR7/.staging"

echo "== an unset or default root selects nothing =="
got=$( ( HOME="$HOME"; export HOME; unset UNSLOTH_HOME; . "$HELPERS_FILE"; _master_root ) )
assert_eq "unset UNSLOTH_HOME yields nothing" "$got" ""
got=$( ( HOME="$HOME"; UNSLOTH_HOME="   "; export HOME UNSLOTH_HOME; . "$HELPERS_FILE"; _master_root ) )
assert_eq "blank UNSLOTH_HOME yields nothing" "$got" ""
mkdir -p "$HOME/.unsloth"
got=$( ( HOME="$HOME"; UNSLOTH_HOME="$HOME/.unsloth"; export HOME UNSLOTH_HOME; . "$HELPERS_FILE"; _master_root ) )
assert_eq "the default root is left to the blocks that own it" "$got" ""

echo "== a deny-listed root is refused =="
got=$( ( HOME="$HOME"; UNSLOTH_HOME="/"; export HOME UNSLOTH_HOME; . "$HELPERS_FILE"; _master_root ) )
assert_eq "/ yields nothing" "$got" ""

echo
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
