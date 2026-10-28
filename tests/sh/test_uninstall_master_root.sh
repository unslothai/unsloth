#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# UNSLOTH_HOME master-root removal in scripts/uninstall.sh: runtimes are children of the master
# root, which is user-chosen, so only trees with the owner marker may be deleted.
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
assert_link()  { _l="$1"; [ -L "$2" ] && { echo "  PASS: $_l"; PASS=$((PASS+1)); } || { echo "  FAIL: $_l (link missing $2)"; FAIL=$((FAIL+1)); }; }
assert_present() { _l="$1"; { [ -e "$2" ] || [ -L "$2" ]; } && { echo "  PASS: $_l"; PASS=$((PASS+1)); } || { echo "  FAIL: $_l (missing $2)"; FAIL=$((FAIL+1)); }; }
assert_eq()    { _l="$1"; [ "$2" = "$3" ] && { echo "  PASS: $_l"; PASS=$((PASS+1)); } || { echo "  FAIL: $_l (got '$2', want '$3')"; FAIL=$((FAIL+1)); }; }

HELPERS_FILE=$(mktemp -p "$_TMP_ROOT")
{
    sed -n '/^_remove_path() {/,/^}/p'      "$UNINSTALL_SH"
    sed -n '/^_remove_lock_file() {/,/^}/p' "$UNINSTALL_SH"
    sed -n '/^_is_unsafe_root() {/,/^}/p'   "$UNINSTALL_SH"
    sed -n '/^_master_root() {/,/^}/p'      "$UNINSTALL_SH"
    sed -n '/^_set_marker() {/,/^}/p'       "$UNINSTALL_SH"
} > "$HELPERS_FILE"
# A missing helper fails inside a swallowed subshell and reads as "nothing removed".
for _h in _remove_path _remove_lock_file _is_unsafe_root _master_root _set_marker; do
    grep -q "^$_h() {" "$HELPERS_FILE" || { echo "FAIL: helpers missing $_h"; exit 1; }
done

BLOCK_FILE=$(mktemp -p "$_TMP_ROOT")
sed -n '/^[[:space:]]*_mr_root="\$_MASTER_ROOT_SAVED"/,/^[[:space:]]*# end master-root children$/p' "$UNINSTALL_SH" > "$BLOCK_FILE"
[ -s "$BLOCK_FILE" ] || { echo "FAIL: master-root removal block not extracted"; exit 1; }
grep -q 'unsloth-studio-owned' "$BLOCK_FILE" || { echo "FAIL: extracted block lost the marker gate"; exit 1; }

run_block() {
    # The real script resolves the master root ONCE before removal; seed it the same way.
    ( set -e; HOME="$1"; UNSLOTH_HOME="$2"; export HOME UNSLOTH_HOME
      . "$HELPERS_FILE"; _MASTER_ROOT_SAVED="$(_master_root)"; . "$BLOCK_FILE" ) || true
}

echo "== an owned master root loses its runtime children =="
MR="$_TMP_ROOT/portable"
mkdir -p "$MR"/{studio,llama.cpp,node,whisper.cpp,audio.cpp}
for d in llama.cpp node whisper.cpp audio.cpp; do : > "$MR/$d/.unsloth-studio-owned"; done
: > "$MR/.llama.cpp.install.lock"
: > "$MR/.node.install.lock"
: > "$MR/.audio.cpp.install.lock"
: > "$MR/.audio.cpp.install.lock.stale.31337"
run_block "$HOME" "$MR"
assert_nodir "llama.cpp removed"   "$MR/llama.cpp"
assert_nodir "node removed"        "$MR/node"
assert_nodir "whisper.cpp removed" "$MR/whisper.cpp"
assert_nodir "audio.cpp removed"   "$MR/audio.cpp"
assert_nodir "llama lock removed"  "$MR/.llama.cpp.install.lock"
assert_nodir "node lock removed"   "$MR/.node.install.lock"
assert_nodir "audio.cpp lock removed" "$MR/.audio.cpp.install.lock"
assert_nodir "stale audio.cpp lock removed" "$MR/.audio.cpp.install.lock.stale.31337"
assert_dir   "studio root left to the Studio removal" "$MR/studio"

echo "== only a FILE is removed from an install-lock path =="
# Locks are always regular files (O_CREAT|O_EXCL); a directory at that name is the user's.
MRL="$_TMP_ROOT/lockshapes"
mkdir -p "$MRL/studio" "$MRL/.node.install.lock/keep"
: > "$MRL/.node.install.lock/keep/user-file"
: > "$MRL/.llama.cpp.install.lock"
# A link at a lock name is the user's too, matching uninstall.ps1's reparse-point rule.
ln -s "$MRL/.llama.cpp.install.lock" "$MRL/.whisper.cpp.install.lock"
ln -s "$MRL/nowhere" "$MRL/.sd.cpp.install.lock"
: > "$MRL/.sd.cpp.install.lock.stale.4242"
run_block "$HOME" "$MRL"
assert_nodir "a real lock file is removed"          "$MRL/.llama.cpp.install.lock"
assert_nodir "a stale lock file is removed"         "$MRL/.sd.cpp.install.lock.stale.4242"
assert_link  "a linked lock is kept"                "$MRL/.whisper.cpp.install.lock"
assert_link  "a dangling linked lock is kept too"   "$MRL/.sd.cpp.install.lock"
assert_dir   "a directory at a lock path is kept"   "$MRL/.node.install.lock"
assert_dir   "and so is everything under it"        "$MRL/.node.install.lock/keep/user-file"

echo "== an unmarked tree is somebody else's and is kept =="
MR2="$_TMP_ROOT/mixed"
mkdir -p "$MR2"/{llama.cpp,node,audio.cpp}
: > "$MR2/node/.unsloth-studio-owned"
: > "$MR2/llama.cpp/my-own-build"
: > "$MR2/audio.cpp/CMakeLists.txt"
run_block "$HOME" "$MR2"
assert_dir   "unmarked llama.cpp kept" "$MR2/llama.cpp"
assert_dir   "its contents kept"       "$MR2/llama.cpp/my-own-build"
assert_dir   "unmarked audio.cpp kept" "$MR2/audio.cpp/CMakeLists.txt"
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

echo "== a root that is gone does not end the uninstall =="
# Under set -e the canonicalizing `cd` fails when the root is absent; bash hides that, so
# test the shells `| sh` reaches. REACHED_END is the assertion.
for _shell in dash "bash --posix" sh; do
    command -v ${_shell%% *} > /dev/null 2>&1 || continue
    out=$( env -i HOME="$HOME" PATH="$PATH" UNSLOTH_HOME="$_TMP_ROOT/never-existed" \
        $_shell -c "set -e; . \"$HELPERS_FILE\"; r=\$(_master_root); echo REACHED_END" 2>&1 \
        || true )
    assert_eq "$_shell keeps going past a missing root" "$out" "REACHED_END"
done

echo "== a dangling runtime symlink is not ours to unlink =="
# An unmarked dangling llama.cpp link must not be removed: -e misses it, _remove_path does not.
MRD="$_TMP_ROOT/dangling"
mkdir -p "$MRD/studio" "$MRD/node"
: > "$MRD/node/.unsloth-studio-owned"
ln -s "$_TMP_ROOT/no-such-volume/llama.cpp" "$MRD/llama.cpp"
run_block "$HOME" "$MRD"
assert_present "an unmarked dangling link is kept" "$MRD/llama.cpp"
assert_nodir "a marked tree beside it still goes" "$MRD/node"

echo "== a whitespace-only Studio override does not suppress the master root =="
# studio_root() trims, so a whitespace-only Studio override is unset to every resolver.
ROOTS_FILE=$(mktemp -p "$_TMP_ROOT")
sed -n '/^_custom_studio_roots() {/,/^}/p' "$UNINSTALL_SH" > "$ROOTS_FILE" 2>/dev/null || : > "$ROOTS_FILE"
if grep -q '_custom_studio_roots() {' "$ROOTS_FILE"; then
    MRW="$_TMP_ROOT/blankoverride"
    mkdir -p "$MRW/studio"
    got=$( ( HOME="$HOME"; UNSLOTH_HOME="$MRW"; UNSLOTH_STUDIO_HOME="   "
             export HOME UNSLOTH_HOME UNSLOTH_STUDIO_HOME
             . "$HELPERS_FILE"
             _emit() { printf '%s\n' "$1"; }
             _from_conf() { :; }
             . "$ROOTS_FILE"
             _custom_studio_roots 2>/dev/null ) | grep -c "$MRW/studio" ) || got=0
    assert_eq "a blank override still reaches the master root" "$got" "1"
else
    echo "  FAIL: _custom_studio_roots could not be extracted"; FAIL=$((FAIL+1))
fi

echo "== the master root is remembered when the environment does not carry it =="
# setup.sh writes a master-root note in the Studio tree so a later bare uninstall finds the runtimes.
NOTED="$_TMP_ROOT/noted"
mkdir -p "$HOME/.unsloth/studio/share" "$NOTED/studio/share" "$HOME/.local/share/unsloth"
printf "UNSLOTH_EXE='%s'\n" "$NOTED/studio/unsloth_studio/bin/unsloth" \
    > "$HOME/.local/share/unsloth/studio.conf"
printf '%s\n' "$NOTED" > "$NOTED/studio/share/.unsloth-master-root"
got=$( ( HOME="$HOME"; unset UNSLOTH_HOME; export HOME; . "$HELPERS_FILE"; _master_root ) )
assert_eq "the note names the master root" "$got" "$NOTED"
# The environment still outranks it, and the deny list still applies to whatever it names.
got=$( ( HOME="$HOME"; UNSLOTH_HOME="$_TMP_ROOT/from-env"; export HOME UNSLOTH_HOME
         . "$HELPERS_FILE"; _master_root ) )
assert_eq "the environment outranks the note" "$got" "$_TMP_ROOT/from-env"
printf '%s\n' "$HOME/.unsloth" > "$NOTED/studio/share/.unsloth-master-root"
got=$( ( HOME="$HOME"; unset UNSLOTH_HOME; export HOME; . "$HELPERS_FILE"; _master_root ) )
assert_eq "a note naming the default root is still refused" "$got" ""
# A copied tree's note names the ORIGINAL master root; accepting it would delete its runtimes.
COPIED="$_TMP_ROOT/copied"
mkdir -p "$COPIED/studio/share"
printf '%s\n' "$NOTED" > "$COPIED/studio/share/.unsloth-master-root"
got=$( ( HOME="$HOME"; unset UNSLOTH_HOME; UNSLOTH_STUDIO_HOME="$COPIED/studio"
         export HOME UNSLOTH_STUDIO_HOME; . "$HELPERS_FILE"; _master_root ) )
assert_eq "a note carried in from another master root is refused" "$got" ""
rm -f "$NOTED/studio/share/.unsloth-master-root" "$HOME/.local/share/unsloth/studio.conf"

echo "== a named Studio root does not borrow another install's note =="
# The named tree's note must win over the legacy one, or the other install's runtimes go.
BORROWED="$_TMP_ROOT/borrowed"
NAMED_MASTER="$_TMP_ROOT/named-master"
NAMED="$NAMED_MASTER/studio"
mkdir -p "$HOME/.unsloth/studio/share" "$BORROWED/studio" "$NAMED/share"
printf '%s\n' "$BORROWED" > "$HOME/.unsloth/studio/share/.unsloth-master-root"
got=$( ( HOME="$HOME"; unset UNSLOTH_HOME; UNSLOTH_STUDIO_HOME="$NAMED"
         export HOME UNSLOTH_STUDIO_HOME; . "$HELPERS_FILE"; _master_root ) )
assert_eq "an unrelated install's note is not adopted" "$got" ""
printf '%s\n' "$NAMED_MASTER" > "$NAMED/share/.unsloth-master-root"
got=$( ( HOME="$HOME"; unset UNSLOTH_HOME UNSLOTH_STUDIO_HOME; STUDIO_HOME="$NAMED"
         export HOME STUDIO_HOME; . "$HELPERS_FILE"; _master_root ) )
assert_eq "the named root's own note is still read" "$got" "$NAMED_MASTER"
rm -f "$HOME/.unsloth/studio/share/.unsloth-master-root" "$NAMED/share/.unsloth-master-root"

echo "== a padded Studio override still finds the note setup.sh wrote =="
# setup.sh trims UNSLOTH_STUDIO_HOME before writing the note, so the reader must trim too.
PADDED_MASTER="$_TMP_ROOT/padded-master"
PADDED="$PADDED_MASTER/studio"
mkdir -p "$PADDED/share"
printf '%s\n' "$PADDED_MASTER" > "$PADDED/share/.unsloth-master-root"
got=$( ( HOME="$HOME"; unset UNSLOTH_HOME; UNSLOTH_STUDIO_HOME="  $PADDED  "
         export HOME UNSLOTH_STUDIO_HOME; . "$HELPERS_FILE"; _master_root ) )
assert_eq "a padded override finds the note" "$got" "$PADDED_MASTER"
got=$( ( HOME="$HOME"; unset UNSLOTH_HOME UNSLOTH_STUDIO_HOME; STUDIO_HOME="  $PADDED  "
         export HOME STUDIO_HOME; . "$HELPERS_FILE"; _master_root ) )
assert_eq "the padded alias finds it too" "$got" "$PADDED_MASTER"

echo "== only the installer's own stale locks are swept =="
# Stale locks are <name>.stale.<pid>; a looser glob matched the user's hidden files.
STALE="$_TMP_ROOT/stale-root"
mkdir -p "$STALE"
: > "$STALE/.node.install.lock.stale.4242"
: > "$STALE/.backup.install.lock.stale.copy"
: > "$STALE/.llama.cpp.install.lock.stale.notapid"
run_block "$HOME" "$STALE"
assert_nodir "the installer's own stale lock goes" "$STALE/.node.install.lock.stale.4242"
assert_present "an unrelated hidden file stays" "$STALE/.backup.install.lock.stale.copy"
assert_present "a non-numeric suffix stays" "$STALE/.llama.cpp.install.lock.stale.notapid"

echo "== a note with a second line is refused, as the comment beside it promises =="
# The Python readers strip the whole file, so a multi-line note is not a directory here either.
MULTI_MASTER="$_TMP_ROOT/multiline-master"
MULTI="$MULTI_MASTER/studio"
mkdir -p "$MULTI/share" "$_TMP_ROOT/elsewhere"
printf '%s\n%s\n' "$MULTI_MASTER" "$_TMP_ROOT/elsewhere" > "$MULTI/share/.unsloth-master-root"
got=$( ( HOME="$HOME"; unset UNSLOTH_HOME; UNSLOTH_STUDIO_HOME="$MULTI"
         export HOME UNSLOTH_STUDIO_HOME; . "$HELPERS_FILE"; _master_root ) )
assert_eq "a two-line note names no master root" "$got" ""

printf '%s\n\n' "$MULTI_MASTER" > "$MULTI/share/.unsloth-master-root"
got=$( ( HOME="$HOME"; unset UNSLOTH_HOME; UNSLOTH_STUDIO_HOME="$MULTI"
         export HOME UNSLOTH_STUDIO_HOME; . "$HELPERS_FILE"; _master_root ) )
assert_eq "a trailing blank line is still one line" "$got" "$MULTI_MASTER"

printf '%s\n' "$MULTI_MASTER" > "$MULTI/share/.unsloth-master-root"
got=$( ( HOME="$HOME"; unset UNSLOTH_HOME; UNSLOTH_STUDIO_HOME="$MULTI"
         export HOME UNSLOTH_STUDIO_HOME; . "$HELPERS_FILE"; _master_root ) )
assert_eq "a one-line note is still honoured" "$got" "$MULTI_MASTER"

echo "== a note in the legacy tree cannot aim the uninstall at a second install =="
# A legacy note naming $HOME must be declined, as storage_roots._is_legacy_studio_tree does,
# or a second install at $HOME/studio is removed.
LEGACY_HOME="$_TMP_ROOT/legacy-note-home"
mkdir -p "$LEGACY_HOME/.unsloth/studio/share"
printf '%s\n' "$LEGACY_HOME" > "$LEGACY_HOME/.unsloth/studio/share/.unsloth-master-root"
mkdir -p "$LEGACY_HOME/studio/share" "$LEGACY_HOME/studio/outputs"
: > "$LEGACY_HOME/studio/.unsloth-studio-owned"
: > "$LEGACY_HOME/studio/studio.db"
got=$( ( HOME="$LEGACY_HOME"; unset UNSLOTH_HOME UNSLOTH_STUDIO_HOME STUDIO_HOME
         export HOME; . "$HELPERS_FILE"; _master_root ) )
assert_eq "a legacy-tree note names no master root" "$got" ""
assert_dir "the second install is untouched" "$LEGACY_HOME/studio/studio.db"

echo
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
