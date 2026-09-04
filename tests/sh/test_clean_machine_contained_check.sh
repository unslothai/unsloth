#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# The "contained" check runs on the macOS legs, where /bin/bash is 3.2.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ASSERT_SH="$SCRIPT_DIR/../../.github/scripts/clean-machine-assert.sh"

ROOT=$(mktemp -d)
trap 'rm -rf "$ROOT"' EXIT
PASS=0
FAIL=0

check() {
    _label="$1"; _ok="$2"
    if [ "$_ok" = 0 ]; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label"
        FAIL=$((FAIL + 1))
    fi
}

# macOS bash 3.2 ends a $( ) at the first ")" closing a case pattern unless the pattern
# also opens with "("; without it the whole script dies at load time.
# Any pattern-closing ")" counts, not only the ones on a "... ) continue ;;" line:
# a pattern whose body sits on the next line still ends its own line with ")".
_bad=$(awk '
  /^        strays=\$\(/ { inside = 1 }
  inside && /^          done\)/ { inside = 0 }
  inside && /^ *#/ { next }
  inside && /\)( continue ;;)?$/ && $0 !~ /^ *\(/ { print NR ": " $0 }
' "$ASSERT_SH")
if [ -z "$_bad" ]; then
    check "case patterns inside \$( ) open with '(' for bash 3.2" 0
else
    check "case patterns inside \$( ) open with '(' for bash 3.2" 1
    printf '%s\n' "$_bad"
fi

HOME_A="$ROOT/home_a"
ROOT_A="$HOME_A/unsloth"
mkdir -p "$HOME_A"
touch "$ROOT/stamp"
sleep 1
mkdir -p "$ROOT_A/cache/uv" "$ROOT_A/bin"
: > "$ROOT_A/bin/unsloth"
set +e
HOME="$HOME_A" CONTAINED_ROOT="$ROOT_A" CONTAINED_STAMP="$ROOT/stamp" \
    bash "$ASSERT_SH" contained > "$ROOT/out1.log" 2>&1
_rc=$?
set -e
[ "$_rc" -eq 0 ] || cat "$ROOT/out1.log"
check "a contained install passes" $((_rc == 0 ? 0 : 1))

mkdir -p "$HOME_A/.cache/uv/archive-v0"
set +e
HOME="$HOME_A" CONTAINED_ROOT="$ROOT_A" CONTAINED_STAMP="$ROOT/stamp" \
    bash "$ASSERT_SH" contained > "$ROOT/out2.log" 2>&1
_rc=$?
set -e
check "a leak into \$HOME fails the check" $((_rc != 0 ? 0 : 1))
grep -q "archive-v0" "$ROOT/out2.log" \
    && check "the failure names the leaked path" 0 \
    || check "the failure names the leaked path" 1

# A symlink is how an interpreter/launcher leak lands (an unpinned UV_PYTHON_BIN_DIR
# links managed Pythons into ~/.local/bin, uv drops ~/.local/bin/unsloth). Placed
# directly in an allowed directory NODE, only the link itself is ever newer, so a
# scan that skips symlinks reports OK on exactly the leak this mode exists to catch.
HOME_S="$ROOT/home_s"
ROOT_S="$HOME_S/unsloth"
mkdir -p "$HOME_S/.local" "$HOME_S/.config"
touch "$ROOT/stamp_s"
sleep 1
mkdir -p "$ROOT_S/bin"
: > "$ROOT_S/bin/unsloth"
ln -s "$ROOT_S/bin/unsloth" "$HOME_S/.local/leaked-python3.12"
set +e
HOME="$HOME_S" CONTAINED_ROOT="$ROOT_S" CONTAINED_STAMP="$ROOT/stamp_s" \
    bash "$ASSERT_SH" contained > "$ROOT/out_s.log" 2>&1
_rc=$?
set -e
check "a symlink leak into an allowed \$HOME directory fails the check" $((_rc != 0 ? 0 : 1))
grep -q "leaked-python3.12" "$ROOT/out_s.log" \
    && check "the failure names the leaked symlink" 0 \
    || { check "the failure names the leaked symlink" 1; cat "$ROOT/out_s.log"; }

# The allowance for ~/.local, ~/.cache and ~/.config is for an mtime bump on a real
# directory. The same name as a symlink redirects the whole subtree out of $HOME and
# hides every child from the scan, so it must not inherit the allowance.
rm -f "$HOME_S/.local/leaked-python3.12"
rmdir "$HOME_S/.config"
ln -s "$ROOT/elsewhere" "$HOME_S/.config"
set +e
HOME="$HOME_S" CONTAINED_ROOT="$ROOT_S" CONTAINED_STAMP="$ROOT/stamp_s" \
    bash "$ASSERT_SH" contained > "$ROOT/out_s2.log" 2>&1
_rc=$?
set -e
check "an allowed directory node replaced by a symlink fails the check" $((_rc != 0 ? 0 : 1))

# The converse: containment is about writing outside the root. A symlink INSIDE the
# root stays allowed however it points, or the portable install's own shims fail.
rm -f "$HOME_S/.config"
mkdir -p "$HOME_S/.config"
touch "$HOME_S/.local" "$HOME_S/.config"
ln -s /opt/somewhere/python3.12 "$ROOT_S/bin/python3.12"
set +e
HOME="$HOME_S" CONTAINED_ROOT="$ROOT_S" CONTAINED_STAMP="$ROOT/stamp_s" \
    bash "$ASSERT_SH" contained > "$ROOT/out_s3.log" 2>&1
_rc=$?
set -e
[ "$_rc" -eq 0 ] || cat "$ROOT/out_s3.log"
check "a symlink inside the root, and a touched allowed directory, still pass" $((_rc == 0 ? 0 : 1))

HOME_B="$ROOT/home_b"
ROOT_B="$ROOT/root_b"
mkdir -p "$HOME_B" "$ROOT_B"
touch "$ROOT/stamp2"
set +e
HOME="$HOME_B" CONTAINED_ROOT="$ROOT_B" CONTAINED_STAMP="$ROOT/stamp2" \
    bash "$ASSERT_SH" contained > "$ROOT/out3.log" 2>&1
_rc=$?
set -e
check "an empty root cannot pass by proving nothing" $((_rc != 0 ? 0 : 1))

echo
echo "ran $((PASS + FAIL)); failed: ${FAIL:-none}"
[ "$FAIL" -eq 0 ]
