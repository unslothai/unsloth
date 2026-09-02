#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# The "contained" check backs the single-root install promise (issue #8865), and it runs on
# the macOS legs too, where /bin/bash is 3.2. Cover both halves: it has to parse there, and
# it has to actually distinguish a contained install from a leaking one.
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

# ── 1. bash 3.2 parses the case arms inside the command substitution ──
# macOS ships bash 3.2.57, whose parser ends a $( ) at the first ")" that closes a case
# pattern unless the pattern also opens with "(". Without the leading paren the whole
# script dies at load time with "syntax error near unexpected token ';;'", and every
# other check in it goes down with it.
_bad=$(awk '
  /^        strays=\$\(/ { inside = 1 }
  inside && /^          done\)/ { inside = 0 }
  inside && /\) continue ;;$/ && $0 !~ /^ *\(/ { print NR ": " $0 }
' "$ASSERT_SH")
if [ -z "$_bad" ]; then
    check "case patterns inside \$( ) open with '(' for bash 3.2" 0
else
    check "case patterns inside \$( ) open with '(' for bash 3.2" 1
    printf '%s\n' "$_bad"
fi

# ── 2. a contained install passes ──
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

# ── 3. a leak outside the root fails, and the message names the offender ──
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

# ── 4. a run that proved nothing must not pass ──
# An empty root with no writes newer than the stamp means the install never happened or
# the stamp was lost, and a naive implementation reports a clean home either way.
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
