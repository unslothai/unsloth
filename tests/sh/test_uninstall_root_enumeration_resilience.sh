#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# _custom_studio_roots must enumerate EVERY root it knows about, even when an earlier one is
# unusable. It runs under `set -e`, so two ordinary-looking constructs used to abort the whole
# enumeration mid-stream and take the remaining roots with them, silently and with no
# diagnostic -- the uninstaller reported success having removed less than it found:
#
#   * `_canon=$(CDPATH= cd -P -- "$_r" ... && pwd -P)` -- an assignment carries the exit status
#     of its command substitution, so a root that cannot be canonicalized killed the subshell.
#     The `[ -n "$_canon" ]` line written to handle that case never ran.
#   * `[ -n "$_master" ] && _emit "$_master"` as the LAST statement of _from_conf -- a studio.conf
#     with no `export UNSLOTH_HOME` line (which is every non-portable install) made the function
#     return non-zero, and `set -e` acts on the call.
#
# Both are asserted through the real script by their OUTCOME: a LATER root still gets removed.
# The interpreter matters here (the abort exits 2 under dash/sh and 1 under bash), so each case
# runs under every shell present.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UNINSTALL_SH="$SCRIPT_DIR/../../scripts/uninstall.sh"
PASS=0
FAIL=0

# Same reason as tests/sh/test_uninstall_prebuilt_artifacts.sh: on WSL the uninstall
# body reaches Windows-side state and sudo, neither contained by a fixture HOME.
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "  SKIP: WSL -- the uninstall body reaches Windows-side state outside the fixture"
    exit 0
fi

SHELLS=""
for _sh in sh dash bash; do
    if command -v "$_sh" >/dev/null 2>&1; then
        SHELLS="$SHELLS $_sh"
    fi
done

assert_gone() {
    if [ -e "$2" ]; then
        echo "  FAIL: $1 (still present: $2)"; FAIL=$((FAIL + 1))
    else
        echo "  PASS: $1"; PASS=$((PASS + 1))
    fi
}

assert_present() {
    if [ -e "$2" ]; then
        echo "  PASS: $1"; PASS=$((PASS + 1))
    else
        echo "  FAIL: $1 (removed: $2)"; FAIL=$((FAIL + 1))
    fi
}

# A complete custom install at $1, the shape install.sh writes for a non-portable custom root:
# UNSLOTH_EXE in share/studio.conf and no `export UNSLOTH_HOME` line, since there is no master
# root. That absence is what case 2 turns on.
make_install() {
    mkdir -p "$1/unsloth_studio" "$1/share" "$1/bin"
    : > "$1/unsloth_studio/.unsloth-studio-owned"
    printf '%s\n' "$2" > "$1/studio.db"
    printf "UNSLOTH_EXE='%s'\n" "$1/unsloth_studio/bin/unsloth" > "$1/share/studio.conf"
}

# env -i: the caller's own UNSLOTH_* / XDG_* would send removals outside the fixture.
run_uninstall() { # $1 interpreter, $2 fixture dir, then env assignments
    _ru_sh="$1"; _ru_dir="$2"; shift 2
    mkdir -p "$_ru_dir/home/.local/share/unsloth" "$_ru_dir/home/.local/bin" \
        "$_ru_dir/home/Desktop"
    env -i PATH="$PATH" HOME="$_ru_dir/home" TMPDIR="$_ru_dir" \
        XDG_RUNTIME_DIR="$_ru_dir/run" XDG_DATA_HOME="$_ru_dir/home/.local/share" \
        XDG_CACHE_HOME="$_ru_dir/home/.cache" XDG_CONFIG_HOME="$_ru_dir/home/.config" \
        XDG_STATE_HOME="$_ru_dir/home/.local/state" "$@" \
        "$_ru_sh" "$UNINSTALL_SH" > "$_ru_dir/uninstall.log" 2>&1
}

# The install the DEFAULT-mode studio.conf names. It is enumerated last, after every variable,
# so it is the one an early abort silently strands.
seed_later_root() { # $1 fixture dir
    make_install "$1/later" later-root-history
    printf "UNSLOTH_EXE='%s'\n" "$1/later/unsloth_studio/bin/unsloth" \
        > "$1/home/.local/share/unsloth/studio.conf"
}

TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$TMP_ROOT"' EXIT

# Case 1: UNSLOTH_STUDIO_HOME names a directory that is not there (the user removed the install
# by hand and left the variable exported). Nothing about that should stop the default-mode conf
# from being read. A missing path is never deleted, so this case has no removal ordering in it.
for SH in $SHELLS; do
    CASE1="$TMP_ROOT/unresolvable-$SH"
    mkdir -p "$CASE1/home/.local/share/unsloth"
    seed_later_root "$CASE1"
    run_uninstall "$SH" "$CASE1" UNSLOTH_STUDIO_HOME="$CASE1/does-not-exist"
    assert_gone "$SH: an unresolvable root does not strand the one enumerated after it" \
        "$CASE1/later"
done

# Case 2: a studio.conf carrying UNSLOTH_EXE but no `export UNSLOTH_HOME`, which is what every
# non-portable install writes. The root is $HOME so the hard deny list refuses to remove it:
# that keeps the fixture intact for the whole run, so the assertion is about the enumeration
# reaching the later root and not about which rm won a race.
for SH in $SHELLS; do
    CASE2="$TMP_ROOT/nomaster-$SH"
    mkdir -p "$CASE2/home/.local/share/unsloth" "$CASE2/home/share"
    seed_later_root "$CASE2"
    printf "UNSLOTH_EXE='%s'\n" "$CASE2/home/unsloth_studio/bin/unsloth" \
        > "$CASE2/home/share/studio.conf"
    run_uninstall "$SH" "$CASE2" UNSLOTH_STUDIO_HOME="$CASE2/home"
    assert_gone "$SH: a conf with no master root does not strand the next conf's install" \
        "$CASE2/later"
    assert_present "$SH: and the deny-listed root it named is still refused" "$CASE2/home"
done

# Case 3: the ordinary single-root uninstall still works and still exits 0 under every shell,
# so the guards above cannot have turned an abort into a different kind of silent skip.
for SH in $SHELLS; do
    CASE3="$TMP_ROOT/plain-$SH"
    mkdir -p "$CASE3/home/.local/share"
    make_install "$CASE3/custom" custom-history
    if run_uninstall "$SH" "$CASE3" UNSLOTH_STUDIO_HOME="$CASE3/custom"; then
        echo "  PASS: $SH: a plain custom-root uninstall exits 0"; PASS=$((PASS + 1))
    else
        echo "  FAIL: $SH: a plain custom-root uninstall exits 0"; FAIL=$((FAIL + 1))
    fi
    assert_gone "$SH: a plain custom-root uninstall removes the root" "$CASE3/custom"
done

echo ""
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
