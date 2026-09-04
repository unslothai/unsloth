#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# A whitespace-only root variable is what an UNSET one looks like everywhere else: install.sh
# runs UNSLOTH_HOME through _trim_ws before it decides portable mode, storage_roots.py reads
# every root as (os.environ.get(...) or "").strip(), and unsloth_cli/commands/studio.py asks
# _env_unset_or_blank. The uninstaller's precedence chain used the RAW values, so
# `UNSLOTH_HOME=" " UNSLOTH_STUDIO_HOME=/real/install` took the first branch, emitted a root
# made of spaces, and never looked at the variable naming the install that environment really
# produced -- an UNDER-delete, leaving the requested custom root and its studio.db on disk.
#
# Every case runs the real scripts/uninstall.sh against a fixture HOME and asserts the removal
# OUTCOME. Whitespace-only, empty and genuinely-set values are covered for each of the three
# variables, so the fix cannot pass by ignoring the variables altogether.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UNINSTALL_SH="$SCRIPT_DIR/../../scripts/uninstall.sh"
PASS=0
FAIL=0

# Same reason as tests/sh/test_uninstall_home_precedence.sh: on WSL the uninstall body reaches
# Windows-side state and sudo, neither contained by a fixture HOME.
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "  SKIP: WSL -- the uninstall body reaches Windows-side state outside the fixture"
    exit 0
fi

assert_gone() {
    if [ -e "$2" ] || [ -L "$2" ]; then
        echo "  FAIL: $1 (still present: $2)"; FAIL=$((FAIL + 1))
    else
        echo "  PASS: $1"; PASS=$((PASS + 1))
    fi
}

assert_content() {
    if [ -f "$2" ] && [ "$(cat "$2")" = "$3" ]; then
        echo "  PASS: $1"; PASS=$((PASS + 1))
    else
        echo "  FAIL: $1 (missing or rewritten: $2)"; FAIL=$((FAIL + 1))
    fi
}

assert_grep() {
    if grep -q "$3" "$2"; then
        echo "  PASS: $1"; PASS=$((PASS + 1))
    else
        echo "  FAIL: $1 (no match for '$3' in $2)"; FAIL=$((FAIL + 1))
    fi
}

# A complete custom install at $1, with $2 in its studio.db so a deletion is visible in the
# content rather than only in the path.
make_flat_install() {
    mkdir -p "$1/unsloth_studio" "$1/share" "$1/bin"
    : > "$1/unsloth_studio/.unsloth-studio-owned"
    printf '%s\n' "$2" > "$1/studio.db"
    printf "UNSLOTH_EXE='%s'\n" "$1/unsloth_studio/bin/unsloth" > "$1/share/studio.conf"
}

# A portable master root at $1: the layout install.sh --root writes.
make_portable_install() {
    mkdir -p "$1/studio/unsloth_studio" "$1/bin" "$1/share"
    : > "$1/studio/unsloth_studio/.unsloth-studio-owned"
    printf '%s\n' "$1" > "$1/.unsloth-portable-root"
    printf '%s\n' "$2" > "$1/studio/studio.db"
    printf "UNSLOTH_EXE='%s'\nexport UNSLOTH_HOME='%s'\n" \
        "$1/studio/unsloth_studio/bin/unsloth" "$1" > "$1/share/studio.conf"
}

# env -i: the caller's own UNSLOTH_* / XDG_* would send removals outside the fixture. The
# variables under test are appended by the caller as NAME=VALUE arguments.
run_uninstall() { # $1 fixture dir, $2 log name, then env assignments
    _ru_dir="$1"; _ru_log="$2"; shift 2
    mkdir -p "$_ru_dir/home/.local/share" "$_ru_dir/home/.local/bin" "$_ru_dir/home/Desktop"
    env -i PATH="$PATH" HOME="$_ru_dir/home" TMPDIR="$_ru_dir" \
        XDG_RUNTIME_DIR="$_ru_dir/run" XDG_DATA_HOME="$_ru_dir/home/.local/share" \
        XDG_CACHE_HOME="$_ru_dir/home/.cache" XDG_CONFIG_HOME="$_ru_dir/home/.config" \
        XDG_STATE_HOME="$_ru_dir/home/.local/state" "$@" \
        sh "$UNINSTALL_SH" > "$_ru_dir/$_ru_log" 2>&1
}

_CASES=""
new_case() { # prints a fresh fixture dir and keeps it on the cleanup list
    _nc=$(mktemp -d)
    _CASES="$_CASES $_nc"
    # shellcheck disable=SC2064
    trap "rm -rf $_CASES" EXIT
    printf '%s\n' "$_nc"
}

# Case 1: the reported defect. A whitespace-only UNSLOTH_HOME must not claim the chain and
# hide the UNSLOTH_STUDIO_HOME install this environment actually made.
C1=$(new_case)
make_flat_install "$C1/custom" custom-history
run_uninstall "$C1" ws.log UNSLOTH_HOME='   ' UNSLOTH_STUDIO_HOME="$C1/custom"
assert_gone "a whitespace-only UNSLOTH_HOME falls through to UNSLOTH_STUDIO_HOME" "$C1/custom"

# Case 2: the same with an empty UNSLOTH_HOME, the value that always worked.
C2=$(new_case)
make_flat_install "$C2/custom" custom-history
run_uninstall "$C2" empty.log UNSLOTH_HOME='' UNSLOTH_STUDIO_HOME="$C2/custom"
assert_gone "an empty UNSLOTH_HOME falls through to UNSLOTH_STUDIO_HOME" "$C2/custom"

# Case 3: a tab/newline-only value is whitespace too.
C3=$(new_case)
make_flat_install "$C3/custom" custom-history
run_uninstall "$C3" tab.log UNSLOTH_HOME="$(printf '\t')" UNSLOTH_STUDIO_HOME="$C3/custom"
assert_gone "a tab-only UNSLOTH_HOME falls through to UNSLOTH_STUDIO_HOME" "$C3/custom"

# Case 4: the precedence this must not regress. A GENUINELY set UNSLOTH_HOME still wins, so a
# stale UNSLOTH_STUDIO_HOME naming a second install keeps its tree and its history.
C4=$(new_case)
make_portable_install "$C4/portable" portable-history
make_flat_install "$C4/other/studio" second-install-history
run_uninstall "$C4" precedence.log \
    UNSLOTH_HOME="$C4/portable" UNSLOTH_STUDIO_HOME="$C4/other/studio"
assert_gone "a set UNSLOTH_HOME still removes its portable root" "$C4/portable"
assert_content "and still suppresses a stale UNSLOTH_STUDIO_HOME" \
    "$C4/other/studio/studio.db" second-install-history

# Case 5: a set UNSLOTH_HOME still suppresses STUDIO_HOME as well.
C5=$(new_case)
make_portable_install "$C5/portable" portable-history
make_flat_install "$C5/other/studio" second-install-history
run_uninstall "$C5" alias_precedence.log \
    UNSLOTH_HOME="$C5/portable" STUDIO_HOME="$C5/other/studio"
assert_gone "a set UNSLOTH_HOME still removes its portable root beside STUDIO_HOME" \
    "$C5/portable"
assert_content "and still suppresses a stale STUDIO_HOME" \
    "$C5/other/studio/studio.db" second-install-history

# Case 6: blank in the first TWO slots reaches the STUDIO_HOME alias.
C6=$(new_case)
make_flat_install "$C6/custom" alias-history
run_uninstall "$C6" alias.log \
    UNSLOTH_HOME='  ' UNSLOTH_STUDIO_HOME=' ' STUDIO_HOME="$C6/custom"
assert_gone "blank UNSLOTH_HOME and UNSLOTH_STUDIO_HOME reach STUDIO_HOME" "$C6/custom"

# Case 7: a whitespace-only UNSLOTH_STUDIO_HOME alone does not suppress STUDIO_HOME, and a set
# UNSLOTH_STUDIO_HOME still does.
C7=$(new_case)
make_flat_install "$C7/aliased" alias-history
run_uninstall "$C7" blank_studio.log UNSLOTH_STUDIO_HOME='	 ' STUDIO_HOME="$C7/aliased"
assert_gone "a blank UNSLOTH_STUDIO_HOME does not suppress STUDIO_HOME" "$C7/aliased"
make_flat_install "$C7/primary" primary-history
make_flat_install "$C7/alias" kept-history
run_uninstall "$C7" set_studio.log \
    UNSLOTH_STUDIO_HOME="$C7/primary" STUDIO_HOME="$C7/alias"
assert_gone "a set UNSLOTH_STUDIO_HOME still wins over STUDIO_HOME" "$C7/primary"
assert_content "and the STUDIO_HOME install keeps its history" \
    "$C7/alias/studio.db" kept-history

# Case 8: padding around a real path is trimmed rather than making every -d test fail, which is
# what install.sh does with the same value (it installs at the trimmed path).
C8=$(new_case)
make_flat_install "$C8/custom" padded-history
run_uninstall "$C8" padded.log UNSLOTH_STUDIO_HOME="  $C8/custom  "
assert_gone "a padded UNSLOTH_STUDIO_HOME still names its install" "$C8/custom"

# Case 9: with every root variable blank the run behaves like a bare one -- it removes the
# default install, touches nothing else, and still prints the hint that explains how to reach a
# custom root, which a raw -z test suppressed.
C9=$(new_case)
mkdir -p "$C9/home/.unsloth/studio/unsloth_studio"
printf 'default-history\n' > "$C9/home/.unsloth/studio/studio.db"
make_flat_install "$C9/untouched" untouched-history
run_uninstall "$C9" blank_all.log UNSLOTH_HOME=' ' UNSLOTH_STUDIO_HOME=' ' STUDIO_HOME=' '
assert_gone "blank roots still remove the default install" "$C9/home/.unsloth/studio"
assert_content "and remove nothing outside it" "$C9/untouched/studio.db" untouched-history
assert_grep "and the custom-root hint is still printed" \
    "$C9/blank_all.log" 'UNSLOTH_STUDIO_HOME or STUDIO_HOME'

echo ""
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
