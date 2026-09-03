#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Regression test: the legacy stable-diffusion.cpp sibling cleanup in scripts/uninstall.sh must
# not walk up from a portable MASTER root.
#
# _custom_studio_roots emits two kinds of entry: Studio roots (the UNSLOTH_EXE walk,
# UNSLOTH_STUDIO_HOME) and, for a portable install, the master root above them (UNSLOTH_HOME and
# studio.conf's `export UNSLOTH_HOME`). The sibling cleanup reads every entry as a Studio root and
# removes <parent>/stable-diffusion.cpp when it carries the Unsloth owner marker, because an older
# build derived that path from UNSLOTH_STUDIO_HOME.parent. For a nested portable root
# (UNSLOTH_HOME=/parent/portable, Studio root /parent/portable/studio) the master root's parent is
# one level too high: /parent/stable-diffusion.cpp is a path that install never used, and a
# SEPARATE installation sitting beside the portable root can own it. It was deleted.
#
# The uninstaller runs for real against a fixture HOME, one mktemp -d per case, so what is
# asserted is the removal OUTCOME. The flat cases at the end are the other half of the test: a
# root that holds the venv directly IS the Studio root, and its legacy sibling must still go.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UNINSTALL_SH="$SCRIPT_DIR/../../scripts/uninstall.sh"
PASS=0
FAIL=0

# Same reason as tests/sh/test_uninstall_portable_root_scope.sh: on WSL the uninstall body
# reaches Windows-side state and sudo, neither contained by a fixture HOME.
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "  SKIP: WSL -- the uninstall body reaches Windows-side state outside the fixture"
    exit 0
fi

_TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$_TMP_ROOT"' EXIT

# Explicit template: -p is GNU-only and a bare mktemp -d lands outside _TMP_ROOT on macOS.
new_case() { mktemp -d "$_TMP_ROOT/case.XXXXXX"; }

assert_gone() {
    if [ -e "$2" ] || [ -L "$2" ]; then
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

# An sd.cpp build carrying the marker install_sd_cpp_prebuilt writes. Only marked trees are
# candidates for removal at all, so an unmarked fixture would assert nothing here.
make_owned_sd_cpp() {
    mkdir -p "$1/build/bin"
    : > "$1/build/bin/sd-cli"
    : > "$1/.unsloth-studio-owned"
}

# env -i: the caller's own UNSLOTH_* / XDG_* would send removals outside the fixture.
run_uninstall() { # $1 = fixture HOME, $2 = log name, rest = VAR=VALUE overrides
    _ru_home="$1"; _ru_log="$2"; shift 2
    env -i PATH="$PATH" HOME="$_ru_home" TMPDIR="$_TMP_ROOT" \
        XDG_RUNTIME_DIR="$_TMP_ROOT/run-$_ru_log" XDG_DATA_HOME="$_ru_home/.local/share" \
        XDG_CACHE_HOME="$_ru_home/.cache" XDG_CONFIG_HOME="$_ru_home/.config" \
        XDG_STATE_HOME="$_ru_home/.local/state" "$@" \
        sh "$UNINSTALL_SH" > "$_TMP_ROOT/$_ru_log.log" 2>&1
}

# 1. Nested portable root beside a separate installation. Uninstalling the portable root must
#    take the whole portable tree (including its own sd.cpp, which sits INSIDE it) and leave the
#    neighbour's legacy sd.cpp alone.
C1=$(new_case)
H1="$C1/home"
PARENT1="$C1/parent"
PORTABLE1="$PARENT1/portable"
OTHER1="$PARENT1/other"
mkdir -p "$H1/.local/share" "$H1/.local/bin" \
    "$PORTABLE1/studio/unsloth_studio" "$PORTABLE1/share" "$PORTABLE1/bin" \
    "$OTHER1/unsloth_studio"
echo portable > "$PORTABLE1/studio/studio.db"
: > "$PORTABLE1/studio/unsloth_studio/.unsloth-studio-owned"
printf '%s\n' "$PORTABLE1" > "$PORTABLE1/.unsloth-portable-root"
printf "UNSLOTH_EXE='%s'\nexport UNSLOTH_HOME='%s'\n" \
    "$PORTABLE1/studio/unsloth_studio/bin/unsloth" "$PORTABLE1" \
    > "$PORTABLE1/share/studio.conf"
# The portable install's OWN legacy sibling: the parent of its Studio root, inside the tree.
make_owned_sd_cpp "$PORTABLE1/stable-diffusion.cpp"
# The neighbour: a separate custom install and the legacy sd.cpp an older build of it wrote.
: > "$OTHER1/unsloth_studio/.unsloth-studio-owned"
make_owned_sd_cpp "$PARENT1/stable-diffusion.cpp"

run_uninstall "$H1" nested UNSLOTH_HOME="$PORTABLE1"

assert_gone    "nested portable master root removed"          "$PORTABLE1"
assert_gone    "its own nested sd.cpp went with the tree"     "$PORTABLE1/stable-diffusion.cpp"
assert_present "a neighbour's legacy sd.cpp is kept"          "$PARENT1/stable-diffusion.cpp"
assert_present "and its build is intact"                      "$PARENT1/stable-diffusion.cpp/build/bin/sd-cli"
assert_present "the neighbouring install itself is kept"      "$OTHER1/unsloth_studio"

# 2. Same layout reached through studio.conf rather than UNSLOTH_HOME: the master root is emitted
#    from `export UNSLOTH_HOME` there too, so pointing UNSLOTH_STUDIO_HOME at the Studio root
#    still puts the master root through the same loop.
C2=$(new_case)
H2="$C2/home"
PARENT2="$C2/parent"
PORTABLE2="$PARENT2/portable"
mkdir -p "$H2/.local/share" "$H2/.local/bin" \
    "$PORTABLE2/studio/unsloth_studio" "$PORTABLE2/share"
: > "$PORTABLE2/studio/unsloth_studio/.unsloth-studio-owned"
printf '%s\n' "$PORTABLE2" > "$PORTABLE2/.unsloth-portable-root"
printf "UNSLOTH_EXE='%s'\nexport UNSLOTH_HOME='%s'\n" \
    "$PORTABLE2/studio/unsloth_studio/bin/unsloth" "$PORTABLE2" \
    > "$PORTABLE2/share/studio.conf"
make_owned_sd_cpp "$PARENT2/stable-diffusion.cpp"

run_uninstall "$H2" conf UNSLOTH_STUDIO_HOME="$PORTABLE2"

assert_gone    "master root from studio.conf removed"              "$PORTABLE2"
assert_present "and the sd.cpp above it is kept"                   "$PARENT2/stable-diffusion.cpp"
assert_present "with its build intact"                             "$PARENT2/stable-diffusion.cpp/build/bin/sd-cli"

# 3. The other half: a FLAT root, holding the venv directly, IS the Studio root, so
#    <parent>/stable-diffusion.cpp is the legacy path that install really used and must still be
#    removed. Without this the fix could pass by never walking to a parent at all.
C3=$(new_case)
H3="$C3/home"
PARENT3="$C3/parent"
FLAT3="$PARENT3/studio"
mkdir -p "$H3/.local/share" "$H3/.local/bin" "$FLAT3/unsloth_studio" "$FLAT3/share"
: > "$FLAT3/unsloth_studio/.unsloth-studio-owned"
printf "UNSLOTH_EXE='%s'\n" "$FLAT3/unsloth_studio/bin/unsloth" > "$FLAT3/share/studio.conf"
make_owned_sd_cpp "$PARENT3/stable-diffusion.cpp"

run_uninstall "$H3" flat UNSLOTH_STUDIO_HOME="$FLAT3"

assert_gone "flat custom root removed"                        "$FLAT3"
assert_gone "flat root's legacy sd.cpp sibling still removed" "$PARENT3/stable-diffusion.cpp"

# 4. A FLAT portable root (UNSLOTH_HOME naming a root that holds the venv itself, the layout
#    storage_roots.studio_root() calls flat) is a Studio root as well: its sibling still goes.
C4=$(new_case)
H4="$C4/home"
PARENT4="$C4/parent"
FLAT4="$PARENT4/portable"
mkdir -p "$H4/.local/share" "$H4/.local/bin" "$FLAT4/unsloth_studio" "$FLAT4/share"
: > "$FLAT4/unsloth_studio/.unsloth-studio-owned"
printf '%s\n' "$FLAT4" > "$FLAT4/.unsloth-portable-root"
printf "UNSLOTH_EXE='%s'\nexport UNSLOTH_HOME='%s'\n" \
    "$FLAT4/unsloth_studio/bin/unsloth" "$FLAT4" > "$FLAT4/share/studio.conf"
make_owned_sd_cpp "$PARENT4/stable-diffusion.cpp"

run_uninstall "$H4" flatportable UNSLOTH_HOME="$FLAT4"

assert_gone "flat portable root removed"                          "$FLAT4"
assert_gone "flat portable root's legacy sd.cpp sibling removed"   "$PARENT4/stable-diffusion.cpp"

# 5. The same parent walk feeds _sd_cpp_sibling_bases, which decides whose sd-server is SIGTERMed
#    before a tree is deleted. From a master root it reaches the neighbour's build and kills a
#    server this uninstall has no business touching. pkill is stubbed to record its argv (and to
#    exit 1 like the real one on no match, so a missing `|| true` would still show up).
C5=$(new_case)
H5="$C5/home"
PARENT5="$C5/parent"
PORTABLE5="$PARENT5/portable"
mkdir -p "$H5/.local/share" "$H5/.local/bin" \
    "$PORTABLE5/studio/unsloth_studio" "$PORTABLE5/share"
: > "$PORTABLE5/studio/unsloth_studio/.unsloth-studio-owned"
printf '%s\n' "$PORTABLE5" > "$PORTABLE5/.unsloth-portable-root"
printf "UNSLOTH_EXE='%s'\nexport UNSLOTH_HOME='%s'\n" \
    "$PORTABLE5/studio/unsloth_studio/bin/unsloth" "$PORTABLE5" \
    > "$PORTABLE5/share/studio.conf"
# Where install_sd_cpp_prebuilt.default_install_dir() puts it today for this layout:
# <UNSLOTH_STUDIO_HOME>/stable-diffusion.cpp, and the Studio root is <master>/studio.
make_owned_sd_cpp "$PORTABLE5/studio/stable-diffusion.cpp"
# ... and where the older build hung it, off the Studio root's parent.
make_owned_sd_cpp "$PORTABLE5/stable-diffusion.cpp"
make_owned_sd_cpp "$PARENT5/stable-diffusion.cpp"

STUB_BIN="$C5/stubbin"
PKILL_LOG="$C5/pkill.args"
mkdir -p "$STUB_BIN"
cat > "$STUB_BIN/pkill" <<EOF
#!/bin/sh
printf '%s\n' "\$*" >> "$PKILL_LOG"
exit 1
EOF
chmod +x "$STUB_BIN/pkill"
: > "$PKILL_LOG"

env -i PATH="$STUB_BIN:$PATH" HOME="$H5" TMPDIR="$_TMP_ROOT" \
    XDG_RUNTIME_DIR="$_TMP_ROOT/run-pkill" XDG_DATA_HOME="$H5/.local/share" \
    XDG_CACHE_HOME="$H5/.cache" XDG_CONFIG_HOME="$H5/.config" \
    XDG_STATE_HOME="$H5/.local/state" UNSLOTH_HOME="$PORTABLE5" \
    sh "$UNINSTALL_SH" > "$_TMP_ROOT/pkill.log" 2>&1

# The recorded patterns are BRE-escaped by _pkill_escape, so strip the backslashes before
# matching; a fixture path under mktemp -d has none of its own.
PKILL_SEEN="$C5/pkill.plain"
sed -e 's/\\//g' "$PKILL_LOG" > "$PKILL_SEEN"
# The stub has to have been reached at all, or the assertions below pass vacuously.
if [ -s "$PKILL_SEEN" ]; then
    echo "  PASS: the pkill stub was used"; PASS=$((PASS + 1))
else
    echo "  FAIL: the pkill stub was never called"; FAIL=$((FAIL + 1))
fi
if grep -qF -- "$PORTABLE5/studio/stable-diffusion.cpp/" "$PKILL_SEEN"; then
    echo "  PASS: the portable install's own sd-server is stopped"; PASS=$((PASS + 1))
else
    echo "  FAIL: the portable install's own sd-server is not stopped"; FAIL=$((FAIL + 1))
fi
if grep -qF -- "$PORTABLE5/stable-diffusion.cpp/" "$PKILL_SEEN"; then
    echo "  PASS: its legacy sd-server inside the tree is stopped too"; PASS=$((PASS + 1))
else
    echo "  FAIL: its legacy sd-server inside the tree is not stopped"; FAIL=$((FAIL + 1))
fi
# Distinct paths, not prefixes: the nested one is <parent>/portable/stable-diffusion.cpp.
if grep -qF -- "$PARENT5/stable-diffusion.cpp/" "$PKILL_SEEN"; then
    echo "  FAIL: a neighbour's sd-server is signalled"; FAIL=$((FAIL + 1))
else
    echo "  PASS: the neighbour's sd-server is left alone"; PASS=$((PASS + 1))
fi

echo ""
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
