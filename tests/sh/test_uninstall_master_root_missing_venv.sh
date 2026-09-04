#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Regression test: absence of a venv is not evidence that a root is FLAT.
#
# scripts/uninstall.sh decides, for every root _custom_studio_roots emits, whether the entry is a
# Studio root or the MASTER root above one, because the legacy stable-diffusion.cpp cleanup takes
# each entry's PARENT and that is one level too high for a master root. The test used to be
# `[ ! -d "<root>/unsloth_studio" ] && [ -d "<root>/studio/unsloth_studio" ]`, which reads a
# nested portable root whose venv is missing or damaged as flat: the uninstaller then walked to
# <parent>/stable-diffusion.cpp and removed it whenever it carried the Unsloth owner marker, even
# though a SEPARATE installation sitting beside the portable root can own that tree. The root
# still has its portable marker and its share/studio.conf in that state, so the layout is decided
# from those rather than from what is left of the venv.
#
# The mirror-image half is pinned here too: a genuine flat install must still be recognized as
# flat, or the ownership requirement collapses into "never flat" and the legacy sibling of every
# flat root is silently kept forever. tests/sh/test_uninstall_master_root_sd_sibling.sh covers
# the healthy nested and flat layouts; this file covers what is left when the venv is gone.
#
# The uninstaller runs for real against a fixture HOME, one mktemp -d per case, so what is
# asserted is the removal OUTCOME.
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

# 1. The reported case: a nested portable install whose studio/unsloth_studio is gone. Its root
#    still carries the portable marker and share/studio.conf, so it is still a master root and
#    the walk to <parent>/stable-diffusion.cpp must not happen.
C1=$(new_case)
H1="$C1/home"
PARENT1="$C1/parent"
PORTABLE1="$PARENT1/portable"
OTHER1="$PARENT1/other"
mkdir -p "$H1/.local/share" "$H1/.local/bin" \
    "$PORTABLE1/studio" "$PORTABLE1/share" "$PORTABLE1/bin" "$OTHER1/unsloth_studio"
echo portable > "$PORTABLE1/studio/studio.db"
printf '%s\n' "$PORTABLE1" > "$PORTABLE1/.unsloth-portable-root"
printf "UNSLOTH_EXE='%s'\nexport UNSLOTH_HOME='%s'\n" \
    "$PORTABLE1/studio/unsloth_studio/bin/unsloth" "$PORTABLE1" \
    > "$PORTABLE1/share/studio.conf"
# The neighbour: a separate installation beside the portable root, plus the legacy sd.cpp an
# older build of THAT install wrote. Marked, which is the only reason it is a candidate at all.
: > "$OTHER1/unsloth_studio/.unsloth-studio-owned"
make_owned_sd_cpp "$PARENT1/stable-diffusion.cpp"

run_uninstall "$H1" missingvenv UNSLOTH_HOME="$PORTABLE1"

assert_gone    "the selected portable root is still removed"      "$PORTABLE1"
assert_present "a neighbour's legacy sd.cpp survives"             "$PARENT1/stable-diffusion.cpp"
assert_present "with its build intact"                            "$PARENT1/stable-diffusion.cpp/build/bin/sd-cli"
assert_present "and the neighbouring install itself"              "$OTHER1/unsloth_studio"

# 2. Nothing of the Studio root left at all, not even the studio/ directory. Absence still is
#    not evidence: the marker names this root as a master root either way.
C2=$(new_case)
H2="$C2/home"
PARENT2="$C2/parent"
PORTABLE2="$PARENT2/portable"
mkdir -p "$H2/.local/share" "$H2/.local/bin" "$PORTABLE2/share" "$PORTABLE2/bin"
printf '%s\n' "$PORTABLE2" > "$PORTABLE2/.unsloth-portable-root"
printf "UNSLOTH_EXE='%s'\nexport UNSLOTH_HOME='%s'\n" \
    "$PORTABLE2/studio/unsloth_studio/bin/unsloth" "$PORTABLE2" \
    > "$PORTABLE2/share/studio.conf"
make_owned_sd_cpp "$PARENT2/stable-diffusion.cpp"

run_uninstall "$H2" novenvatall UNSLOTH_HOME="$PORTABLE2"

assert_gone    "the root is removed with no Studio dir at all"    "$PORTABLE2"
assert_present "and the neighbour's sd.cpp is still kept"         "$PARENT2/stable-diffusion.cpp"

# 3. A stray directory called unsloth_studio inside a healthy NESTED master root must not make
#    it read as flat either. The venv is only evidence of a flat install when it is OURS.
C3=$(new_case)
H3="$C3/home"
PARENT3="$C3/parent"
PORTABLE3="$PARENT3/portable"
mkdir -p "$H3/.local/share" "$H3/.local/bin" \
    "$PORTABLE3/studio/unsloth_studio" "$PORTABLE3/unsloth_studio/bin" "$PORTABLE3/share"
: > "$PORTABLE3/studio/unsloth_studio/.unsloth-studio-owned"
printf '%s\n' "$PORTABLE3" > "$PORTABLE3/.unsloth-portable-root"
printf "UNSLOTH_EXE='%s'\nexport UNSLOTH_HOME='%s'\n" \
    "$PORTABLE3/studio/unsloth_studio/bin/unsloth" "$PORTABLE3" \
    > "$PORTABLE3/share/studio.conf"
make_owned_sd_cpp "$PARENT3/stable-diffusion.cpp"

run_uninstall "$H3" strayvenv UNSLOTH_HOME="$PORTABLE3"

assert_gone    "the nested master root is removed"                "$PORTABLE3"
assert_present "a stray unsloth_studio does not flatten it"       "$PARENT3/stable-diffusion.cpp"

# 4. THE OTHER HALF. A genuine flat portable install -- venv directly in the root, carrying the
#    install-time owner marker -- IS the Studio root, so <parent>/stable-diffusion.cpp is the
#    path an older build of THIS install really used and must still be removed. Without this the
#    ownership requirement could pass by never walking to a parent at all.
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

run_uninstall "$H4" flatowned UNSLOTH_HOME="$FLAT4"

assert_gone "a genuine flat portable root is removed"             "$FLAT4"
assert_gone "and its legacy sd.cpp sibling still goes with it"    "$PARENT4/stable-diffusion.cpp"

# 5. The same for a flat CUSTOM (non-portable) root, which has no portable marker to consult:
#    its behaviour must be exactly what it was.
C5=$(new_case)
H5="$C5/home"
PARENT5="$C5/parent"
FLAT5="$PARENT5/studio"
mkdir -p "$H5/.local/share" "$H5/.local/bin" "$FLAT5/unsloth_studio" "$FLAT5/share"
: > "$FLAT5/unsloth_studio/.unsloth-studio-owned"
printf "UNSLOTH_EXE='%s'\n" "$FLAT5/unsloth_studio/bin/unsloth" > "$FLAT5/share/studio.conf"
make_owned_sd_cpp "$PARENT5/stable-diffusion.cpp"

run_uninstall "$H5" flatcustom UNSLOTH_STUDIO_HOME="$FLAT5"

assert_gone "a flat custom root is removed"                       "$FLAT5"
assert_gone "and its legacy sd.cpp sibling too"                   "$PARENT5/stable-diffusion.cpp"

# 6. The same predicate gates _sd_cpp_sibling_bases, which decides whose sd-server is SIGTERMed
#    before a tree is deleted. With the venv missing, the old test reached the neighbour's build
#    and signalled a server this uninstall has no business touching. pkill is stubbed to record
#    its argv (and to exit 1 like the real one on no match, so a missing `|| true` still shows).
C6=$(new_case)
H6="$C6/home"
PARENT6="$C6/parent"
PORTABLE6="$PARENT6/portable"
mkdir -p "$H6/.local/share" "$H6/.local/bin" "$PORTABLE6/studio" "$PORTABLE6/share"
printf '%s\n' "$PORTABLE6" > "$PORTABLE6/.unsloth-portable-root"
printf "UNSLOTH_EXE='%s'\nexport UNSLOTH_HOME='%s'\n" \
    "$PORTABLE6/studio/unsloth_studio/bin/unsloth" "$PORTABLE6" \
    > "$PORTABLE6/share/studio.conf"
# Inside the tree (what this install owns) and beside it (the neighbour's).
make_owned_sd_cpp "$PORTABLE6/stable-diffusion.cpp"
make_owned_sd_cpp "$PARENT6/stable-diffusion.cpp"

STUB_BIN="$C6/stubbin"
PKILL_LOG="$C6/pkill.args"
mkdir -p "$STUB_BIN"
cat > "$STUB_BIN/pkill" <<EOF
#!/bin/sh
printf '%s\n' "\$*" >> "$PKILL_LOG"
exit 1
EOF
chmod +x "$STUB_BIN/pkill"
: > "$PKILL_LOG"

env -i PATH="$STUB_BIN:$PATH" HOME="$H6" TMPDIR="$_TMP_ROOT" \
    XDG_RUNTIME_DIR="$_TMP_ROOT/run-pkill6" XDG_DATA_HOME="$H6/.local/share" \
    XDG_CACHE_HOME="$H6/.cache" XDG_CONFIG_HOME="$H6/.config" \
    XDG_STATE_HOME="$H6/.local/state" UNSLOTH_HOME="$PORTABLE6" \
    sh "$UNINSTALL_SH" > "$_TMP_ROOT/pkill6.log" 2>&1

# The recorded patterns are BRE-escaped by _pkill_escape, so strip the backslashes before
# matching; a fixture path under mktemp -d has none of its own.
PKILL_SEEN="$C6/pkill.plain"
sed -e 's/\\//g' "$PKILL_LOG" > "$PKILL_SEEN"
if [ -s "$PKILL_SEEN" ]; then
    echo "  PASS: the pkill stub was used"; PASS=$((PASS + 1))
else
    echo "  FAIL: the pkill stub was never called"; FAIL=$((FAIL + 1))
fi
if grep -qF -- "$PORTABLE6/stable-diffusion.cpp/" "$PKILL_SEEN"; then
    echo "  PASS: the install's own sd-server inside the tree is stopped"; PASS=$((PASS + 1))
else
    echo "  FAIL: the install's own sd-server inside the tree is not stopped"; FAIL=$((FAIL + 1))
fi
if grep -qF -- "$PARENT6/stable-diffusion.cpp/" "$PKILL_SEEN"; then
    echo "  FAIL: a neighbour's sd-server is signalled"; FAIL=$((FAIL + 1))
else
    echo "  PASS: the neighbour's sd-server is left alone"; PASS=$((PASS + 1))
fi

echo ""
echo "  $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
