#!/usr/bin/env bash
# Regression test: setup.sh's portable probe reads a marker one level up only
# when the Studio root is the `studio` child install.sh writes.
#
# install.sh produces exactly two portable shapes. FLAT: the Studio root IS the
# master root and the marker sits inside it. NESTED: the Studio root is literally
# <master>/studio and the marker sits one level up. So a parent marker names THIS
# install under one spelling only; any other direct child of a marked root is an
# unrelated tree. install.sh's own _clear_stale_portable_marker already refuses to
# touch a parent marker unless $STUDIO_HOME matches */studio, and
# storage_roots._inherits_parent_portable_marker applies the same rule.
#
# Reading a neighbour's marker made `unsloth studio update` against a plain
# install at <master>/other report portable, which skips the WebView cache clear
# that install's own update is supposed to perform.
#
# Runs the real _setup_portable_mode block out of setup.sh against fake roots.
set -u
HERE="$(CDPATH= cd -P -- "$(dirname "$0")" && pwd -P)"
SETUP="$HERE/../../studio/setup.sh"
fails=0
check() { # name expected actual
    if [ "$2" = "$3" ]; then printf '  PASS  %s\n' "$1"
    else printf '  FAIL  %s : expected [%s] got [%s]\n' "$1" "$2" "$3"; fails=$((fails+1)); fi
}

blockP="$(awk '
    /^_setup_portable_mode\(\) \{$/ {grab = 1}
    grab {print}
    grab && /^\}$/ {exit}
' "$SETUP")"
# _setup_portable_mode normalizes UNSLOTH_PORTABLE through this helper.
blockT="$(grep '^_setup_trim_ws() ' "$SETUP")"

# Self-validate: a refactor must fail here, not silently test "".
case "$blockP" in *"UNSLOTH_PORTABLE"*) : ;; *) echo "FAIL: blockP extraction broke"; exit 1 ;; esac
case "$blockP" in *".unsloth-portable-root"*) : ;; *) echo "FAIL: blockP lost the marker probe"; exit 1 ;; esac
case "$blockP" in *'$STUDIO_HOME/..'*) : ;; *) echo "FAIL: blockP lost the parent lookup"; exit 1 ;; esac
case "$blockT" in *"[[:space:]]"*) : ;; *) echo "FAIL: blockT extraction broke"; exit 1 ;; esac

SNIP="$blockT"'
'"$blockP"'
if _setup_portable_mode; then printf portable; else printf plain; fi'

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

probe() { # studio_home [unsloth_home] [unsloth_portable]
    env -i PATH="$PATH" HOME="$TMP/home" \
        STUDIO_HOME="$1" UNSLOTH_HOME="${2:-}" UNSLOTH_PORTABLE="${3:-}" \
        bash -c "$SNIP" _
}

mkdir -p "$TMP/home"

# The case-fold only runs on macOS, so provoke it: a uname earlier on PATH than
# the real one. Only uname is shadowed; tr and the rest still resolve normally.
mkdir -p "$TMP/fakebin"
printf '#!/bin/sh\nprintf "Darwin\\n"\n' > "$TMP/fakebin/uname"
chmod +x "$TMP/fakebin/uname"
probe_darwin() { # studio_home
    env -i PATH="$TMP/fakebin:$PATH" HOME="$TMP/home" \
        STUDIO_HOME="$1" UNSLOTH_HOME="" UNSLOTH_PORTABLE="" \
        bash -c "$SNIP" _
}

# The nested shape `install.sh --root <master>` writes.
MASTER="$TMP/master"
mkdir -p "$MASTER/studio/unsloth_studio/bin" "$MASTER/bin" "$MASTER/share"
printf '%s\n' "$MASTER" > "$MASTER/.unsloth-portable-root"

# A second, unrelated install that merely sits beside studio/.
SIBLING="$MASTER/other"
mkdir -p "$SIBLING/unsloth_studio/bin"

echo
echo "[1] the studio child still reads as portable"
check "nested: portable" portable "$(probe "$MASTER/studio")"

echo
echo "[2] a sibling of studio/ does not inherit the marker"
check "sibling: plain" plain "$(probe "$SIBLING")"

echo
echo "[3] the flat layout is untouched"
# Master root IS the Studio root and the marker is INSIDE it, so the restricted
# parent lookup must not reach this one. Its name is whatever the user called the
# volume, which is the reason the first probe cannot be name-gated too.
FLAT="$TMP/vol"
mkdir -p "$FLAT/unsloth_studio/bin"
printf '%s\n' "$FLAT" > "$FLAT/.unsloth-portable-root"
check "flat: portable" portable "$(probe "$FLAT")"

echo
echo "[4] the other two signals are unaffected"
# UNSLOTH_HOME names the master root outright, so no name rule applies to it.
check "explicit UNSLOTH_HOME: portable" portable "$(probe "$SIBLING" "$MASTER")"
check "UNSLOTH_PORTABLE=1: portable" portable "$(probe "$SIBLING" "" 1)"
check "UNSLOTH_PORTABLE=' True ': portable" portable "$(probe "$SIBLING" "" " True ")"

echo
echo "[5] a plain install under no marker at all"
PLAIN="$TMP/plain/studio"
mkdir -p "$PLAIN/unsloth_studio/bin"
check "unmarked studio child: plain" plain "$(probe "$PLAIN")"

echo
echo "[6] a marker two levels up reaches nothing"
DEEP="$MASTER/studio/nested"
mkdir -p "$DEEP"
check "grandchild: plain" plain "$(probe "$DEEP")"

echo
echo "[7] \`Studio\` is the same directory where the filesystem says so"
# The installer writes `studio`, but a user typing UNSLOTH_STUDIO_HOME by hand on
# macOS can spell it `Studio` and open the very same directory; `cd -P` hands the
# spelling straight through. Rejecting it would break a real nested install, so
# this matches storage_roots._inherits_parent_portable_marker.
CASED="$TMP/cased"
mkdir -p "$CASED/Studio/unsloth_studio/bin"
printf '%s\n' "$CASED" > "$CASED/.unsloth-portable-root"
check "darwin: Studio inherits the parent marker" portable "$(probe_darwin "$CASED/Studio")"
check "linux: Studio is a distinct directory"     plain    "$(probe "$CASED/Studio")"
# Lowercase must not depend on the fold, or the fold becomes the whole rule.
check "darwin: lowercase studio still inherits"   portable "$(probe_darwin "$MASTER/studio")"
# And the fold must not turn every child into a match.
check "darwin: a sibling is still not adopted"    plain    "$(probe_darwin "$SIBLING")"

echo
if [ "$fails" -ne 0 ]; then
    printf '\n%d check(s) failed\n' "$fails"
    exit 1
fi
echo "ALL SETUP PORTABLE-MARKER SCOPE CHECKS PASSED"
