#!/usr/bin/env bash
# Regression test: setup.sh's portable probe reads a marker one level up only
# when the Studio root really is the `studio` child install.sh writes, and only
# when that marker is the MASTER root's rather than a flat install's own.
#
# install.sh produces exactly two portable shapes. FLAT: the Studio root IS the
# master root and the marker sits inside it. NESTED: the Studio root is literally
# <master>/studio and the marker sits one level up. So a parent marker names THIS
# install under one spelling only; any other direct child of a marked root is an
# unrelated tree. install.sh's own _clear_stale_portable_marker applies the same
# rule, as do storage_roots._inherits_parent_portable_marker and its CLI twin.
#
# Reading a neighbour's marker made `unsloth studio update` against a plain
# install report portable, which skips the WebView cache clear that install's own
# update is supposed to perform, leaving the desktop app serving the previous
# frontend. Two shapes did that, and sections [7] and [8] cover one each:
#
#   [7] NAME. The rule used to fold `Studio` to `studio` whenever `uname` said
#       Darwin. macOS is case-insensitive by DEFAULT, not by rule, and a
#       case-sensitive APFS volume is what an external disk carrying a portable
#       install tends to be formatted as -- there <master>/studio and
#       <master>/Studio are two separate installs. The question asked is now
#       whether the two paths identify the same directory (st_dev/st_ino, `-ef`),
#       so every case runs under BOTH a faked Darwin uname and the real one and
#       must give the same answer.
#
#   [8] OWNERSHIP. A flat portable install occupying <root> keeps its venv at
#       <root>/unsloth_studio and its marker at <root>. A separate normal install
#       at <root>/studio passed the name test and inherited that marker. It is
#       excluded by the one question that separates the two layouts -- does the
#       parent own a venv DIRECTLY -- using install.sh's four ownership tests in
#       install.sh's order, which is what storage_roots._flat_venv_is_owned
#       already did on the Python side.
#
# A case-insensitive volume cannot be created on the CI host, so the `studio`
# spelling is made a symlink to the real `Studio`. That gives the two spellings
# the one property the predicate tests -- same st_dev/st_ino -- without claiming
# to simulate case-insensitive name lookup in general, which the predicate never
# relies on because it only ever probes the literal name `studio`.
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

# Anchor uniqueness: a second line matching either of these would silently lift a
# different block and leave every assertion below testing the wrong text.
# Prefix match, since that is what the two lifts below use: awk anchors the
# function header, and grep takes _setup_trim_ws's whole one-line body.
for _a in '_setup_portable_mode() {' '_setup_trim_ws() '; do
    _n=$(awk -v s="$_a" 'index($0, s) == 1 {n++} END {print n+0}' "$SETUP")
    if [ "$_n" -ne 1 ]; then
        printf 'FAIL: anchor [%s] matches %s lines, expected 1\n' "$_a" "$_n"; exit 1
    fi
done

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
case "$blockP" in *'${STUDIO_HOME%/*}'*) : ;; *) echo "FAIL: blockP lost the parent lookup"; exit 1 ;; esac
case "$blockT" in *"[[:space:]]"*) : ;; *) echo "FAIL: blockT extraction broke"; exit 1 ;; esac
# The three halves of the rule, asserted against CODE only: the prose in the
# block names `uname` to explain why it is not consulted, and matching that would
# fail the block for saying so. Without any one of them a whole section below
# would pass for the wrong reason.
blockP_code="$(printf '%s\n' "$blockP" | grep -v '^[[:space:]]*#')"
case "$blockP_code" in *"tr '[:upper:]' '[:lower:]'"*) : ;; *) echo "FAIL: blockP lost the case fold"; exit 1 ;; esac
case "$blockP_code" in *'-ef "$_spm_parent/studio"'*) : ;; *) echo "FAIL: blockP lost the identity probe"; exit 1 ;; esac
case "$blockP_code" in *'.unsloth-studio-owned'*) : ;; *) echo "FAIL: blockP lost the flat-parent exclusion"; exit 1 ;; esac
# The identity probe replaced a `uname` gate; it must not come back.
case "$blockP_code" in *'uname'*) echo "FAIL: blockP gates the fold on the platform again"; exit 1 ;; *) : ;; esac
# A `..` component would make the two grep -qxF sentinels compare a path
# install.sh never wrote, so the flat-parent exclusion would go silently inert
# while every assertion in section [8] that does not depend on them stayed green.
case "$blockP_code" in *'$STUDIO_HOME/..'*) echo "FAIL: blockP derives the parent with a .. component"; exit 1 ;; *) : ;; esac
# ${var,,} is a bashism, and macOS has neither readlink -f nor realpath.
case "$blockP_code" in *',,}'*) echo "FAIL: blockP uses a bash-only case fold"; exit 1 ;; *) : ;; esac
case "$blockP_code" in *'readlink -f'*|*'realpath '*) echo "FAIL: blockP uses a tool macOS lacks"; exit 1 ;; *) : ;; esac
# Lifted out with only _setup_trim_ws for company, so any other helper would be
# a 127 that reads as false inside an `if` and takes the guard silently inert.
case "$blockP_code" in *'_flat_venv_is_owned'*|*'_setup_abs_path'*) echo "FAIL: blockP calls a helper it does not define"; exit 1 ;; *) : ;; esac

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

# A uname earlier on PATH than the real one. The rule must not consult it at all,
# so section [7] runs every case through both this and the real uname and asserts
# the same answer. Only uname is shadowed; tr and the rest still resolve normally.
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
echo "[7] \`Studio\` is the same directory where the FILESYSTEM says so"
# The installer writes `studio`, but a user typing UNSLOTH_STUDIO_HOME by hand can
# spell it `Studio` and, on a case-insensitive volume, open the very same
# directory; `cd -P` hands the spelling straight through. Rejecting it would break
# a real nested install.
#
# The old rule folded whenever `uname` said Darwin. macOS is case-insensitive by
# DEFAULT, not by rule, and a case-sensitive APFS volume is exactly what an
# external disk carrying a portable install tends to be formatted as -- there
# `studio` and `Studio` are two separate installs. So the question asked is
# whether the two paths identify the same directory (st_dev/st_ino, `-ef`), which
# is what storage_roots._inherits_parent_portable_marker, its CLI twin and
# install.sh's _clear_stale_portable_marker all ask now. Every case is run under
# BOTH a faked Darwin uname and the real one and must give the SAME answer, so a
# platform gate creeping back in fails here.
#
# A case-insensitive volume cannot be created on the CI host, so `studio` is made
# a symlink to the real `Studio`. That gives the two spellings the one property
# the predicate tests -- same st_dev/st_ino -- without claiming to simulate
# case-insensitive name lookup in general, which the predicate never relies on
# because it only ever probes the literal name `studio`.
CASED="$TMP/cased"
mkdir -p "$CASED/Studio/unsloth_studio/bin"
ln -s Studio "$CASED/studio"
printf '%s\n' "$CASED" > "$CASED/.unsloth-portable-root"
# Two REAL directories under one marked root: <split>/studio is the genuine nested
# portable install, <split>/Studio a separate normal one. This is the shape the
# platform fold got wrong.
SPLIT="$TMP/split"
mkdir -p "$SPLIT/studio/unsloth_studio/bin" "$SPLIT/Studio/unsloth_studio/bin"
printf '%s\n' "$SPLIT" > "$SPLIT/.unsloth-portable-root"
for _p in darwin linux; do
    if [ "$_p" = darwin ]; then _pr=probe_darwin; else _pr=probe; fi
    check "$_p: one directory, two spellings, inherits" portable "$($_pr "$CASED/Studio")"
    check "$_p: two directories, the cased one does not" plain    "$($_pr "$SPLIT/Studio")"
    # ...and the nested portable install beside it keeps working. Without this the
    # rule could collapse into "never portable" and every check above still pass.
    check "$_p: two directories, the nested one still does" portable "$($_pr "$SPLIT/studio")"
    # Lowercase must not depend on the fold, or the fold becomes the whole rule.
    check "$_p: lowercase studio still inherits"   portable "$($_pr "$MASTER/studio")"
    # And the fold must not turn every child into a match.
    check "$_p: a sibling is still not adopted"    plain    "$($_pr "$SIBLING")"
done

echo
echo "[8] a flat install at <root> does not lend its marker to a normal <root>/studio"
# install.sh --portable over an existing root makes THAT directory the master
# root: the venv sits directly at <root>/unsloth_studio and the marker at <root>.
# A separate normal install pointed at <root>/studio through UNSLOTH_STUDIO_HOME
# is then a different tree, and reading the flat neighbour's marker made its
# `unsloth studio update` skip _clear_webview_caches, leaving the desktop WebView
# serving the previous frontend. The parent's marker only counts when the parent
# does NOT own a venv directly -- the same question install.sh's
# _clear_stale_portable_marker and storage_roots._flat_venv_is_owned ask, with the
# same four ownership tests in the same order.
flat_case() { # label sentinel_setup
    _fc_dir="$TMP/flat_$1"
    mkdir -p "$_fc_dir/unsloth_studio/bin" "$_fc_dir/bin" "$_fc_dir/share"
    mkdir -p "$_fc_dir/studio/unsloth_studio/bin"
    printf '%s\n' "$_fc_dir" > "$_fc_dir/.unsloth-portable-root"
    printf '%s' "$_fc_dir"
}
# (a) the in-venv owner marker
FA="$(flat_case owned)"; : > "$FA/unsloth_studio/.unsloth-studio-owned"
check "flat parent (owned marker): child is plain" plain "$(probe "$FA/studio")"
# (b) share/studio.conf naming this venv
FB="$(flat_case conf)"; printf "UNSLOTH_EXE='%s/unsloth_studio/bin/unsloth'\n" "$FB" > "$FB/share/studio.conf"
check "flat parent (studio.conf):  child is plain" plain "$(probe "$FB/studio")"
# (c) bin/unsloth symlinked at this venv
FC="$(flat_case shim)"; : > "$FC/unsloth_studio/bin/unsloth"; ln -s "$FC/unsloth_studio/bin/unsloth" "$FC/bin/unsloth"
check "flat parent (bin symlink):  child is plain" plain "$(probe "$FC/studio")"
# (d) the generated portable wrapper exec'ing this venv
FD="$(flat_case wrapper)"; printf "exec '%s/unsloth_studio/bin/unsloth' \"\$@\"\n" "$FD" > "$FD/bin/unsloth"
check "flat parent (wrapper exec): child is plain" plain "$(probe "$FD/studio")"
# The exclusion must not swallow the legitimate nested install: an unsloth_studio
# at the parent that NAMES NOTHING is not one of ours, and $MASTER has none at all.
FE="$(flat_case unowned)"
check "unowned parent venv: child still portable" portable "$(probe "$FE/studio")"
check "no parent venv at all: child still portable" portable "$(probe "$MASTER/studio")"

echo
if [ "$fails" -ne 0 ]; then
    printf '\n%d check(s) failed\n' "$fails"
    exit 1
fi
echo "ALL SETUP PORTABLE-MARKER SCOPE CHECKS PASSED"
