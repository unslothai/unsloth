#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# UNSLOTH_STUDIO_FULL_DEPS is the documented way to force the whole dependency pass to run
# again -- install_python_stack.py calls it "the escape hatch" and says a skip nobody can
# turn off is a bug nobody can work around. But it is read INSIDE install_python_stack.py,
# and setup.sh's fast path is the branch that never starts it: when the installed version
# equals the one on PyPI the pass is skipped up in the shell, so the hatch could not be
# reached on exactly the install it exists for and `UNSLOTH_STUDIO_FULL_DEPS=1 unsloth
# studio update` printed "dependencies up to date" and did nothing.
#
# The escape now lives in _fast_path_escapes, which is also what the UV_OFFLINE branch
# calls, so both branches are covered by one copy and cannot disagree. This drives the real
# blocks and the real helper out of setup.sh, the same way
# test_setup_desktop_backend_version_fastpath.sh does.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

BLK="$WORK/fastpath_blk.sh"
OFFLINE_BLK="$WORK/offline_blk.sh"
HELPERS="$WORK/helpers.sh"

# Every slice assumption is checked and reported as drift rather than left to surface as a
# syntax error from a temp file the reader has never heard of. A block that never sourced
# leaves _SKIP_PYTHON_DEPS at its false default, which is the answer half the cases below
# want, so silent extraction failure would look like a pass.
drift() {
    echo "FATAL: the fast-path extraction no longer matches $SETUP_SH -- $1" >&2
    echo "       Fix the extraction in $0 (or the block in setup.sh), do not silence it." >&2
    exit 1
}

FASTPATH_COND='[ -n "$INSTALLED_VER" ] && [ -n "$LATEST_VER" ] && [ "$INSTALLED_VER" = "$LATEST_VER" ]; then'
OFFLINE_COND='[ -z "$LATEST_VER" ]; then'

# $1 condition literal, $2 end-anchor regex, $3 keep|drop the anchor line, $4 destination.
extract_branch() {
    _extract_status=0
    awk -v COND="$1" -v ENDRE="$2" -v KEEP="$3" '
        index($0, COND) > 0 { starts++ }
        !on && index($0, COND) > 0 {
            prefix = substr($0, 1, index($0, COND) - 1)
            if (prefix !~ /^[ \t]*(el)?if $/) { bad_prefix = prefix; next }
            on = 1
            print "if " COND
            next
        }
        on && !ended && $0 ~ ENDRE { ended = 1; if (KEEP == "keep") { body++; print } next }
        on && !ended { body++; print }
        END {
            if (starts != 1) { print starts + 0 > "/dev/stderr"; exit 3 }
            if (!on) { print bad_prefix > "/dev/stderr"; exit 4 }
            if (!ended) { exit 5 }
            if (body == 0) { exit 6 }
            print "fi"
        }
    ' "$SETUP_SH" > "$4" 2> "$WORK/extract_err" || _extract_status=$?

    case "$_extract_status" in
        0) ;;
        3) drift "expected exactly 1 line holding '$1', found $(cat "$WORK/extract_err")" ;;
        4) drift "'$1' is no longer introduced by if/elif (leading text: '$(cat "$WORK/extract_err")')" ;;
        5) drift "the end anchor ($2) no longer follows '$1'" ;;
        6) drift "the block after '$1' is empty" ;;
        *) drift "the extraction of '$1' failed with status $_extract_status" ;;
    esac
}

extract_branch "$FASTPATH_COND" '^[ \t]*_fast_path_escapes$' keep "$BLK"
extract_branch "$OFFLINE_COND"  '^    fi$'                   drop "$OFFLINE_BLK"

grep -q '^[[:space:]]*_fast_path_escapes$' "$BLK" \
    || drift "the up-to-date block no longer calls _fast_path_escapes"
grep -q '^[[:space:]]*_fast_path_escapes$' "$OFFLINE_BLK" \
    || drift "the offline block no longer calls _fast_path_escapes"

: > "$HELPERS"
for _fn in _setup_install_is_verified _uv_offline_requested _fast_path_escapes; do
    awk -v FN="^${_fn}\\\\(\\\\) \\\\{" '
        $0 ~ FN { grab = 1 }
        grab { print }
        grab && /^}/ { grab = 0 }
    ' "$SETUP_SH" >> "$HELPERS"
    grep -q "^${_fn}() {" "$HELPERS" \
        || drift "$_fn is no longer a top-level function in setup.sh"
done

# The escape has to live in the SHARED helper, not in one branch: an escape inlined into
# the version compare alone would leave the offline branch answering differently, which is
# the whole reason _fast_path_escapes exists.
#
# Sliced to a FILE and grepped from there, never `awk ... | grep -q`: this suite runs under
# `set -o pipefail` and grep -q exits on its first match, which kills awk with SIGPIPE and
# fails the pipeline whenever the match is not near the end. That is the same defect the uv
# cache scan carried, and writing this check the tempting way reproduced it here.
FPE="$WORK/fast_path_escapes.sh"
awk '/^_fast_path_escapes\(\) \{/ { grab = 1 } grab { print } grab && /^}/ { grab = 0 }' \
    "$SETUP_SH" > "$FPE"
grep -q 'UNSLOTH_STUDIO_FULL_DEPS' "$FPE" \
    || drift "_fast_path_escapes no longer consults UNSLOTH_STUDIO_FULL_DEPS -- the hatch is
       unreachable on an up-to-date install, which is the only install it exists for"
grep -q 'substep "UNSLOTH_STUDIO_FULL_DEPS' "$FPE" \
    || drift "the full-deps escape no longer announces itself; a pass nobody asked for and
       nothing explains is indistinguishable from the fast path being broken"

for _blk in "$BLK" "$OFFLINE_BLK" "$HELPERS"; do
    if ! _syntax_err=$(bash -n "$_blk" 2>&1); then
        cat -n "$_blk" >&2
        echo "$_syntax_err" >&2
        drift "the extracted block $_blk is not valid bash (see above)"
    fi
done

# setup.sh is bash, but this region is written in POSIX sh and the neighbouring boolish
# helpers are too, so a bash-only construct here would be a style break the next reader
# copies. dash parses it or this fails.
if command -v dash >/dev/null 2>&1; then
    if ! _dash_err=$(dash -n "$FPE" 2>&1); then
        echo "$_dash_err" >&2
        drift "the shared escapes are no longer POSIX sh"
    fi
fi

PASS=0
FAIL=0
check() {
    if [ "$2" = "$3" ]; then
        echo "  PASS: $1 (got=$2)"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $1 (got=$2, want=$3)"
        FAIL=$((FAIL + 1))
    fi
}

# A venv whose python runs without site-packages, so every probe inside the escapes takes
# its "cannot answer" path and only the variable under test moves the answer.
VENV_DIR="$WORK/mock_venv"
mkdir -p "$VENV_DIR/bin"
cat << 'EOF' > "$VENV_DIR/bin/python"
#!/bin/sh
exec python3 -S "$@"
EOF
chmod +x "$VENV_DIR/bin/python"
printf 'def verify_install(**kwargs):\n    return {"ok": True}\n' > "$WORK/install_manifest.py"

_common_env() {
    _PKG_NAME="unsloth"
    SCRIPT_DIR="$WORK"
    # A torch pin or a desktop floor leaking in from the ambient environment would fire a
    # different arm of the escapes and every "true" case here would read false for the
    # wrong reason, so they are cleared rather than assumed absent.
    UNSLOTH_TORCH_INDEX_URL=""
    UNSLOTH_TORCH_INDEX_FAMILY=""
    UNSLOTH_DESKTOP_BACKEND_VERSION=""
    _SKIP_PYTHON_DEPS=false
}

# $1 is passed to the block as the environment variable, or the literal <unset> to remove it.
eval_fastpath() {
    (
        INSTALLED_VER="2026.8.15"
        LATEST_VER="2026.8.15"
        if [ "$1" = "<unset>" ]; then unset UNSLOTH_STUDIO_FULL_DEPS; else UNSLOTH_STUDIO_FULL_DEPS="$1"; fi
        _common_env
        # false is also what a block that never ran leaves behind, so the skip cases would
        # pass on a block that did nothing at all. Both ways that can happen report here.
        _STEP_CALLS=0
        step() { _STEP_CALLS=$((_STEP_CALLS + 1)); }
        substep() { :; }
        # shellcheck disable=SC1090
        . "$HELPERS"
        # shellcheck disable=SC1090
        . "$BLK" || { echo "BLOCK_FAILED_TO_RUN"; exit 0; }
        [ "$_STEP_CALLS" -gt 0 ] || { echo "BLOCK_NOT_ENTERED"; exit 0; }
        echo "$_SKIP_PYTHON_DEPS"
    )
}

# The offline branch prints no `step`, so "did it run at all" is answered by its substeps.
eval_offline() {
    (
        INSTALLED_VER="2026.8.15"
        LATEST_VER=""
        UV_OFFLINE="1"
        if [ "$1" = "<unset>" ]; then unset UNSLOTH_STUDIO_FULL_DEPS; else UNSLOTH_STUDIO_FULL_DEPS="$1"; fi
        _common_env
        _SUBSTEPS=""
        substep() { _SUBSTEPS="$_SUBSTEPS
$1"; }
        step() { :; }
        # shellcheck disable=SC1090
        . "$HELPERS"
        # shellcheck disable=SC1090
        . "$OFFLINE_BLK" || { echo "BLOCK_FAILED_TO_RUN"; exit 0; }
        case "$_SUBSTEPS" in
            *"keeping the verified install"* | *"could not reach PyPI"*) ;;
            *) echo "BLOCK_NOT_ENTERED"; exit 0 ;;
        esac
        echo "$_SKIP_PYTHON_DEPS"
    )
}

# What the user is told to run, and what it has to print while doing it.
eval_message() {
    (
        INSTALLED_VER="2026.8.15"
        LATEST_VER="2026.8.15"
        UNSLOTH_STUDIO_FULL_DEPS="$1"
        _common_env
        _SUBSTEPS=""
        step() { :; }
        substep() { _SUBSTEPS="$_SUBSTEPS
$1"; }
        # shellcheck disable=SC1090
        . "$HELPERS"
        # shellcheck disable=SC1090
        . "$BLK" || { echo "BLOCK_FAILED_TO_RUN"; exit 0; }
        case "$_SUBSTEPS" in
            *"UNSLOTH_STUDIO_FULL_DEPS is set"*) echo "announced" ;;
            *) echo "silent" ;;
        esac
    )
}

echo "Testing UNSLOTH_STUDIO_FULL_DEPS against setup.sh's up-to-date fast path:"

# The baseline the hatch has to be able to override: same version, nothing else wrong.
check "unset leaves the fast path alone" "$(eval_fastpath '<unset>')" "true"

# Accepted exactly as install_python_stack.py's _full_deps_requested accepts it:
# .strip().lower() in ("1", "true", "yes", "on"). A shell that took a narrower set would
# make the same command mean different things depending on which half of the update it hit.
for truthy in 1 true TRUE True yes YES on ON " 1 " "  on  " "	true	"; do
    check "UNSLOTH_STUDIO_FULL_DEPS=[$truthy] forces the dependency pass" \
        "$(eval_fastpath "$truthy")" "false"
done

# ...and nothing wider. "maybe" is not consent, and 0/false/empty are the values a script
# that wanted the default writes on purpose.
for falsy in 0 false FALSE no off "" " " maybe 2 "1x" "on!" "yes please"; do
    check "UNSLOTH_STUDIO_FULL_DEPS=[$falsy] leaves the fast path alone" \
        "$(eval_fastpath "$falsy")" "true"
done

echo "The offline branch shares the escape, so it answers the same:"
check "offline, verified, hatch unset" "$(eval_offline '<unset>')" "true"
check "offline, verified, hatch set"   "$(eval_offline '1')"       "false"
check "offline, verified, hatch=maybe" "$(eval_offline 'maybe')"   "true"

echo "And it says why it is doing the work:"
check "a forced pass names the variable that forced it" "$(eval_message '1')" "announced"

echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
