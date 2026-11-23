#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# setup.sh's fast path must escape when UNSLOTH_DESKTOP_BACKEND_VERSION requires an upgrade,
# and the UV_OFFLINE branch is held to the same bar via the shared _fast_path_escapes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

BLK="$WORK/fastpath_blk.sh"
OFFLINE_BLK="$WORK/offline_blk.sh"

# Every slice assumption is checked and reported as drift: a block that never sourced leaves
# _SKIP_PYTHON_DEPS at its false default, so half the cases would pass wrongly.
drift() {
    echo "FATAL: the fast-path extraction no longer matches $SETUP_SH -- $1" >&2
    echo "       Fix the extraction in $0 (or the block in setup.sh), do not silence it:" >&2
    echo "       start anchors are the INSTALLED_VER = LATEST_VER condition and the" >&2
    echo "       -z LATEST_VER condition (any if/elif keyword); each branch ends where it" >&2
    echo "       hands off, at the _fast_path_escapes call and at the chain's own fi." >&2
    exit 1
}

# Matched without the leading if/elif keyword; it is normalised to `if` so the slice parses alone.
FASTPATH_COND='[ -n "$INSTALLED_VER" ] && [ -n "$LATEST_VER" ] && [ "$INSTALLED_VER" = "$LATEST_VER" ]; then'
OFFLINE_COND='[ -z "$LATEST_VER" ]; then'

# $1 condition literal, $2 end-anchor regex, $3 destination. KEEP consumes the anchor line.
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

extract_branch "$FASTPATH_COND" '^[ \t]*_fast_path_escapes$'  keep "$BLK"
extract_branch "$OFFLINE_COND"  '^    fi$'                    drop "$OFFLINE_BLK"

# Without these the extraction could shrink to nothing and every case would pass.
grep -q '_SKIP_PYTHON_DEPS=true' "$BLK" \
    || drift "the extracted up-to-date block never sets _SKIP_PYTHON_DEPS=true"
grep -q '^[[:space:]]*_fast_path_escapes$' "$BLK" \
    || drift "the extracted up-to-date block no longer calls _fast_path_escapes"
grep -q '_SKIP_PYTHON_DEPS=true' "$OFFLINE_BLK" \
    || drift "the extracted offline block never sets _SKIP_PYTHON_DEPS=true"
grep -q '^[[:space:]]*_fast_path_escapes$' "$OFFLINE_BLK" \
    || drift "the extracted offline block no longer calls _fast_path_escapes -- an offline
       skip that ducks the shared escapes reports success and repairs nothing"
grep -q 'could not reach PyPI' "$OFFLINE_BLK" \
    || drift "the extracted offline block lost its updating-to-be-safe default"

for _blk in "$BLK" "$OFFLINE_BLK"; do
    if ! _syntax_err=$(bash -n "$_blk" 2>&1); then
        echo "--- extracted block ($_blk) ---" >&2
        cat -n "$_blk" >&2
        echo "--- bash -n ---" >&2
        echo "$_syntax_err" >&2
        drift "the extracted block $_blk is not valid bash (see above)"
    fi
done

# A missing helper is a "command not found" that reads as an incomplete install, which three
# cases expect for a different reason.
HELPERS="$WORK/helpers.sh"
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
grep -q 'UNSLOTH_DESKTOP_BACKEND_VERSION' "$HELPERS" \
    || drift "the shared escapes no longer consult UNSLOTH_DESKTOP_BACKEND_VERSION"
if ! _syntax_err=$(bash -n "$HELPERS" 2>&1); then
    echo "$_syntax_err" >&2
    drift "the extracted helpers are not valid bash"
fi

PASS=0
FAIL=0

check() {
    local label="$1"
    local got="$2"
    local want="$3"
    if [ "$got" = "$want" ]; then
        echo "  PASS: $label (got=$got)"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $label (got=$got, want=$want)"
        FAIL=$((FAIL + 1))
    fi
}

# A venv without site-packages, exercising setup's fallback parser.
VENV_DIR="$WORK/mock_venv"
mkdir -p "$VENV_DIR/bin"
cat << 'EOF' > "$VENV_DIR/bin/python"
#!/bin/sh
exec python3 -S "$@"
EOF
chmod +x "$VENV_DIR/bin/python"

printf 'def verify_install(**kwargs):\n    return {"ok": True}\n' > "$WORK/install_manifest.py"

# An ambient torch pin would fire the XPU arm and every case would read false wrongly.
_common_env() {
    _PKG_NAME="unsloth"
    SCRIPT_DIR="$WORK"
    UNSLOTH_TORCH_INDEX_URL=""
    UNSLOTH_TORCH_INDEX_FAMILY=""
    _SKIP_PYTHON_DEPS=false
}

eval_fastpath() {
    local installed_ver="$1"
    local latest_ver="$2"
    local desktop_ver="${3:-}"
    (
        INSTALLED_VER="$installed_ver"
        LATEST_VER="$latest_ver"
        UNSLOTH_DESKTOP_BACKEND_VERSION="$desktop_ver"
        _common_env
        # false is also what a block that never ran leaves behind, so both ways report themselves.
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

# The offline branch prints no `step`: exactly one of its two substep messages must appear.
eval_offline() {
    local installed_ver="$1"
    local uv_offline="$2"
    local desktop_ver="${3:-}"
    (
        INSTALLED_VER="$installed_ver"
        LATEST_VER=""
        UV_OFFLINE="$uv_offline"
        UNSLOTH_DESKTOP_BACKEND_VERSION="$desktop_ver"
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

echo "Testing UNSLOTH_DESKTOP_BACKEND_VERSION fastpath escape in setup.sh:"

check "matching versions, no desktop requirement" \
    "$(eval_fastpath '2026.8.15' '2026.8.15' '')" "true"

check "installed satisfies desktop requirement" \
    "$(eval_fastpath '2026.8.15' '2026.8.15' '2026.8.15')" "true"

check "installed exceeds desktop requirement" \
    "$(eval_fastpath '2026.8.16' '2026.8.16' '2026.8.15')" "true"

check "installed older than desktop requirement (2026.8.4 < 2026.8.15)" \
    "$(eval_fastpath '2026.8.4' '2026.8.4' '2026.8.15')" "false"

check "installed older than desktop requirement (2026.8.14 < 2026.8.15)" \
    "$(eval_fastpath '2026.8.14' '2026.8.14' '2026.8.15')" "false"

# Without packaging, a suffix cannot be ordered safely, so force the dependency pass.
check "post-release requirement forces dependency pass without packaging" \
    "$(eval_fastpath '2026.8.15' '2026.8.15' '2026.8.15.post1')" "false"

echo "The offline branch is held to the same bar:"
# UV_OFFLINE keeps a verified tree, which can still be below the desktop floor.
check "offline, verified, no desktop requirement" \
    "$(eval_offline '2026.8.15' '1' '')" "true"

check "offline, verified, satisfies desktop requirement" \
    "$(eval_offline '2026.8.15' '1' '2026.8.15')" "true"

check "offline, verified, BELOW desktop requirement" \
    "$(eval_offline '2026.8.4' '1' '2026.8.15')" "false"

check "offline, verified, unorderable desktop requirement" \
    "$(eval_offline '2026.8.15' '1' '2026.8.15.post1')" "false"

check "unreachable PyPI without UV_OFFLINE still updates" \
    "$(eval_offline '2026.8.15' '' '')" "false"

check "unreachable PyPI, UV_OFFLINE off, below requirement" \
    "$(eval_offline '2026.8.4' 'false' '2026.8.15')" "false"

check "offline with nothing installed still updates" \
    "$(eval_offline '' '1' '')" "false"

echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
