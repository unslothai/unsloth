#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Tests that setup.sh's fastpath escapes when UNSLOTH_DESKTOP_BACKEND_VERSION
# requires a backend upgrade even if INSTALLED_VER == LATEST_VER.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

BLK="$WORK/fastpath_blk.sh"

# This test runs the fast path by slicing it out of setup.sh, so every slice
# assumption below is checked and reported as drift. An unchecked slice fails
# as a bare "syntax error near unexpected token" from a temp file the reader
# has never heard of, which is how the elif of #8515 sat red on main: the
# condition text was unchanged, only the keyword in front of it moved, and
# three of the six cases still "passed" because a block that never sourced
# leaves _SKIP_PYTHON_DEPS at its false default.
drift() {
    echo "FATAL: the fast-path extraction no longer matches $SETUP_SH -- $1" >&2
    echo "       Fix the extraction in $0 (or the block in setup.sh), do not silence it:" >&2
    echo "       start anchor is the INSTALLED_VER = LATEST_VER condition (any if/elif keyword)," >&2
    echo "       end anchor is the first _setup_pin= assignment after it." >&2
    exit 1
}

# Matched as a literal, and deliberately without the leading keyword: the block
# is reached by an elif today and was reached by an if before #8515, and which
# one it is has no bearing on what this test exercises. The keyword is checked
# separately below and normalised to a plain `if` on the way out, so the slice
# is always a standalone, parseable construct.
FASTPATH_COND='[ -n "$INSTALLED_VER" ] && [ -n "$LATEST_VER" ] && [ "$INSTALLED_VER" = "$LATEST_VER" ]; then'

_extract_status=0
awk -v COND="$FASTPATH_COND" '
    index($0, COND) > 0 { starts++ }
    !on && index($0, COND) > 0 {
        prefix = substr($0, 1, index($0, COND) - 1)
        if (prefix !~ /^[ \t]*(el)?if $/) { bad_prefix = prefix; next }
        on = 1
        print "if " COND
        next
    }
    on && !ended && $0 ~ /^[ \t]*_setup_pin=/ { ended = 1; next }
    on && !ended { body++; print }
    END {
        if (starts != 1) { print starts + 0 > "/dev/stderr"; exit 3 }
        if (!on) { print bad_prefix > "/dev/stderr"; exit 4 }
        if (!ended) { exit 5 }
        if (body == 0) { exit 6 }
        print "fi"
    }
' "$SETUP_SH" > "$BLK" 2> "$WORK/extract_err" || _extract_status=$?

case "$_extract_status" in
    0) ;;
    3) drift "expected exactly 1 line holding the fast-path condition, found $(cat "$WORK/extract_err")" ;;
    4) drift "the fast-path condition is no longer introduced by if/elif (leading text: '$(cat "$WORK/extract_err")')" ;;
    5) drift "the end anchor (_setup_pin=) no longer follows the fast-path condition" ;;
    6) drift "the fast-path block is empty" ;;
    *) drift "the extraction failed with status $_extract_status" ;;
esac

# The slice has to still contain what this test claims to test. Without these
# the extraction could shrink to nothing meaningful and every case would pass.
grep -q '_SKIP_PYTHON_DEPS=true' "$BLK" \
    || drift "the extracted block never sets _SKIP_PYTHON_DEPS=true"
grep -q 'UNSLOTH_DESKTOP_BACKEND_VERSION' "$BLK" \
    || drift "the extracted block no longer consults UNSLOTH_DESKTOP_BACKEND_VERSION"

# The part that matters: a slice that does not parse is drift, not a test failure.
if ! _syntax_err=$(bash -n "$BLK" 2>&1); then
    echo "--- extracted block ---" >&2
    cat -n "$BLK" >&2
    echo "--- bash -n ---" >&2
    echo "$_syntax_err" >&2
    drift "the extracted block is not valid bash (see above)"
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

# Create a mock venv that runs Python without site-packages, exercising setup's fallback parser.
VENV_DIR="$WORK/mock_venv"
mkdir -p "$VENV_DIR/bin"
cat << 'EOF' > "$VENV_DIR/bin/python"
#!/bin/sh
exec python3 -S "$@"
EOF
chmod +x "$VENV_DIR/bin/python"

# Mock install_manifest to return ok: True so manifest check passes
printf 'def verify_install():\n    return {"ok": True}\n' > "$WORK/install_manifest.py"

eval_fastpath() {
    local installed_ver="$1"
    local latest_ver="$2"
    local desktop_ver="${3:-}"
    (
        INSTALLED_VER="$installed_ver"
        LATEST_VER="$latest_ver"
        UNSLOTH_DESKTOP_BACKEND_VERSION="$desktop_ver"
        _PKG_NAME="unsloth"
        SCRIPT_DIR="$WORK"
        _SKIP_PYTHON_DEPS=false
        # false is also what a block that never ran leaves behind, so three of
        # the six cases below would pass on a block that did nothing at all.
        # Both ways that can happen report themselves instead.
        _STEP_CALLS=0
        step() { _STEP_CALLS=$((_STEP_CALLS + 1)); }
        substep() { :; }

        # Execute extracted block
        # shellcheck disable=SC1090
        . "$BLK" || { echo "BLOCK_FAILED_TO_RUN"; exit 0; }
        [ "$_STEP_CALLS" -gt 0 ] || { echo "BLOCK_NOT_ENTERED"; exit 0; }
        echo "$_SKIP_PYTHON_DEPS"
    )
}

echo "Testing UNSLOTH_DESKTOP_BACKEND_VERSION fastpath escape in setup.sh:"

# 1. When versions match and no desktop version required -> skips python deps
check "matching versions, no desktop requirement" \
    "$(eval_fastpath '2026.8.15' '2026.8.15' '')" "true"

# 2. When installed version satisfies desktop requirement -> skips python deps
check "installed satisfies desktop requirement" \
    "$(eval_fastpath '2026.8.15' '2026.8.15' '2026.8.15')" "true"

check "installed exceeds desktop requirement" \
    "$(eval_fastpath '2026.8.16' '2026.8.16' '2026.8.15')" "true"

# 3. When installed version is older than desktop requirement -> escapes fastpath (_SKIP_PYTHON_DEPS=false)
check "installed older than desktop requirement (2026.8.4 < 2026.8.15)" \
    "$(eval_fastpath '2026.8.4' '2026.8.4' '2026.8.15')" "false"

check "installed older than desktop requirement (2026.8.14 < 2026.8.15)" \
    "$(eval_fastpath '2026.8.14' '2026.8.14' '2026.8.15')" "false"

# 4. Without packaging, a suffix cannot be ordered safely, so force the dependency pass.
check "post-release requirement forces dependency pass without packaging" \
    "$(eval_fastpath '2026.8.15' '2026.8.15' '2026.8.15.post1')" "false"

echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
