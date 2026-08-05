#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Behaviour tests for the #7803 fix: the Python request handed to uv, and the
# guard that recreates a venv left on a skipped interpreter by an earlier run.
# The real helpers and the real guard block are extracted from install.sh and
# executed against a stubbed uv, so this cannot drift into testing a copy.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"

PASS=0
FAIL=0

assert_eq() {
    _label="$1"; _expected="$2"; _actual="$3"
    if [ "$_actual" = "$_expected" ]; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected '$_expected', got '$_actual')"
        FAIL=$((FAIL + 1))
    fi
}

_HELPERS=$(mktemp)
{
    sed -n '/^PYTHON_SKIP=/p' "$INSTALL_SH"
    sed -n '/^_python_is_skipped()/,/^}/p' "$INSTALL_SH"
    sed -n '/^_python_request()/,/^}/p' "$INSTALL_SH"
} > "$_HELPERS"
# shellcheck disable=SC1090
. "$_HELPERS"

echo "=== the request handed to uv ==="

assert_eq "a bare 3.13 asks for a range that excludes the bad patch" \
    ">=3.13.9,<3.14" "$(_python_request 3.13)"
assert_eq "other versions are passed through untouched" \
    "3.12" "$(_python_request 3.12)"
assert_eq "an explicit patch from --python is the user's choice" \
    "3.13.8" "$(_python_request 3.13.8)"

echo "=== the skip list ==="

if _python_is_skipped "3.13.8"; then
    assert_eq "3.13.8 is skipped" "yes" "yes"
else
    assert_eq "3.13.8 is skipped" "yes" "no"
fi
if _python_is_skipped "3.13.12"; then
    assert_eq "a good patch is not skipped" "no" "yes"
else
    assert_eq "a good patch is not skipped" "no" "no"
fi
if _python_is_skipped ""; then
    assert_eq "an unreadable version is not skipped" "no" "yes"
else
    assert_eq "an unreadable version is not skipped" "no" "no"
fi

echo "=== the venv guard ==="

_GUARD=$(mktemp)
sed -n '/^# The request above only decides/,/^fi$/p' "$INSTALL_SH" > "$_GUARD"
[ -s "$_GUARD" ] || { echo "  FAIL: could not extract the venv guard"; exit 1; }

# Runs the guard against a fake venv whose python reports $1, with uv stubbed.
# Echoes the request uv was asked for, or nothing when the guard did not fire.
run_guard() {
    _reported="$1"
    _user_python="${2:-}"
    _work=$(mktemp -d)
    mkdir -p "$_work/venv/bin"
    cat > "$_work/venv/bin/python" <<EOF
#!/bin/sh
echo "$_reported"
EOF
    chmod +x "$_work/venv/bin/python"

    (
        set -e
        VENV_DIR="$_work/venv"
        _USER_PYTHON="$_user_python"
        PYTHON_VERSION="3.13"
        # shellcheck disable=SC1090
        . "$_HELPERS"
        run_install_cmd() {
            shift  # label
            shift  # uv
            shift  # venv
            shift  # target dir
            shift  # --python
            echo "REQUEST=$1" >&2
            mkdir -p "$VENV_DIR/bin"
            printf '#!/bin/sh\necho 3.13.12\n' > "$VENV_DIR/bin/python"
            chmod +x "$VENV_DIR/bin/python"
        }
        # shellcheck disable=SC1090
        . "$_GUARD"
    ) 2>&1 >/dev/null | sed -n 's/^REQUEST=//p'
    rm -rf "$_work"
}

assert_eq "a venv left on 3.13.8 is recreated on the range" \
    ">=3.13.9,<3.14" "$(run_guard 3.13.8)"
assert_eq "a healthy venv is left alone" \
    "" "$(run_guard 3.13.12)"
assert_eq "an unreadable interpreter is left alone" \
    "" "$(run_guard '')"
assert_eq "--python is honoured even on a skipped version" \
    "" "$(run_guard 3.13.8 /usr/bin/python3.13)"

rm -f "$_HELPERS" "$_GUARD"

echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
