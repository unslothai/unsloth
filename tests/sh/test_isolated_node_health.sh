#!/bin/bash
# Unit tests for _isolated_node_layout_complete / _quarantine_broken_isolated_node
# from studio/setup.sh. Same extract-function style as test_node_decision.sh.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
PASS=0
FAIL=0

_FUNC_FILE=$(mktemp)
{
    sed -n '/^_isolated_node_layout_complete()/,/^}/p' "$SETUP_SH"
    sed -n '/^_quarantine_broken_isolated_node()/,/^}/p' "$SETUP_SH"
} > "$_FUNC_FILE"
if ! grep -q '^_isolated_node_layout_complete()' "$_FUNC_FILE" \
    || ! grep -q '^_quarantine_broken_isolated_node()' "$_FUNC_FILE"; then
    echo "FAIL: could not extract isolated-node helpers from $SETUP_SH"
    rm -f "$_FUNC_FILE"
    exit 1
fi

substep() { :; }
step() { :; }
C_ERR=""

# shellcheck disable=SC1090
. "$_FUNC_FILE"
rm -f "$_FUNC_FILE"

assert_rc() {
    _label="$1"
    _expected="$2"
    shift 2
    set +e
    "$@"
    _actual=$?
    set -e
    if [ "$_actual" -eq "$_expected" ]; then
        echo " PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo " FAIL: $_label (expected rc $_expected, got $_actual)"
        FAIL=$((FAIL + 1))
    fi
}

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

assert_rc "missing dir" 1 _isolated_node_layout_complete "$WORKDIR/nope"

mkdir -p "$WORKDIR/empty"
assert_rc "empty dir" 1 _isolated_node_layout_complete "$WORKDIR/empty"

BROKEN="$WORKDIR/broken"
mkdir -p "$BROKEN/bin"
printf '%s\n' '#!/usr/bin/env node' "require('../lib/cli.js')" > "$BROKEN/bin/npm"
chmod +x "$BROKEN/bin/npm"
printf '%s\n' '#!/bin/sh' 'echo v24.0.0' > "$BROKEN/bin/node"
chmod +x "$BROKEN/bin/node"
assert_rc "broken npm shim, no lib/" 1 _isolated_node_layout_complete "$BROKEN"

GOOD="$WORKDIR/good"
mkdir -p "$GOOD/bin" "$GOOD/lib/node_modules/npm/bin"
printf '%s\n' '#!/bin/sh' 'echo v24.0.0' > "$GOOD/bin/node"
chmod +x "$GOOD/bin/node"
echo "console.log('11.0.0')" > "$GOOD/lib/node_modules/npm/bin/npm-cli.js"
ln -s "../lib/node_modules/npm/bin/npm-cli.js" "$GOOD/bin/npm"
assert_rc "official unix layout" 0 _isolated_node_layout_complete "$GOOD"

WIN="$WORKDIR/win"
mkdir -p "$WIN/node_modules/npm/bin"
echo fake > "$WIN/node.exe"
echo fake > "$WIN/node_modules/npm/bin/npm-cli.js"
assert_rc "official windows layout" 0 _isolated_node_layout_complete "$WIN"

assert_rc "quarantine broken" 0 _quarantine_broken_isolated_node "$BROKEN"
if [ ! -d "$BROKEN" ] && ls -d "$BROKEN".broken.* >/dev/null 2>&1; then
    echo " PASS: quarantine renamed stub"
    PASS=$((PASS + 1))
else
    echo " FAIL: quarantine did not rename stub"
    FAIL=$((FAIL + 1))
fi

assert_rc "quarantine leaves good tree" 0 _quarantine_broken_isolated_node "$GOOD"
if [ -d "$GOOD" ]; then
    echo " PASS: quarantine kept official tree"
    PASS=$((PASS + 1))
else
    echo " FAIL: quarantine moved an official tree"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "Passed: $PASS Failed: $FAIL"
[ "$FAIL" -eq 0 ] || exit 1
