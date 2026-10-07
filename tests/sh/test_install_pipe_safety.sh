#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# A piped install must not report a bogus transport error: the body lives in _unsloth_main so
# sh parses the whole file before running. The installer's own exit code must still surface.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
echo "=== structure ==="

# The wrapper must be invoked on the LAST executable line, or sh runs before draining the pipe.
if grep -q '^_unsloth_main() {' "$INSTALL_SH"; then
    echo "  PASS: _unsloth_main is defined at top level"
    PASS=$((PASS + 1))
else
    echo "  FAIL: install.sh is not wrapped in _unsloth_main -- curl-pipe safety is gone"
    FAIL=$((FAIL + 1))
fi

_last="$(grep -vE '^\s*(#|$)' "$INSTALL_SH" | tail -1)"
assert_eq "last statement invokes the wrapper" '_unsloth_main "$@"' "$_last"

# Below one pipe buffer this test would prove nothing.
_bytes="$(wc -c < "$INSTALL_SH" | tr -d ' ')"
if [ "$_bytes" -gt 65536 ]; then
    echo "  PASS: install.sh ($_bytes bytes) exceeds a 64KiB pipe buffer, so this matters"
    PASS=$((PASS + 1))
else
    echo "  FAIL: install.sh is only $_bytes bytes; re-derive whether pipe safety still applies"
    FAIL=$((FAIL + 1))
fi

echo "=== behaviour: an early exit must not kill the writer ==="

# `--python` with no argument exits 1 before any work. PIPESTATUS must be read on the next
# line, so drop errexit rather than appending `|| true`.
set +e
cat "$INSTALL_SH" | sh -s -- --python >/dev/null 2>&1
_pipe=("${PIPESTATUS[@]}")
set -e
_writer_rc="${_pipe[0]}"
_reader_rc="${_pipe[1]}"

# 141 (128 + SIGPIPE) is what curl reports as (56)/(23).
assert_eq "writer survives the early exit (not SIGPIPE)" "0" "$_writer_rc"
assert_eq "installer's own exit code still propagates" "1" "$_reader_rc"

echo "=== behaviour: the same holds for a mid-file exit ==="
# A later validation block, so the property is not specific to one early branch.
set +e
cat "$INSTALL_SH" | sh -s -- --package '-evil' >/dev/null 2>&1
_pipe2=("${PIPESTATUS[@]}")
set -e
assert_eq "writer survives a later exit" "0" "${_pipe2[0]}"
assert_eq "later exit code propagates" "1" "${_pipe2[1]}"

echo ""
echo "=== $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] || exit 1
