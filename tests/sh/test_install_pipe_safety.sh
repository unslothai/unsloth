#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Guards that `curl ... | sh` cannot report a bogus transport error.
#
# History: install.sh was ~150KB of top-level statements. A top-level `exit` left most
# of the file unread, the write end failed, and curl appended "(56) Failure writing
# output to destination" (or "(23) Failed writing body") after our own error message,
# so users read a real diagnosis as a broken download. The fix is structural: the body
# lives in _unsloth_main, so sh parses the whole file before running anything.
#
# This pins both halves of that contract: the writer must not be killed, AND the
# installer's own exit code must still reach the caller.
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

echo "=== structure ==="

# The wrapper must be invoked on the LAST executable line, or sh starts executing
# before it has drained the pipe.
if grep -q '^_unsloth_main() {' "$INSTALL_SH"; then
    echo "  PASS: _unsloth_main is defined at top level"
    PASS=$((PASS + 1))
else
    echo "  FAIL: install.sh is not wrapped in _unsloth_main -- curl-pipe safety is gone"
    FAIL=$((FAIL + 1))
fi

_last="$(grep -vE '^\s*(#|$)' "$INSTALL_SH" | tail -1)"
assert_eq "last statement invokes the wrapper" '_unsloth_main "$@"' "$_last"

# Below one pipe buffer the file would fit in the kernel's buffer and this test would
# prove nothing, so fail loudly instead of passing vacuously.
_bytes="$(wc -c < "$INSTALL_SH" | tr -d ' ')"
if [ "$_bytes" -gt 65536 ]; then
    echo "  PASS: install.sh ($_bytes bytes) exceeds a 64KiB pipe buffer, so this matters"
    PASS=$((PASS + 1))
else
    echo "  FAIL: install.sh is only $_bytes bytes; re-derive whether pipe safety still applies"
    FAIL=$((FAIL + 1))
fi

echo "=== behaviour: an early exit must not kill the writer ==="

# `--python` with no argument exits 1 from argument validation having done no work: no
# venv, no downloads, no filesystem writes. Deterministic and safe to run for real.
#
# PIPESTATUS must be read on the very next line, so drop errexit around the pipeline
# rather than appending `|| true`, which would clobber it with the status of `true`.
set +e
cat "$INSTALL_SH" | sh -s -- --python >/dev/null 2>&1
_pipe=("${PIPESTATUS[@]}")
set -e
_writer_rc="${_pipe[0]}"
_reader_rc="${_pipe[1]}"

# A writer rc of 141 (128 + SIGPIPE) is the failure mode curl reports as (56)/(23).
assert_eq "writer survives the early exit (not SIGPIPE)" "0" "$_writer_rc"
assert_eq "installer's own exit code still propagates" "1" "$_reader_rc"

echo "=== behaviour: the same holds for a mid-file exit ==="
# `--package '-evil'` exits from a later validation block, still before any filesystem
# work, so the property is not specific to one early branch.
set +e
cat "$INSTALL_SH" | sh -s -- --package '-evil' >/dev/null 2>&1
_pipe2=("${PIPESTATUS[@]}")
set -e
assert_eq "writer survives a later exit" "0" "${_pipe2[0]}"
assert_eq "later exit code propagates" "1" "${_pipe2[1]}"

echo ""
echo "=== $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ] || exit 1
