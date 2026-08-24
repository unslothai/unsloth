#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Test setup.sh wiring: exit 0 forces the pass and all other statuses keep it.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
PASS=0
FAIL=0

# The escape block, from its guard to the first dedented `fi`.
awk '/^if \[ "\$_SKIP_PYTHON_DEPS" = true \] && \[ -x "\$VENV_DIR\/bin\/python" \]; then/ {on=1}
     on {print}
     on && /^fi$/ {exit}' "$SETUP_SH" > "$WORK/escape.sh"
grep -q -- "--amd-torch-needs-dependency-pass" "$WORK/escape.sh" || {
    echo "FATAL: AMD escape block not found in $SETUP_SH" >&2; exit 1; }

assert_eq() {
    _label="$1"; _expected="$2"; _actual="$3"
    if [ "$_actual" = "$_expected" ]; then
        echo "  PASS: $_label"; PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected '$_expected', got '$_actual')"; FAIL=$((FAIL + 1))
    fi
}

VENV_DIR="$WORK/venv"
mkdir -p "$VENV_DIR/bin"
: > "$WORK/install_python_stack.py"

# A stand-in interpreter that records its arguments and exits with $PROBE_RC.
cat > "$VENV_DIR/bin/python" <<'STUB'
#!/bin/sh
printf '%s\n' "$*" >> "$PROBE_LOG"
exit "$PROBE_RC"
STUB
chmod +x "$VENV_DIR/bin/python"

run_escape() {
    (
        PROBE_RC="$1"
        _SKIP_PYTHON_DEPS="${2:-true}"
        export PROBE_RC PROBE_LOG
        SCRIPT_DIR="$WORK"
        substep() { :; }
        # shellcheck disable=SC1090
        . "$WORK/escape.sh"
        echo "$_SKIP_PYTHON_DEPS"
    )
}

PROBE_LOG="$WORK/calls.txt"
: > "$PROBE_LOG"
export PROBE_LOG

echo "=== only a conclusive answer forces the pass ==="
assert_eq "exit 0 forces the dependency pass" "false" "$(run_escape 0)"
assert_eq "exit 1 keeps the fast path" "true" "$(run_escape 1)"

echo "=== every probe failure keeps the fast path ==="
# Probe errors keep the fast path, including the deadline: 124 on GNU, 143 on BusyBox.
for rc in 2 124 125 126 127 137 143; do
    assert_eq "exit $rc keeps the fast path" "true" "$(run_escape "$rc")"
done

echo "=== the fallback branch, on hosts without timeout ==="
# PATH holds only the stub bin, so `command -v timeout` fails and the elif runs.
run_no_timeout() { (PATH="$VENV_DIR/bin"; run_escape "$1"); }
assert_eq "exit 0 forces the pass without timeout" "false" "$(run_no_timeout 0)"
assert_eq "exit 1 keeps the fast path without timeout" "true" "$(run_no_timeout 1)"
: > "$PROBE_LOG"
run_no_timeout 0 >/dev/null
assert_eq "and still probes exactly once" "1" "$(wc -l < "$PROBE_LOG" | tr -d ' ')"

echo "=== the block does nothing it was not asked to ==="
assert_eq "a pass already forced is left forced" "false" "$(run_escape 1 false)"
assert_eq "and the probe is not run at all in that case" \
    "0" "$(: > "$PROBE_LOG"; run_escape 1 false >/dev/null; wc -l < "$PROBE_LOG" | tr -d ' ')"
: > "$PROBE_LOG"
run_escape 0 >/dev/null
assert_eq "the probe is asked exactly once" "1" "$(wc -l < "$PROBE_LOG" | tr -d ' ')"
assert_eq "with the module and the flag, and nothing else" \
    "$WORK/install_python_stack.py --amd-torch-needs-dependency-pass" \
    "$(cat "$PROBE_LOG")"

echo "=== a missing interpreter is not probed ==="
mv "$VENV_DIR/bin/python" "$WORK/python.away"
: > "$PROBE_LOG"
assert_eq "no venv python keeps the fast path" "true" "$(run_escape 0)"
assert_eq "and spawns nothing" "0" "$(wc -l < "$PROBE_LOG" | tr -d ' ')"
mv "$WORK/python.away" "$VENV_DIR/bin/python"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
