#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Argument and pipe-safety tests for scripts/uninstall.sh.
#
# Behavioral cases use an instrumented copy that aborts before uninstalling.
# The real no-argument case is skipped on WSL because it reaches host state.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_UNINSTALL_SH="$SCRIPT_DIR/../../scripts/uninstall.sh"
PASS=0
FAIL=0
BODY_MARKER="__UNSLOTH_TEST_BODY_REACHED__"

_TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$_TMP_ROOT"' EXIT

# Abort the test copy immediately before the first uninstall action.
_body_entries=$(grep -c '^[[:space:]]*echo "Stopping any running Unsloth Studio servers\.\.\."$' "$SOURCE_UNINSTALL_SH" || true)
[ "$_body_entries" = "1" ] || {
    echo "FATAL: expected one uninstall body entry, found $_body_entries" >&2
    exit 1
}
UNINSTALL_SH="$_TMP_ROOT/uninstall.sh"
sed 's/^[[:space:]]*echo "Stopping any running Unsloth Studio servers\.\.\."$/    echo "__UNSLOTH_TEST_BODY_REACHED__"; return 99/' \
    "$SOURCE_UNINSTALL_SH" > "$UNINSTALL_SH"

# Isolate every environment-controlled removal path.
unset UNSLOTH_STUDIO_HOME STUDIO_HOME UNSLOTH_UNINSTALL_ROCM
XDG_RUNTIME_DIR="$_TMP_ROOT/run"
export XDG_RUNTIME_DIR
mkdir -p "$XDG_RUNTIME_DIR"

ok()   { echo "  PASS: $1"; PASS=$((PASS + 1)); }
nope() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

check() {
    if [ "$3" = "$2" ]; then ok "$1"; else nope "$1 (expected '$2', got '$3')"; fi
}

assert_says() {
    case "$3" in
        *"$2"*) ok "$1" ;;
        *) nope "$1 (no '$2' in: $(printf '%s' "$3" | head -n1))" ;;
    esac
}

assert_not_says() {
    case "$3" in
        *"$2"*) nope "$1 (unexpected '$2')" ;;
        *) ok "$1" ;;
    esac
}

# Check the fixture as a unit.
FIXTURE_PATHS=".unsloth/studio .unsloth/studio/auth/.desktop_secret .local/bin/unsloth .local/share/unsloth"

# assert_fixture <label> <home> present|gone
assert_fixture() {
    _missing=""
    for _rel in $FIXTURE_PATHS; do
        if [ -e "$2/$_rel" ] || [ -L "$2/$_rel" ]; then
            [ "$3" = "gone" ] && _missing="$_missing $_rel"
        else
            [ "$3" = "present" ] && _missing="$_missing $_rel"
        fi
    done
    if [ -z "$_missing" ]; then ok "$1"; else nope "$1 (wrong state:$_missing)"; fi
}

# Publish a validated fixture path. A global keeps setup failures from being
# hidden by command-substitution errexit behavior.
FIXTURE_HOME=""
make_home() {
    # Explicit template, not TMPDIR: BSD mktemp with no template implies -t and
    # resolves the directory itself, so on macOS TMPDIR= lands a sibling of
    # _TMP_ROOT and the check below aborts the suite.
    FIXTURE_HOME=$(mktemp -d "$_TMP_ROOT/home.XXXXXX") || FIXTURE_HOME=""
    case "$FIXTURE_HOME" in
        "$_TMP_ROOT"/*) ;;
        *) echo "FATAL: no fixture home under $_TMP_ROOT (got '$FIXTURE_HOME')" >&2; exit 1 ;;
    esac
    mkdir -p "$FIXTURE_HOME/.unsloth/studio/unsloth_studio/bin" \
             "$FIXTURE_HOME/.unsloth/studio/auth" \
             "$FIXTURE_HOME/.local/share/unsloth" \
             "$FIXTURE_HOME/.local/bin"
    : > "$FIXTURE_HOME/.unsloth/studio/unsloth_studio/bin/unsloth"
    : > "$FIXTURE_HOME/.unsloth/studio/auth/.desktop_secret"
    ln -s "$FIXTURE_HOME/.unsloth/studio/unsloth_studio/bin/unsloth" "$FIXTURE_HOME/.local/bin/unsloth"
}

RC=0
OUT=""
run_script() {
    _script="$1"
    _fake_home="$2"
    shift 2
    OUT=$(HOME="$_fake_home" sh "$_script" "$@" 2>&1) && RC=0 || RC=$?
}

run_uninstall() {
    run_script "$UNINSTALL_SH" "$@"
}

echo "=== structure: the wrapper that makes an early exit pipe-safe ==="

# Portable structural checks complement the Linux-only pipe test.
if grep -q '^_unsloth_uninstall_main() {' "$SOURCE_UNINSTALL_SH"; then
    ok "_unsloth_uninstall_main is defined at top level"
else
    nope "uninstall.sh is not wrapped in _unsloth_uninstall_main -- curl-pipe safety is gone"
fi
_tail="$(grep -vE '^[[:space:]]*(#|$)' "$SOURCE_UNINSTALL_SH" | tail -3)"
_expected_tail='{
    _unsloth_uninstall_main "$@"
}'
check "final entry is a complete compound block" "$_expected_tail" "$_tail"

echo "=== help: prints usage, removes nothing ==="

for _flag in --help -h; do
    make_home
    run_uninstall "$FIXTURE_HOME" "$_flag"
    check "$_flag exits 0" "0" "$RC"
    assert_says "$_flag prints usage" "Unsloth Studio uninstaller" "$OUT"
done
assert_fixture "help keeps the install" "$FIXTURE_HOME" present

echo "=== unknown arguments: abort before touching anything ==="

make_home
run_uninstall "$FIXTURE_HOME" --dry-run
check "'--dry-run' exits 2" "2" "$RC"
assert_says "the rejected argument is named" "unrecognized argument: --dry-run" "$OUT"
assert_says "the error states nothing was removed" "Nothing was removed" "$OUT"
assert_fixture "a rejected option keeps the install" "$FIXTURE_HOME" present

make_home
run_uninstall "$FIXTURE_HOME" uninstall
check "a positional argument exits 2" "2" "$RC"
assert_fixture "a rejected positional argument keeps the install" "$FIXTURE_HOME" present

make_home
run_uninstall "$FIXTURE_HOME" --dry-run --help
check "'--dry-run --help' exits 2" "2" "$RC"

make_home
run_uninstall "$FIXTURE_HOME" --help --dry-run
check "'--help --dry-run' exits 0" "0" "$RC"

echo "=== no arguments: the guard must not have broken the uninstall ==="

make_home
run_uninstall "$FIXTURE_HOME"
check "no arguments reach the instrumented body" "99" "$RC"
assert_says "the instrumented body announces itself" "$BODY_MARKER" "$OUT"
assert_fixture "instrumentation keeps the install" "$FIXTURE_HOME" present

# The body reaches host state on WSL, so do not run it there.
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "  SKIP: WSL -- the uninstall body reaches Windows-side state outside the fixture"
else
    make_home
    run_script "$SOURCE_UNINSTALL_SH" "$FIXTURE_HOME"
    check "no arguments still uninstalls (exit 0)" "0" "$RC"
    assert_fixture "no arguments removes the install" "$FIXTURE_HOME" gone
fi

echo "=== behaviour: a piped --help must not break the writer ==="

# Shrink the pipe so an unread script tail produces a broken writer. Skip where
# Linux F_SETPIPE_SZ is unavailable.
if python3 -c 'import fcntl, sys; sys.exit(0 if sys.platform.startswith("linux") else 1)' 2>/dev/null; then
    make_home
    verdict=$(UNINSTALL_SH="$UNINSTALL_SH" FAKE_HOME="$FIXTURE_HOME" python3 - <<'PY'
import fcntl, os, subprocess
F_SETPIPE_SZ, F_GETPIPE_SZ = 1031, 1032
with open(os.environ["UNINSTALL_SH"], "rb") as fh:
    src = fh.read()
r, w = os.pipe()
try:
    fcntl.fcntl(w, F_SETPIPE_SZ, 4096)
except OSError as exc:
    print(f"skip: cannot resize the pipe ({exc})")
    raise SystemExit(0)
# Reject a pipe size large enough to make the test vacuous.
if len(src) <= fcntl.fcntl(w, F_GETPIPE_SZ):
    print("vacuous: uninstall.sh now fits in one pipe buffer, re-derive this test")
    raise SystemExit(0)
proc = subprocess.Popen(
    ["sh", "-s", "--", "--help"], stdin = r,
    stdout = subprocess.PIPE, stderr = subprocess.STDOUT,
    env = dict(os.environ, HOME = os.environ["FAKE_HOME"]),
)
os.close(r)
try:
    with os.fdopen(w, "wb") as pipe:
        pipe.write(src)
    writer = "ok"
except BrokenPipeError:
    writer = "epipe"
out = proc.communicate(timeout = 60)[0]
body = "body" if b"__UNSLOTH_TEST_BODY_REACHED__" in out else "guarded"
print(f"{writer}:{proc.returncode}:{body}")
PY
    )
    case "$verdict" in
        skip:*|vacuous:*) echo "  SKIP: ${verdict#*: }" ;;
        *)
            # Check both the writer and script exit codes.
            check "piped --help stays guarded" "ok:0:guarded" "$verdict"
            ;;
    esac
else
    echo "  SKIP: pipe-buffer case needs Linux F_SETPIPE_SZ"
fi

echo "=== behaviour: the vulnerable final-call prefix is inert ==="

# Drop the closing brace and truncate the call immediately after its function
# name. Without the compound block this is a valid no-argument uninstall.
_tail_prefix="$_TMP_ROOT/tail-after-function-name.sh"
sed '$d' "$UNINSTALL_SH" | sed '$s/ "\$@"$//' > "$_tail_prefix"
make_home
OUT=$(HOME="$FIXTURE_HOME" sh -s -- --help < "$_tail_prefix" 2>&1) || true
assert_not_says "the incomplete final call never reaches the body" "$BODY_MARKER" "$OUT"
assert_fixture "tail truncation keeps the install" "$FIXTURE_HOME" present

echo "=== behaviour: a truncated download must remove nothing ==="

# One generic near-complete truncation complements the exact final-call boundary.
make_home
_bytes=$(wc -c < "$UNINSTALL_SH")
OUT=$(head -c "$((_bytes * 99 / 100))" "$UNINSTALL_SH" \
    | HOME="$FIXTURE_HOME" sh 2>&1) || true
assert_not_says "a 99% download never reaches the body" "$BODY_MARKER" "$OUT"
assert_fixture "a 99% download removes nothing" "$FIXTURE_HOME" present

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
