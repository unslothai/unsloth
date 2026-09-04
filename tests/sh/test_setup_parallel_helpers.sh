#!/bin/bash
# Unit tests for parallel setup helpers from studio/setup.sh (issue #8818).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
PASS=0
FAIL=0

_FUNC_FILE=$(mktemp)
sed -n '/^_setup_parallel_reset()/,/^}/p' "$SETUP_SH" > "$_FUNC_FILE"
sed -n '/^_setup_parallel_run()/,/^}/p' "$SETUP_SH" >> "$_FUNC_FILE"
sed -n '/^_setup_parallel_wait()/,/^}/p' "$SETUP_SH" >> "$_FUNC_FILE"
sed -n '/^step()/,/^}/p' "$SETUP_SH" | head -n 1 >> "$_FUNC_FILE"
sed -n '/^setup_fail()/,/^}/p' "$SETUP_SH" >> "$_FUNC_FILE"

if [ ! -s "$_FUNC_FILE" ]; then
    echo "FAIL: could not extract parallel helpers from $SETUP_SH"
    exit 1
fi

# shellcheck disable=SC1090
. "$_FUNC_FILE"

setup_fail() {
    return 1
}

assert_parallel_ok() {
    local label="$1"
    _setup_parallel_reset
    _setup_parallel_run "$label" true
    if _setup_parallel_wait; then
        echo "  PASS: $label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $label"
        FAIL=$((FAIL + 1))
    fi
}

assert_parallel_fail() {
    local label="$1"
    _setup_parallel_reset
    _setup_parallel_run "$label" false
    if _setup_parallel_wait; then
        echo "  FAIL: $label (expected failure)"
        FAIL=$((FAIL + 1))
    else
        echo "  PASS: $label (failed as expected)"
        PASS=$((PASS + 1))
    fi
}

echo "setup parallel helpers"
assert_parallel_ok "single successful job"
assert_parallel_fail "single failing job"

_setup_parallel_reset
_setup_parallel_run "job-a" true
_setup_parallel_run "job-b" true
if _setup_parallel_wait; then
    echo "  PASS: two successful jobs"
    PASS=$((PASS + 1))
else
    echo "  FAIL: two successful jobs"
    FAIL=$((FAIL + 1))
fi

if grep -q '_setup_parallel_wait_fail_open' "$SETUP_SH"; then
    echo "  FAIL: unused _setup_parallel_wait_fail_open is still defined"
    FAIL=$((FAIL + 1))
else
    echo "  PASS: unused _setup_parallel_wait_fail_open is gone"
    PASS=$((PASS + 1))
fi

rm -f "$_FUNC_FILE"

# ── gitignore hide/restore scoped to npm run build ──
_GI_FILE=$(mktemp)
sed -n '/^_setup_hide_star_gitignores_from()/,/^}/p' "$SETUP_SH" > "$_GI_FILE"
sed -n '/^_setup_restore_star_gitignores()/,/^}/p' "$SETUP_SH" >> "$_GI_FILE"
sed -n '/^_setup_restore_twbuild_gitignores_from()/,/^}/p' "$SETUP_SH" >> "$_GI_FILE"
# shellcheck disable=SC1090
. "$_GI_FILE"
_HIDDEN_GITIGNORES=()

_GI_ROOT=$(mktemp -d)
mkdir -p "$_GI_ROOT/frontend"
printf '*\n' > "$_GI_ROOT/.gitignore"
_setup_hide_star_gitignores_from "$_GI_ROOT/frontend"
if [ -f "$_GI_ROOT/.gitignore._twbuild" ] && [ ! -f "$_GI_ROOT/.gitignore" ]; then
    echo "  PASS: hide star gitignore during build window"
    PASS=$((PASS + 1))
else
    echo "  FAIL: hide star gitignore during build window"
    FAIL=$((FAIL + 1))
fi
_setup_restore_star_gitignores
if [ -f "$_GI_ROOT/.gitignore" ] && [ ! -f "$_GI_ROOT/.gitignore._twbuild" ]; then
    echo "  PASS: restore star gitignore after build"
    PASS=$((PASS + 1))
else
    echo "  FAIL: restore star gitignore after build"
    FAIL=$((FAIL + 1))
fi

printf '*\n' > "$_GI_ROOT/.gitignore"
mv "$_GI_ROOT/.gitignore" "$_GI_ROOT/.gitignore._twbuild"
_setup_restore_twbuild_gitignores_from "$_GI_ROOT/frontend"
if [ -f "$_GI_ROOT/.gitignore" ] && [ ! -f "$_GI_ROOT/.gitignore._twbuild" ]; then
    echo "  PASS: abort path restores leftover ._twbuild gitignore"
    PASS=$((PASS + 1))
else
    echo "  FAIL: abort path restores leftover ._twbuild gitignore"
    FAIL=$((FAIL + 1))
fi
rm -rf "$_GI_ROOT"
rm -f "$_GI_FILE"

# ── abort kills the background frontend job ──
_ABORT_FILE=$(mktemp)
sed -n '/^_setup_restore_twbuild_gitignores_from()/,/^}/p' "$SETUP_SH" > "$_ABORT_FILE"
sed -n '/^_setup_abort_frontend_job()/,/^}/p' "$SETUP_SH" >> "$_ABORT_FILE"
# shellcheck disable=SC1090
. "$_ABORT_FILE"
sleep 30 &
_SETUP_FRONTEND_BG_PID=$!
_sleep_pid=$_SETUP_FRONTEND_BG_PID
_setup_abort_frontend_job
if kill -0 "$_sleep_pid" 2>/dev/null; then
    echo "  FAIL: abort should kill the frontend job"
    FAIL=$((FAIL + 1))
    kill -KILL "$_sleep_pid" 2>/dev/null || true
    wait "$_sleep_pid" 2>/dev/null || true
else
    echo "  PASS: abort kills the frontend job"
    PASS=$((PASS + 1))
fi
rm -f "$_ABORT_FILE"

echo ""
echo "Passed: $PASS  Failed: $FAIL"
[ "$FAIL" -eq 0 ] || exit 1
