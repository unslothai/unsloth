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
sed -n '/^_setup_frontend_reap_if_exited()/,/^}/p' "$SETUP_SH" >> "$_FUNC_FILE"

if [ ! -s "$_FUNC_FILE" ]; then
    echo "FAIL: could not extract parallel helpers from $SETUP_SH"
    exit 1
fi

# shellcheck disable=SC1090
. "$_FUNC_FILE"

step() { :; }
substep() { :; }
# A zombie answers kill -0; under a PID 1 that never reaps it would read as still running.
_alive() { kill -0 "$1" 2>/dev/null && [ "$(ps -o stat= -p "$1" 2>/dev/null | cut -c1)" != Z ]; }

_setup_bg_fail() {
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

rm -f "$_FUNC_FILE"

_SETE_FILE=$(mktemp)
for _fn in setup_fail _setup_parallel_reset _setup_parallel_run _setup_bg_fail _setup_parallel_wait; do
    sed -n "/^$_fn()/,/^}/p" "$SETUP_SH" >> "$_SETE_FILE"
done
_SETE_OUT=$(
    bash -c '
        set -euo pipefail
        C_ERR=; step() { echo "STEP $*"; }
        _setup_frontend_reap_if_exited() { :; }; _setup_abort_frontend_job() { echo ABORTED; }
        . "$1"
        _setup_parallel_reset
        _setup_parallel_run "T5 a" bash -c "exit 5"
        _setup_parallel_run "T5 b" bash -c "sleep 0.3; echo B_DONE"
        _setup_parallel_wait
    ' _ "$_SETE_FILE" 2>&1
) && _sete_rc=0 || _sete_rc=$?
if [ "$_sete_rc" -ne 0 ] && printf '%s\n' "$_SETE_OUT" | grep -q 'T5 a failed' \
    && printf '%s\n' "$_SETE_OUT" | grep -q B_DONE && printf '%s\n' "$_SETE_OUT" | grep -q ABORTED; then
    echo "  PASS: set -e parent joins every job and aborts the frontend"
    PASS=$((PASS + 1))
else
    echo "  FAIL: set -e parallel wait (rc=$_sete_rc): $_SETE_OUT"
    FAIL=$((FAIL + 1))
fi
rm -f "$_SETE_FILE"

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

_ABORT_FILE=$(mktemp)
sed -n '/^_setup_restore_twbuild_gitignores_from()/,/^}/p' "$SETUP_SH" > "$_ABORT_FILE"
sed -n '/^_setup_pid_tree()/,/^}/p' "$SETUP_SH" >> "$_ABORT_FILE"
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

# npm and vite run as grandchildren of the job; a bash waiting on them defers SIGTERM.
( bash -c 'bash -c "sleep 31; true"; true'; true ) &
_SETUP_FRONTEND_BG_PID=$!
_tree_pids=""
for _ in 1 2 3 4 5 6 7 8 9 10; do
    _c1=$(pgrep -P "$_SETUP_FRONTEND_BG_PID" 2>/dev/null | head -1)
    _c2=$( [ -n "$_c1" ] && pgrep -P "$_c1" 2>/dev/null | head -1 )
    _c3=$( [ -n "$_c2" ] && pgrep -P "$_c2" 2>/dev/null | head -1 )
    [ -n "$_c3" ] && { _tree_pids="$_c1 $_c2 $_c3"; break; }
    sleep 0.2
done
_setup_abort_frontend_job
_orphans=""
for _p in $_tree_pids; do _alive "$_p" && _orphans="$_orphans $_p"; done
if [ -n "$_tree_pids" ] && [ -z "$_orphans" ]; then
    echo "  PASS: abort kills the job's grandchildren"
    PASS=$((PASS + 1))
else
    echo "  FAIL: abort left descendants running (tree='$_tree_pids' orphans='$_orphans')"
    FAIL=$((FAIL + 1))
    for _p in $_orphans; do kill -KILL "$_p" 2>/dev/null || true; done
fi
rm -f "$_ABORT_FILE"

_REAP_FILE=$(mktemp)
sed -n '/^_setup_frontend_reap_if_exited()/,/^}/p' "$SETUP_SH" >> "$_REAP_FILE"
# shellcheck disable=SC1090
. "$_REAP_FILE"
_SETUP_FAIL_CALLED=0
_setup_bg_fail() {
    _SETUP_FAIL_CALLED=1
    return 1
}
false &
_SETUP_FRONTEND_BG_PID=$!
wait "$_SETUP_FRONTEND_BG_PID" 2>/dev/null || true
if _setup_frontend_reap_if_exited; then
    echo "  FAIL: reap should propagate frontend failure"
    FAIL=$((FAIL + 1))
else
    if [ "$_SETUP_FAIL_CALLED" -eq 1 ]; then
        echo "  PASS: reap fails when frontend job failed"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: reap did not fail"
        FAIL=$((FAIL + 1))
    fi
fi
rm -f "$_REAP_FILE"

_MARK_FILE=$(mktemp)
for _fn in setup_fail _setup_bg_fail _setup_abort_frontend_job _setup_frontend_reap_if_exited; do
    sed -n "/^$_fn()/,/^}/p" "$SETUP_SH" >> "$_MARK_FILE"
done
_MARK_OUT=$(
    UNSLOTH_TAURI_UPDATE=1 bash -c '
        C_ERR=; step() { :; }; _setup_restore_twbuild_gitignores_from() { :; }
        . "$1"
        ( setup_fail 7 "OXC validator dependency installation failed" ) &
        _SETUP_FRONTEND_BG_PID=$!
        while kill -0 "$_SETUP_FRONTEND_BG_PID" 2>/dev/null; do sleep 0.05; done
        _setup_frontend_reap_if_exited
    ' _ "$_MARK_FILE" 2>/dev/null
) && _mark_rc=0 || _mark_rc=$?
if [ "$_mark_rc" -eq 7 ] && [ "$(printf '%s\n' "$_MARK_OUT" | grep -c '^\[TAURI:ERROR\]')" -eq 1 ] \
    && printf '%s\n' "$_MARK_OUT" | grep -q '^\[TAURI:ERROR\] OXC validator'; then
    echo "  PASS: one specific TAURI error marker, exit code kept"
    PASS=$((PASS + 1))
else
    echo "  FAIL: TAURI markers (rc=$_mark_rc): $_MARK_OUT"
    FAIL=$((FAIL + 1))
fi
rm -f "$_MARK_FILE"

_nested_defs=$(awk '
    /^if / {depth++}
    /^fi( |$)/ {depth--}
    /^(_setup_frontend_reap_if_exited|_setup_frontend_join|_setup_abort_frontend_job|_setup_pid_tree|_setup_launch_frontend_build_and_oxc|_setup_frontend_build_and_oxc)\(\) \{/ && depth != 0 {print $1}
' "$SETUP_SH")
_defs_found=$(grep -cE '^(_setup_frontend_reap_if_exited|_setup_frontend_join|_setup_abort_frontend_job)\(\) \{' "$SETUP_SH" || true)
if [ -z "$_nested_defs" ] && [ "$_defs_found" -eq 3 ]; then
    echo "  PASS: frontend job helpers are defined unconditionally"
    PASS=$((PASS + 1))
else
    echo "  FAIL: frontend job helpers defined inside a conditional: $_nested_defs (found $_defs_found)"
    FAIL=$((FAIL + 1))
fi

_EXIT_FILE=$(mktemp)
for _fn in _setup_restore_twbuild_gitignores_from _setup_pid_tree _setup_abort_frontend_job _setup_launch_frontend_build_and_oxc; do
    sed -n "/^$_fn()/,/^}/p" "$SETUP_SH" >> "$_EXIT_FILE"
done
_EXIT_PIDS=$(mktemp)
bash -c '
    set -euo pipefail
    SCRIPT_DIR=/nonexistent
    . "$1"
    _setup_frontend_build_and_oxc() { bash -c "sleep 32; true"; true; }
    _setup_launch_frontend_build_and_oxc
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        _c=$(pgrep -P "$_SETUP_FRONTEND_BG_PID" 2>/dev/null | head -1)
        _g=$( [ -n "$_c" ] && pgrep -P "$_c" 2>/dev/null | head -1 )
        [ -n "$_g" ] && break
        sleep 0.2
    done
    echo "$_SETUP_FRONTEND_BG_PID $_c $_g" > "$2"
    false
' _ "$_EXIT_FILE" "$_EXIT_PIDS" 2>/dev/null && _exit_rc=0 || _exit_rc=$?
_exit_left=""
for _p in $(cat "$_EXIT_PIDS"); do _alive "$_p" && _exit_left="$_exit_left $_p"; done
if [ "$_exit_rc" -ne 0 ] && [ "$(wc -w < "$_EXIT_PIDS")" -eq 3 ] && [ -z "$_exit_left" ]; then
    echo "  PASS: set -e exit after launch stops the frontend job"
    PASS=$((PASS + 1))
else
    echo "  FAIL: set -e exit left the job running (rc=$_exit_rc pids=$(cat "$_EXIT_PIDS") left=$_exit_left)"
    FAIL=$((FAIL + 1))
    for _p in $_exit_left; do kill -KILL "$_p" 2>/dev/null || true; done
fi
rm -f "$_EXIT_FILE" "$_EXIT_PIDS"

# A failing sidecar worker leaves the frontend sibling running and its own exit code stands.
_SIB_FILE=$(mktemp)
for _fn in setup_fail _setup_parallel_reset _setup_parallel_run _setup_bg_fail _setup_parallel_wait \
    _setup_restore_twbuild_gitignores_from _setup_frontend_reap_if_exited _setup_pid_tree _setup_abort_frontend_job; do
    sed -n "/^$_fn()/,/^}/p" "$SETUP_SH" >> "$_SIB_FILE"
done
_SIB_OUT=$(
    bash -c '
        set -euo pipefail
        C_ERR=; SCRIPT_DIR=/nonexistent; step() { echo "STEP $*"; }
        . "$1"
        ( sleep 6; echo FRONTEND_DONE ) &
        _SETUP_FRONTEND_BG_PID=$!
        _setup_parallel_reset
        _setup_parallel_run "T5 5.5.0" setup_fail 9 "install transformers 5.5.0 failed"
        _setup_parallel_wait
    ' _ "$_SIB_FILE" 2>&1
) && _sib_rc=0 || _sib_rc=$?
if [ "$_sib_rc" -eq 1 ] && ! printf '%s\n' "$_SIB_OUT" | grep -q 'Frontend build or OXC install failed'; then
    echo "  PASS: a failing sidecar worker does not abort the frontend sibling"
    PASS=$((PASS + 1))
else
    echo "  FAIL: sidecar failure hit the frontend job (rc=$_sib_rc): $_SIB_OUT"
    FAIL=$((FAIL + 1))
fi
rm -f "$_SIB_FILE"

# An installed package (no pyproject.toml beside studio/) joins npm before the core reinstall.
if awk '
    /^if \[ -f "\$REPO_ROOT\/pyproject.toml" \]; then$/ {g=1; next}
    g == 1 && /_setup_frontend_reap_if_exited/ {g=2; next}
    g == 2 && /^else$/ {g=3; next}
    g == 3 && /^    _setup_frontend_join$/ {g=4; next}
    g == 4 && /^if \[ "\$_SKIP_PYTHON_DEPS" = false \]; then$/ {g=5; next}
    g == 5 && /^    install_python_stack$/ {found=1; exit}
    END {exit !found}
' "$SETUP_SH"; then
    echo "  PASS: installed-package runs join the frontend job before install_python_stack"
    PASS=$((PASS + 1))
else
    echo "  FAIL: installed-package runs can overlap npm with the core reinstall"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "Passed: $PASS  Failed: $FAIL"
[ "$FAIL" -eq 0 ] || exit 1
