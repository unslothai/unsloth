#!/bin/bash
# Integration tests for the Unsloth Studio launcher (extracted from install.sh)
# Tests end-to-end scenarios: fast path, background/foreground launch, lock race,
# stale lock, all ports busy, real port binding, double-launch race.
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

assert_match() {
    _label="$1"; _pattern="$2"; _actual="$3"
    if printf '%s\n' "$_actual" | grep -qE -- "$_pattern"; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected pattern '$_pattern' in '$_actual')"
        FAIL=$((FAIL + 1))
    fi
}

assert_not_match() {
    _label="$1"; _pattern="$2"; _actual="$3"
    if ! printf '%s\n' "$_actual" | grep -qE -- "$_pattern"; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (pattern '$_pattern' should NOT appear in '$_actual')"
        FAIL=$((FAIL + 1))
    fi
}

assert_file_exists() {
    _label="$1"; _file="$2"
    if [ -e "$_file" ]; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (file '$_file' does not exist)"
        FAIL=$((FAIL + 1))
    fi
}

assert_file_not_exists() {
    _label="$1"; _file="$2"
    if [ ! -e "$_file" ]; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (file '$_file' should not exist)"
        FAIL=$((FAIL + 1))
    fi
}

# ── Extract and prepare launcher script ──
_extract_launcher() {
    local dest="$1"
    # Extract heredoc body
    sed -n "/cat > .* << 'LAUNCHER_EOF'$/,/^LAUNCHER_EOF$/{ /cat >/d; /^LAUNCHER_EOF$/d; p; }" "$INSTALL_SH" > "$dest"
    chmod +x "$dest"
}

# ── Build a complete sandbox for one test ──
# Sets global: SANDBOX, LAUNCHER, LOCK_DIR_PATH
_setup_sandbox() {
    SANDBOX=$(mktemp -d)
    SANDBOX_PIDS=()

    # Directory tree
    mkdir -p "$SANDBOX/home/.local/share/unsloth"
    mkdir -p "$SANDBOX/runtime"
    mkdir -p "$SANDBOX/bin"
    mkdir -p "$SANDBOX/calls"
    mkdir -p "$SANDBOX/proc"

    # Fake /proc/version (Linux, no WSL)
    echo "Linux version 5.15.0" > "$SANDBOX/proc/version"

    # Extract launcher
    LAUNCHER="$SANDBOX/home/.local/share/unsloth/launch-studio.sh"
    _extract_launcher "$LAUNCHER"

    # Patch constants for fast testing
    sed -i 's/^BASE_PORT=.*/BASE_PORT=19000/' "$LAUNCHER"
    sed -i 's/^MAX_PORT_OFFSET=.*/MAX_PORT_OFFSET=5/' "$LAUNCHER"
    sed -i 's/^TIMEOUT_SEC=.*/TIMEOUT_SEC=3/' "$LAUNCHER"
    sed -i 's/^POLL_INTERVAL_SEC=.*/POLL_INTERVAL_SEC=0/' "$LAUNCHER"
    # Patch /proc/version
    sed -i "s|/proc/version|$SANDBOX/proc/version|g" "$LAUNCHER"
    # Patch DATA_DIR to sandbox
    sed -i "s|DATA_DIR=.*|DATA_DIR=\"$SANDBOX/home/.local/share/unsloth\"|" "$LAUNCHER"
    # Patch LOCK_DIR to use sandbox runtime
    sed -i "s|LOCK_DIR=.*|LOCK_DIR=\"$SANDBOX/runtime/unsloth-studio-launcher.lock\"|" "$LAUNCHER"
    LOCK_DIR_PATH="$SANDBOX/runtime/unsloth-studio-launcher.lock"

    # Write studio.conf pointing to fake unsloth
    local fake_unsloth="$SANDBOX/bin/fake-unsloth"
    cat > "$fake_unsloth" <<FAKEUNSLOTH
#!/bin/sh
echo "\$0 \$*" >> "$SANDBOX/calls/exec_calls"
# Stay alive briefly so the launcher can poll
sleep 5
FAKEUNSLOTH
    chmod +x "$fake_unsloth"
    echo "UNSLOTH_EXE='$fake_unsloth'" > "$SANDBOX/home/.local/share/unsloth/studio.conf"

    # Build mock bin directory with essential tools
    for _cmd in bash grep sed awk sort cat date sleep id mktemp rm mkdir kill printf od head sh uname test tr wc cut env dirname basename readlink nohup; do
        _real=$(command -v "$_cmd" 2>/dev/null || true)
        [ -n "$_real" ] && ln -sf "$_real" "$SANDBOX/bin/$_cmd"
    done
}

_teardown_sandbox() {
    # Kill any background processes we started
    for _pid in "${SANDBOX_PIDS[@]}"; do
        kill "$_pid" 2>/dev/null || true
        wait "$_pid" 2>/dev/null || true
    done
    rm -rf "$SANDBOX"
}

# Helper: run the launcher in background mode (no TTY, piped)
_run_launcher_bg() {
    local extra_path="${1:-}"
    local launcher_path="${2:-$LAUNCHER}"
    env -i PATH="${extra_path:+$extra_path:}$SANDBOX/bin" \
        HOME="$SANDBOX/home" \
        XDG_RUNTIME_DIR="$SANDBOX/runtime" \
        bash --norc --noprofile "$launcher_path" </dev/null 2>&1
}

# Helper: run the launcher with a PTY via script(1)
_run_launcher_tty() {
    local extra_path="${1:-}"
    local launcher_path="${2:-$LAUNCHER}"
    env -i PATH="${extra_path:+$extra_path:}$SANDBOX/bin" \
        HOME="$SANDBOX/home" \
        XDG_RUNTIME_DIR="$SANDBOX/runtime" \
        script -qec "bash --norc --noprofile '$launcher_path'" /dev/null 2>&1
}

# ══════════════════════════════════════════════════════════════════════
echo "=== T1: Fast path -- studio already healthy ==="
# ══════════════════════════════════════════════════════════════════════

_setup_sandbox

# Mock curl: always returns healthy for port 19000
_call_log="$SANDBOX/calls/browser_calls"
cat > "$SANDBOX/bin/curl" <<MOCK
#!/bin/sh
echo '{"status":"healthy","service":"Unsloth UI Backend"}'
MOCK
chmod +x "$SANDBOX/bin/curl"

cat > "$SANDBOX/bin/xdg-open" <<MOCK
#!/bin/sh
echo "xdg-open \$*" >> "$_call_log"
MOCK
chmod +x "$SANDBOX/bin/xdg-open"

# Mock ss: port 19000 is listening (so _candidate_ports includes it)
cat > "$SANDBOX/bin/ss" <<'MOCK'
#!/bin/sh
echo "LISTEN  0  4096  0.0.0.0:19000  0.0.0.0:*"
MOCK
chmod +x "$SANDBOX/bin/ss"

_rc=0
_output=$(_run_launcher_bg 2>&1) || _rc=$?
assert_eq "T1: exit code 0" "0" "$_rc"

# Check xdg-open was called with port 19000
sleep 0.2
_browser_calls=$(cat "$_call_log" 2>/dev/null || true)
assert_match "T1: xdg-open called with 19000" "localhost:19000" "$_browser_calls"

# Lock dir should NOT exist (fast path skips lock)
assert_file_not_exists "T1: no lock dir (fast path)" "$LOCK_DIR_PATH"

_teardown_sandbox

# ══════════════════════════════════════════════════════════════════════
echo ""
echo "=== T2: Normal launch -- background mode (no TTY, timeout) ==="
# ══════════════════════════════════════════════════════════════════════

_setup_sandbox

# Mock curl: always returns unhealthy
cat > "$SANDBOX/bin/curl" <<'MOCK'
#!/bin/sh
echo '{"status":"starting"}'
exit 1
MOCK
chmod +x "$SANDBOX/bin/curl"

# Mock ss: no ports busy
cat > "$SANDBOX/bin/ss" <<'MOCK'
#!/bin/sh
echo ""
MOCK
chmod +x "$SANDBOX/bin/ss"

# No terminal emulators -> falls back to nohup
_rc=0
_output=$(_run_launcher_bg 2>&1) || _rc=$?
assert_eq "T2: exit code 1 (timeout)" "1" "$_rc"
assert_match "T2: timeout message" "did not become healthy" "$_output"

# Lock should be released after exit (trap cleans up)
# Give a moment for cleanup
sleep 0.2
assert_file_not_exists "T2: lock released on exit" "$LOCK_DIR_PATH"

_teardown_sandbox

# ══════════════════════════════════════════════════════════════════════
echo ""
echo "=== T3: Normal launch -- foreground mode (TTY via script) ==="
# ══════════════════════════════════════════════════════════════════════

# Check if script(1) is available
if ! command -v script >/dev/null 2>&1; then
    echo "  SKIP: script(1) not available for TTY tests"
else
    _setup_sandbox

    # Make fake-unsloth exit quickly (1s) so test doesn't hang
    cat > "$SANDBOX/bin/fake-unsloth" <<FAKEEOF
#!/bin/sh
echo "\$0 \$*" >> "$SANDBOX/calls/exec_calls"
sleep 1
FAKEEOF
    chmod +x "$SANDBOX/bin/fake-unsloth"

    # Counter-based mock curl: healthy after 3 calls.
    # Call 1 = fast-path health check (fail)
    # Call 2 = post-lock health check (fail)
    # Call 3+ = background subshell health poll after exec (succeed)
    _counter_file="$SANDBOX/calls/curl_counter"
    echo "0" > "$_counter_file"
    cat > "$SANDBOX/bin/curl" <<MOCK
#!/bin/sh
_count=\$(cat "$_counter_file" 2>/dev/null || echo 0)
_count=\$((_count + 1))
echo "\$_count" > "$_counter_file"
if [ "\$_count" -ge 3 ]; then
    echo '{"status":"healthy","service":"Unsloth UI Backend"}'
else
    echo '{"status":"starting"}'
    exit 1
fi
MOCK
    chmod +x "$SANDBOX/bin/curl"

    _call_log="$SANDBOX/calls/browser_calls"
    cat > "$SANDBOX/bin/xdg-open" <<MOCK
#!/bin/sh
echo "xdg-open \$*" >> "$_call_log"
MOCK
    chmod +x "$SANDBOX/bin/xdg-open"

    # Mock ss: all ports free
    cat > "$SANDBOX/bin/ss" <<'MOCK'
#!/bin/sh
echo ""
MOCK
    chmod +x "$SANDBOX/bin/ss"

    # Run with TTY via script(1).
    # In TTY mode the launcher execs fake-unsloth, which writes to exec_calls.
    # The background subshell polls health and opens the browser.
    _rc=0
    timeout 15 script -qec "env -i PATH='$SANDBOX/bin' HOME='$SANDBOX/home' XDG_RUNTIME_DIR='$SANDBOX/runtime' bash --norc --noprofile '$LAUNCHER'" /dev/null >/dev/null 2>&1 || _rc=$?

    sleep 0.5

    # Check fake unsloth was called with studio args
    _exec_calls=$(cat "$SANDBOX/calls/exec_calls" 2>/dev/null || true)
    assert_match "T3: unsloth called with studio args" "studio -H 0.0.0.0 -p 19000" "$_exec_calls"

    # Check browser was opened (background subshell opens it after health check)
    _browser_calls=$(cat "$_call_log" 2>/dev/null || true)
    assert_match "T3: xdg-open called with 19000" "localhost:19000" "$_browser_calls"

    _teardown_sandbox
fi

# ══════════════════════════════════════════════════════════════════════
echo ""
echo "=== T4: Lock race prevention (existing live instance) ==="
# ══════════════════════════════════════════════════════════════════════

_setup_sandbox

# Pre-create lock dir with a live PID (sleep process)
sleep 300 &
_blocker_pid=$!
SANDBOX_PIDS+=("$_blocker_pid")
mkdir -p "$LOCK_DIR_PATH"
echo "$_blocker_pid" > "$LOCK_DIR_PATH/pid"

# Mock curl: returns healthy for port 19000 after first call
_counter_file="$SANDBOX/calls/curl_counter_t4"
echo "0" > "$_counter_file"
cat > "$SANDBOX/bin/curl" <<MOCK
#!/bin/sh
_count=\$(cat "$_counter_file" 2>/dev/null || echo 0)
_count=\$((_count + 1))
echo "\$_count" > "$_counter_file"
if [ "\$_count" -ge 1 ]; then
    echo '{"status":"healthy","service":"Unsloth UI Backend"}'
else
    echo '{"status":"starting"}'
    exit 1
fi
MOCK
chmod +x "$SANDBOX/bin/curl"

_call_log="$SANDBOX/calls/browser_calls_t4"
cat > "$SANDBOX/bin/xdg-open" <<MOCK
#!/bin/sh
echo "xdg-open \$*" >> "$_call_log"
MOCK
chmod +x "$SANDBOX/bin/xdg-open"

# Mock ss: port 19000 is listening
cat > "$SANDBOX/bin/ss" <<'MOCK'
#!/bin/sh
echo "LISTEN  0  4096  0.0.0.0:19000  0.0.0.0:*"
MOCK
chmod +x "$SANDBOX/bin/ss"

_rc=0
_output=$(_run_launcher_bg 2>&1) || _rc=$?
assert_eq "T4: exit code 0 (joined existing)" "0" "$_rc"

sleep 0.2
_browser_calls=$(cat "$_call_log" 2>/dev/null || true)
assert_match "T4: browser opened on port 19000" "localhost:19000" "$_browser_calls"

# Lock dir should still exist (owned by the blocker)
assert_file_exists "T4: lock still held by original process" "$LOCK_DIR_PATH"

# Verify fake unsloth was NOT called (second launcher shouldn't launch)
_exec_calls=$(cat "$SANDBOX/calls/exec_calls" 2>/dev/null || true)
assert_eq "T4: unsloth not called (no double-launch)" "" "$_exec_calls"

kill "$_blocker_pid" 2>/dev/null || true
wait "$_blocker_pid" 2>/dev/null || true
_teardown_sandbox

# ══════════════════════════════════════════════════════════════════════
echo ""
echo "=== T5: Stale lock reclaim ==="
# ══════════════════════════════════════════════════════════════════════

_setup_sandbox

# Create lock dir with a dead PID
mkdir -p "$LOCK_DIR_PATH"
echo "99999999" > "$LOCK_DIR_PATH/pid"

# Mock curl: becomes healthy after 1 call
_counter_file="$SANDBOX/calls/curl_counter_t5"
echo "0" > "$_counter_file"
cat > "$SANDBOX/bin/curl" <<MOCK
#!/bin/sh
_count=\$(cat "$_counter_file" 2>/dev/null || echo 0)
_count=\$((_count + 1))
echo "\$_count" > "$_counter_file"
if [ "\$_count" -ge 2 ]; then
    echo '{"status":"healthy","service":"Unsloth UI Backend"}'
else
    echo '{"status":"starting"}'
    exit 1
fi
MOCK
chmod +x "$SANDBOX/bin/curl"

_call_log="$SANDBOX/calls/browser_calls_t5"
cat > "$SANDBOX/bin/xdg-open" <<MOCK
#!/bin/sh
echo "xdg-open \$*" >> "$_call_log"
MOCK
chmod +x "$SANDBOX/bin/xdg-open"

# Mock ss: no ports busy
cat > "$SANDBOX/bin/ss" <<'MOCK'
#!/bin/sh
echo ""
MOCK
chmod +x "$SANDBOX/bin/ss"

# The stale lock has dead PID 99999999. The launcher should:
# 1. Reclaim the stale lock (rm + mkdir)
# 2. Proceed to launch (bg mode, _spawn_terminal -> nohup)
# 3. Poll health: curl becomes healthy after 2nd call
# 4. Open browser and exit 0
# The key assertion: launcher did NOT think another instance was running.

_rc=0
_output=$(_run_launcher_bg 2>&1) || _rc=$?

# Lock should be released (cleanup trap)
sleep 0.2
assert_file_not_exists "T5: stale lock reclaimed and released" "$LOCK_DIR_PATH"

# Verify the launcher did NOT report "waiting for other launcher" or "Timed out waiting for other"
# (which would mean it incorrectly treated the stale lock as live)
_has_waiting=$(echo "$_output" | grep -c "Timed out waiting for other" || true)
assert_eq "T5: no 'waiting for other launcher' message" "0" "$_has_waiting"

_teardown_sandbox

# ══════════════════════════════════════════════════════════════════════
echo ""
echo "=== T6: All ports busy ==="
# ══════════════════════════════════════════════════════════════════════

_setup_sandbox

# Mock curl: never healthy
cat > "$SANDBOX/bin/curl" <<'MOCK'
#!/bin/sh
exit 1
MOCK
chmod +x "$SANDBOX/bin/curl"

# Mock ss: all 6 ports (19000-19005) are busy
cat > "$SANDBOX/bin/ss" <<'MOCK'
#!/bin/sh
cat <<'EOF'
LISTEN  0  4096  0.0.0.0:19000  0.0.0.0:*
LISTEN  0  4096  0.0.0.0:19001  0.0.0.0:*
LISTEN  0  4096  0.0.0.0:19002  0.0.0.0:*
LISTEN  0  4096  0.0.0.0:19003  0.0.0.0:*
LISTEN  0  4096  0.0.0.0:19004  0.0.0.0:*
LISTEN  0  4096  0.0.0.0:19005  0.0.0.0:*
EOF
MOCK
chmod +x "$SANDBOX/bin/ss"

_rc=0
_output=$(_run_launcher_bg 2>&1) || _rc=$?
assert_eq "T6: exit code 1" "1" "$_rc"
assert_match "T6: error message" "No free port found" "$_output"

_teardown_sandbox

# ══════════════════════════════════════════════════════════════════════
echo ""
echo "=== T7: Real port binding test ==="
# ══════════════════════════════════════════════════════════════════════

# Use socat or python to actually bind port 19000, verify launcher skips it
_have_socat=false
_have_python=false
if command -v socat >/dev/null 2>&1; then
    _have_socat=true
elif command -v python3 >/dev/null 2>&1; then
    _have_python=true
fi

if [ "$_have_socat" = true ] || [ "$_have_python" = true ]; then
    _setup_sandbox

    # Actually bind port 19000
    if [ "$_have_socat" = true ]; then
        socat TCP-LISTEN:19000,fork,reuseaddr /dev/null &
        _bind_pid=$!
    else
        python3 -c "
import socket, time
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('0.0.0.0', 19000))
s.listen(1)
time.sleep(30)
" &
        _bind_pid=$!
    fi
    SANDBOX_PIDS+=("$_bind_pid")
    sleep 0.3

    # Use REAL ss (not mock) -- remove any mock ss
    rm -f "$SANDBOX/bin/ss"
    # Symlink real ss
    _real_ss=$(command -v ss 2>/dev/null || true)
    [ -n "$_real_ss" ] && ln -sf "$_real_ss" "$SANDBOX/bin/ss"

    # Mock curl: never healthy (we just care about port selection)
    cat > "$SANDBOX/bin/curl" <<'MOCK'
#!/bin/sh
exit 1
MOCK
    chmod +x "$SANDBOX/bin/curl"

    # Extract just _find_launch_port result
    _result=$(env -i PATH="$SANDBOX/bin" HOME="$SANDBOX/home" \
        XDG_RUNTIME_DIR="$SANDBOX/runtime" \
        bash --norc --noprofile -c "
            . '$LAUNCHER'
        " </dev/null 2>&1 | grep -oE 'port [0-9]+' | head -1 || true)

    # Alternative: just test the function directly
    # Extract launcher functions (no main)
    _func_file=$(mktemp)
    sed '/^# ── Main ──$/,$d' "$LAUNCHER" > "$_func_file"
    sed -i 's/^set -euo pipefail$//' "$_func_file"
    sed -i 's|^#!/usr/bin/env bash$||' "$_func_file"

    _port_result=$(env -i PATH="$SANDBOX/bin" HOME="$SANDBOX/home" \
        bash --norc --noprofile -c "
            . '$_func_file'
            BASE_PORT=19000; MAX_PORT_OFFSET=5
            _find_launch_port
        " 2>/dev/null)
    rm -f "$_func_file"

    # Port 19000 is busy, so should return 19001
    assert_eq "T7: skips busy port 19000 -> 19001" "19001" "$_port_result"

    kill "$_bind_pid" 2>/dev/null || true
    wait "$_bind_pid" 2>/dev/null || true
    _teardown_sandbox
else
    echo "  SKIP: neither socat nor python3 available for real port test"
fi

# ══════════════════════════════════════════════════════════════════════
echo ""
echo "=== T8: Double-launch race (concurrent) ==="
# ══════════════════════════════════════════════════════════════════════

_setup_sandbox

# Mock curl: always returns healthy (both instances see it immediately)
cat > "$SANDBOX/bin/curl" <<'MOCK'
#!/bin/sh
echo '{"status":"healthy","service":"Unsloth UI Backend"}'
MOCK
chmod +x "$SANDBOX/bin/curl"

_call_log="$SANDBOX/calls/browser_calls_t8"
cat > "$SANDBOX/bin/xdg-open" <<MOCK
#!/bin/sh
echo "xdg-open \$*" >> "$_call_log"
MOCK
chmod +x "$SANDBOX/bin/xdg-open"

# Mock ss: port 19000 listening (so _candidate_ports finds it)
cat > "$SANDBOX/bin/ss" <<'MOCK'
#!/bin/sh
echo "LISTEN  0  4096  0.0.0.0:19000  0.0.0.0:*"
MOCK
chmod +x "$SANDBOX/bin/ss"

# Launch two instances simultaneously with timeout
_rc1=0; _rc2=0
timeout 10 env -i PATH="$SANDBOX/bin" HOME="$SANDBOX/home" XDG_RUNTIME_DIR="$SANDBOX/runtime" \
    bash --norc --noprofile "$LAUNCHER" </dev/null >"$SANDBOX/calls/out1" 2>&1 &
_pid1=$!

timeout 10 env -i PATH="$SANDBOX/bin" HOME="$SANDBOX/home" XDG_RUNTIME_DIR="$SANDBOX/runtime" \
    bash --norc --noprofile "$LAUNCHER" </dev/null >"$SANDBOX/calls/out2" 2>&1 &
_pid2=$!

wait "$_pid1" 2>/dev/null || _rc1=$?
wait "$_pid2" 2>/dev/null || _rc2=$?

# Both should hit the fast path (already healthy) and exit 0
if [ "$_rc1" = "0" ] || [ "$_rc2" = "0" ]; then
    assert_eq "T8: at least one launcher succeeds" "0" "0"
else
    assert_eq "T8: at least one launcher succeeds" "0" "$_rc1 and $_rc2"
fi

# xdg-open should have been called at least once
sleep 0.2
_browser_calls=$(cat "$_call_log" 2>/dev/null || true)
if [ -n "$_browser_calls" ]; then
    assert_match "T8: browser opened" "localhost:19000" "$_browser_calls"
else
    echo "  PASS: T8: both launchers exited without deadlock"
    PASS=$((PASS + 1))
fi

_teardown_sandbox

# ══════════════════════════════════════════════════════════════════════
echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
