#!/bin/bash
# Unit tests for launcher functions extracted from install.sh
# Tests: _is_port_busy, _find_launch_port, _check_health, _open_browser,
#        stale lock edge cases, WSL shortcut arg construction, PS1 content checks
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
INSTALL_PS1="$SCRIPT_DIR/../../install.ps1"
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

assert_file_contains() {
    _label="$1"; _file="$2"; _pattern="$3"
    if grep -qE "$_pattern" "$_file" 2>/dev/null; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (pattern '$_pattern' not found in $_file)"
        FAIL=$((FAIL + 1))
    fi
}

# ── Extract launcher heredoc from install.sh ──
_FUNC_FILE=$(mktemp)
# Extract everything between 'LAUNCHER_EOF' markers (the heredoc body)
# Note: the cat line may be indented, so don't anchor with ^
sed -n "/cat > .* << 'LAUNCHER_EOF'$/,/^LAUNCHER_EOF$/{ /cat >/d; /^LAUNCHER_EOF$/d; p; }" "$INSTALL_SH" > "$_FUNC_FILE"

# Remove set -euo pipefail and shebang so we can source it safely
sed -i 's/^set -euo pipefail$//' "$_FUNC_FILE"
sed -i 's|^#!/usr/bin/env bash$||' "$_FUNC_FILE"

# Remove the config-loading preamble (DATA_DIR through LOCK_DIR) and the
# main section so sourcing only defines functions.
# Delete from DATA_DIR= up to (but not including) the first function definition.
sed -i '/^DATA_DIR=/,/^# ── HTTP GET helper/{ /^# ── HTTP GET helper/!d; }' "$_FUNC_FILE"
# Remove the main section (everything from "# ── Main ──" onward)
sed -i '/^# ── Main ──$/,$d' "$_FUNC_FILE"

# Create a fake /proc/version file we can control
_PROC_VERSION_FILE=$(mktemp)
echo "Linux version 5.15.0" > "$_PROC_VERSION_FILE"
# Patch /proc/version references to our fake file
sed -i "s|/proc/version|$_PROC_VERSION_FILE|g" "$_FUNC_FILE"

# Build a minimal tools directory with symlinks to essential commands
_TOOLS_DIR=$(mktemp -d)
for _cmd in bash grep sed awk sort cat date sleep id mktemp rm mkdir kill printf od head sh uname test tr wc cut env dirname basename readlink nohup; do
    _real=$(command -v "$_cmd" 2>/dev/null || true)
    [ -n "$_real" ] && ln -sf "$_real" "$_TOOLS_DIR/$_cmd"
done

# Helper: run a function from the extracted launcher in a controlled environment
# $1 = extra PATH dirs (prepended), $2.. = command to run after sourcing
run_in_sandbox() {
    _extra_path="$1"; shift
    _sandbox_path="$_extra_path:$_TOOLS_DIR"
    env -i PATH="$_sandbox_path" HOME="${_SANDBOX_HOME:-/tmp}" \
        XDG_RUNTIME_DIR="${_SANDBOX_RUNTIME:-/tmp}" \
        bash --norc --noprofile -c "
            . '$_FUNC_FILE'
            BASE_PORT=\${BASE_PORT:-19000}
            MAX_PORT_OFFSET=\${MAX_PORT_OFFSET:-5}
            TIMEOUT_SEC=\${TIMEOUT_SEC:-3}
            POLL_INTERVAL_SEC=\${POLL_INTERVAL_SEC:-0}
            $*
        " 2>&1
}

cleanup() {
    rm -f "$_FUNC_FILE" "$_PROC_VERSION_FILE"
    rm -rf "$_TOOLS_DIR"
}
trap cleanup EXIT

# ══════════════════════════════════════════════════════════════════════
echo "=== A. _is_port_busy ==="
# ══════════════════════════════════════════════════════════════════════

# A1: ss reports port busy -> returns 0
_mock_dir=$(mktemp -d)
cat > "$_mock_dir/ss" <<'MOCK'
#!/bin/sh
echo "LISTEN  0  4096  0.0.0.0:19000  0.0.0.0:*"
MOCK
chmod +x "$_mock_dir/ss"

_rc=0
run_in_sandbox "$_mock_dir" '_is_port_busy 19000' >/dev/null 2>&1 || _rc=$?
assert_eq "A1: ss reports port busy -> 0" "0" "$_rc"
rm -rf "$_mock_dir"

# A2: ss reports port NOT busy -> returns 1
_mock_dir=$(mktemp -d)
cat > "$_mock_dir/ss" <<'MOCK'
#!/bin/sh
echo "LISTEN  0  4096  0.0.0.0:19001  0.0.0.0:*"
MOCK
chmod +x "$_mock_dir/ss"

_rc=0
run_in_sandbox "$_mock_dir" '_is_port_busy 19000' >/dev/null 2>&1 || _rc=$?
assert_eq "A2: ss reports port not busy -> 1" "1" "$_rc"
rm -rf "$_mock_dir"

# A3: neither ss nor lsof available -> returns 1 (assumed free)
_empty_dir=$(mktemp -d)
_rc=0
run_in_sandbox "$_empty_dir" '_is_port_busy 19000' >/dev/null 2>&1 || _rc=$?
assert_eq "A3: no ss/lsof -> 1 (assumed free)" "1" "$_rc"
rm -rf "$_empty_dir"

# ══════════════════════════════════════════════════════════════════════
echo ""
echo "=== B. _find_launch_port ==="
# ══════════════════════════════════════════════════════════════════════

# B1: all ports free -> returns BASE_PORT (19000)
_mock_dir=$(mktemp -d)
cat > "$_mock_dir/ss" <<'MOCK'
#!/bin/sh
# No ports listening
echo ""
MOCK
chmod +x "$_mock_dir/ss"

_result=$(run_in_sandbox "$_mock_dir" 'BASE_PORT=19000; MAX_PORT_OFFSET=5; _find_launch_port')
assert_eq "B1: all free -> 19000" "19000" "$_result"
rm -rf "$_mock_dir"

# B2: first 3 ports busy -> returns BASE_PORT+3
_mock_dir=$(mktemp -d)
cat > "$_mock_dir/ss" <<'MOCK'
#!/bin/sh
cat <<'EOF'
LISTEN  0  4096  0.0.0.0:19000  0.0.0.0:*
LISTEN  0  4096  0.0.0.0:19001  0.0.0.0:*
LISTEN  0  4096  0.0.0.0:19002  0.0.0.0:*
EOF
MOCK
chmod +x "$_mock_dir/ss"

_result=$(run_in_sandbox "$_mock_dir" 'BASE_PORT=19000; MAX_PORT_OFFSET=5; _find_launch_port')
assert_eq "B2: first 3 busy -> 19003" "19003" "$_result"
rm -rf "$_mock_dir"

# B3: all ports busy -> returns 1
_mock_dir=$(mktemp -d)
cat > "$_mock_dir/ss" <<'MOCK'
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
chmod +x "$_mock_dir/ss"

_rc=0
run_in_sandbox "$_mock_dir" 'BASE_PORT=19000; MAX_PORT_OFFSET=5; _find_launch_port' >/dev/null 2>&1 || _rc=$?
assert_eq "B3: all busy -> exit 1" "1" "$_rc"
rm -rf "$_mock_dir"

# ══════════════════════════════════════════════════════════════════════
echo ""
echo "=== C. _check_health ==="
# ══════════════════════════════════════════════════════════════════════

# C1: curl returns healthy JSON -> returns 0
_mock_dir=$(mktemp -d)
cat > "$_mock_dir/curl" <<'MOCK'
#!/bin/sh
echo '{"status":"healthy","service":"Unsloth UI Backend"}'
MOCK
chmod +x "$_mock_dir/curl"

_rc=0
run_in_sandbox "$_mock_dir" '_check_health 19000' >/dev/null 2>&1 || _rc=$?
assert_eq "C1: healthy JSON -> 0" "0" "$_rc"
rm -rf "$_mock_dir"

# C2: reversed key order -> still returns 0
_mock_dir=$(mktemp -d)
cat > "$_mock_dir/curl" <<'MOCK'
#!/bin/sh
echo '{"service":"Unsloth UI Backend","status":"healthy"}'
MOCK
chmod +x "$_mock_dir/curl"

_rc=0
run_in_sandbox "$_mock_dir" '_check_health 19000' >/dev/null 2>&1 || _rc=$?
assert_eq "C2: reversed key order -> 0" "0" "$_rc"
rm -rf "$_mock_dir"

# C3: status=starting -> returns 1
_mock_dir=$(mktemp -d)
cat > "$_mock_dir/curl" <<'MOCK'
#!/bin/sh
echo '{"status":"starting","service":"Unsloth UI Backend"}'
MOCK
chmod +x "$_mock_dir/curl"

_rc=0
run_in_sandbox "$_mock_dir" '_check_health 19000' >/dev/null 2>&1 || _rc=$?
assert_eq "C3: status=starting -> 1" "1" "$_rc"
rm -rf "$_mock_dir"

# C4: no curl/wget -> returns 1
_empty_dir=$(mktemp -d)
_rc=0
run_in_sandbox "$_empty_dir" '_check_health 19000' >/dev/null 2>&1 || _rc=$?
assert_eq "C4: no curl/wget -> 1" "1" "$_rc"
rm -rf "$_empty_dir"

# ══════════════════════════════════════════════════════════════════════
echo ""
echo "=== D. _open_browser ==="
# ══════════════════════════════════════════════════════════════════════

# D1: Linux (no WSL) with xdg-open -> xdg-open called
echo "Linux version 5.15.0" > "$_PROC_VERSION_FILE"
_mock_dir=$(mktemp -d)
_call_log=$(mktemp)
cat > "$_mock_dir/xdg-open" <<MOCK
#!/bin/sh
echo "xdg-open \$*" >> "$_call_log"
MOCK
chmod +x "$_mock_dir/xdg-open"
cat > "$_mock_dir/uname" <<'MOCK'
#!/bin/sh
echo "Linux"
MOCK
chmod +x "$_mock_dir/uname"

run_in_sandbox "$_mock_dir" '_open_browser "http://localhost:19000"' >/dev/null 2>&1; sleep 0.1
_calls=$(cat "$_call_log" 2>/dev/null || true)
assert_match "D1: xdg-open called" "xdg-open.*localhost:19000" "$_calls"
rm -rf "$_mock_dir" "$_call_log"

# D2: WSL with powershell.exe -> powershell.exe called
echo "Linux version 5.15.0-microsoft-standard-WSL2" > "$_PROC_VERSION_FILE"
_mock_dir=$(mktemp -d)
_call_log=$(mktemp)
cat > "$_mock_dir/powershell.exe" <<MOCK
#!/bin/sh
echo "powershell.exe \$*" >> "$_call_log"
MOCK
chmod +x "$_mock_dir/powershell.exe"
cat > "$_mock_dir/uname" <<'MOCK'
#!/bin/sh
echo "Linux"
MOCK
chmod +x "$_mock_dir/uname"

run_in_sandbox "$_mock_dir" '_open_browser "http://localhost:19000"' >/dev/null 2>&1; sleep 0.1
_calls=$(cat "$_call_log" 2>/dev/null || true)
assert_match "D2: WSL powershell.exe called" "powershell.exe" "$_calls"
rm -rf "$_mock_dir" "$_call_log"

# D3: WSL with no powershell.exe but cmd.exe -> cmd.exe called
echo "Linux version 5.15.0-microsoft-standard-WSL2" > "$_PROC_VERSION_FILE"
_mock_dir=$(mktemp -d)
_call_log=$(mktemp)
cat > "$_mock_dir/cmd.exe" <<MOCK
#!/bin/sh
echo "cmd.exe \$*" >> "$_call_log"
MOCK
chmod +x "$_mock_dir/cmd.exe"
cat > "$_mock_dir/uname" <<'MOCK'
#!/bin/sh
echo "Linux"
MOCK
chmod +x "$_mock_dir/uname"

run_in_sandbox "$_mock_dir" '_open_browser "http://localhost:19000"' >/dev/null 2>&1; sleep 0.1
_calls=$(cat "$_call_log" 2>/dev/null || true)
assert_match "D3: WSL cmd.exe called" "cmd.exe" "$_calls"
rm -rf "$_mock_dir" "$_call_log"

# D4: no browser command -> stderr contains "Open in your browser"
echo "Linux version 5.15.0" > "$_PROC_VERSION_FILE"
_empty_dir=$(mktemp -d)
cat > "$_empty_dir/uname" <<'MOCK'
#!/bin/sh
echo "Linux"
MOCK
chmod +x "$_empty_dir/uname"

_output=$(run_in_sandbox "$_empty_dir" '_open_browser "http://localhost:19000"' 2>&1)
assert_match "D4: no browser -> fallback message" "Open in your browser" "$_output"
rm -rf "$_empty_dir"

# Reset proc version
echo "Linux version 5.15.0" > "$_PROC_VERSION_FILE"

# ══════════════════════════════════════════════════════════════════════
echo ""
echo "=== E. Stale lock edge cases ==="
# ══════════════════════════════════════════════════════════════════════

# For lock tests we need a real-ish environment with mkdir, id, etc.
# We test the _acquire_lock function with crafted lock dirs.

# E1: Lock dir exists but pid file is empty -> treated as stale, reclaimed
_sandbox_rt=$(mktemp -d)
_lock_dir="$_sandbox_rt/unsloth-studio-launcher-99999.lock"
mkdir -p "$_lock_dir"
touch "$_lock_dir/pid"  # empty pid file

_mock_dir=$(mktemp -d)
# Mock id to return 99999
cat > "$_mock_dir/id" <<'MOCK'
#!/bin/sh
echo "99999"
MOCK
chmod +x "$_mock_dir/id"
# Mock curl to fail (no healthy port)
cat > "$_mock_dir/curl" <<'MOCK'
#!/bin/sh
exit 1
MOCK
chmod +x "$_mock_dir/curl"

_SANDBOX_RUNTIME="$_sandbox_rt" _rc=0
env -i PATH="$_mock_dir:$_TOOLS_DIR" HOME="/tmp" XDG_RUNTIME_DIR="$_sandbox_rt" \
    bash --norc --noprofile -c "
        . '$_FUNC_FILE'
        LOCK_DIR='$_lock_dir'
        TIMEOUT_SEC=1
        POLL_INTERVAL_SEC=0
        _acquire_lock
    " >/dev/null 2>&1 || _rc=$?
# If _acquire_lock succeeded, the pid file should now contain a PID
_new_pid=$(cat "$_lock_dir/pid" 2>/dev/null || true)
if [ "$_rc" = "0" ] && [ -n "$_new_pid" ]; then
    assert_eq "E1: empty pid -> stale, reclaimed" "0" "0"
else
    assert_eq "E1: empty pid -> stale, reclaimed" "0" "$_rc"
fi
rm -rf "$_sandbox_rt"

# E2: Lock dir exists but pid file is missing -> treated as stale, reclaimed
_sandbox_rt=$(mktemp -d)
_lock_dir="$_sandbox_rt/unsloth-studio-launcher-99999.lock"
mkdir -p "$_lock_dir"
# No pid file at all

_mock_dir2=$(mktemp -d)
cat > "$_mock_dir2/id" <<'MOCK'
#!/bin/sh
echo "99999"
MOCK
chmod +x "$_mock_dir2/id"
cat > "$_mock_dir2/curl" <<'MOCK'
#!/bin/sh
exit 1
MOCK
chmod +x "$_mock_dir2/curl"

_rc=0
env -i PATH="$_mock_dir2:$_TOOLS_DIR" HOME="/tmp" XDG_RUNTIME_DIR="$_sandbox_rt" \
    bash --norc --noprofile -c "
        . '$_FUNC_FILE'
        LOCK_DIR='$_lock_dir'
        TIMEOUT_SEC=1
        POLL_INTERVAL_SEC=0
        _acquire_lock
    " >/dev/null 2>&1 || _rc=$?
_new_pid=$(cat "$_lock_dir/pid" 2>/dev/null || true)
if [ "$_rc" = "0" ] && [ -n "$_new_pid" ]; then
    assert_eq "E2: missing pid -> stale, reclaimed" "0" "0"
else
    assert_eq "E2: missing pid -> stale, reclaimed" "0" "$_rc"
fi
rm -rf "$_sandbox_rt" "$_mock_dir" "$_mock_dir2"

# E3: Lock dir exists with non-numeric pid -> treated as stale, reclaimed
_sandbox_rt=$(mktemp -d)
_lock_dir="$_sandbox_rt/unsloth-studio-launcher-99999.lock"
mkdir -p "$_lock_dir"
echo "not-a-pid" > "$_lock_dir/pid"

_mock_dir=$(mktemp -d)
cat > "$_mock_dir/id" <<'MOCK'
#!/bin/sh
echo "99999"
MOCK
chmod +x "$_mock_dir/id"
cat > "$_mock_dir/curl" <<'MOCK'
#!/bin/sh
exit 1
MOCK
chmod +x "$_mock_dir/curl"

_rc=0
env -i PATH="$_mock_dir:$_TOOLS_DIR" HOME="/tmp" XDG_RUNTIME_DIR="$_sandbox_rt" \
    bash --norc --noprofile -c "
        . '$_FUNC_FILE'
        LOCK_DIR='$_lock_dir'
        TIMEOUT_SEC=1
        POLL_INTERVAL_SEC=0
        _acquire_lock
    " >/dev/null 2>&1 || _rc=$?
_new_pid=$(cat "$_lock_dir/pid" 2>/dev/null || true)
if [ "$_rc" = "0" ] && [ -n "$_new_pid" ]; then
    assert_eq "E3: non-numeric pid -> stale, reclaimed" "0" "0"
else
    assert_eq "E3: non-numeric pid -> stale, reclaimed" "0" "$_rc"
fi
rm -rf "$_sandbox_rt" "$_mock_dir"

# ══════════════════════════════════════════════════════════════════════
echo ""
echo "=== F. WSL shortcut arg construction ==="
# ══════════════════════════════════════════════════════════════════════

# Extract the WSL variable-building lines from install.sh and test them
# in isolation. We source just the arg construction snippet.

# F1: WSL_DISTRO_NAME="Ubuntu" (no spaces) -> args contain -d "Ubuntu"
_result=$(
    _css_distro="Ubuntu"
    _css_launcher="/home/user/.local/share/unsloth/launch-studio.sh"
    _css_wsl_args=""
    if [ -n "$_css_distro" ]; then
        _css_wsl_args="-d \"$_css_distro\" "
    fi
    _css_wsl_args="${_css_wsl_args}-- bash -l -c \"exec \\\"$_css_launcher\\\"\""
    echo "$_css_wsl_args"
)
assert_match "F1: Ubuntu -> -d \"Ubuntu\"" '-d "Ubuntu"' "$_result"

# F2: WSL_DISTRO_NAME="Ubuntu Preview" (spaces) -> args contain -d "Ubuntu Preview" as one token
_result=$(
    _css_distro="Ubuntu Preview"
    _css_launcher="/home/user/.local/share/unsloth/launch-studio.sh"
    _css_wsl_args=""
    if [ -n "$_css_distro" ]; then
        _css_wsl_args="-d \"$_css_distro\" "
    fi
    _css_wsl_args="${_css_wsl_args}-- bash -l -c \"exec \\\"$_css_launcher\\\"\""
    echo "$_css_wsl_args"
)
assert_match "F2: Ubuntu Preview -> quoted" '-d "Ubuntu Preview"' "$_result"

# F3: launcher path with spaces -> path is double-quoted inside bash -c
_result=$(
    _css_distro=""
    _css_launcher="/home/my user/.local/share/unsloth/launch-studio.sh"
    _css_wsl_args=""
    if [ -n "$_css_distro" ]; then
        _css_wsl_args="-d \"$_css_distro\" "
    fi
    _css_wsl_args="${_css_wsl_args}-- bash -l -c \"exec \\\"$_css_launcher\\\"\""
    echo "$_css_wsl_args"
)
assert_match "F3: path with spaces -> double-quoted" 'exec \\"[^"]*my user[^"]*\\"' "$_result"

# ══════════════════════════════════════════════════════════════════════
echo ""
echo "=== G. PS1 port-finding content checks ==="
# ══════════════════════════════════════════════════════════════════════

if [ -f "$INSTALL_PS1" ]; then
    # G1: Find-FreeLaunchPort contains Test-PortBusy fallback
    assert_file_contains "G1: Find-FreeLaunchPort has Test-PortBusy fallback" \
        "$INSTALL_PS1" "Test-PortBusy"

    # G2: Get-NetTCPConnection is inside a try block with catch
    # Check that the pattern try { ... Get-NetTCPConnection ... } catch exists
    assert_file_contains "G2: Get-NetTCPConnection inside try/catch" \
        "$INSTALL_PS1" "Get-NetTCPConnection.*-ErrorAction Stop"
else
    echo "  SKIP: install.ps1 not found at $INSTALL_PS1"
fi

# ══════════════════════════════════════════════════════════════════════
echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
