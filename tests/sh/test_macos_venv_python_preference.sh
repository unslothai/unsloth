#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# _uv_venv_arm64: managed-only, so uv never executes the PATH pythons (the Xcode CLT
# dialog on a Mac without the tools), with an unflagged retry so a host that cannot
# resolve a managed build keeps its system Python.
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

assert_contains() {
    _label="$1"; _haystack="$2"; _needle="$3"
    if echo "$_haystack" | grep -qF "$_needle"; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected to find '$_needle')"
        FAIL=$((FAIL + 1))
    fi
}

_FN=$(mktemp)
sed -n '/^_uv_venv_arm64()/,/^}/p' "$INSTALL_SH" > "$_FN"
[ -s "$_FN" ] || { echo "  FAIL: _uv_venv_arm64 not found in install.sh"; exit 1; }

# $1 = shell, $2 = exit code for the only-managed attempt. The stub echoes which
# form it got, so ordering and fallback both show up in one trace.
_run() {
    "$1" -c '
        . "'"$_FN"'"
        VENV_DIR=/tmp/venv; PYTHON_VERSION=3.12
        _run_uv_venv() {
            shift
            case " $* " in
                *" only-managed "*) echo "managed"; return '"$2"' ;;
            esac
            echo "unflagged"; return 0
        }
        _uv_venv_arm64 "create venv" && echo "rc=0" || echo "rc=$?"
    ' 2>&1 | tr '\n' ' '
}

for _sh in sh bash; do
    echo "=== _uv_venv_arm64 under $_sh ==="

    _out=$(_run "$_sh" 0)
    assert_eq "managed request succeeds, no fallback" "managed rc=0 " "$_out"

    _out=$(_run "$_sh" 2)
    assert_eq "managed request fails, falls back unflagged" "managed unflagged rc=0 " "$_out"

    # Both failing must stay non-zero: the caller is under set -e and the
    # rollback trap depends on it.
    _out=$("$_sh" -c '
        . "'"$_FN"'"
        VENV_DIR=/tmp/venv; PYTHON_VERSION=3.12
        _run_uv_venv() { return 2; }
        _uv_venv_arm64 "create venv" && echo "rc=0" || echo "rc=$?"
    ' 2>&1)
    assert_eq "both attempts fail, non-zero propagates" "rc=2" "$_out"
done

echo "=== install.sh call sites ==="

# The last call site re-assigns PYTHON_VERSION to 3.12 first, so the helper has to
# read it at call time.
assert_contains "helper expands PYTHON_VERSION at call time" \
    "$(cat "$_FN")" 'cpython-${PYTHON_VERSION}-macos-aarch64-none'

_direct=$(grep -c 'uv venv .*cpython-\${PYTHON_VERSION}-macos-aarch64-none' "$INSTALL_SH" || true)
assert_eq "arm64 sites go through the helper only" "0" "$_direct"

_calls=$(grep -c '^ *_uv_venv_arm64 ' "$INSTALL_SH" || true)
assert_eq "all three arm64 venv sites routed" "3" "$_calls"

# _python_request carries a user --python or an interpreter path, which only-managed
# would ignore.
_out=$(grep -A 1 '_python_request "\$PYTHON_VERSION"' "$INSTALL_SH" || true)
if echo "$_out" | grep -q 'only-managed'; then
    echo "  FAIL: only-managed leaked onto a _python_request call site"
    FAIL=$((FAIL + 1))
else
    echo "  PASS: _python_request call sites left unflagged"
    PASS=$((PASS + 1))
fi

echo "=== Unsloth installer stream ==="

# install.rs turns [TAURI:ERROR_OUTPUT] into "Installation failed" until a later
# [TAURI:ERROR_CLEAR]. A recovered fallback must emit one or Unsloth reports a
# failure it already recovered from.
_STREAM=$(mktemp)
{
    printf 'C_ERR=""; TAURI_MODE=true; UNSLOTH_VERBOSE=false\n'
    printf 'step() { :; }\ntauri_log() { :; }\n'
    for _f in _is_verbose tauri_stream_log tauri_clear_install_error _redact_install_output \
              run_install_cmd _macos_has_selected_install_name_tool _run_uv_venv _uv_venv_arm64; do
        sed -n "/^$_f()/,/^}/p" "$INSTALL_SH"
    done
} > "$_STREAM"

_UVDIR=$(mktemp -d)
cat > "$_UVDIR/uv" << 'UV_EOF'
#!/bin/sh
case " $* " in *" only-managed "*) [ "$UV_FAIL_MANAGED" = 1 ] && exit 2 ;; esac
mkdir -p "$2/bin" && printf '#!/bin/sh\n' > "$2/bin/python" && chmod +x "$2/bin/python"
UV_EOF
chmod +x "$_UVDIR/uv"

_emit() {  # UV_FAIL_MANAGED
    _sd=$(mktemp -d)
    PATH="$_UVDIR:$PATH" OS=linux VENV_DIR="$_sd/venv" PYTHON_VERSION=3.12 UV_FAIL_MANAGED="$1" \
        sh -c ". '$_STREAM'; _uv_venv_arm64 'create venv'; echo RC=\$?" 2>&1
    rm -rf "$_sd"
}

_out=$(_emit 0)
assert_contains "managed attempt succeeds, returns 0" "$_out" "RC=0"
if echo "$_out" | grep -q ERROR_OUTPUT; then
    echo "  FAIL: clean run must not report a failure"
    FAIL=$((FAIL + 1))
else
    echo "  PASS: clean run reports no failure"
    PASS=$((PASS + 1))
fi

_out=$(_emit 1)
assert_contains "fallback run still returns 0" "$_out" "RC=0"
assert_contains "recovery clears the Unsloth failure" "$_out" "ERROR_CLEAR"
assert_eq "ERROR_CLEAR is the last error-state line" "ERROR_CLEAR" \
    "$(echo "$_out" | grep -o 'ERROR_OUTPUT\|ERROR_CLEAR' | tail -1)"

rm -rf "$_UVDIR"
rm -f "$_FN" "$_STREAM"
echo ""
echo "Passed: $PASS, Failed: $FAIL"
[ "$FAIL" -eq 0 ]
