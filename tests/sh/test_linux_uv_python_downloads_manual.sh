#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Exercise Fedora's manual-download recovery and unchanged failure behavior.
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

assert_not_contains() {
    _label="$1"; _haystack="$2"; _needle="$3"
    if echo "$_haystack" | grep -qF "$_needle"; then
        echo "  FAIL: $_label (found '$_needle' but should not)"
        FAIL=$((FAIL + 1))
    else
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    fi
}

_FN=$(mktemp)
{
    sed -n '/^PYTHON_SKIP=/p' "$INSTALL_SH"
    sed -n '/^_python_skip_applies()/,/^}/p' "$INSTALL_SH"
    sed -n '/^_python_request()/,/^}/p' "$INSTALL_SH"
    sed -n '/^_uv_venv_requested()/,/^}/p' "$INSTALL_SH"
} > "$_FN"
[ -s "$_FN" ] || { echo "  FAIL: _uv_venv_requested not found in install.sh"; exit 1; }
grep -q '^_uv_venv_requested()' "$_FN" || { echo "  FAIL: _uv_venv_requested not extracted"; exit 1; }

_FEDORA_HINT="hint: A managed Python download is available for Python >=3.13, !=3.13.8, <3.14, but Python downloads are set to 'manual', use \`uv python install >=3.13, !=3.13.8, <3.14\` to install the required version"

# $1 = shell, $2 = first-venv mode (clean|fedora|other), $3 = python-install rc
# Pass the hint as data so sh -c does not evaluate its backticked example.
_run() {
    FEDORA_HINT="$_FEDORA_HINT" "$1" -c '
        . "'"$_FN"'"
        VENV_DIR=/tmp/venv; PYTHON_VERSION=3.13
        _uvvr_n=0
        _run_uv_venv() {
            _uvvr_n=$((_uvvr_n + 1))
            echo "venv-$_uvvr_n python=$4"
            case "'"$2"'" in
                clean) return 0 ;;
                fedora)
                    if [ "$_uvvr_n" -eq 1 ]; then
                        printf "%s\n" "$FEDORA_HINT" >&2
                        return 2
                    fi
                    return 0
                    ;;
                other)
                    echo "error: Failed to create virtual environment" >&2
                    return 2
                    ;;
            esac
        }
        run_install_cmd() {
            echo "python-install:$5"
            return '"$3"'
        }
        _uv_venv_requested "create venv" && echo "rc=0" || echo "rc=$?"
    ' 2>&1 | tr '\n' ' '
}

for _sh in sh bash; do
    echo "=== _uv_venv_requested under $_sh ==="

    _out=$(_run "$_sh" clean 0)
    assert_eq "clean host: first venv succeeds, no python install" \
        "venv-1 python=>=3.13,<3.14,!=3.13.8 rc=0 " "$_out"
    assert_not_contains "clean host: no uv python install" "$_out" "python-install:"

    _out=$(_run "$_sh" fedora 0)
    assert_contains "fedora: hint reaches the helper verbatim" "$_out" \
        'use `uv python install >=3.13, !=3.13.8, <3.14`'
    assert_contains "fedora: first venv fails then retries" "$_out" "venv-1 python=>=3.13,<3.14,!=3.13.8"
    assert_contains "fedora: uv python install uses the same request" "$_out" \
        "python-install:>=3.13,<3.14,!=3.13.8"
    assert_contains "fedora: retry venv succeeds" "$_out" "venv-2 python=>=3.13,<3.14,!=3.13.8"
    assert_contains "fedora: overall rc=0" "$_out" "rc=0"

    _out=$(_run "$_sh" other 0)
    assert_contains "unrelated failure: first venv attempted" "$_out" "venv-1 "
    assert_not_contains "unrelated failure: no uv python install" "$_out" "python-install:"
    assert_not_contains "unrelated failure: no retry venv" "$_out" "venv-2 "
    assert_contains "unrelated failure: non-zero propagates" "$_out" "rc=2"

    _out=$(_run "$_sh" fedora 1)
    assert_contains "python install failure: attempted install" "$_out" "python-install:"
    assert_not_contains "python install failure: no retry venv" "$_out" "venv-2 "
    assert_contains "python install failure: non-zero propagates" "$_out" "rc=1"
done

echo "=== install.sh call sites ==="

_create=$(grep -c '_uv_venv_requested "create venv"' "$INSTALL_SH" || true)
assert_eq "create venv goes through the helper" "1" "$_create"

_recreate=$(grep -c '_uv_venv_requested "recreate venv"' "$INSTALL_SH" || true)
assert_eq "recreate venv goes through the helper" "1" "$_recreate"

_raw_create=$(grep -c '_run_uv_venv "create venv"' "$INSTALL_SH" || true)
assert_eq "no leftover raw _run_uv_venv create venv" "0" "$_raw_create"

_raw_recreate=$(grep -c '_run_uv_venv "recreate venv"' "$INSTALL_SH" || true)
assert_eq "no leftover raw _run_uv_venv recreate venv" "0" "$_raw_recreate"

if grep -E 'UV_PYTHON_DOWNLOADS=automatic|UV_NO_CONFIG=1' "$INSTALL_SH" | grep -qE '_uv_venv_requested|python install'; then
    echo "  FAIL: helper must not globally override distro uv download policy"
    FAIL=$((FAIL + 1))
else
    echo "  PASS: helper does not set UV_PYTHON_DOWNLOADS/UV_NO_CONFIG"
    PASS=$((PASS + 1))
fi

_helper=$(sed -n '/^_uv_venv_requested()/,/^}/p' "$INSTALL_SH")
assert_contains "helper tees stdout/stderr live" "$_helper" "tee"
assert_contains "helper uses mkfifo rather than redirect-and-cat" "$_helper" "mkfifo"
if echo "$_helper" | grep -q 'cat "$_uvvr_out"'; then
    echo "  FAIL: helper still replays captured stdout with cat"
    FAIL=$((FAIL + 1))
else
    echo "  PASS: helper does not replay captured stdout with cat"
    PASS=$((PASS + 1))
fi

echo "=== Unsloth installer stream ==="

# A successful retry must clear the first attempt's Unsloth failure marker.
_STREAM=$(mktemp)
{
    printf 'C_ERR=""; TAURI_MODE=true; UNSLOTH_VERBOSE=false; UNSLOTH_DL_MARKER_MIN_BYTES=52428800\n'
    printf 'PYTHON_SKIP="3.13.8"; SKIP_TORCH=false\n'
    printf 'step() { :; }\ntauri_log() { :; }\n'
    for _f in _is_verbose tauri_stream_log tauri_clear_install_error _redact_install_output \
              _uv_download_markers run_install_cmd _macos_has_selected_install_name_tool \
              _run_uv_venv _python_skip_applies _python_request _uv_venv_requested; do
        sed -n "/^$_f()/,/^}/p" "$INSTALL_SH"
    done
} > "$_STREAM"

_UVDIR=$(mktemp -d)
_STUB_STATE="$_UVDIR/state"
mkdir -p "$_STUB_STATE"
cat > "$_UVDIR/uv" << 'UV_EOF'
#!/bin/sh
echo "$*" >> "${UV_STUB_LOG:-/dev/null}"
case "$1" in
    python)
        [ "$2" = install ] || exit 1
        [ "${UV_FAIL_INSTALL:-0}" = 1 ] && exit 1
        exit 0
        ;;
    venv)
        if [ -f "${UV_STUB_STATE}/fail_manual" ]; then
            rm -f "${UV_STUB_STATE}/fail_manual"
            echo "error: No interpreter found for Python >=3.13, !=3.13.8, <3.14 in search path or managed installations" >&2
            echo "" >&2
            echo "hint: A managed Python download is available for Python >=3.13, !=3.13.8, <3.14, but Python downloads are set to 'manual', use \`uv python install >=3.13, !=3.13.8, <3.14\` to install the required version" >&2
            if [ "${UV_SLEEP_VENV:-0}" != 0 ]; then
                : > "${UV_STUB_STATE}/sleeping"
                sleep "$UV_SLEEP_VENV"
                rm -f "${UV_STUB_STATE}/sleeping"
            fi
            exit 2
        fi
        if [ -f "${UV_STUB_STATE}/fail_other" ]; then
            echo "error: Failed to create virtual environment" >&2
            exit 2
        fi
        mkdir -p "$2/bin" && printf '#!/bin/sh\n' > "$2/bin/python" && chmod +x "$2/bin/python"
        exit 0
        ;;
esac
exit 1
UV_EOF
chmod +x "$_UVDIR/uv"

_emit() {  # mode: clean|fedora|other|install-fail
    _sd=$(mktemp -d)
    _log=$(mktemp)
    rm -f "$_STUB_STATE/fail_manual" "$_STUB_STATE/fail_other"
    case "$1" in
        fedora|install-fail) : > "$_STUB_STATE/fail_manual" ;;
        other) : > "$_STUB_STATE/fail_other" ;;
    esac
    _fail_install=0
    [ "$1" = install-fail ] && _fail_install=1
    PATH="$_UVDIR:$PATH" OS=linux VENV_DIR="$_sd/venv" PYTHON_VERSION=3.13 \
        UV_STUB_LOG="$_log" UV_STUB_STATE="$_STUB_STATE" UV_FAIL_INSTALL="$_fail_install" \
        sh -c ". '$_STREAM'; _uv_venv_requested 'create venv'; echo RC=\$?" 2>&1
    echo "STUB_LOG=$(tr '\n' '|' < "$_log")"
    rm -rf "$_sd"
    rm -f "$_log"
}

_out=$(_emit clean)
assert_contains "clean run still returns 0" "$_out" "RC=0"
assert_not_contains "clean run: no uv python install" "$_out" "python install"
if echo "$_out" | grep -q ERROR_OUTPUT; then
    echo "  FAIL: clean run must not report a failure"
    FAIL=$((FAIL + 1))
else
    echo "  PASS: clean run reports no failure"
    PASS=$((PASS + 1))
fi

_out=$(_emit fedora)
assert_contains "fedora fallback still returns 0" "$_out" "RC=0"
assert_contains "fedora fallback ran uv python install" "$_out" \
    "python install >=3.13,<3.14,!=3.13.8"
assert_contains "recovery clears the Unsloth failure" "$_out" "ERROR_CLEAR"
assert_eq "ERROR_CLEAR is the last error-state line" "ERROR_CLEAR" \
    "$(echo "$_out" | grep -oE 'ERROR_OUTPUT|ERROR_CLEAR' | tail -1)"

# Verify output is visible before the first, sleeping venv attempt exits.
_sd=$(mktemp -d)
_live=$(mktemp)
_log=$(mktemp)
rm -f "$_STUB_STATE/fail_manual" "$_STUB_STATE/fail_other" "$_STUB_STATE/sleeping"
: > "$_STUB_STATE/fail_manual"
(
    PATH="$_UVDIR:$PATH" OS=linux VENV_DIR="$_sd/venv" PYTHON_VERSION=3.13 \
        UV_STUB_LOG="$_log" UV_STUB_STATE="$_STUB_STATE" UV_FAIL_INSTALL=0 \
        UV_SLEEP_VENV=2 \
        sh -c ". '$_STREAM'; _uv_venv_requested 'create venv'; echo RC=\$?"
) >"$_live" 2>&1 &
_pid=$!
_gave_up=
_t0=$(date +%s)
while [ ! -f "$_STUB_STATE/sleeping" ]; do
    if ! kill -0 "$_pid" 2>/dev/null; then
        _gave_up=exited
        break
    fi
    if [ $(( $(date +%s) - _t0 )) -ge 8 ]; then
        _gave_up=timeout
        break
    fi
done
if [ -n "$_gave_up" ]; then
    echo "  FAIL: live-stream probe never saw the venv sleep ($_gave_up)"
    FAIL=$((FAIL + 1))
    kill "$_pid" 2>/dev/null || true
elif grep -q OUTPUT_CLEAR "$_live" 2>/dev/null && [ -f "$_STUB_STATE/sleeping" ]; then
    echo "  PASS: TAURI markers stream while venv is still running"
    PASS=$((PASS + 1))
else
    echo "  FAIL: TAURI markers were held back until venv creation ended"
    FAIL=$((FAIL + 1))
fi
wait "$_pid" || true
_out=$(cat "$_live")
assert_contains "live fedora fallback still returns 0" "$_out" "RC=0"
assert_contains "live recovery still clears the Unsloth failure" "$_out" "ERROR_CLEAR"
assert_eq "live ERROR_CLEAR is the last error-state line" "ERROR_CLEAR" \
    "$(echo "$_out" | grep -oE 'ERROR_OUTPUT|ERROR_CLEAR' | tail -1)"
rm -rf "$_sd"
rm -f "$_live" "$_log"

_out=$(_emit other)
assert_not_contains "unrelated stream failure: no python install" "$_out" "python install"
assert_contains "unrelated stream failure: non-zero" "$_out" "RC=2"
if echo "$_out" | grep -q ERROR_CLEAR; then
    echo "  FAIL: unrecovered failure must not clear the Unsloth error"
    FAIL=$((FAIL + 1))
else
    echo "  PASS: unrecovered failure does not emit ERROR_CLEAR"
    PASS=$((PASS + 1))
fi

_out=$(_emit install-fail)
assert_contains "python install stream failure: attempted install" "$_out" "python install"
assert_contains "python install stream failure: non-zero" "$_out" "RC=1"

echo "=== capture fallback ==="

# Capture setup failure must preserve the original venv behavior.
rm -f "$_STUB_STATE/fail_manual" "$_STUB_STATE/fail_other"

_sd=$(mktemp -d)
_NOFIFO=$(mktemp -d)
printf '#!/bin/sh\nexit 1\n' > "$_NOFIFO/mkfifo"
chmod +x "$_NOFIFO/mkfifo"
_out=$(PATH="$_NOFIFO:$_UVDIR:$PATH" OS=linux VENV_DIR="$_sd/venv" PYTHON_VERSION=3.13 \
    UV_STUB_STATE="$_STUB_STATE" \
    sh -c ". '$_STREAM'; _uv_venv_requested 'create venv'; echo RC=\$?" 2>&1)
assert_contains "mkfifo failure falls back to a plain venv" "$_out" "RC=0"
if [ -x "$_sd/venv/bin/python" ]; then
    echo "  PASS: mkfifo failure still leaves an interpreter"
    PASS=$((PASS + 1))
else
    echo "  FAIL: mkfifo failure aborted the venv"
    FAIL=$((FAIL + 1))
fi
rm -rf "$_sd" "$_NOFIFO"

_NOTEE=$(mktemp -d)
for _c in sh sed awk grep mktemp mkfifo rm cat mkdir chmod; do
    if _p=$(command -v "$_c" 2>/dev/null); then
        ln -s "$_p" "$_NOTEE/$_c" 2>/dev/null || true
    fi
done
ln -s "$_UVDIR/uv" "$_NOTEE/uv" 2>/dev/null || true
if (PATH="$_NOTEE"; command -v tee >/dev/null 2>&1); then
    echo "  FAIL: no-tee case could not build a tee-free PATH"
    FAIL=$((FAIL + 1))
else
    _sd=$(mktemp -d)
    _out=$(PATH="$_NOTEE" OS=linux VENV_DIR="$_sd/venv" PYTHON_VERSION=3.13 \
        UV_STUB_STATE="$_STUB_STATE" \
        "$(command -v sh)" -c ". '$_STREAM'; _uv_venv_requested 'create venv'; echo RC=\$?" 2>&1)
    assert_contains "missing tee falls back to a plain venv" "$_out" "RC=0"
    if [ -x "$_sd/venv/bin/python" ]; then
        echo "  PASS: missing tee still leaves an interpreter"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: missing tee aborted the venv"
        FAIL=$((FAIL + 1))
    fi
    rm -rf "$_sd"
fi
rm -rf "$_NOTEE"

echo "=== interrupt cleanup ==="

# Signal cleanup must remove the global FIFO capture directory.
case "$(sed -n '/^_cleanup_install_temporaries()/,/^}/p' "$INSTALL_SH")" in
    *_UV_VENV_CAPTURE_DIR*)
        echo "  PASS: EXIT/signal cleanup owns the venv capture directory"
        PASS=$((PASS + 1))
        ;;
    *)
        echo "  FAIL: _cleanup_install_temporaries does not remove the venv capture directory"
        FAIL=$((FAIL + 1))
        ;;
esac

if grep -q '^_UV_VENV_CAPTURE_DIR=""' "$INSTALL_SH"; then
    echo "  PASS: capture path is cleared before the traps are installed"
    PASS=$((PASS + 1))
else
    echo "  FAIL: capture cleanup state is not initialized before the traps"
    FAIL=$((FAIL + 1))
fi

_TRAPS=$(mktemp)
for _f in _cleanup_install_temporaries _on_install_signal; do
    sed -n "/^$_f()/,/^}/p" "$INSTALL_SH" >> "$_TRAPS"
done

for _sh in sh bash; do
    _case=$(mktemp -d)
    mkdir -p "$_case/tmp"
    : > "$_case/ready"
    TMPDIR="$_case/tmp" CASE_DIR="$_case" "$_sh" -c '
        . "'"$_FN"'"
        . "'"$_TRAPS"'"
        VENV_DIR="$CASE_DIR/venv"
        PYTHON_VERSION=3.13
        _restore_studio_venv_replacement() { :; }
        _run_uv_venv() {
            printf "in-venv %s\n" "$_UV_VENV_CAPTURE_DIR" > "$CASE_DIR/ready"
            while :; do :; done
        }
        trap "_on_install_signal 143" TERM
        _uv_venv_requested "create venv"
    ' >/dev/null 2>&1 &
    _pid=$!
    for _ in $(seq 1 300); do [ -s "$_case/ready" ] && break; sleep 0.01; done
    if ! grep -q '^in-venv ' "$_case/ready" 2>/dev/null; then
        echo "  FAIL: $_sh interrupt case never reached uv venv"
        FAIL=$((FAIL + 1))
        kill -KILL "$_pid" 2>/dev/null || true
        wait "$_pid" 2>/dev/null || true
        rm -rf "$_case"
        continue
    fi
    kill -TERM "$_pid"
    set +e
    wait "$_pid"
    _signal_rc=$?
    set -e
    assert_eq "$_sh TERM preserves the signal status" "143" "$_signal_rc"
    # TMPDIR is this case's own, and the capture is the only thing put in it.
    _left=$(ls -A "$_case/tmp" 2>/dev/null || true)
    if [ -z "$_left" ]; then
        echo "  PASS: $_sh TERM removes the venv capture directory"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_sh TERM left $_left behind in TMPDIR"
        FAIL=$((FAIL + 1))
    fi
    rm -rf "$_case"
done
rm -f "$_TRAPS"

rm -rf "$_UVDIR"
rm -f "$_FN" "$_STREAM"
echo ""
echo "Passed: $PASS, Failed: $FAIL"
[ "$FAIL" -eq 0 ]
