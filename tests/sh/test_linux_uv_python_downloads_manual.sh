#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# _uv_venv_requested: when uv venv fails because a distro uv.toml set
# python-downloads = "manual" (Fedora), run `uv python install` for the same
# _python_request and retry. Any other venv failure must stay failed.
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
_run() {
    "$1" -c '
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
                        echo "'"$_FEDORA_HINT"'" >&2
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

if grep -E 'UV_PYTHON_DOWNLOADS=automatic|UV_NO_CONFIG=1' "$INSTALL_SH" | grep -q '_uv_venv_requested\|python install'; then
    echo "  FAIL: helper must not globally override distro uv download policy"
    FAIL=$((FAIL + 1))
else
    echo "  PASS: helper does not set UV_PYTHON_DOWNLOADS/UV_NO_CONFIG"
    PASS=$((PASS + 1))
fi

echo "=== Studio installer stream ==="

# install.rs turns [TAURI:ERROR_OUTPUT] into "Installation failed" until a later
# [TAURI:ERROR_CLEAR]. A recovered fallback must emit one or Studio reports a
# failure it already recovered from.
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
assert_contains "recovery clears the Studio failure" "$_out" "ERROR_CLEAR"
assert_eq "ERROR_CLEAR is the last error-state line" "ERROR_CLEAR" \
    "$(echo "$_out" | grep -o 'ERROR_OUTPUT\|ERROR_CLEAR' | tail -1)"

_out=$(_emit other)
assert_not_contains "unrelated stream failure: no python install" "$_out" "python install"
assert_contains "unrelated stream failure: non-zero" "$_out" "RC=2"
if echo "$_out" | grep -q ERROR_CLEAR; then
    echo "  FAIL: unrecovered failure must not clear the Studio error"
    FAIL=$((FAIL + 1))
else
    echo "  PASS: unrecovered failure does not emit ERROR_CLEAR"
    PASS=$((PASS + 1))
fi

_out=$(_emit install-fail)
assert_contains "python install stream failure: attempted install" "$_out" "python install"
assert_contains "python install stream failure: non-zero" "$_out" "RC=1"

rm -rf "$_UVDIR"
rm -f "$_FN" "$_STREAM"
echo ""
echo "Passed: $PASS, Failed: $FAIL"
[ "$FAIL" -eq 0 ]
