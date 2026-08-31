#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Regression coverage for uv's unconditional macOS install_name_tool patch. A consumer
# Mac without CLT must never execute Apple's developer-tool shim, while a selected CLT
# or full Xcode installation must retain uv's real libpython patch.
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

assert_empty() {
    _label="$1"; _path="$2"
    if [ ! -s "$_path" ]; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (unexpected contents: $(cat "$_path"))"
        FAIL=$((FAIL + 1))
    fi
}

_EXTRACTED_HELPERS=$(mktemp)
for _f in _macos_has_selected_install_name_tool _run_uv_venv; do
    sed -n "/^$_f()/,/^}/p" "$INSTALL_SH" >> "$_EXTRACTED_HELPERS"
done
if ! grep -q '^_macos_has_selected_install_name_tool()' "$_EXTRACTED_HELPERS" \
   || ! grep -q '^_run_uv_venv()' "$_EXTRACTED_HELPERS"; then
    echo "  FAIL: guarded uv venv helpers not found in install.sh"
    rm -f "$_EXTRACTED_HELPERS"
    exit 1
fi

_ROOT=$(mktemp -d)
trap 'rm -rf "$_ROOT"; rm -f "$_EXTRACTED_HELPERS" "${_TRAP_HELPERS:-}"' EXIT
_BIN="$_ROOT/bin"
mkdir -p "$_BIN"

cat > "$_BIN/xcode-select" <<'XCODE_EOF'
#!/bin/sh
printf '%s\n' "$*" >> "$CASE_DIR/xcode-select.log"
if [ -n "${DEVELOPER_DIR:-}" ]; then
    printf '%s\n' "$DEVELOPER_DIR"
    exit 0
fi
[ "${XCODE_SELECT_RC:-0}" -eq 0 ] || exit "$XCODE_SELECT_RC"
printf '%s\n' "$XCODE_DEV"
XCODE_EOF

cat > "$_BIN/install_name_tool" <<'TOOL_EOF'
#!/bin/sh
printf '%s\n' "$*" >> "$CASE_DIR/apple-tool.log"
exit 0
TOOL_EOF

cat > "$_BIN/uv" <<'UV_EOF'
#!/bin/sh
printf '%s\n' "$*" >> "$CASE_DIR/uv.log"

printf 'argc=%s\n' "$#" >> "$CASE_DIR/uv-argv.log"
for arg in "$@"; do printf '<%s>\n' "$arg" >> "$CASE_DIR/uv-argv.log"; done
if [ "${FAKE_UV_PATCH:-1}" = 1 ]; then
    dylib="$CASE_DIR/libpython3.12.dylib"
    install_name_tool -id "$dylib" "$dylib" || :
fi
exit "${FAKE_UV_RC:-0}"
UV_EOF
chmod +x "$_BIN/xcode-select" "$_BIN/install_name_tool" "$_BIN/uv"

# Each named layout owns its expected uv status and real-tool count; callers pass only
# the shell and scenario rather than a positional clump of correlated expectations.
run_case() {
    _shell="$1"; _layout="$2"
    _uv_rc=0; _expected_rc=0; _expected_tool=0; _developer_override=""; _dev=""
    case "$_layout" in
        absent) _select_rc=1 ;;
        stale) _select_rc=0 ;;
        partial)
            _select_rc=0
            _dev="$_ROOT/${_shell##*/}-$_layout/developer"
            mkdir -p "$_dev"
            ;;
        custom-selection)
            _select_rc=0; _expected_tool=1
            _dev="$_ROOT/${_shell##*/}-$_layout/custom-developer-alias"
            _real_dev="$_ROOT/${_shell##*/}-$_layout/real-developer"
            mkdir -p "$_real_dev/usr/bin"
            : > "$_real_dev/usr/bin/install_name_tool"
            chmod +x "$_real_dev/usr/bin/install_name_tool"
            ln -s "$_real_dev" "$_dev"
            _developer_override="$_dev"
            ;;
        clt)
            _select_rc=0; _expected_tool=1
            _dev="$_ROOT/${_shell##*/}-$_layout/Library/Developer/CommandLineTools"
            mkdir -p "$_dev/usr/bin"
            : > "$_dev/usr/bin/install_name_tool"
            chmod +x "$_dev/usr/bin/install_name_tool"
            ;;
        xcode)
            _select_rc=0; _uv_rc=19; _expected_rc=19; _expected_tool=1
            _dev="$_ROOT/${_shell##*/}-$_layout/Applications/Xcode.app/Contents/Developer"
            mkdir -p "$_dev/Toolchains/XcodeDefault.xctoolchain/usr/bin"
            : > "$_dev/Toolchains/XcodeDefault.xctoolchain/usr/bin/install_name_tool"
            chmod +x "$_dev/Toolchains/XcodeDefault.xctoolchain/usr/bin/install_name_tool"
            ;;
        partial-failure)
            _select_rc=0; _uv_rc=23; _expected_rc=23
            _dev="$_ROOT/${_shell##*/}-$_layout/developer"
            mkdir -p "$_dev"
            ;;
    esac

    _case="$_ROOT/${_shell##*/}-$_layout"
    mkdir -p "$_case/tmp"
    : > "$_case/apple-tool.log"
    : > "$_case/xcode-select.log"
    : > "$_case/uv.log"

    : > "$_case/uv-argv.log"
    : "${_dev:=$_case/developer}"

    set +e
    _out=$(CASE_DIR="$_case" TMPDIR="$_case/tmp" PATH="$_BIN:$PATH" \
        DEVELOPER_DIR="$_developer_override" XCODE_DEV="$_dev" XCODE_SELECT_RC="$_select_rc" \
        FAKE_UV_RC="$_uv_rc" "$_shell" -c '
            . "'"$_EXTRACTED_HELPERS"'"
            OS=macos
            _UV_INSTALL_NAME_TOOL_SHIM_DIR=""
            run_install_cmd() { shift; "$@"; }
            if _run_uv_venv "create venv" "$CASE_DIR/venv" --python cpython-3.12-macos-aarch64-none; then
                rc=0
            else
                rc=$?
            fi
            printf "rc=%s guard=%s\n" "$rc" "${_UV_INSTALL_NAME_TOOL_SHIM_DIR-unset}"
            exit "$rc"
        ' 2>&1)
    _actual_rc=$?
    set -e

    assert_eq "$_shell $_layout preserves uv status" "$_expected_rc" "$_actual_rc"
    assert_eq "$_shell $_layout clears guard state" "rc=$_expected_rc guard=" "$_out"
    assert_eq "$_shell $_layout preserves uv argv" \
        "venv $_case/venv --python cpython-3.12-macos-aarch64-none" "$(cat "$_case/uv.log")"
    assert_eq "$_shell $_layout real tool call count" "$_expected_tool" \
        "$(wc -l < "$_case/apple-tool.log" | tr -d ' ')"
    if [ "$_expected_tool" = 0 ]; then
        assert_empty "$_shell $_layout never reaches Apple tool" "$_case/apple-tool.log"
    else
        _dylib="$_case/libpython3.12.dylib"
        assert_eq "$_shell $_layout keeps uv self-ID patch" "-id $_dylib $_dylib" \
            "$(cat "$_case/apple-tool.log")"
    fi
    assert_eq "$_shell $_layout removes temporary shim" "0" \
        "$(find "$_case/tmp" -mindepth 1 -maxdepth 1 | wc -l | tr -d ' ')"
}

for _sh in sh bash; do
    echo "=== install_name_tool guard under $_sh ==="
    run_case "$_sh" absent
    run_case "$_sh" stale
    run_case "$_sh" partial
    run_case "$_sh" custom-selection
    run_case "$_sh" partial-failure
    run_case "$_sh" clt
    run_case "$_sh" xcode
done


echo "=== exact generic/user and non-macOS forwarding ==="
for _sh in sh bash; do
    _case="$_ROOT/${_sh}-forwarding"
    mkdir -p "$_case/tmp"
    : > "$_case/uv.log"; : > "$_case/uv-argv.log"; : > "$_case/xcode-select.log"
    CASE_DIR="$_case" TMPDIR="$_case/tmp" PATH="$_BIN:$PATH" FAKE_UV_PATCH=0 \
      "$_sh" -c '
        . "'"$_EXTRACTED_HELPERS"'"
        _UV_INSTALL_NAME_TOOL_SHIM_DIR=""
        run_install_cmd() { shift; "$@"; }
        OS=linux
        _run_uv_venv "generic user Python" "$CASE_DIR/venv with spaces" \
          --python "$CASE_DIR/Python 3.12/bin/python" --offline
      '
    _expected_argv=$(printf '%s\n' 'argc=5' \
      '<venv>' "<$_case/venv with spaces>" '<--python>' \
      "<$_case/Python 3.12/bin/python>" '<--offline>')
    assert_eq "$_sh non-macOS preserves exact generic/user-Python argv boundaries" \
      "$_expected_argv" "$(cat "$_case/uv-argv.log")"
    assert_empty "$_sh non-macOS skips the developer-tool probe" "$_case/xcode-select.log"
done

echo "=== call-site and trap cleanup coverage ==="
_outside=$(awk '
    /^_run_uv_venv\(\)/ { in_helper=1 }
    in_helper && /^}/ { in_helper=0; next }
    !in_helper && /run_install_cmd .*uv venv/ { print }
' "$INSTALL_SH")
assert_empty "all real uv venv commands are routed through the guard" <(printf '%s' "$_outside")

_cleanup=$(sed -n '/^_cleanup_install_temporaries()/,/^}/p' "$INSTALL_SH")
case "$_cleanup" in
    *'_UV_INSTALL_NAME_TOOL_SHIM_DIR'*)
        echo "  PASS: EXIT/signal cleanup owns the shim directory"
        PASS=$((PASS + 1))
        ;;
    *)
        echo "  FAIL: _cleanup_install_temporaries does not remove the shim directory"
        FAIL=$((FAIL + 1))
        ;;
esac

if grep -q '^_UV_INSTALL_NAME_TOOL_SHIM_DIR=""' "$INSTALL_SH"; then
    echo "  PASS: inherited shim cleanup path is cleared before traps are installed"
    PASS=$((PASS + 1))
else
    echo "  FAIL: shim cleanup state is not initialized safely"
    FAIL=$((FAIL + 1))
fi

_TRAP_HELPERS=$(mktemp)
for _f in _cleanup_install_temporaries _on_install_signal; do
    sed -n "/^$_f()/,/^}/p" "$INSTALL_SH" >> "$_TRAP_HELPERS"
done

for _sh in sh bash; do
    _signal_case="$_ROOT/${_sh}-signal"
    mkdir -p "$_signal_case/tmp"
    : > "$_signal_case/ready"
    CASE_DIR="$_signal_case" TMPDIR="$_signal_case/tmp" PATH="$_BIN:$PATH" \
        XCODE_SELECT_RC=1 "$_sh" -c '
            . "'"$_EXTRACTED_HELPERS"'"
            . "'"$_TRAP_HELPERS"'"
            OS=macos
            _UV_OVERRIDE_TMPDIR=""
            _UV_INSTALL_NAME_TOOL_SHIM_DIR=""
            _UNSLOTH_TORCH_OVERRIDES=""
            _restore_studio_venv_replacement() { :; }
            run_install_cmd() {
                printf "%s\n" "$_UV_INSTALL_NAME_TOOL_SHIM_DIR" > "$CASE_DIR/ready"
                while :; do :; done
            }
            trap "_on_install_signal 143" TERM
            _run_uv_venv "create venv" "$CASE_DIR/venv" --python cpython-3.12-macos-aarch64-none
        ' >/dev/null 2>&1 &
    _signal_pid=$!
    for _ in $(seq 1 100); do [ -s "$_signal_case/ready" ] && break; sleep 0.01; done
    _signal_shim=$(cat "$_signal_case/ready")
    if [ -z "$_signal_shim" ]; then
        echo "  FAIL: $_sh signal case never entered guarded uv command"
        FAIL=$((FAIL + 1))
        kill -KILL "$_signal_pid" 2>/dev/null || true
        wait "$_signal_pid" 2>/dev/null || true
        continue
    fi
    kill -TERM "$_signal_pid"
    set +e
    wait "$_signal_pid"
    _signal_rc=$?
    set -e
    assert_eq "$_sh TERM preserves signal status" "143" "$_signal_rc"
    if [ ! -e "$_signal_shim" ]; then
        echo "  PASS: $_sh TERM removes the active shim directory"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_sh TERM left shim directory $_signal_shim"
        FAIL=$((FAIL + 1))
    fi
done


echo ""
echo "Passed: $PASS, Failed: $FAIL"
[ "$FAIL" -eq 0 ]
