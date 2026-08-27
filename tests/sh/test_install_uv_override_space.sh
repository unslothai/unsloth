#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# uv splits UV_OVERRIDE on whitespace, so a repo cloned under a path with a space
# truncates it and aborts every later uv call (issue #6503). install.sh must hand
# uv a space-free copy. Exercises the real install.sh hardening block.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

ok()  { echo "  PASS: $1"; PASS=$((PASS + 1)); }
bad() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

# Extract the UV_OVERRIDE hardening block (outer case ... esac plus the export)
# and run it directly, so the test tracks install.sh rather than a copy of it.
BLOCK=$(awk '
    /case "[$]_OVERRIDES_FILE" in/ { grab = 1 }
    grab { print }
    grab && /export UV_OVERRIDE="[$]_OVERRIDES_FILE"/ { exit }
' "$INSTALL_SH")
if ! printf '%s' "$BLOCK" | grep -q 'export UV_OVERRIDE'; then
    echo "  FAIL: could not extract UV_OVERRIDE block from install.sh"
    exit 1
fi

run_block() {
    _OVERRIDES_FILE="$1"
    _UV_OVERRIDE_TMPDIR=""
    unset UV_OVERRIDE
    eval "$BLOCK"
}

echo "=== test_install_uv_override_space ==="

# 1. Spaced path -> space-free copy with identical contents, temp dir tracked.
WORK=$(mktemp -d)
mkdir -p "$WORK/Open Source"
SRC="$WORK/Open Source/overrides-darwin-arm64.txt"
printf 'transformers>=4.57.6\n' > "$SRC"
run_block "$SRC"
case "$UV_OVERRIDE" in
    *[[:space:]]*) bad "spaced path: UV_OVERRIDE still contains whitespace ($UV_OVERRIDE)" ;;
    *)             ok  "spaced path: UV_OVERRIDE is whitespace-free" ;;
esac
[ "$UV_OVERRIDE" != "$SRC" ] && ok "spaced path: points at a copy" || bad "spaced path: not copied"
[ "$(cat "$UV_OVERRIDE" 2>/dev/null)" = "transformers>=4.57.6" ] \
    && ok "spaced path: copy contents identical" || bad "spaced path: contents differ"
{ [ -n "$_UV_OVERRIDE_TMPDIR" ] && [ -d "$_UV_OVERRIDE_TMPDIR" ]; } \
    && ok "spaced path: temp dir tracked for cleanup" || bad "spaced path: temp dir not tracked"
# The exit-trap cleanup (_on_install_exit) must then remove it.
[ -n "$_UV_OVERRIDE_TMPDIR" ] && rm -rf "$_UV_OVERRIDE_TMPDIR" 2>/dev/null || true
[ ! -d "$_UV_OVERRIDE_TMPDIR" ] && ok "spaced path: temp dir removable" || bad "spaced path: temp dir lingers"
rm -rf "$WORK"

# 2. No-space path -> passthrough, no temp dir.
PLAIN=$(mktemp -d)
PSRC="$PLAIN/overrides-darwin-arm64.txt"
printf 'transformers>=4.57.6\n' > "$PSRC"
run_block "$PSRC"
[ "$UV_OVERRIDE" = "$PSRC" ] && ok "no-space path: UV_OVERRIDE unchanged" || bad "no-space path: changed ($UV_OVERRIDE)"
[ -z "$_UV_OVERRIDE_TMPDIR" ] && ok "no-space path: no temp dir created" || bad "no-space path: temp dir created"
rm -rf "$PLAIN"

# 3. TMPDIR itself contains a space -> use /tmp for a safe copy, no leak.
WORK2=$(mktemp -d)
mkdir -p "$WORK2/Open Source" "$WORK2/tmp dir"
SRC2="$WORK2/Open Source/overrides-darwin-arm64.txt"
printf 'transformers>=4.57.6\n' > "$SRC2"
RES=$( TMPDIR="$WORK2/tmp dir"; export TMPDIR; run_block "$SRC2"
       if printf '%s' "$UV_OVERRIDE" | grep -q '[[:space:]]'; then _safe=no; else _safe=yes; fi
       [ "$(cat "$UV_OVERRIDE" 2>/dev/null)" = "transformers>=4.57.6" ] && _copy=yes || _copy=no
       { [ -n "$_UV_OVERRIDE_TMPDIR" ] && [ -d "$_UV_OVERRIDE_TMPDIR" ]; } && _tracked=yes || _tracked=no
       printf 'SAFE=%s\nCOPY=%s\nTRACKED=%s\n' "$_safe" "$_copy" "$_tracked"
       [ -n "$_UV_OVERRIDE_TMPDIR" ] && rm -rf "$_UV_OVERRIDE_TMPDIR" 2>/dev/null || true )
echo "$RES" | grep -qx "SAFE=yes" \
    && ok "spaced TMPDIR: UV_OVERRIDE is whitespace-free" || bad "spaced TMPDIR: unsafe override path ($RES)"
echo "$RES" | grep -qx "COPY=yes" \
    && ok "spaced TMPDIR: safe copy contents identical" || bad "spaced TMPDIR: copy contents differ ($RES)"
echo "$RES" | grep -qx "TRACKED=yes" \
    && ok "spaced TMPDIR: temp dir tracked for cleanup" || bad "spaced TMPDIR: temp dir not tracked ($RES)"
# The unsafe TMPDIR itself must stay unused.
_leftover=$(find "$WORK2/tmp dir" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -n1)
[ -z "$_leftover" ] && ok "spaced TMPDIR: no leaked temp dir" || bad "spaced TMPDIR: leaked $_leftover"
rm -rf "$WORK2"

# 4. A tab in the path is whitespace uv also splits on -> copied like a space.
WORK3=$(mktemp -d)
TABDIR=$(printf 'Open\tSource')
mkdir -p "$WORK3/$TABDIR"
SRC3="$WORK3/$TABDIR/overrides-darwin-arm64.txt"
printf 'transformers>=4.57.6\n' > "$SRC3"
run_block "$SRC3"
case "$UV_OVERRIDE" in
    *[[:space:]]*) bad "tab path: UV_OVERRIDE still contains whitespace" ;;
    *)             ok  "tab path: UV_OVERRIDE is whitespace-free" ;;
esac
[ -n "$_UV_OVERRIDE_TMPDIR" ] && rm -rf "$_UV_OVERRIDE_TMPDIR" 2>/dev/null || true
rm -rf "$WORK3"

# 5. install.sh must clear _UV_OVERRIDE_TMPDIR before registering the exit trap,
# so an inherited value can never reach the trap's rm -rf.
_init_line=$(grep -n '^_UV_OVERRIDE_TMPDIR=""' "$INSTALL_SH" | head -n1 | cut -d: -f1)
_trap_line=$(grep -n '^trap _on_install_exit EXIT' "$INSTALL_SH" | head -n1 | cut -d: -f1)
{ [ -n "$_init_line" ] && [ -n "$_trap_line" ] && [ "$_init_line" -lt "$_trap_line" ]; } \
    && ok "init: _UV_OVERRIDE_TMPDIR cleared before exit trap" \
    || bad "init: _UV_OVERRIDE_TMPDIR not cleared before exit trap (init=$_init_line trap=$_trap_line)"

# 6. Piped/standalone installs bootstrap the wheel without dependencies, then
# activate its packaged override before any with-dependencies Unsloth resolve.
BOOTSTRAP=$(sed -n '/^_bootstrap_packaged_mlx_override() {/,/^}/p' "$INSTALL_SH")
if [ -z "$BOOTSTRAP" ]; then
    bad "early packaged-override bootstrap is missing"
else
    printf '%s' "$BOOTSTRAP" | grep -q 'preparing Apple Silicon model support' \
        && ok "bootstrap: progress message is user-friendly" \
        || bad "bootstrap: user-friendly progress message is missing"
    if printf '%s' "$BOOTSTRAP" | grep -q 'mlx-vlm may be backtracked'; then
        bad "bootstrap: warning exposes resolver jargon"
    else
        ok "bootstrap: warning avoids resolver jargon"
    fi
    WORK4=$(mktemp -d)
    mkdir -p "$WORK4/Packaged Overrides"
    PACKAGED_OVERRIDE="$WORK4/Packaged Overrides/overrides-darwin-arm64.txt"
    FAKE_PY="$WORK4/python"
    CAPTURE="$WORK4/install-command.txt"
    printf 'transformers>=4.57.6\n' > "$PACKAGED_OVERRIDE"
    printf '#!/bin/sh\nprintf "%%s\\n" "$FAKE_OVERRIDE_PATH"\n' > "$FAKE_PY"
    chmod +x "$FAKE_PY"
    export FAKE_OVERRIDE_PATH="$PACKAGED_OVERRIDE"
    export CAPTURE
    OS=macos
    _ARCH=arm64
    _VENV_PY="$FAKE_PY"
    PACKAGE_NAME=unsloth
    C_WARN=warn
    SKIP_TORCH=false
    _OVERRIDES_FILE="$WORK4/missing-repository-override.txt"
    _UV_OVERRIDE_TMPDIR=""
    unset UV_OVERRIDE
    substep() { :; }
    run_install_cmd_retry() { printf '%s\n' "$*" > "$CAPTURE"; }
    eval "$BOOTSTRAP"
    _bootstrap_packaged_mlx_override
    _bootstrap_cmd=$(cat "$CAPTURE" 2>/dev/null || true)
    case "$_bootstrap_cmd" in
        *"uv pip install --python $FAKE_PY --no-deps --upgrade-package unsloth -- unsloth"*)
            ok "bootstrap: wheel is installed without dependencies" ;;
        *) bad "bootstrap: no-deps wheel command missing ($_bootstrap_cmd)" ;;
    esac
    case "$UV_OVERRIDE" in
        *[[:space:]]*) bad "bootstrap: UV_OVERRIDE contains whitespace ($UV_OVERRIDE)" ;;
        *) ok "bootstrap: packaged override path is whitespace-safe" ;;
    esac
    [ "$(cat "$UV_OVERRIDE" 2>/dev/null)" = "transformers>=4.57.6" ] \
        && ok "bootstrap: packaged override contents are preserved" \
        || bad "bootstrap: packaged override contents differ"
    printf '%s' "$BOOTSTRAP" | grep -q '"$_VENV_PY" -I -c' \
        && ok "bootstrap: packaged resource lookup ignores local modules" \
        || bad "bootstrap: packaged resource lookup is not isolated"
    [ -n "$_UV_OVERRIDE_TMPDIR" ] && rm -rf "$_UV_OVERRIDE_TMPDIR" 2>/dev/null || true

    EXTERNAL_OVERRIDE="$WORK4/external-override.txt"
    printf 'transformers==5.5.0\n' > "$EXTERNAL_OVERRIDE"
    UV_OVERRIDE="$EXTERNAL_OVERRIDE"
    export UV_OVERRIDE
    _OVERRIDES_FILE="$WORK4/still-missing.txt"
    _UV_OVERRIDE_TMPDIR=""
    : > "$CAPTURE"
    _bootstrap_packaged_mlx_override
    { [ ! -s "$CAPTURE" ] && [ "$UV_OVERRIDE" = "$EXTERNAL_OVERRIDE" ]; } \
        && ok "bootstrap: caller-provided UV_OVERRIDE is preserved" \
        || bad "bootstrap: caller-provided UV_OVERRIDE was replaced"
    unset UV_OVERRIDE

    check_bootstrap_skip() {  # label, os, arch, sibling override, skip torch
        _skip_label="$1"
        OS="$2"
        _ARCH="$3"
        _OVERRIDES_FILE="$4"
        SKIP_TORCH="$5"
        _UV_OVERRIDE_TMPDIR=""
        unset UV_OVERRIDE
        : > "$CAPTURE"
        _bootstrap_packaged_mlx_override
        { [ ! -s "$CAPTURE" ] && [ -z "${UV_OVERRIDE:-}" ]; } \
            && ok "bootstrap: $_skip_label is unchanged" \
            || bad "bootstrap: $_skip_label unexpectedly bootstrapped"
    }
    check_bootstrap_skip "Linux" linux arm64 "$WORK4/missing.txt" false
    check_bootstrap_skip "Intel macOS" macos x86_64 "$WORK4/missing.txt" false
    check_bootstrap_skip "no-torch install" macos arm64 "$WORK4/missing.txt" true
    check_bootstrap_skip "repository install" macos arm64 "$PACKAGED_OVERRIDE" false
    rm -rf "$WORK4"
fi

_bootstrap_line=$(grep -n '^_bootstrap_packaged_mlx_override$' "$INSTALL_SH" | head -n1 | cut -d: -f1)
_with_deps_line=$(grep -n '^_build_unsloth_torch_overrides()' "$INSTALL_SH" | head -n1 | cut -d: -f1)
{ [ -n "$_bootstrap_line" ] && [ -n "$_with_deps_line" ] \
  && [ "$_bootstrap_line" -lt "$_with_deps_line" ]; } \
    && ok "bootstrap: packaged override activates before dependency resolution" \
    || bad "bootstrap: activation is not before dependency resolution"

echo ""
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo "FAILED"
    exit 1
fi
echo "ALL PASSED"
