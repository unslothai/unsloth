#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# The top-level installer must keep uv's install-time cache under the resolved
# Studio root unless the caller supplied a nonblank override. Exercise the real
# helper in both POSIX sh and bash where available.
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

ok() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
bad() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

HELPER=$(awk '
    /^_configure_uv_cache\(\) \{/ { grab = 1 }
    grab { print }
    grab && /^}/ { exit }
' "$INSTALL_SH")
if ! printf '%s\n' "$HELPER" | grep -q '^_configure_uv_cache() {'; then
    echo "  FAIL: could not extract _configure_uv_cache from install.sh"
    exit 1
fi

WORK=$(mktemp -d)
PROBE="$WORK/probe.sh"
trap 'rm -rf "$WORK"' EXIT INT TERM
printf '%s\n' "$HELPER" > "$PROBE"
cat >> "$PROBE" <<'PROBE'
case "$1" in
    unset) unset UV_CACHE_DIR ;;
    value) UV_CACHE_DIR=$2 ;;
    *) exit 2 ;;
esac
STUDIO_HOME=$3
_configure_uv_cache
_child=$($4 -c 'printf "%s" "${UV_CACHE_DIR+x}:$UV_CACHE_DIR"')
printf 'value=%s\nchild=%s\n' "$UV_CACHE_DIR" "$_child"
PROBE

run_case() { # shell, label, state, input, expected, root
    _shell=$1
    _label=$2
    _state=$3
    _input=$4
    _expected=$5
    _root=$6
    _actual=$($_shell "$PROBE" "$_state" "$_input" "$_root" "$_shell")
    _wanted=$(printf 'value=%s\nchild=x:%s' "$_expected" "$_expected")
    if [ "$_actual" = "$_wanted" ]; then
        ok "$_shell: $_label"
    else
        bad "$_shell: $_label (expected [$_wanted], got [$_actual])"
    fi
}

echo "=== test_install_uv_cache_root ==="
for shell in sh bash; do
    command -v "$shell" >/dev/null 2>&1 || continue
    ROOT="$WORK/$shell studio root"
    DEFAULT="$ROOT/cache/uv"
    OVERRIDE="$WORK/$shell caller cache/uv artifacts"
    run_case "$shell" "unset defaults under Studio" unset "" "$DEFAULT" "$ROOT"
    run_case "$shell" "empty defaults under Studio" value "" "$DEFAULT" "$ROOT"
    run_case "$shell" "spaces default under Studio" value "   " "$DEFAULT" "$ROOT"
    run_case "$shell" "tab defaults under Studio" value "$(printf '\t')" "$DEFAULT" "$ROOT"
    run_case "$shell" "explicit spaced override is exact" value "$OVERRIDE" "$OVERRIDE" "$ROOT"
done

_resolve_line=$(grep -n '^_resolve_studio_destinations$' "$INSTALL_SH" | head -n1 | cut -d: -f1)
_configure_line=$(grep -n '^_configure_uv_cache$' "$INSTALL_SH" | head -n1 | cut -d: -f1)
_uv_line=$(grep -n '^# ── Install uv ──$' "$INSTALL_SH" | head -n1 | cut -d: -f1)
if [ -n "$_resolve_line" ] && [ -n "$_configure_line" ] && [ -n "$_uv_line" ] \
   && [ "$_resolve_line" -lt "$_configure_line" ] && [ "$_configure_line" -lt "$_uv_line" ]; then
    ok "helper runs after destination resolution and before uv setup"
else
    bad "helper ordering (resolve=$_resolve_line configure=$_configure_line uv=$_uv_line)"
fi

echo ""
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo "FAILED"
    exit 1
fi
echo "ALL PASSED"
