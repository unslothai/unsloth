#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# See /studio/LICENSE.AGPL-3.0
#
# A local npm errno gets a local hint, not "registry blocked"; an EACCES on the HTTP socket
# (FetchError) gets the "OS refused node's connection" variant.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
PASS=0
FAIL=0

_FUNC_FILE=$(mktemp)
_log_file=$(mktemp)
trap 'rm -f "$_FUNC_FILE" "$_log_file"' EXIT
{
    sed -n '/^_NPM_LOCAL_FAILURE_RE=/p' "$SETUP_SH"
    sed -n '/^_suggest_npm_local_failure()/,/^}/p' "$SETUP_SH"
    sed -n '/^_suggest_npm_registry()/,/^}/p' "$SETUP_SH"
} > "$_FUNC_FILE"
for _needle in '_NPM_LOCAL_FAILURE_RE=' '_suggest_npm_local_failure()' '_suggest_npm_registry()'; do
    grep -q -- "$_needle" "$_FUNC_FILE" || { echo "FAIL: could not extract $_needle from $SETUP_SH"; exit 1; }
done

step()    { printf '  %-15s%s\n' "$1" "$2" >&2; }
substep() { printf '  %-15s%s\n' "" "$1" >&2; }
C_WARN=""
# shellcheck disable=SC1090
. "$_FUNC_FILE"

_EACCES_LOG="npm error code EACCES
npm error FetchError: request to https://registry.npmjs.org/oxlint/-/oxlint-1.65.0.tgz failed, reason:
npm error The operation was rejected by your operating system."

_EPERM_LOG="npm ERR! code EPERM
npm ERR! syscall rename
npm ERR! Error: EPERM: operation not permitted, rename 'C:\\npm-cache\\_cacache\\tmp\\x'"

# Real npm 10.9 output for a deny-write cache ACL: FetchError, but with a path.
_CACHE_ACL_LOG="npm error code EPERM
npm error syscall mkdir
npm error path D:\\a\\_temp\\x\\cache\\_cacache
npm error errno EPERM
npm error FetchError: Invalid response body while trying to fetch https://registry.npmjs.org/is-number: EPERM: operation not permitted, mkdir 'D:\\a\\_temp\\x\\cache\\_cacache'"

_ESC=$(printf '\033')
_COLOR_EPERM_LOG="${_ESC}[31mnpm${_ESC}[39m ${_ESC}[31merror${_ESC}[39m ${_ESC}[90mcode${_ESC}[39m EPERM
${_ESC}[31mnpm${_ESC}[39m ${_ESC}[31merror${_ESC}[39m syscall rename"

_NETWORK_LOG="npm error code ENOTFOUND
npm error network request to https://registry.npmjs.org/oxlint failed, reason: getaddrinfo ENOTFOUND registry.npmjs.org"

# Windows npm warns about EPERM during cleanup after a genuine network failure.
_NETWORK_CLEANUP_LOG="npm warn cleanup Failed to remove some directories [
npm warn cleanup   [Error: EPERM: operation not permitted, rmdir 'C:\\x\\node_modules\\oxlint']
npm warn cleanup ]
npm error code ETIMEDOUT
npm error network request to https://registry.npmjs.org/oxlint failed"

_PROXY_LOG="npm error code E403
npm error 403 Forbidden - GET https://registry.npmjs.org/oxlint"

_UNRELATED_LOG="npm error code ELIFECYCLE
npm error oxc-validator@1.0.0 postinstall script failed"

# check <name> <log> <expected hint: file|socket|registry|none>
check() {
    printf '%s\n' "$2" > "$_log_file"
    [ -n "$2" ] || : > "$_log_file"
    local _out _got=""
    _out="$( _suggest_npm_registry "$_log_file" 2>&1 )"
    case "$_out" in *"local file error"*) _got="$_got file" ;; esac
    case "$_out" in *"refused node's connection"*) _got="$_got socket" ;; esac
    case "$_out" in *"looks blocked (corporate firewall/proxy?)"*) _got="$_got registry" ;; esac
    _got="${_got# }"
    if [ "${_got:-none}" = "$3" ]; then
        PASS=$((PASS + 1)); echo "ok   $1"
    else
        FAIL=$((FAIL + 1)); echo "FAIL $1 (want $3, got ${_got:-none})"
        echo "     output: $_out"
    fi
}

check "#8725 socket EACCES gets the connection hint" "$_EACCES_LOG" socket
check "EPERM on a cache file gets the file hint" "$_EPERM_LOG" file
check "unwritable cache (FetchError + path) gets the file hint" "$_CACHE_ACL_LOG" file
check "colored (color=always) EPERM log gets the file hint" "$_COLOR_EPERM_LOG" file
check "ENOTFOUND log gets the registry hint" "$_NETWORK_LOG" registry
check "network failure with EPERM cleanup warnings gets the registry hint" "$_NETWORK_CLEANUP_LOG" registry
check "403 log gets the registry hint" "$_PROXY_LOG" registry
check "unrelated failure stays quiet" "$_UNRELATED_LOG" none
check "empty log keeps the registry hint" "" registry
UNSLOTH_NPM_REGISTRY="https://mirror.example/api/npm/" \
    check "local hint survives UNSLOTH_NPM_REGISTRY" "$_EACCES_LOG" socket
UNSLOTH_NPM_REGISTRY="https://mirror.example/api/npm/" \
    check "UNSLOTH_NPM_REGISTRY still silences the registry hint" "$_NETWORK_LOG" none

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
