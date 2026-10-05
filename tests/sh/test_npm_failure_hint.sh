#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# See /studio/LICENSE.AGPL-3.0
#
# _suggest_npm_registry in studio/setup.sh: a local npm errno (#8725) gets the
# local hint, not "registry.npmjs.org looks blocked".

set -u

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

# check <name> <log> <expect-local:yes|no> <expect-registry:yes|no>
check() {
    printf '%s\n' "$2" > "$_log_file"
    [ -n "$2" ] || : > "$_log_file"
    local _out _local=no _registry=no
    _out="$( _suggest_npm_registry "$_log_file" 2>&1 )"
    case "$_out" in *"local file error"*) _local=yes ;; esac
    case "$_out" in *"looks blocked (corporate firewall/proxy?)"*) _registry=yes ;; esac
    if [ "$_local" = "$3" ] && [ "$_registry" = "$4" ]; then
        PASS=$((PASS + 1)); echo "ok   $1"
    else
        FAIL=$((FAIL + 1)); echo "FAIL $1 (want local=$3 registry=$4, got local=$_local registry=$_registry)"
        echo "     output: $_out"
    fi
}

check "EACCES log gets the local hint" "$_EACCES_LOG" yes no
check "EPERM log gets the local hint" "$_EPERM_LOG" yes no
check "ENOTFOUND log gets the registry hint" "$_NETWORK_LOG" no yes
check "network failure with EPERM cleanup warnings gets the registry hint" "$_NETWORK_CLEANUP_LOG" no yes
check "403 log gets the registry hint" "$_PROXY_LOG" no yes
check "unrelated failure stays quiet" "$_UNRELATED_LOG" no no
check "empty log keeps the registry hint" "" no yes
UNSLOTH_NPM_REGISTRY="https://mirror.example/api/npm/" \
    check "local hint survives UNSLOTH_NPM_REGISTRY" "$_EACCES_LOG" yes no
UNSLOTH_NPM_REGISTRY="https://mirror.example/api/npm/" \
    check "UNSLOTH_NPM_REGISTRY still silences the registry hint" "$_NETWORK_LOG" no no

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
