#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# See /studio/LICENSE.AGPL-3.0
#
# Regression tests for the npm-failure hint in studio/setup.sh (issue #8725).
#
# A Windows user installing Studio hit this, with Node 24 on PATH:
#
#     npm error code EACCES
#     npm error FetchError: request to https://registry.npmjs.org/oxlint/-/oxlint-1.65.0.tgz failed
#     npm error The operation was rejected by your operating system.
#
# That is a local failure -- a locked or unwritable npm cache -- yet the
# installer answered "registry.npmjs.org looks blocked (corporate
# firewall/proxy?)" and sent them hunting for a corporate proxy that did not
# exist. Re-running, and running as Administrator, changed nothing.
#
# The gate here was already log-aware, but its network regex includes the
# literal registry.npmjs.org, and npm's FetchError line names that host even
# when the cause is a permission error -- so this log matched and the wrong
# hint printed. The fix classifies local errno failures (EACCES/EPERM/EBUSY/
# ENOSPC) BEFORE the network markers.
#
# These tests extract the two functions and drive them with captured logs, so
# nothing here touches npm or the network.

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
PASS=0
FAIL=0

_FUNC_FILE=$(mktemp)
{
    sed -n '/^_NPM_LOCAL_FAILURE_RE=/p' "$SETUP_SH"
    sed -n '/^_suggest_npm_local_failure()/,/^}/p' "$SETUP_SH"
    sed -n '/^_suggest_npm_registry()/,/^}/p' "$SETUP_SH"
} > "$_FUNC_FILE"

for _needle in '_NPM_LOCAL_FAILURE_RE=' '_suggest_npm_local_failure()' '_suggest_npm_registry()'; do
    if ! grep -q -- "$_needle" "$_FUNC_FILE"; then
        echo "FAIL: could not extract $_needle from $SETUP_SH"
        exit 1
    fi
done

# Minimal stand-ins for the installer's output helpers; both write to stderr,
# which is where the real ones send this guidance.
step()    { printf '  %-15s%s\n' "$1" "$2" >&2; }
substep() { printf '  %-15s%s\n' "" "$1" >&2; }
C_WARN=""

# shellcheck disable=SC1090
. "$_FUNC_FILE"

_log_file=$(mktemp)

# The log from issue #8725, trimmed to the lines that matter. Note that it
# mentions registry.npmjs.org: that is exactly what used to mislead the gate.
_EACCES_LOG='npm error code EACCES
npm error errno EACCES
npm error FetchError: request to https://registry.npmjs.org/oxlint/-/oxlint-1.65.0.tgz failed, reason:
npm error   code: '"'"'EACCES'"'"',
npm error   type: '"'"'system'"'"'
npm error The operation was rejected by your operating system.'

_NETWORK_LOG='npm error code ENOTFOUND
npm error network request to https://registry.npmjs.org/oxlint failed, reason: getaddrinfo ENOTFOUND registry.npmjs.org'

_PROXY_LOG='npm error code E403
npm error 403 Forbidden - GET https://registry.npmjs.org/oxlint
npm error tunneling socket could not be established'

_UNRELATED_LOG='npm error code ELIFECYCLE
npm error errno 1
npm error oxc-validator@1.0.0 postinstall script failed'

check() {
    # check <name> <log> <expect-local:yes|no> <expect-registry:yes|no>
    local _name="$1" _log="$2" _want_local="$3" _want_registry="$4"
    printf '%s\n' "$_log" > "$_log_file"

    local _out
    _out="$( _suggest_npm_registry "$_log_file" 2>&1 )"

    local _got_local="no" _got_registry="no"
    case "$_out" in *"blocked by the operating system"*) _got_local="yes" ;; esac
    case "$_out" in *"looks blocked (corporate firewall/proxy?)"*) _got_registry="yes" ;; esac

    if [ "$_got_local" = "$_want_local" ] && [ "$_got_registry" = "$_want_registry" ]; then
        PASS=$((PASS + 1))
        echo "ok   $_name"
    else
        FAIL=$((FAIL + 1))
        echo "FAIL $_name"
        echo "     expected local=$_want_local registry=$_want_registry"
        echo "     got      local=$_got_local registry=$_got_registry"
        echo "     output: $_out"
    fi
}

# The bug: a permission error must not be reported as a blocked registry.
check "EACCES log gets the local hint, not the registry one" "$_EACCES_LOG" yes no

# Genuine network failures keep the behaviour the hint exists for.
check "ENOTFOUND log still gets the registry hint" "$_NETWORK_LOG" no yes
check "403/tunnelling log still gets the registry hint" "$_PROXY_LOG" no yes

# Neither local nor network: stay quiet, the raw npm error is better than a guess.
check "unrelated failure stays quiet" "$_UNRELATED_LOG" no no

# A local failure is not about registries, so the mirror opt-out must not silence it.
UNSLOTH_NPM_REGISTRY="https://mirror.example/api/npm/" \
    check "local hint survives UNSLOTH_NPM_REGISTRY" "$_EACCES_LOG" yes no

# EPERM is the other errno Windows reports for a locked file.
check "EPERM log gets the local hint" \
    'npm error code EPERM
npm error syscall unlink
npm error EPERM: operation not permitted, unlink '"'"'C:\Users\u\AppData\Local\npm-cache\_cacache\tmp\x'"'"'' \
    yes no

# No captured log at all: unchanged from before the fix -- the registry hint is
# the only guidance available, so it still prints.
: > "$_log_file"
_out="$( _suggest_npm_registry "$_log_file" 2>&1 )"
case "$_out" in
    *"looks blocked (corporate firewall/proxy?)"*)
        PASS=$((PASS + 1)); echo "ok   empty log keeps the old registry hint" ;;
    *)
        FAIL=$((FAIL + 1)); echo "FAIL empty log keeps the old registry hint"; echo "     output: $_out" ;;
esac

rm -f "$_FUNC_FILE" "$_log_file"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
