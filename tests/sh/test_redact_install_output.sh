#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# uv/pip failure text embeds the --index-url verbatim, so _redact_install_output strips
# user:token@ and ?token= secrets. Mirrors install_python_stack.py and the ps1 copies.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
_FUNC_FILE=$(mktemp)
sed -n '/^_redact_install_output()/,/^}/p' "$INSTALL_SH" > "$_FUNC_FILE"
# shellcheck disable=SC1090
. "$_FUNC_FILE"
rm -f "$_FUNC_FILE"

redact_str() {
    _rs_tmp=$(mktemp)
    printf '%s\n' "$1" > "$_rs_tmp"
    _rs_out=$(_redact_install_output "$_rs_tmp")
    rm -f "$_rs_tmp"
    printf '%s' "$_rs_out"
}

echo "=== _redact_install_output ==="
assert_eq "userinfo user:token@ redacted" \
    "ERROR: failed https://<redacted>@download.pytorch.org/whl/cu128" \
    "$(redact_str 'ERROR: failed https://alice:s3cr3t@download.pytorch.org/whl/cu128')"

assert_eq "bare-token@ userinfo redacted" \
    "fetch https://<redacted>@host/whl/cu128 failed" \
    "$(redact_str 'fetch https://ghp_deadbeef@host/whl/cu128 failed')"

assert_eq "single ?token= query redacted" \
    "url https://host/whl/cu128?token=<redacted> unreachable" \
    "$(redact_str 'url https://host/whl/cu128?token=abcd1234 unreachable')"

assert_eq "multiple query values redacted" \
    "https://host/whl/cu128?token=<redacted>&channel=<redacted>" \
    "$(redact_str 'https://host/whl/cu128?token=abcd1234&channel=beta')"

assert_eq "http (not https) userinfo redacted" \
    "http://<redacted>@host/simple" \
    "$(redact_str 'http://u:p@host/simple')"

assert_eq "fragment token redacted" \
    "ERROR: could not fetch https://mirror.local/whl/cu128#<redacted> (403)" \
    "$(redact_str 'ERROR: could not fetch https://mirror.local/whl/cu128#token=SECRET123 (403)')"

assert_eq "query and fragment both redacted" \
    "https://host/whl/cu128?token=<redacted>#<redacted> done" \
    "$(redact_str 'https://host/whl/cu128?token=abc#sig=xyz done')"

assert_eq "plain line untouched" \
    "Resolved 42 packages in 1.2s" \
    "$(redact_str 'Resolved 42 packages in 1.2s')"
assert_eq "plain url without creds untouched" \
    "downloading https://download.pytorch.org/whl/cu128/torch-2.8.0.whl" \
    "$(redact_str 'downloading https://download.pytorch.org/whl/cu128/torch-2.8.0.whl')"
assert_eq "bare hash comment untouched" \
    "# retrying with --no-cache-dir" \
    "$(redact_str '# retrying with --no-cache-dir')"

_leak=$(redact_str 'https://alice:s3cr3t@host/whl/cu128?token=SUPERSECRET#frag=ALSOSECRET')
case "$_leak" in
    *s3cr3t*|*SUPERSECRET*|*ALSOSECRET*) assert_eq "no secret leak" "clean" "leaked:$_leak" ;;
    *)                                   assert_eq "no secret leak" "clean" "clean" ;;
esac

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
