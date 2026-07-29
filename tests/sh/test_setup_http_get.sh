#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# studio/setup.sh fetch helpers: curl preferred, wget accepted, neither is an error.
# install.sh takes either transport everywhere, so a wget-only box installs fine
# and used to stall in setup.sh, where curl was the only way to fetch anything.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
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
    if echo "$_haystack" | grep -qF -- "$_needle"; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected to find '$_needle' in '$_haystack')"
        FAIL=$((FAIL + 1))
    fi
}

# ── Extract the two helpers from setup.sh ──
_FN_FILE=$(mktemp)
sed -n '/^_setup_http_get()/,/^}/p'       "$SETUP_SH" >  "$_FN_FILE"
sed -n '/^_setup_http_get_timed()/,/^}/p' "$SETUP_SH" >> "$_FN_FILE"

for _fn in _setup_http_get _setup_http_get_timed; do
    grep -q "^$_fn()" "$_FN_FILE" || { echo "  FAIL: $_fn not found in setup.sh"; exit 1; }
done

_MOCK=$(mktemp -d)
_LOG="$_MOCK/argv.log"

_make_shim() {
    cat > "$_MOCK/$1" <<EOF
#!/bin/sh
echo "$1 \$*" >> "$_LOG"
echo "$1-body"
EOF
    chmod +x "$_MOCK/$1"
}

# PATH holds only the shims, so an absent shim is genuinely absent (not a stub
# that command -v still finds).
_run() {
    _have="$1"; _call="$2"
    rm -f "$_MOCK"/curl "$_MOCK"/wget "$_MOCK"/timeout "$_LOG"
    for _t in $_have; do _make_shim "$_t"; done
    ( PATH="$_MOCK"; export PATH; . "$_FN_FILE"; $_call "https://example.invalid/x" ) 2>/dev/null
}

_argv() { cat "$_LOG" 2>/dev/null | tr '\n' ' '; }

echo "=== _setup_http_get ==="

assert_eq "curl preferred when both exist" "curl-body" "$(_run 'curl wget' _setup_http_get)"
assert_contains "curl call keeps -LsSf" "$(_argv)" "curl -LsSf"

assert_eq "wget used when curl is missing" "wget-body" "$(_run 'wget' _setup_http_get)"
assert_contains "wget call writes to stdout" "$(_argv)" "wget -qO-"

_out=$(_run '' _setup_http_get || true)
assert_eq "no transport: empty output" "" "$_out"
_rc=0; _run '' _setup_http_get >/dev/null 2>&1 || _rc=$?
assert_eq "no transport: non-zero exit" "1" "$_rc"

echo ""
echo "=== _setup_http_get_timed ==="

assert_eq "curl preferred when both exist" "curl-body" "$(_run 'curl wget' _setup_http_get_timed)"
assert_contains "curl bounds the whole transfer" "$(_argv)" "--max-time 5"

assert_eq "wget used when curl is missing" "wget-body" "$(_run 'wget' _setup_http_get_timed)"
assert_contains "wget sets the timeout" "$(_argv)" "--timeout=5"
# wget's --timeout is per operation and it retries 20 times by default, so
# without --tries=1 a stalling server turns this bounded check into minutes.
assert_contains "wget limited to one attempt" "$(_argv)" "--tries=1"

# --timeout is per operation, so a drip response never ends the transfer; the
# outer timeout is what actually matches curl's --max-time.
assert_eq "timeout wraps wget when available" "timeout-body" "$(_run 'wget timeout' _setup_http_get_timed)"
assert_contains "wall clock deadline on wget" "$(_argv)" "timeout 5 wget -qO- --timeout=5 --tries=1"
assert_eq "no timeout binary: wget still runs" "wget-body" "$(_run 'wget' _setup_http_get_timed)"

_rc=0; _run '' _setup_http_get_timed >/dev/null 2>&1 || _rc=$?
assert_eq "no transport: non-zero exit" "1" "$_rc"

rm -rf "$_MOCK" "$_FN_FILE"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
