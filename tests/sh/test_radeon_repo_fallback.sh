#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# _radeon_fetch_listing separates "no such release" (HTTP 404/410) from "host unreachable" (#7264, #10657).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"

_FUNC_FILE=$(mktemp)
sed -n '/^_radeon_fetch_listing()/,/^}/p' "$INSTALL_SH" > "$_FUNC_FILE"
[ -s "$_FUNC_FILE" ] || { echo "  FAIL: could not extract _radeon_fetch_listing"; exit 1; }

_STUB_DIR=$(mktemp -d)
cat > "$_STUB_DIR/fakepy" <<'STUB'
#!/bin/sh
echo cp312
STUB
chmod +x "$_STUB_DIR/fakepy"

# $1 = stub curl exit code, $2 = the %{http_code} it writes out, as curl -w does even on failure.
run_fetch() {
    _curl_dir=$(mktemp -d)
    cat > "$_curl_dir/curl" <<STUB
#!/bin/sh
printf '\\n%s' '$2'
exit $1
STUB
    chmod +x "$_curl_dir/curl"
    PATH="$_curl_dir:$PATH" bash -c "
        . '$_FUNC_FILE'
        _VENV_PY='$_STUB_DIR/fakepy'
        _RADEON_LISTING=''
        _RADEON_HOST_ANSWERED=false
        if _radeon_fetch_listing 'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.14/'; then
            _rc=ok
        else
            _rc=fail
        fi
        echo \"\$_rc answered=\$_RADEON_HOST_ANSWERED\"
    "
    rm -rf "$_curl_dir"
}

echo "=== test_radeon_repo_fallback ==="

# 1) curl -f exits 22 on HTTP >= 400; only 404/410 mean the release is absent.
assert_eq "HTTP 404 -> fetch fails, host recorded as answering" \
    "fail answered=true" "$(run_fetch 22 404)"
assert_eq "HTTP 410 -> fetch fails, host recorded as answering" \
    "fail answered=true" "$(run_fetch 22 410)"
assert_eq "HTTP 503 -> outage, host not recorded as answering" \
    "fail answered=false" "$(run_fetch 22 503)"
assert_eq "HTTP 429 -> rate limit, host not recorded as answering" \
    "fail answered=false" "$(run_fetch 22 429)"

# 2) Connection refused (7).
assert_eq "connection refused -> fetch fails, host not recorded as answering" \
    "fail answered=false" "$(run_fetch 7 000)"

# 3) Timeout (28).
assert_eq "timeout -> fetch fails, host not recorded as answering" \
    "fail answered=false" "$(run_fetch 28 000)"

# 4) Served listing.
_curl_dir=$(mktemp -d)
cat > "$_curl_dir/curl" <<'STUB'
#!/bin/sh
echo '<a href="torch-2.11.0%2Brocm7.2.4-cp312-cp312-linux_x86_64.whl">torch</a>'
printf '\n200'
STUB
chmod +x "$_curl_dir/curl"
_result=$(PATH="$_curl_dir:$PATH" bash -c "
    . '$_FUNC_FILE'
    _VENV_PY='$_STUB_DIR/fakepy'
    _RADEON_LISTING=''
    _RADEON_HOST_ANSWERED=false
    if _radeon_fetch_listing 'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.4/'; then
        printf 'ok answered=%s pytag=%s\n' \"\$_RADEON_HOST_ANSWERED\" \"\$_RADEON_PYTAG\"
    else
        echo fail
    fi
")
assert_eq "a served listing -> fetch succeeds" "ok answered=false pytag=cp312" "$_result"
rm -rf "$_curl_dir"

# 5) Sticky: 404 on X.Y.Z then a transport failure on X.Y still means the host is up.
_curl_dir=$(mktemp -d)
cat > "$_curl_dir/curl" <<STUB
#!/bin/sh
_n=\$(cat "$_curl_dir/calls" 2>/dev/null || echo 0)
echo \$((_n + 1)) > "$_curl_dir/calls"
[ "\$_n" = 0 ] && { printf '\\n404'; exit 22; }
printf '\\n000'
exit 7
STUB
chmod +x "$_curl_dir/curl"
_result=$(PATH="$_curl_dir:$PATH" bash -c "
    . '$_FUNC_FILE'
    _VENV_PY='$_STUB_DIR/fakepy'
    _RADEON_LISTING=''
    _RADEON_HOST_ANSWERED=false
    _radeon_fetch_listing 'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.14.60850/' || true
    _RADEON_LISTING=''
    _radeon_fetch_listing 'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.14/' || true
    printf '%s calls=%s\n' \"\$_RADEON_HOST_ANSWERED\" \"\$(cat '$_curl_dir/calls')\"
")
assert_eq "one answer across two attempts is enough" "true calls=2" "$_result"
rm -rf "$_curl_dir"

# 6) wget (only reached without curl): its 8 covers every HTTP error, so it stays "unreachable".
run_fetch_wget() {
    _wget_dir=$(mktemp -d)
    cat > "$_wget_dir/wget" <<STUB
#!/bin/sh
exit $1
STUB
    chmod +x "$_wget_dir/wget"
    PATH="$_wget_dir" "$(command -v bash)" -c "
        . '$_FUNC_FILE'
        _VENV_PY='$_STUB_DIR/fakepy'
        _RADEON_LISTING=''
        _RADEON_HOST_ANSWERED=false
        if _radeon_fetch_listing 'https://repo.radeon.com/rocm/manylinux/rocm-rel-7.14/'; then
            _rc=ok
        else
            _rc=fail
        fi
        echo \"\$_rc answered=\$_RADEON_HOST_ANSWERED\"
    "
    rm -rf "$_wget_dir"
}
assert_eq "no curl, wget HTTP error -> host not recorded as answering" \
    "fail answered=false" "$(run_fetch_wget 8)"
assert_eq "no curl, wget network failure -> host not recorded as answering" \
    "fail answered=false" "$(run_fetch_wget 4)"

# 7) The not-published arm precedes the unreachable arm.
_answered_line=$(grep -n 'elif \[ "\$_RADEON_HOST_ANSWERED" = true \]; then' "$INSTALL_SH" | head -1 | cut -d: -f1)
_unreachable_line=$(grep -n 'Radeon repo unreachable' "$INSTALL_SH" | head -1 | cut -d: -f1)
assert_eq "not-published arm precedes the unreachable arm" "yes" \
    "$([ -n "$_answered_line" ] && [ -n "$_unreachable_line" ] && \
       [ "$_answered_line" -lt "$_unreachable_line" ] && echo yes)"
assert_eq "the not-published message names the release asked for" "yes" \
    "$(grep -q 'repo.radeon.com has no \$_radeon_rel wheels' "$INSTALL_SH" && echo yes)"

rm -f "$_FUNC_FILE"
rm -rf "$_STUB_DIR"

summary
