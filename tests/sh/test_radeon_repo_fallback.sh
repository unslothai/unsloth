#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# _radeon_fetch_listing tells its caller WHY it came back empty, so the fallback to the
# pytorch.org index can say "AMD publishes no wheels for this release" instead of implying the
# repo is down. AMD only ever published some releases on repo.radeon.com (nothing past
# rocm-rel-7.2.4 as of 2026-09), so on a newer host the miss is the normal case, and calling it
# "unavailable" is what sent the reporters of #7264 and #10657 looking for a broken host.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"

_FUNC_FILE=$(mktemp)
sed -n '/^_radeon_fetch_listing()/,/^}/p' "$INSTALL_SH" > "$_FUNC_FILE"
[ -s "$_FUNC_FILE" ] || { echo "  FAIL: could not extract _radeon_fetch_listing"; exit 1; }

# A fake interpreter, so the python tag does not depend on the runner's python.
_STUB_DIR=$(mktemp -d)
cat > "$_STUB_DIR/fakepy" <<'STUB'
#!/bin/sh
echo cp312
STUB
chmod +x "$_STUB_DIR/fakepy"

# $1 is the exit code the stub curl reports; it writes nothing, as a failed fetch does not.
run_fetch() {
    _curl_dir=$(mktemp -d)
    cat > "$_curl_dir/curl" <<STUB
#!/bin/sh
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

# 1) curl -f exits 22 on any HTTP >= 400: the host answered, it just has no such release.
assert_eq "HTTP 404 -> fetch fails, host recorded as answering" \
    "fail answered=true" "$(run_fetch 22)"

# 2) Connection refused (7). Nothing answered, so the caller must keep saying "unreachable".
assert_eq "connection refused -> fetch fails, host not recorded as answering" \
    "fail answered=false" "$(run_fetch 7)"

# 3) Timeout (28), the other common transport failure, is likewise not an answer.
assert_eq "timeout -> fetch fails, host not recorded as answering" \
    "fail answered=false" "$(run_fetch 28)"

# 4) A listing that comes back is still a plain success, with the flag untouched.
_curl_dir=$(mktemp -d)
cat > "$_curl_dir/curl" <<'STUB'
#!/bin/sh
echo '<a href="torch-2.11.0%2Brocm7.2.4-cp312-cp312-linux_x86_64.whl">torch</a>'
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

# 5) The flag is sticky across the caller's second attempt on the shorter X.Y path: a 404 on
# X.Y.Z followed by a transport failure on X.Y still means the host is up. Two real calls in
# one shell, as install.sh makes them, with a stub that answers differently each time.
_curl_dir=$(mktemp -d)
cat > "$_curl_dir/curl" <<STUB
#!/bin/sh
_n=\$(cat "$_curl_dir/calls" 2>/dev/null || echo 0)
echo \$((_n + 1)) > "$_curl_dir/calls"
[ "\$_n" = 0 ] && exit 22
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

# 6) The wget half. curl is checked first, so this arm is only reachable on a host without
# it: PATH holds the stub wget and nothing else. wget's 8 is its "server error response".
run_fetch_wget() {
    _wget_dir=$(mktemp -d)
    cat > "$_wget_dir/wget" <<STUB
#!/bin/sh
exit $1
STUB
    chmod +x "$_wget_dir/wget"
    # bash by absolute path: PATH holds only the stub, so the name would not resolve.
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
assert_eq "no curl, wget HTTP error -> host recorded as answering" \
    "fail answered=true" "$(run_fetch_wget 8)"
assert_eq "no curl, wget network failure -> host not recorded as answering" \
    "fail answered=false" "$(run_fetch_wget 4)"

# 7) Structure at the call site: the not-published arm is gated on the flag and comes before
# the unreachable arm, so a reachable host never reads as down.
_answered_line=$(grep -n 'elif \[ "\$_RADEON_HOST_ANSWERED" = true \]; then' "$INSTALL_SH" | head -1 | cut -d: -f1)
_unreachable_line=$(grep -n 'Radeon repo unreachable' "$INSTALL_SH" | head -1 | cut -d: -f1)
assert_eq "not-published arm precedes the unreachable arm" "yes" \
    "$([ -n "$_answered_line" ] && [ -n "$_unreachable_line" ] && \
       [ "$_answered_line" -lt "$_unreachable_line" ] && echo yes)"
assert_eq "the not-published message names the release asked for" "yes" \
    "$(grep -q 'repo.radeon.com publishes no \$_radeon_rel wheels' "$INSTALL_SH" && echo yes)"

rm -f "$_FUNC_FILE"
rm -rf "$_STUB_DIR"

summary
