#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Guards install.sh's uv bootstrap.
#
# It used to download astral's install.sh to a temp file, run it and delete the file, which
# is shape for shape what a dropper does. It now fetches the pinned release archive and
# verifies a hardcoded SHA-256 first, the move install.ps1 already made.
#
# The fallback must stay: musl, armv7 and hosts without a digest tool keep the old path
# rather than risk a wrong triple. These tests pin the digest check (a mismatched archive
# installs nothing), the extraction, and the fallback being reachable but not primary.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

ok()  { echo "  PASS: $1"; PASS=$((PASS + 1)); }
bad() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# ── source contract ──
echo "=== source contract ==="

if grep -q '_uv_install_pinned' "$INSTALL_SH"; then
    ok "install.sh has a pinned-release uv path"
else
    bad "install.sh has a pinned-release uv path"
fi

# The pinned attempt must precede the fallback, or the fallback is what actually runs.
_pinned_at=$(grep -n 'if _uv_install_pinned; then' "$INSTALL_SH" | head -1 | cut -d: -f1)
_fallback_at=$(grep -n 'download "https://astral.sh/uv/install.sh"' "$INSTALL_SH" | head -1 | cut -d: -f1)
if [ -n "$_pinned_at" ] && [ -n "$_fallback_at" ] && [ "$_pinned_at" -lt "$_fallback_at" ]; then
    ok "the pinned path is tried before the astral fallback"
else
    bad "the pinned path is tried before the astral fallback (pinned=$_pinned_at fallback=$_fallback_at)"
fi

# A truncated digest would silently never match and route every host to the fallback.
_bad_digests=$(grep -oE 'uv-[a-z0-9_]+-[a-z0-9.-]+\.tar\.gz [0-9a-f]*' "$INSTALL_SH" \
    | awk '{ if (length($2) != 64) print }' | wc -l | tr -d ' ')
if [ "$_bad_digests" = "0" ]; then
    ok "every pinned archive digest is a full sha256"
else
    bad "every pinned archive digest is a full sha256 ($_bad_digests malformed)"
fi

# One version constant, quoted into every URL.
if grep -q '^UV_PINNED_VERSION="' "$INSTALL_SH"; then
    ok "the pinned uv version is a single constant"
else
    bad "the pinned uv version is a single constant"
fi

# astral's destination priority puts XDG_DATA_HOME/../bin between XDG_BIN_HOME and the home
# default. An implementation that skips that tier drops uv under ~/.local/bin on a host that
# configured an XDG location, where no later shell looks for it.
for _impl in "$INSTALL_SH" "$SCRIPT_DIR/../../studio/setup.sh" \
             "$SCRIPT_DIR/../../install.ps1" "$SCRIPT_DIR/../../studio/setup.ps1"; do
    if grep -q 'XDG_DATA_HOME' "$_impl"; then
        ok "${_impl##*/} honours the XDG_DATA_HOME destination tier"
    else
        bad "${_impl##*/} honours the XDG_DATA_HOME destination tier"
    fi
done

# ── behaviour ──
echo "=== behaviour ==="

# Drive the helper block from install.sh with a stubbed downloader: offline, sandboxed.
awk '/^# ── uv from a pinned release ──$/,/^if ! command -v uv /' "$INSTALL_SH" \
    | sed '$d' > "$WORK/uvfns.sh"

# Stand-in for the real archive: same uv-<triple>/{uv,uvx} layout.
mkdir -p "$WORK/src/uv-fake-triple"
printf '#!/bin/sh\necho "uv 0.12.1 (fake)"\n' > "$WORK/src/uv-fake-triple/uv"
printf '#!/bin/sh\necho "uvx"\n' > "$WORK/src/uv-fake-triple/uvx"
tar -czf "$WORK/uv-fake.tar.gz" -C "$WORK/src" uv-fake-triple
if command -v sha256sum >/dev/null 2>&1; then
    FIXTURE_SHA=$(sha256sum "$WORK/uv-fake.tar.gz" | awk '{print $1}')
else
    FIXTURE_SHA=$(shasum -a 256 "$WORK/uv-fake.tar.gz" | awk '{print $1}')
fi

run_case() {
    # $ADVERTISED = the digest the pin claims, $1 = HOME for the run
    _rc_home="$1"
    mkdir -p "$_rc_home"
    (
        set +e
        tauri_log() { :; }
        # shellcheck disable=SC1090
        . "$WORK/uvfns.sh"
        # Stub the host lookup and the transport, not the installer: the point under test
        # is the verify/extract/place path.
        _uv_pinned_asset() { echo "uv-fake.tar.gz $ADVERTISED"; }
        download() { cp -f "$WORK/uv-fake.tar.gz" "$2"; }
        HOME="$_rc_home"
        unset UV_INSTALL_DIR UV_UNMANAGED_INSTALL XDG_BIN_HOME
        if [ -n "${CASE_XDG_DATA_HOME:-}" ]; then
            export XDG_DATA_HOME="$CASE_XDG_DATA_HOME"
        else
            unset XDG_DATA_HOME
        fi
        _uv_install_pinned
        echo "rc=$?"
    )
}

ADVERTISED="$FIXTURE_SHA" run_case "$WORK/home_ok" > "$WORK/out_ok" 2>&1 || true
if grep -q '^rc=0$' "$WORK/out_ok" && [ -x "$WORK/home_ok/.local/bin/uv" ]; then
    ok "a matching digest installs uv into the default destination"
else
    bad "a matching digest installs uv into the default destination"
    sed 's/^/      /' "$WORK/out_ok"
fi

if [ -x "$WORK/home_ok/.local/bin/uvx" ]; then
    ok "uvx is installed alongside uv"
else
    bad "uvx is installed alongside uv"
fi

# XDG_DATA_HOME/../bin wins over the home default, as it does for astral's installer.
ADVERTISED="$FIXTURE_SHA" CASE_XDG_DATA_HOME="$WORK/home_xdg/share" \
    run_case "$WORK/home_xdg" > "$WORK/out_xdg" 2>&1 || true
if [ -x "$WORK/home_xdg/bin/uv" ] && [ ! -e "$WORK/home_xdg/.local/bin/uv" ]; then
    ok "XDG_DATA_HOME redirects the install away from the home default"
else
    bad "XDG_DATA_HOME redirects the install away from the home default"
    sed 's/^/      /' "$WORK/out_xdg"
fi

ADVERTISED="0000000000000000000000000000000000000000000000000000000000000000" \
    run_case "$WORK/home_bad" > "$WORK/out_bad" 2>&1 || true
if grep -q '^rc=0$' "$WORK/out_bad"; then
    bad "a mismatched digest is rejected"
else
    ok "a mismatched digest is rejected"
fi
if [ -e "$WORK/home_bad/.local/bin/uv" ]; then
    bad "a rejected archive installs nothing"
else
    ok "a rejected archive installs nothing"
fi

# Repeat application. The installer is re-run on every upgrade and every repair, so the second
# and third pass over the same HOME must land on the same tree, not accumulate or half-replace.
_sig() { printf '%s|%s' "$(cd "$1" && find . -type f | LC_ALL=C sort | tr '\n' ' ')" \
                        "$(cat "$1/.local/bin/uv" 2>/dev/null)"; }
_idem_ok=1
_idem_sig=""
for _i in 1 2 3; do
    ADVERTISED="$FIXTURE_SHA" run_case "$WORK/home_idem" > "$WORK/out_idem_$_i" 2>&1 || true
    grep -q '^rc=0$' "$WORK/out_idem_$_i" || _idem_ok=0
    _s=$(_sig "$WORK/home_idem")
    [ -z "$_idem_sig" ] && _idem_sig="$_s"
    [ "$_s" = "$_idem_sig" ] || _idem_ok=0
done
if [ "$_idem_ok" = 1 ]; then
    ok "runs 1..3 over the same HOME leave an identical tree"
else
    bad "repeat application is not idempotent"
    sed 's/^/      /' "$WORK/out_idem_3"
fi

# A stale uv from an older install is replaced in place. Two copies under one destination would
# leave PATH order deciding which one runs.
printf 'stale binary' > "$WORK/home_idem/.local/bin/uv"
ADVERTISED="$FIXTURE_SHA" run_case "$WORK/home_idem" > "$WORK/out_replace" 2>&1 || true
if grep -q 'fake' "$WORK/home_idem/.local/bin/uv" 2>/dev/null \
   && [ "$(cd "$WORK/home_idem" && find . -name uv -type f | wc -l | tr -d ' ')" = 1 ]; then
    ok "an existing uv at the destination is replaced in place"
else
    bad "an existing uv at the destination is replaced in place"
fi

# Host matrix. A wrong triple installs a binary that cannot execute, which is worse than not
# installing at all, so every host must either get its own triple or decline to the fallback.
_probe_asset() { # $1 fn, $2 os, $3 arch, $4 libc
    (
        set +e
        # Bind before the stubs: inside a function body $2/$3/$4 are the stub's own arguments.
        _pa_os="$2"; _pa_arch="$3"; _pa_libc="$4"
        tauri_log() { :; }
        # shellcheck disable=SC1090
        . "$WORK/uvfns.sh"
        uname() { case "${1:-}" in -m) echo "$_pa_arch" ;; *) echo "$_pa_os" ;; esac; }
        ldd() { if [ "$_pa_libc" = musl ]; then echo "musl libc 1.2.4"; else echo "ldd (GNU libc) 2.35"; fi; }
        "$1" 2>/dev/null
    )
}
_matrix_bad=0
# "<os> <arch> <libc> <expected triple, or - to decline>"
while read -r _m_os _m_arch _m_libc _m_want; do
    [ -n "$_m_os" ] || continue
    # `|| _m_got=`: a declining host exits non-zero, and set -e would end the run here.
    _m_got=$(_probe_asset _uv_pinned_asset "$_m_os" "$_m_arch" "$_m_libc") || _m_got=""
    if [ "$_m_want" = "-" ]; then
        [ -z "$_m_got" ] || { echo "      $_m_os/$_m_arch/$_m_libc should decline, got '$_m_got'"; _matrix_bad=$((_matrix_bad + 1)); }
    else
        case "$_m_got" in
            "uv-$_m_want.tar.gz "*) : ;;
            *) echo "      $_m_os/$_m_arch/$_m_libc expected $_m_want, got '$_m_got'"; _matrix_bad=$((_matrix_bad + 1)) ;;
        esac
    fi
done <<'MATRIX'
Linux x86_64 gnu x86_64-unknown-linux-gnu
Linux amd64 gnu x86_64-unknown-linux-gnu
Linux aarch64 gnu aarch64-unknown-linux-gnu
Linux arm64 gnu aarch64-unknown-linux-gnu
Darwin x86_64 gnu x86_64-apple-darwin
Darwin arm64 gnu aarch64-apple-darwin
Darwin aarch64 gnu aarch64-apple-darwin
Linux x86_64 musl -
Linux aarch64 musl -
Linux armv7l gnu -
Linux i686 gnu -
Linux ppc64le gnu -
Linux riscv64 gnu -
Linux s390x gnu -
Darwin i386 gnu -
FreeBSD x86_64 gnu -
SunOS x86_64 gnu -
MINGW64_NT-10.0 x86_64 gnu -
CYGWIN_NT-10.0 x86_64 gnu -
unknown unknown gnu -
MATRIX
if [ "$_matrix_bad" = 0 ]; then
    ok "every host either gets its own triple or declines to the fallback"
else
    bad "the host matrix has $_matrix_bad wrong outcomes"
fi

# An unpinned host must decline, so the caller falls back instead of installing nothing.
(
    set +e
    tauri_log() { :; }
    # shellcheck disable=SC1090
    . "$WORK/uvfns.sh"
    uname() { if [ "$1" = "-s" ]; then echo Linux; else echo sparc64; fi; }
    _uv_pinned_asset >/dev/null 2>&1
    echo "rc=$?"
) > "$WORK/out_arch" 2>&1 || true
if grep -q '^rc=0$' "$WORK/out_arch"; then
    bad "an unpinned architecture declines so the fallback runs"
else
    ok "an unpinned architecture declines so the fallback runs"
fi

echo
echo "passed: $PASS  failed: $FAIL"
[ "$FAIL" -eq 0 ]
