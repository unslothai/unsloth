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

# All four installers must pin the same uv, or which one a user ends up with depends on which
# script reached the machine first.
_pinned_versions=$(
    grep -hoE '(UV_PINNED_VERSION|_SETUP_UV_PINNED_VERSION)="[0-9.]+"' \
        "$INSTALL_SH" "$SCRIPT_DIR/../../studio/setup.sh"
    grep -hoE '\$UvPinnedVersion += +"[0-9.]+"' \
        "$SCRIPT_DIR/../../install.ps1" "$SCRIPT_DIR/../../studio/setup.ps1"
)
_pinned_distinct=$(printf '%s\n' "$_pinned_versions" | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | sort -u)
_pinned_count=$(printf '%s\n' "$_pinned_distinct" | grep -c .)
if [ "$(printf '%s\n' "$_pinned_versions" | grep -c .)" = "4" ] && [ "$_pinned_count" = "1" ]; then
    ok "all four installers pin the same uv ($_pinned_distinct)"
else
    bad "the pinned uv version disagrees across installers: $(printf '%s' "$_pinned_distinct" | tr '\n' ' ')"
fi

# The pin has to clear every version floor in the tree. Before the pin, astral's endpoint always
# delivered the newest uv, so raising a floor was safe on its own; now a floor above the pin would
# install a uv the same script immediately judges too old. This is the check that catches it.
_floors=$(
    grep -hoE '^UV_MIN_VERSION="[0-9.]+"|^UV_OFFLINE_MIN_VERSION="[0-9.]+"' "$INSTALL_SH"
    grep -hoE '\$UvMinVersion += +"[0-9.]+"' "$SCRIPT_DIR/../../install.ps1"
)
_floor_bad=0
for _floor in $(printf '%s\n' "$_floors" | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?'); do
    # sort -V: the lower of the two sorts first, so the pin must not be it unless they are equal.
    _lowest=$(printf '%s\n%s\n' "$_floor" "$_pinned_distinct" | sort -V | head -1)
    if [ "$_lowest" != "$_floor" ] && [ "$_floor" != "$_pinned_distinct" ]; then
        echo "      floor $_floor is above the pin $_pinned_distinct"
        _floor_bad=$((_floor_bad + 1))
    fi
done
if [ "$_floor_bad" = 0 ]; then
    ok "the pinned uv clears every version floor in the tree"
else
    bad "$_floor_bad version floor(s) sit above the pinned uv"
fi

# astral's installer wrote its own shell-profile line; the pinned path does not, so install.sh's
# own profile write is now the only thing that puts _LOCAL_BIN on a NEW shell's PATH. That guard
# has to read the PATH we inherited: by the time it runs, this process has prepended the directory
# for the uv bootstrap and the venv, so testing the live $PATH answers yes for a login shell that
# would answer no, the profile line never gets written, and `unsloth` is missing from the next
# terminal.
_snapshot_at=$(grep -n '^_UNSLOTH_LOGIN_PATH="\$PATH"' "$INSTALL_SH" | head -1 | cut -d: -f1)
_first_mutation=$(grep -n 'export PATH=' "$INSTALL_SH" | head -1 | cut -d: -f1)
if [ -n "$_snapshot_at" ] && [ -n "$_first_mutation" ] && [ "$_snapshot_at" -lt "$_first_mutation" ]; then
    ok "the login PATH is snapshotted before anything prepends to PATH"
else
    bad "the login PATH snapshot is missing or too late (snapshot=$_snapshot_at first mutation=$_first_mutation)"
fi
if grep -q '_path_has_dir "\$_UNSLOTH_LOGIN_PATH"' "$INSTALL_SH"; then
    ok "the shell-profile guard tests the inherited PATH, not the one we prepended to"
else
    bad "the shell-profile guard tests the inherited PATH, not the one we prepended to"
fi

# ...and having found that a new shell would NOT see the directory, it has to actually persist it.
# A fresh account can have no rc file at all; astral's installer used to create one, the pinned
# path does not, so an empty _SHELL_PROFILE means the next terminal resolves neither unsloth nor
# uv. Run the real block from install.sh rather than grepping it.
# The guard now calls _persist_login_path_dir, so the extract has to carry the function too.
_fn_start=$(grep -n '^_path_has_dir() {' "$INSTALL_SH" | head -1 | cut -d: -f1)
# Through the sentinel that closes the block: the two persistence decisions are one contract
# and the test drives both.
_guard_end=$(grep -n '^# end of the PATH persistence block$' "$INSTALL_SH" | head -1 | cut -d: -f1)
sed -n "${_fn_start},${_guard_end}p" "$INSTALL_SH" > "$WORK/path_guard.sh"
mkdir -p "$WORK/fresh_home/.local/bin"
(
    set +e
    step() { :; }
    HOME="$WORK/fresh_home"; export HOME
    SHELL="/bin/bash"; export SHELL
    unset ZSH_VERSION
    _LOCAL_BIN="$HOME/.local/bin"
    _STUDIO_HOME_REDIRECT="none"
    _UNSLOTH_LOGIN_PATH="/usr/bin:/bin"
    # shellcheck disable=SC1090
    . "$WORK/path_guard.sh"
) >/dev/null 2>&1 || true
if grep -q '\.local/bin' "$WORK/fresh_home/.profile" 2>/dev/null; then
    ok "an account with no rc file still gets ~/.local/bin persisted"
else
    bad "an account with no rc file still gets ~/.local/bin persisted"
fi
# And it must stay idempotent: a second run over the file it just wrote adds nothing.
(
    set +e
    step() { :; }
    HOME="$WORK/fresh_home"; export HOME
    SHELL="/bin/bash"; export SHELL
    unset ZSH_VERSION
    _LOCAL_BIN="$HOME/.local/bin"
    _STUDIO_HOME_REDIRECT="none"
    _UNSLOTH_LOGIN_PATH="/usr/bin:/bin"
    # shellcheck disable=SC1090
    . "$WORK/path_guard.sh"
) >/dev/null 2>&1 || true
if [ "$(grep -c 'Added by Unsloth installer' "$WORK/fresh_home/.profile")" = "1" ]; then
    ok "the created profile is written once, not once per run"
else
    bad "the created profile is written once, not once per run"
fi

# The digest belongs inside the mirror loop. A proxy answering 200 with its own body is a
# successful download by every measure Invoke-WebRequest has, and checking once afterwards spends
# the only attempt on it and never reaches the mirror that would have served the real archive.
for _ps in "$SCRIPT_DIR/../../install.ps1" "$SCRIPT_DIR/../../studio/setup.ps1"; do
    _dl_line=$(grep -n 'if (-not \$downloaded) { return \$false }' "$_ps" | head -1 | cut -d: -f1)
    _hash_line=$(grep -n 'Get-FileHash -LiteralPath \$zip' "$_ps" | head -1 | cut -d: -f1)
    if [ -n "$_dl_line" ] && [ -n "$_hash_line" ] && [ "$_hash_line" -lt "$_dl_line" ]; then
        ok "${_ps##*/} verifies each uv mirror before it stops trying them"
    else
        bad "${_ps##*/} verifies each uv mirror before it stops trying them"
    fi
done

# A configured uv mirror is exclusive: a restricted network sets one because the public hosts are
# unreachable, and download() has no timeout, so trying them first stalls instead of falling back.
for _impl in "$INSTALL_SH" "$SCRIPT_DIR/../../studio/setup.sh"; do
    if grep -q 'UV_INSTALLER_GHE_BASE_URL' "$_impl" && grep -q 'UV_INSTALLER_GITHUB_BASE_URL' "$_impl"; then
        ok "${_impl##*/} honours a configured uv mirror"
    else
        bad "${_impl##*/} honours a configured uv mirror"
    fi
done

# The staging name must be unique per process. Two installers sharing one fixed staging path let
# the loser keep writing through its open descriptor after the winner renamed that inode into
# place, publishing a truncated uv. mktemp in the destination directory is what makes the rename
# a swap of a file nobody else can still be writing.
for _impl in "$INSTALL_SH" "$SCRIPT_DIR/../../studio/setup.sh"; do
    if grep -qE 'mktemp "\$_[a-z]+_dest/\.\$_[a-z]+_exe\.XXXXXX"' "$_impl" \
       && ! grep -q 'unsloth-new' "$_impl"; then
        ok "${_impl##*/} stages the uv copy under a per-process name"
    else
        bad "${_impl##*/} stages the uv copy under a per-process name"
    fi
done

# astral honours UV_DOWNLOAD_URL and its alias INSTALLER_DOWNLOAD_URL ahead of the mirror
# variables, and a host that sets one usually cannot reach the public endpoints, so trying those
# first stalls instead of falling back. All four implementations have to agree on that order.
for _impl in "$INSTALL_SH" "$SCRIPT_DIR/../../studio/setup.sh" \
             "$SCRIPT_DIR/../../install.ps1" "$SCRIPT_DIR/../../studio/setup.ps1"; do
    if grep -q 'UV_DOWNLOAD_URL' "$_impl" && grep -q 'INSTALLER_DOWNLOAD_URL' "$_impl"; then
        ok "${_impl##*/} honours astral's primary download override"
    else
        bad "${_impl##*/} honours astral's primary download override"
    fi
done

# 0755, not the umask default: astral ships these executable for everyone, and a umask of 077
# would otherwise leave uv unusable for other accounts on a shared machine.
for _impl in "$INSTALL_SH" "$SCRIPT_DIR/../../studio/setup.sh"; do
    if grep -q 'chmod 0755' "$_impl" && ! grep -qE 'chmod \+x "\$_[a-z]+_stage"' "$_impl"; then
        ok "${_impl##*/} stages uv with an explicit 0755"
    else
        bad "${_impl##*/} stages uv with an explicit 0755"
    fi
done

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

# A destination that is a symlink must be replaced, not written through. `~/.local/bin/uv ->
# /opt/homebrew/bin/uv` is an ordinary layout, and a plain cp there rewrites Homebrew's binary.
mkdir -p "$WORK/home_link/.local/bin" "$WORK/elsewhere"
printf 'other package manager owns this' > "$WORK/elsewhere/uv"
_link_before=$(sha256sum "$WORK/elsewhere/uv" 2>/dev/null || shasum -a 256 "$WORK/elsewhere/uv")
ln -sf "$WORK/elsewhere/uv" "$WORK/home_link/.local/bin/uv"
ADVERTISED="$FIXTURE_SHA" run_case "$WORK/home_link" > "$WORK/out_link" 2>&1 || true
_link_after=$(sha256sum "$WORK/elsewhere/uv" 2>/dev/null || shasum -a 256 "$WORK/elsewhere/uv")
if [ "$_link_before" = "$_link_after" ]; then
    ok "a symlinked destination is not written through"
else
    bad "the install rewrote the file the destination symlink pointed at"
fi
if [ ! -L "$WORK/home_link/.local/bin/uv" ] && grep -q 'fake' "$WORK/home_link/.local/bin/uv" 2>/dev/null; then
    ok "the symlink itself is replaced by the installed uv"
else
    bad "the symlink itself is replaced by the installed uv"
fi
# The staging file must not survive a run, or the destination collects debris on every upgrade.
if [ -z "$(find "$WORK/home_link/.local/bin" -name '.uv.*' 2>/dev/null)" ]; then
    ok "no staging file is left behind"
else
    bad "no staging file is left behind"
fi

# A binary that cannot execute must decline, not report success. The executable bit says nothing
# about whether the loader a GNU binary asks for exists: a stripped NixOS-derived image reads a
# glibc version from getconf, passes every static check, and then fails on first use with the
# fallback already skipped. Stand in for that with an archive whose uv cannot run.
mkdir -p "$WORK/src_bad/uv-fake-triple"
printf '\177ELF not a real loader target\n' > "$WORK/src_bad/uv-fake-triple/uv"
printf '#!/bin/sh\necho uvx\n' > "$WORK/src_bad/uv-fake-triple/uvx"
tar -czf "$WORK/uv-bad.tar.gz" -C "$WORK/src_bad" uv-fake-triple
if command -v sha256sum >/dev/null 2>&1; then
    BAD_EXEC_SHA=$(sha256sum "$WORK/uv-bad.tar.gz" | awk '{print $1}')
else
    BAD_EXEC_SHA=$(shasum -a 256 "$WORK/uv-bad.tar.gz" | awk '{print $1}')
fi
mkdir -p "$WORK/home_noexec"
(
    set +e
    tauri_log() { :; }
    # shellcheck disable=SC1090
    . "$WORK/uvfns.sh"
    _uv_pinned_asset() { echo "uv-bad.tar.gz $BAD_EXEC_SHA"; }
    download() { cp -f "$WORK/uv-bad.tar.gz" "$2"; }
    HOME="$WORK/home_noexec"; export HOME
    unset UV_INSTALL_DIR UV_UNMANAGED_INSTALL XDG_BIN_HOME XDG_DATA_HOME
    _uv_install_pinned
    echo "rc=$?"
) > "$WORK/out_noexec" 2>&1 || true
if grep -q '^rc=0$' "$WORK/out_noexec"; then
    bad "a uv that cannot execute declines to the fallback"
else
    ok "a uv that cannot execute declines to the fallback"
fi

# And it must not have destroyed the uv the host was already using. The rename publishes over the
# destination, so validating the new binary only after that point would leave a host whose loader
# is missing with neither its old working uv nor a usable new one, while _uv_present_before still
# says one is installed.
mkdir -p "$WORK/home_keep/.local/bin"
printf '#!/bin/sh\necho "uv 0.9.9 (incumbent)"\n' > "$WORK/home_keep/.local/bin/uv"
chmod +x "$WORK/home_keep/.local/bin/uv"
(
    set +e
    tauri_log() { :; }
    # shellcheck disable=SC1090
    . "$WORK/uvfns.sh"
    _uv_pinned_asset() { echo "uv-bad.tar.gz $BAD_EXEC_SHA"; }
    download() { cp -f "$WORK/uv-bad.tar.gz" "$2"; }
    HOME="$WORK/home_keep"; export HOME
    unset UV_INSTALL_DIR UV_UNMANAGED_INSTALL XDG_BIN_HOME XDG_DATA_HOME
    _uv_install_pinned
) >/dev/null 2>&1 || true
if "$WORK/home_keep/.local/bin/uv" 2>/dev/null | grep -q incumbent; then
    ok "a uv that cannot execute never replaces the working one"
else
    bad "a uv that cannot execute replaced the working one"
fi
if [ -z "$(find "$WORK/home_keep/.local/bin" -name '.uv.*' 2>/dev/null)" ]; then
    ok "the rejected staging file is cleaned up"
else
    bad "the rejected staging file is cleaned up"
fi
# uv and uvx ship as a set. When uv is rejected the loop must abandon the whole placement, not
# carry on and publish the pinned uvx beside whatever older uv the host already had, which is a
# pairing we never build or test.
if [ -f "$WORK/home_keep/.local/bin/uvx" ]; then
    bad "a rejected uv must not publish its uvx"
else
    ok "a rejected uv must not publish its uvx"
fi

# The probe runs a binary that was just downloaded, so it has to be bounded on both of the ways
# such a binary can fail to return: reading stdin, and never exiting. Neither may hold an
# unattended install open.
mkdir -p "$WORK/src_hang/uv-fake-triple"
# Reads a line from stdin, so with the installer's console attached it would block forever;
# with </dev/null the read hits EOF at once. Then sleeps past the ceiling, so an unbounded
# wait would hang here instead.
printf '#!/bin/sh\nread _line\nsleep 120\n' > "$WORK/src_hang/uv-fake-triple/uv"
chmod +x "$WORK/src_hang/uv-fake-triple/uv"
printf '#!/bin/sh\necho uvx\n' > "$WORK/src_hang/uv-fake-triple/uvx"
tar -czf "$WORK/uv-hang.tar.gz" -C "$WORK/src_hang" uv-fake-triple
if command -v sha256sum >/dev/null 2>&1; then
    HANG_SHA=$(sha256sum "$WORK/uv-hang.tar.gz" | awk '{print $1}')
else
    HANG_SHA=$(shasum -a 256 "$WORK/uv-hang.tar.gz" | awk '{print $1}')
fi
mkdir -p "$WORK/home_hang"
_HANG_START=$(date +%s)
(
    set +e
    tauri_log() { :; }
    # shellcheck disable=SC1090
    . "$WORK/uvfns.sh"
    _uv_pinned_asset() { echo "uv-hang.tar.gz $HANG_SHA"; }
    download() { cp -f "$WORK/uv-hang.tar.gz" "$2"; }
    HOME="$WORK/home_hang"; export HOME
    unset UV_INSTALL_DIR UV_UNMANAGED_INSTALL XDG_BIN_HOME XDG_DATA_HOME
    _uv_install_pinned
    echo "rc=$?"
) > "$WORK/out_hang" 2>&1 || true
_HANG_ELAPSED=$(( $(date +%s) - _HANG_START ))
# 40s, not 20: the ceiling only applies where `timeout` exists, and a host without it still has
# to come back because stdin is closed. Either way this must not run for two minutes.
if [ "$_HANG_ELAPSED" -lt 40 ] && grep -q '^rc=1$' "$WORK/out_hang"; then
    ok "the executable probe is bounded and declines (${_HANG_ELAPSED}s)"
else
    bad "the executable probe is bounded and declines (${_HANG_ELAPSED}s, $(cat "$WORK/out_hang"))"
fi

# Host matrix. A wrong triple installs a binary that cannot execute, which is worse than not
# installing at all, so every host must either get its own triple or decline to the fallback.
# $4 libc: musl | none (no ldd and no getconf) | a glibc version | rosetta (Darwin only)
# $5 bits: what getconf LONG_BIT reports, so a 32-bit userland on a 64-bit kernel is covered.
_probe_asset() { # $1 fn, $2 os, $3 arch, $4 libc, $5 bits
    (
        set +e
        # Bind before the stubs: inside a function body $2..$5 are the stub's own arguments.
        _pa_os="$2"; _pa_arch="$3"; _pa_libc="$4"; _pa_bits="$5"
        tauri_log() { :; }
        # shellcheck disable=SC1090
        . "$WORK/uvfns.sh"
        uname() { case "${1:-}" in -m) echo "$_pa_arch" ;; *) echo "$_pa_os" ;; esac; }
        ldd() {
            case "$_pa_libc" in
                musl) echo "musl libc (x86_64)" ;;
                none) return 127 ;;
                *) echo "ldd (Ubuntu GLIBC $_pa_libc-0ubuntu1) $_pa_libc" ;;
            esac
        }
        getconf() {
            case "${1:-}" in
                LONG_BIT) echo "$_pa_bits" ;;
                GNU_LIBC_VERSION) [ "$_pa_libc" = none ] && return 1; echo "glibc $_pa_libc" ;;
            esac
        }
        sysctl() { [ "$_pa_libc" = rosetta ] && echo 1; }
        "$1" 2>/dev/null
    )
}
_matrix_bad=0
# "<os> <arch> <libc> <bits> <expected triple, or - to decline>"
while read -r _m_os _m_arch _m_libc _m_bits _m_want; do
    [ -n "$_m_os" ] || continue
    # `|| _m_got=`: a declining host exits non-zero, and set -e would end the run here.
    _m_got=$(_probe_asset _uv_pinned_asset "$_m_os" "$_m_arch" "$_m_libc" "$_m_bits") || _m_got=""
    _m_label="$_m_os/$_m_arch/$_m_libc/$_m_bits"
    if [ "$_m_want" = "-" ]; then
        [ -z "$_m_got" ] || { echo "      $_m_label should decline, got '$_m_got'"; _matrix_bad=$((_matrix_bad + 1)); }
    else
        case "$_m_got" in
            "uv-$_m_want.tar.gz "*) : ;;
            *) echo "      $_m_label expected $_m_want, got '$_m_got'"; _matrix_bad=$((_matrix_bad + 1)) ;;
        esac
    fi
done <<'MATRIX'
Linux x86_64 2.35 64 x86_64-unknown-linux-gnu
Linux amd64 2.35 64 x86_64-unknown-linux-gnu
Linux aarch64 2.35 64 aarch64-unknown-linux-gnu
Linux arm64 2.28 64 aarch64-unknown-linux-gnu
Linux x86_64 2.17 64 x86_64-unknown-linux-gnu
Darwin x86_64 2.35 64 x86_64-apple-darwin
Darwin arm64 2.35 64 aarch64-apple-darwin
Darwin aarch64 2.35 64 aarch64-apple-darwin
Darwin x86_64 rosetta 64 aarch64-apple-darwin
Linux x86_64 musl 64 -
Linux aarch64 musl 64 -
Linux x86_64 none 64 -
Linux aarch64 none 64 -
Linux x86_64 2.12 64 -
Linux aarch64 2.27 64 -
Linux x86_64 2.35 32 -
Linux aarch64 2.35 32 -
Linux armv7l 2.35 64 -
Linux i686 2.35 64 -
Linux ppc64le 2.35 64 -
Linux riscv64 2.35 64 -
Linux s390x 2.35 64 -
Darwin i386 2.35 64 -
FreeBSD x86_64 2.35 64 -
SunOS x86_64 2.35 64 -
MINGW64_NT-10.0 x86_64 2.35 64 -
CYGWIN_NT-10.0 x86_64 2.35 64 -
unknown unknown 2.35 64 -
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


# An interrupted install must not leave the pinned path's temporaries behind: the work
# directory holds a ~40 MB unpacked archive, and the staging file sits inside a directory that
# is on PATH. The helper's own cleanup only runs when it returns normally, so both have to be
# reachable from the signal and exit traps.
_cleanup_body=$(awk '/^_cleanup_install_temporaries\(\) \{/,/^\}/' "$INSTALL_SH")
if printf '%s' "$_cleanup_body" | grep -q '_UIP_WORK' && printf '%s' "$_cleanup_body" | grep -q '_UIP_STAGE'; then
    ok "the pinned uv temporaries are removed by the interrupt cleanup"
else
    bad "the pinned uv temporaries are removed by the interrupt cleanup"
fi
if grep -q '^_UIP_WORK=""' "$INSTALL_SH" && grep -q '_UIP_WORK="\$_uip_work"' "$INSTALL_SH" \
   && grep -q '_UIP_STAGE="\$_uip_stage"' "$INSTALL_SH"; then
    ok "the pinned uv temporaries are published to the trap as they are created"
else
    bad "the pinned uv temporaries are published to the trap as they are created"
fi
# fish sources none of the POSIX rc files, so an `export` line in ~/.profile is a no-op for a
# fish user: the install works in this process and the next session resolves neither uv nor the
# unsloth shim.
mkdir -p "$WORK/fish_home/.local/bin"
(
    set +e
    step() { :; }
    HOME="$WORK/fish_home"; export HOME
    SHELL="/usr/bin/fish"; export SHELL
    unset ZSH_VERSION XDG_CONFIG_HOME
    _LOCAL_BIN="$HOME/.local/bin"
    _STUDIO_HOME_REDIRECT="none"
    _UNSLOTH_LOGIN_PATH="/usr/bin:/bin"
    # shellcheck disable=SC1090
    . "$WORK/path_guard.sh"
) >/dev/null 2>&1 || true
if grep -q 'fish_add_path' "$WORK/fish_home/.config/fish/conf.d/unsloth.fish" 2>/dev/null \
   && [ ! -f "$WORK/fish_home/.profile" ]; then
    ok "a fish login shell gets a fish drop-in, not an ignored ~/.profile"
else
    bad "a fish login shell gets a fish drop-in, not an ignored ~/.profile"
fi

# uv can land somewhere other than ~/.local/bin (UV_INSTALL_DIR and friends outrank it), and
# astral's installer wrote a PATH line for whichever directory it picked. The pinned path
# replaces that installer, so it has to persist its own destination too.
mkdir -p "$WORK/uvdir_home/.local/bin" "$WORK/uvdir_home/opt/uvbin"
: > "$WORK/uvdir_home/.bashrc"
(
    set +e
    step() { :; }
    HOME="$WORK/uvdir_home"; export HOME
    SHELL="/bin/bash"; export SHELL
    unset ZSH_VERSION UV_NO_MODIFY_PATH
    _LOCAL_BIN="$HOME/.local/bin"
    _STUDIO_HOME_REDIRECT="none"
    _UNSLOTH_LOGIN_PATH="/usr/bin:/bin"
    _UNSLOTH_UV_BIN_DIR="$HOME/opt/uvbin"
    # shellcheck disable=SC1090
    . "$WORK/path_guard.sh"
) >/dev/null 2>&1 || true
if grep -qF "$WORK/uvdir_home/opt/uvbin" "$WORK/uvdir_home/.bashrc" 2>/dev/null; then
    ok "a custom uv install directory reaches the next shell"
else
    bad "a custom uv install directory reaches the next shell"
fi
# ...and UV_NO_MODIFY_PATH is astral's opt-out, so it has to be honoured here for the same
# reason it is honoured there.
mkdir -p "$WORK/uvopt_home/.local/bin" "$WORK/uvopt_home/opt/uvbin"
: > "$WORK/uvopt_home/.bashrc"
(
    set +e
    step() { :; }
    HOME="$WORK/uvopt_home"; export HOME
    SHELL="/bin/bash"; export SHELL
    unset ZSH_VERSION
    UV_NO_MODIFY_PATH=1; export UV_NO_MODIFY_PATH
    _LOCAL_BIN="$HOME/.local/bin"
    _STUDIO_HOME_REDIRECT="none"
    _UNSLOTH_LOGIN_PATH="/usr/bin:/bin"
    _UNSLOTH_UV_BIN_DIR="$HOME/opt/uvbin"
    # shellcheck disable=SC1090
    . "$WORK/path_guard.sh"
) >/dev/null 2>&1 || true
if grep -qF "$WORK/uvopt_home/opt/uvbin" "$WORK/uvopt_home/.bashrc" 2>/dev/null; then
    bad "UV_NO_MODIFY_PATH still suppresses the uv directory write"
else
    ok "UV_NO_MODIFY_PATH still suppresses the uv directory write"
fi

# A path with a space is two arguments to fish_add_path and neither of them exists, so the
# drop-in has to quote. The home directory alone is enough to hit this.
mkdir -p "$WORK/fish sp home/.local/bin"
(
    set +e
    step() { :; }
    HOME="$WORK/fish sp home"; export HOME
    SHELL="/usr/bin/fish"; export SHELL
    unset ZSH_VERSION XDG_CONFIG_HOME
    _LOCAL_BIN="$HOME/.local/bin"
    _STUDIO_HOME_REDIRECT="none"
    _UNSLOTH_LOGIN_PATH="/usr/bin:/bin"
    # shellcheck disable=SC1090
    . "$WORK/path_guard.sh"
) >/dev/null 2>&1 || true
_fish_line=$(grep 'fish_add_path' "$WORK/fish sp home/.config/fish/conf.d/unsloth.fish" 2>/dev/null)
if [ "$_fish_line" = "fish_add_path '$WORK/fish sp home/.local/bin'" ]; then
    ok "a fish path with a space is written as one quoted argument"
else
    bad "a fish path with a space is written as one quoted argument ($_fish_line)"
fi

# The rc line is written inside double quotes, so a custom uv directory holding $ or a backtick
# would be expanded by the shell that reads it rather than treated as a path.
mkdir -p "$WORK/uvmeta_home/.local/bin" "$WORK/uvmeta_home/opt/a\$b/bin"
: > "$WORK/uvmeta_home/.bashrc"
(
    set +e
    step() { :; }
    HOME="$WORK/uvmeta_home"; export HOME
    SHELL="/bin/bash"; export SHELL
    unset ZSH_VERSION UV_NO_MODIFY_PATH
    _LOCAL_BIN="$HOME/.local/bin"
    _STUDIO_HOME_REDIRECT="none"
    _UNSLOTH_LOGIN_PATH="/usr/bin:/bin"
    _UNSLOTH_UV_BIN_DIR="$HOME/opt/a\$b/bin"
    # shellcheck disable=SC1090
    . "$WORK/path_guard.sh"
) >/dev/null 2>&1 || true
# Read it back the way a login shell would: source the file and ask what PATH holds.
_uvmeta_path=$(HOME="$WORK/uvmeta_home"; PATH="/usr/bin:/bin"; . "$WORK/uvmeta_home/.bashrc" 2>/dev/null; printf '%s' "$PATH")
case ":$_uvmeta_path:" in
    *":$WORK/uvmeta_home/opt/a\$b/bin:"*)
        ok "a uv directory holding shell metacharacters survives the rc round trip" ;;
    *)
        bad "a uv directory holding shell metacharacters survives the rc round trip" ;;
esac

# Publishing uv and then failing to publish uvx must put the incumbent uv back. The caller
# treats a non-zero return as "fall back to astral's installer", but that installer can be
# unreachable, and a host that had a working pair must not be left with a new uv beside its old
# uvx. Made to fail by pointing uvx's destination at a directory, which no rename can replace.
mkdir -p "$WORK/home_half/.local/bin"
printf '#!/bin/sh\necho "uv 0.9.9 (incumbent)"\n' > "$WORK/home_half/.local/bin/uv"
printf '#!/bin/sh\necho "uvx 0.9.9 (incumbent)"\n' > "$WORK/home_half/.local/bin/uvx"
chmod +x "$WORK/home_half/.local/bin/uv" "$WORK/home_half/.local/bin/uvx"
(
    set +e
    tauri_log() { :; }
    # Fail the second rename only. A directory at the destination would not do it: `mv f d`
    # moves f INTO d and reports success, which is itself worth knowing.
    mv() {
        case "$*" in
            *"/uvx") return 1 ;;
        esac
        command mv "$@"
    }
    # shellcheck disable=SC1090
    . "$WORK/uvfns.sh"
    _uv_pinned_asset() { echo "uv-fake.tar.gz $FIXTURE_SHA"; }
    download() { cp -f "$WORK/uv-fake.tar.gz" "$2"; }
    HOME="$WORK/home_half"; export HOME
    unset UV_INSTALL_DIR UV_UNMANAGED_INSTALL XDG_BIN_HOME XDG_DATA_HOME
    _uv_install_pinned
    echo "rc=$?"
) > "$WORK/out_half" 2>&1 || true
if grep -q '^rc=0$' "$WORK/out_half"; then
    bad "a half-published pair declines to the fallback"
else
    ok "a half-published pair declines to the fallback"
fi
if [ -z "$(find "$WORK/home_half/.local/bin" -maxdepth 1 -name '.uv*' 2>/dev/null)" ]; then
    ok "the half-published staging and undo files are cleaned up"
else
    bad "the half-published staging and undo files are cleaned up"
fi

# studio/setup.sh is run directly for local and Colab setup, and astral's installer used to
# write a profile line for whichever destination it chose. Without one the export dies with that
# shell and every later run reinstalls uv.
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
SETUP_PS1="$SCRIPT_DIR/../../studio/setup.ps1"
_sp_start=$(grep -n '^_setup_persist_uv_path() {' "$SETUP_SH" | head -1 | cut -d: -f1)
_sp_end=$(awk -v s="$_sp_start" 'NR>s && /^\}$/ {print NR; exit}' "$SETUP_SH")
sed -n "${_sp_start},${_sp_end}p" "$SETUP_SH" > "$WORK/setup_path.sh"
mkdir -p "$WORK/setup_home" "$WORK/setup_home/opt/bin"
: > "$WORK/setup_home/.bashrc"
(
    set +e
    HOME="$WORK/setup_home"; export HOME
    SHELL="/bin/bash"; export SHELL
    unset ZSH_VERSION UV_NO_MODIFY_PATH UV_UNMANAGED_INSTALL
    _SETUP_LOGIN_PATH="/usr/bin:/bin"
    # shellcheck disable=SC1090
    . "$WORK/setup_path.sh"
    _setup_persist_uv_path "$HOME/opt/bin"
) >/dev/null 2>&1 || true
if grep -qF "$WORK/setup_home/opt/bin" "$WORK/setup_home/.bashrc" 2>/dev/null; then
    ok "a direct setup.sh run persists the uv destination"
else
    bad "a direct setup.sh run persists the uv destination"
fi
# ...and the same mention-only trap as install.sh: PYTHONPATH holds the text PATH, so only a
# name boundary keeps it from passing for an entry.
mkdir -p "$WORK/setup_mention" "$WORK/setup_mention/opt/bin"
printf 'export PYTHONPATH="%s/opt/bin"\n' "$WORK/setup_mention" > "$WORK/setup_mention/.bashrc"
(
    set +e
    HOME="$WORK/setup_mention"; export HOME
    SHELL="/bin/bash"; export SHELL
    unset ZSH_VERSION UV_NO_MODIFY_PATH UV_UNMANAGED_INSTALL
    _SETUP_LOGIN_PATH="/usr/bin:/bin"
    # shellcheck disable=SC1090
    . "$WORK/setup_path.sh"
    _setup_persist_uv_path "$HOME/opt/bin"
) >/dev/null 2>&1 || true
if grep -q 'export PATH=' "$WORK/setup_mention/.bashrc" 2>/dev/null; then
    ok "setup.sh writes past a non-PATH mention of the destination"
else
    bad "setup.sh writes past a non-PATH mention of the destination"
fi

mkdir -p "$WORK/setup_unmanaged" "$WORK/setup_unmanaged/opt/bin"
: > "$WORK/setup_unmanaged/.bashrc"
(
    set +e
    HOME="$WORK/setup_unmanaged"; export HOME
    SHELL="/bin/bash"; export SHELL
    unset ZSH_VERSION UV_NO_MODIFY_PATH
    UV_UNMANAGED_INSTALL=1; export UV_UNMANAGED_INSTALL
    _SETUP_LOGIN_PATH="/usr/bin:/bin"
    # shellcheck disable=SC1090
    . "$WORK/setup_path.sh"
    _setup_persist_uv_path "$HOME/opt/bin"
) >/dev/null 2>&1 || true
if grep -qF "$WORK/setup_unmanaged/opt/bin" "$WORK/setup_unmanaged/.bashrc" 2>/dev/null; then
    bad "UV_UNMANAGED_INSTALL suppresses the setup.sh profile write"
else
    ok "UV_UNMANAGED_INSTALL suppresses the setup.sh profile write"
fi

# A destination holding a glob character is a pattern inside a case arm, so `/opt/*` counted as
# present whenever the inherited PATH held any /opt entry and the persistence was skipped.
mkdir -p "$WORK/glob_home/.local/bin" "$WORK/glob_home/opt/star"
: > "$WORK/glob_home/.bashrc"
(
    set +e
    step() { :; }
    HOME="$WORK/glob_home"; export HOME
    SHELL="/bin/bash"; export SHELL
    unset ZSH_VERSION UV_NO_MODIFY_PATH UV_UNMANAGED_INSTALL
    _LOCAL_BIN="$HOME/.local/bin"
    _STUDIO_HOME_REDIRECT="none"
    _UNSLOTH_LOGIN_PATH="$HOME/opt/star:/usr/bin"
    _UNSLOTH_UV_BIN_DIR="$HOME/opt/*"
    # shellcheck disable=SC1090
    . "$WORK/path_guard.sh"
) >/dev/null 2>&1 || true
if grep -qF 'opt/*' "$WORK/glob_home/.bashrc" 2>/dev/null; then
    ok "a uv directory holding a glob character is compared literally"
else
    bad "a uv directory holding a glob character is compared literally"
fi

# The fish escapers have to be valid sed. Run them rather than reading them: the setup.sh copy
# reached sed as `s/\/\\/g` and would have killed setup under set -e after uv was published.
for _impl in "$INSTALL_SH" "$SETUP_SH"; do
    _esc_line=$(grep -h 'sed "s/' "$_impl" | grep 'fish\|_quoted=' | head -1)
    _esc_expr=${_esc_line#*| }
    if [ -n "$_esc_expr" ] && printf '%s' "/opt/a b" | eval "${_esc_expr%\)}" >/dev/null 2>&1; then
        ok "${_impl##*/} escapes fish paths with a valid sed expression"
    else
        bad "${_impl##*/} escapes fish paths with a valid sed expression"
    fi
done

# A commented-out old export is not an active PATH entry, and neither is a directory that merely
# starts with ours. Taking either for one leaves the next shell unable to resolve uv.
for _case in comment prefix; do
    _ch="$WORK/entry_$_case"
    mkdir -p "$_ch/.local/bin" "$_ch/opt/uv"
    if [ "$_case" = comment ]; then
        printf '# export PATH="%s/opt/uv:$PATH"\n' "$_ch" > "$_ch/.bashrc"
    else
        printf 'export PATH="%s/opt/uv-old:$PATH"\n' "$_ch" > "$_ch/.bashrc"
    fi
    (
        set +e
        step() { :; }
        HOME="$_ch"; export HOME
        SHELL="/bin/bash"; export SHELL
        unset ZSH_VERSION UV_NO_MODIFY_PATH UV_UNMANAGED_INSTALL
        _LOCAL_BIN="$HOME/.local/bin"
        _STUDIO_HOME_REDIRECT="none"
        _UNSLOTH_LOGIN_PATH="/usr/bin:/bin"
        _UNSLOTH_UV_BIN_DIR="$HOME/opt/uv"
        # shellcheck disable=SC1090
        . "$WORK/path_guard.sh"
    ) >/dev/null 2>&1 || true
    # Count only ACTIVE lines naming the uv directory exactly: the ~/.local/bin line is written
    # too and would mask the answer.
    # `|| _n=0`: grep -c exits non-zero on no match and set -e would take the suite down.
    _n=$(grep -v '^[[:space:]]*#' "$_ch/.bashrc" | grep -cE "(^|[^[:alnum:]_.~/-])$_ch/opt/uv([^[:alnum:]_.~/-]|$)") || _n=0
    if [ "$_n" = "1" ]; then
        ok "an inactive entry ($_case) does not suppress the uv PATH write"
    else
        bad "an inactive entry ($_case) does not suppress the uv PATH write (active lines: $_n)"
    fi
done
# ...and a genuinely active entry still suppresses it, so the write stays idempotent.
_ch="$WORK/entry_active"
mkdir -p "$_ch/.local/bin" "$_ch/opt/uv"
printf 'export PATH="%s/opt/uv:$PATH"\n' "$_ch" > "$_ch/.bashrc"
(
    set +e
    step() { :; }
    HOME="$_ch"; export HOME
    SHELL="/bin/bash"; export SHELL
    unset ZSH_VERSION UV_NO_MODIFY_PATH UV_UNMANAGED_INSTALL
    _LOCAL_BIN="$HOME/.local/bin"
    _STUDIO_HOME_REDIRECT="none"
    _UNSLOTH_LOGIN_PATH="/usr/bin:/bin"
    _UNSLOTH_UV_BIN_DIR="$HOME/opt/uv"
    # shellcheck disable=SC1090
    . "$WORK/path_guard.sh"
) >/dev/null 2>&1 || true
_n=$(grep -v '^[[:space:]]*#' "$_ch/.bashrc" | grep -cE "(^|[^[:alnum:]_.~/-])$_ch/opt/uv([^[:alnum:]_.~/-]|$)") || _n=0
if [ "$_n" = "1" ]; then
    ok "an active entry still suppresses the uv PATH write"
else
    bad "an active entry still suppresses the uv PATH write (active lines: $_n)"
fi

# Both installers put the pinned destination back in front of PATH after the ~/.local/bin
# prepend that follows them, or a stale uv there shadows the one that was just verified.
if grep -q 'export PATH="\$_UNSLOTH_UV_BIN_DIR:\$PATH"' "$INSTALL_SH"; then
    ok "install.sh keeps the pinned uv destination ahead of ~/.local/bin"
else
    bad "install.sh keeps the pinned uv destination ahead of ~/.local/bin"
fi
if grep -q '\[ "\$_SETUP_UV_PINNED_OK" = true \] || export PATH="\$HOME/.local/bin:\$PATH"' "$SETUP_SH"; then
    ok "setup.sh does not prepend ~/.local/bin over a pinned destination"
else
    bad "setup.sh does not prepend ~/.local/bin over a pinned destination"
fi

# The NSIS hooks run before the user can cancel and $INSTDIR can be a directory they chose, so
# the tidy-up only applies where our own executable already is.
_hooks="$SCRIPT_DIR/../../studio/src-tauri/windows/hooks.nsh"
_h_gates=$(grep -c 'FileExists} "$INSTDIR\\${MAINBINARYNAME}.exe"' "$_hooks") || _h_gates=0
_h_deletes=$(grep -c 'Delete "$INSTDIR\\install.sh"' "$_hooks") || _h_deletes=0
if [ "$_h_gates" = "2" ] && [ "$_h_deletes" = "2" ]; then
    ok "the NSIS hooks only tidy a directory that already holds an Unsloth install"
else
    bad "the NSIS hooks only tidy a directory that already holds an Unsloth install"
fi

# astral's installer wired EVERY startup file it knew: ~/.profile, each bash file that exists,
# zsh under ZDOTDIR, and a fish drop-in. Writing only the one for the current shell would leave
# a bash user whose .bash_profile does not source .bashrc without uv on PATH, so the replacement
# has to cover the same set, once each.
_ph="$WORK/parity_home"
mkdir -p "$_ph/.local/bin" "$_ph/opt/uv"
: > "$_ph/.bashrc"; : > "$_ph/.bash_profile"; : > "$_ph/.zshrc"
for _pass in 1 2; do
    (
        set +e
        step() { :; }
        HOME="$_ph"; export HOME
        SHELL="/bin/bash"; export SHELL
        unset ZSH_VERSION ZDOTDIR UV_NO_MODIFY_PATH UV_UNMANAGED_INSTALL
        _LOCAL_BIN="$HOME/.local/bin"
        _STUDIO_HOME_REDIRECT="none"
        _UNSLOTH_LOGIN_PATH="/usr/bin:/bin"
        _UNSLOTH_UV_BIN_DIR="$HOME/opt/uv"
        # shellcheck disable=SC1090
        . "$WORK/path_guard.sh"
    ) >/dev/null 2>&1 || true
done
_missing=""
for _f in .profile .bashrc .bash_profile .zshrc .config/fish/conf.d/unsloth.fish; do
    _n=$(grep -c "opt/uv" "$_ph/$_f" 2>/dev/null) || _n=0
    [ "$_n" = "1" ] || _missing="$_missing $_f=$_n"
done
# .bash_login and .zshenv did not exist, so they must not have been created.
[ -f "$_ph/.bash_login" ] && _missing="$_missing .bash_login=created"
[ -f "$_ph/.zshenv" ] && _missing="$_missing .zshenv=created"
if [ -z "$_missing" ]; then
    ok "the uv PATH entry reaches every startup file astral wired, once each"
else
    bad "the uv PATH entry reaches every startup file astral wired, once each ($_missing)"
fi

# fish reads none of the POSIX files, so its drop-in is the only thing that puts uv on a fish
# user's PATH. A different directory already in that file must not pass for this one.
_fh="$WORK/fish_entry_home"
mkdir -p "$_fh/.config/fish/conf.d" "$_fh/.local/bin" "$_fh/opt/uv"
printf "# Added by Unsloth installer\nfish_add_path '%s/opt/uv-old'\n" "$_fh" > "$_fh/.config/fish/conf.d/unsloth.fish"
(
    set +e
    step() { :; }
    HOME="$_fh"; export HOME
    SHELL="/bin/bash"; export SHELL
    unset ZSH_VERSION ZDOTDIR UV_NO_MODIFY_PATH UV_UNMANAGED_INSTALL
    _LOCAL_BIN="$HOME/.local/bin"
    _STUDIO_HOME_REDIRECT="none"
    _UNSLOTH_LOGIN_PATH="/usr/bin:/bin"
    _UNSLOTH_UV_BIN_DIR="$HOME/opt/uv"
    # shellcheck disable=SC1090
    . "$WORK/path_guard.sh"
) >/dev/null 2>&1 || true
if grep -qxF "fish_add_path '$_fh/opt/uv'" "$_fh/.config/fish/conf.d/unsloth.fish"; then
    ok "a different fish entry does not suppress this one"
else
    bad "a different fish entry does not suppress this one"
fi

# A profile that merely NAMES the directory is not a profile that puts it on PATH:
# `UV_CACHE=$HOME/.local/bin` must not suppress the export, or the next shell finds no uv.
_mh="$WORK/mention_home"
mkdir -p "$_mh/.local/bin"
# PYTHONPATH is the trap for a naive "does the line mention a path" filter: it holds the text
# PATH, so only a name boundary keeps it out.
printf 'UV_CACHE="$HOME/.local/bin"\nexport PYTHONPATH="$HOME/.local/bin"\n' > "$_mh/.profile"
(
    set +e
    step() { :; }
    HOME="$_mh"; export HOME
    SHELL="/bin/bash"; export SHELL
    unset ZSH_VERSION
    _LOCAL_BIN="$HOME/.local/bin"
    _STUDIO_HOME_REDIRECT="none"
    _UNSLOTH_LOGIN_PATH="/usr/bin:/bin"
    # shellcheck disable=SC1090
    . "$WORK/path_guard.sh"
) >/dev/null 2>&1 || true
if grep -q 'export PATH=' "$_mh/.profile"; then
    ok "a non-PATH mention of the directory does not suppress the export"
else
    bad "a non-PATH mention of the directory does not suppress the export"
fi

# A launcher on a UNC share is a remote script to PowerShell, and RemoteSigned refuses an
# unsigned one, so a roaming profile would get a shortcut that exits without starting Studio.
# A mapped drive is the same share and the same zone, so it must take the same branch.
_ps1="$SCRIPT_DIR/../../install.ps1"
if grep -q '\$launcherIsRemote = \$launcherPs1 -like "\\\\\*"' "$_ps1" \
   && grep -q "DriveType -eq 'Network'" "$_ps1" \
   && grep -q '"-NoProfile -ExecutionPolicy Bypass -File `"$launcherPs1`""' "$_ps1"; then
    ok "a UNC or mapped-drive launcher gets a policy that can actually load it"
else
    bad "a UNC or mapped-drive launcher gets a policy that can actually load it"
fi

# The DEFAULT install puts uv in ~/.local/bin, so the all-profile write must not be gated on the
# destination differing from it: gating there left every ordinary machine with the single-file
# write, which is the case that matters most.
_dh="$WORK/default_home"
mkdir -p "$_dh/.local/bin"
: > "$_dh/.bashrc"; : > "$_dh/.bash_profile"
for _pass in 1 2; do
    (
        set +e
        step() { :; }
        HOME="$_dh"; export HOME
        SHELL="/bin/bash"; export SHELL
        unset ZSH_VERSION ZDOTDIR UV_NO_MODIFY_PATH UV_UNMANAGED_INSTALL
        _LOCAL_BIN="$HOME/.local/bin"
        _STUDIO_HOME_REDIRECT="none"
        _UNSLOTH_LOGIN_PATH="/usr/bin:/bin"
        _UNSLOTH_UV_BIN_DIR="$HOME/.local/bin"
        # shellcheck disable=SC1090
        . "$WORK/path_guard.sh"
    ) >/dev/null 2>&1 || true
done
_dmiss=""
for _f in .profile .bashrc .bash_profile .config/fish/conf.d/unsloth.fish; do
    _n=$(grep -c "local/bin" "$_dh/$_f" 2>/dev/null) || _n=0
    [ "$_n" = "1" ] || _dmiss="$_dmiss $_f=$_n"
done
if [ -z "$_dmiss" ]; then
    ok "a default install wires every startup file, once each"
else
    bad "a default install wires every startup file, once each ($_dmiss)"
fi

# studio/setup.ps1 replaced astral's installer, so a failed pinned install needs somewhere to go
# or the whole setup silently drops to pip for torch and everything after it.
# The flag, not the return value: Invoke-SetupCommand hands back $LASTEXITCODE, so reading the
# function's $true through it always said "failed" and ran the fallback every time.
if grep -q 'if (-not $script:UvPinnedInstalled -and (Get-Command winget' "$SETUP_PS1" \
   && grep -q '$script:UvPinnedInstalled = $true' "$SETUP_PS1"; then
    ok "setup.ps1 still has a uv fallback when the pinned install fails"
else
    bad "setup.ps1 still has a uv fallback when the pinned install fails"
fi

# A directory named uv at the destination must not look like a published binary: `mv f d` moves
# into it and reports success, and a searchable directory passes -x.
mkdir -p "$WORK/home_dir_target/.local/bin/uv"
(
    set +e
    tauri_log() { :; }
    # shellcheck disable=SC1090
    . "$WORK/uvfns.sh"
    _uv_pinned_asset() { echo "uv-fake.tar.gz $FIXTURE_SHA"; }
    download() { cp -f "$WORK/uv-fake.tar.gz" "$2"; }
    HOME="$WORK/home_dir_target"; export HOME
    unset UV_INSTALL_DIR UV_UNMANAGED_INSTALL XDG_BIN_HOME XDG_DATA_HOME
    _uv_install_pinned
    echo "rc=$?"
) > "$WORK/out_dir_target" 2>&1 || true
if grep -q '^rc=0$' "$WORK/out_dir_target"; then
    bad "a directory at the destination declines to the fallback"
else
    ok "a directory at the destination declines to the fallback"
fi

# A destination holding an ERE metacharacter must still match itself on the next run, or every
# reinstall appends another PATH block to every profile.
_mh="$WORK/meta_home"
mkdir -p "$_mh/.local/bin" "$_mh/opt/a+b(c)"
: > "$_mh/.bashrc"
for _pass in 1 2; do
    (
        set +e
        step() { :; }
        HOME="$_mh"; export HOME
        SHELL="/bin/bash"; export SHELL
        unset ZSH_VERSION ZDOTDIR UV_NO_MODIFY_PATH UV_UNMANAGED_INSTALL
        _LOCAL_BIN="$HOME/.local/bin"
        _STUDIO_HOME_REDIRECT="none"
        _UNSLOTH_LOGIN_PATH="/usr/bin:/bin"
        _UNSLOTH_UV_BIN_DIR="$HOME/opt/a+b(c)"
        # shellcheck disable=SC1090
        . "$WORK/path_guard.sh"
    ) >/dev/null 2>&1 || true
done
_n=$(grep -cF 'a+b(c)' "$_mh/.bashrc") || _n=0
if [ "$_n" = "1" ]; then
    ok "a destination holding regex metacharacters is written once, not once per run"
else
    bad "a destination holding regex metacharacters is written once, not once per run ($_n)"
fi

echo
echo "passed: $PASS  failed: $FAIL"
[ "$FAIL" -eq 0 ]
