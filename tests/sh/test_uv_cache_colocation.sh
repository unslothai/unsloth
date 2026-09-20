#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Guards the uv cache co-location in install.sh.
#
# uv's cache defaulted to $HOME/.cache/uv while STUDIO_HOME can be pointed anywhere with
# UNSLOTH_STUDIO_HOME. uv hardlinks wheels out of its cache into the venv when both are on
# one filesystem and COPIES when they are not, so every redirected install paid twice the
# disk and left several GB on the drive the user had deliberately moved off (an SD card or
# second disk being the usual reason to redirect at all).
#
# Measured with torch 2.11.0+cpu: co-located, a 749 MB cache and a 748 MB venv occupy 755 MB
# between them (st_nlink 2 on the shared objects); across a boundary they are duplicated.
#
# The contract:
#   * unset UV_CACHE_DIR  -> $STUDIO_HOME/cache/uv, exported, directory created
#   * caller-set          -> left exactly as-is
#   * follows STUDIO_HOME wherever UNSLOTH_STUDIO_HOME puts it
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
# Lift the block out of install.sh so the real code is what runs here.
_FN_FILE=$(mktemp)
_TMP=$(mktemp -d)
trap 'rm -rf "$_FN_FILE" "$_TMP"' EXIT
# The helper first: the block asks it whether the cache is usable, and leaving it out does not
# fail loudly. `command not found` exits 127, `if !` reads that as "not writable", and every
# case quietly reports an unset UV_CACHE_DIR.
awk '/^_uv_cache_root_is_writable\(\) \{$/,/^\}$/' "$INSTALL_SH" > "$_FN_FILE"
awk '/^# Keep uv.s cache on the same filesystem as the venv it fills\.$/,/^fi$/' \
    "$INSTALL_SH" >> "$_FN_FILE"

if ! grep -q 'UV_CACHE_DIR="\$STUDIO_HOME/cache/uv"' "$_FN_FILE"; then
    echo "FAIL: could not extract the UV_CACHE_DIR block from install.sh"
    exit 1
fi
if ! grep -q '^_uv_cache_root_is_writable() {' "$_FN_FILE"; then
    echo "FAIL: could not extract _uv_cache_root_is_writable from install.sh"
    exit 1
fi

_SH="${BASH:-/bin/bash}"
_run() {  # $1 = STUDIO_HOME, $2 = preset UV_CACHE_DIR ("" for unset)
    "$_SH" -c "
        STUDIO_HOME='$1'
        if [ -n '$2' ]; then UV_CACHE_DIR='$2'; export UV_CACHE_DIR; else unset UV_CACHE_DIR; fi
        . '$_FN_FILE'
        printf '%s' \"\${UV_CACHE_DIR:-<unset>}\"
    "
}

echo "=== unset: defaults under STUDIO_HOME ==="
assert_eq "default home"    "$_TMP/studio/cache/uv" "$(_run "$_TMP/studio" '')"
# The whole point: a redirected STUDIO_HOME takes the cache with it.
assert_eq "redirected home" "$_TMP/sdcard/unsloth/cache/uv" "$(_run "$_TMP/sdcard/unsloth" '')"

echo "=== the directory is actually created (uv would too, but not before we log) ==="
_run "$_TMP/mk" '' >/dev/null
assert_eq "cache dir created" "yes" "$([ -d "$_TMP/mk/cache/uv" ] && echo yes || echo no)"

echo "=== an uncreatable cache falls back to uv's default, it does not stay exported ==="
# uv aborts with "Failed to initialize cache at ..." on a cache path it cannot create, so
# keeping the export after a failed mkdir turns a disk optimisation into a hard install
# failure on a host where uv's own default would have worked.
: > "$_TMP/blocked"          # STUDIO_HOME is a FILE -> $STUDIO_HOME/cache/uv cannot exist
assert_eq "uncreatable cache is dropped" "<unset>" "$(_run "$_TMP/blocked" '')"
mkdir -p "$_TMP/rofile" && : > "$_TMP/rofile/cache"   # writable home, "cache" is a file
assert_eq "cache-as-file is dropped"     "<unset>" "$(_run "$_TMP/rofile" '')"
# mkdir -p exits 0 for a directory that ALREADY exists, so an unwritable leftover cache from
# an earlier install under another account passed the mkdir and then failed uv exactly as an
# uncreatable path does. Writability has to be probed, not inferred.
mkdir -p "$_TMP/leftover/cache/uv" && chmod 500 "$_TMP/leftover/cache/uv"
if [ "$(id -u)" = "0" ]; then
    echo "  SKIP: unwritable-cache case (root writes through the mode bits)"
else
    assert_eq "unwritable existing cache is dropped" "<unset>" "$(_run "$_TMP/leftover" '')"
fi
chmod 700 "$_TMP/leftover/cache/uv"
# ...and the probe file it writes does not survive into the cache uv then fills.
_run "$_TMP/probe" '' >/dev/null
assert_eq "write probe cleaned up" "" "$(ls -A "$_TMP/probe/cache/uv")"
# ... and the fallback really is usable, unlike the path we just refused.
if command -v uv >/dev/null 2>&1; then
    _uv_rc=$("$_SH" -c "
        STUDIO_HOME='$_TMP/rofile'
        unset UV_CACHE_DIR
        . '$_FN_FILE'
        uv venv '$_TMP/rofile/venv' >/dev/null 2>&1
        printf '%s' \"\$?\"
    ")
    assert_eq "uv still runs after the fallback" "0" "$_uv_rc"
fi

echo "=== a caller-set UV_CACHE_DIR is never overridden ==="
assert_eq "explicit value kept" "/custom/uvcache" "$(_run "$_TMP/studio" '/custom/uvcache')"

echo "=== it is exported, not just assigned (uv runs in child processes) ==="
_exported=$("$_SH" -c "
    STUDIO_HOME='$_TMP/studio'
    unset UV_CACHE_DIR
    . '$_FN_FILE'
    sh -c 'printf %s \"\${UV_CACHE_DIR:-<unset>}\"'
")
assert_eq "visible to child processes" "$_TMP/studio/cache/uv" "$_exported"

echo "=== structural: set before uv is first invoked ==="
_set_line=$(grep -n 'UV_CACHE_DIR="\$STUDIO_HOME/cache/uv"' "$INSTALL_SH" | head -1 | cut -d: -f1)
_uv_line=$(grep -n 'installing uv package manager' "$INSTALL_SH" | head -1 | cut -d: -f1)
assert_eq "precedes uv bootstrap" "yes" \
    "$([ -n "$_set_line" ] && [ -n "$_uv_line" ] && [ "$_set_line" -lt "$_uv_line" ] && echo yes || echo no)"
# Match the call that creates the venv, not the label it carries: the literal
# 'run_install_cmd "create venv" uv venv' stopped existing when #8479 moved venv
# creation behind _run_uv_venv, and a label this file cannot find reads as "the cache
# is set too late" rather than "the grep is stale". Comment lines are dropped so the
# prose above the helper does not answer first.
_venv_line=$(grep -nE '(^|[^[:alnum:]_"`])uv venv([[:space:]]|$)' "$INSTALL_SH" \
    | grep -vE '^[0-9]+:[[:space:]]*#' | head -1 | cut -d: -f1)
assert_eq "found the venv creation call" "yes" "$([ -n "$_venv_line" ] && echo yes || echo no)"
assert_eq "precedes venv creation" "yes" \
    "$([ -n "$_set_line" ] && [ -n "$_venv_line" ] && [ "$_set_line" -lt "$_venv_line" ] && echo yes || echo no)"

echo "=== studio/setup.sh sets the SAME cache (it is the standalone update entry point) ==="
# install.sh exports UV_CACHE_DIR for its own process only, and `unsloth studio update` runs
# studio/setup.sh directly, so without the same block there a redirected STUDIO_HOME
# downloads a second cache to $HOME/.cache/uv on the first update and copies every wheel
# across the boundary. Same path, so the update also REUSES what the install fetched.
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
_SETUP_FN=$(mktemp)
trap 'rm -rf "$_FN_FILE" "$_TMP" "$_SETUP_FN"' EXIT
awk '/^# Same uv cache install\.sh chose/,/^fi$/' "$SETUP_SH" > "$_SETUP_FN"
if ! grep -q 'UV_CACHE_DIR="\$STUDIO_HOME/cache/uv"' "$_SETUP_FN"; then
    echo "  FAIL: could not extract the UV_CACHE_DIR block from studio/setup.sh"
    FAIL=$((FAIL + 1))
else
    _run_setup() {  # $1 = STUDIO_HOME, $2 = preset UV_CACHE_DIR ("" for unset)
        "$_SH" -c "
            STUDIO_HOME='$1'
            if [ -n '$2' ]; then UV_CACHE_DIR='$2'; export UV_CACHE_DIR; else unset UV_CACHE_DIR; fi
            . '$_SETUP_FN'
            printf '%s' \"\${UV_CACHE_DIR:-<unset>}\"
        "
    }
    assert_eq "setup.sh defaults under STUDIO_HOME" \
        "$_TMP/upd/cache/uv" "$(_run_setup "$_TMP/upd" '')"
    # The whole point: the update lands on the SAME path install.sh chose, so a redirected
    # home does not grow a second cache on the original filesystem.
    assert_eq "setup.sh matches install.sh for a redirected home" \
        "$_TMP/sdcard/unsloth/cache/uv" "$(_run_setup "$_TMP/sdcard/unsloth" '')"
    assert_eq "setup.sh keeps a caller-set value" \
        "/custom/uvcache" "$(_run_setup "$_TMP/upd" '/custom/uvcache')"
    # Same unwritable-path contract as install.sh, so an update cannot fail where the install
    # succeeded.
    mkdir -p "$_TMP/updro" && : > "$_TMP/updro/cache"
    assert_eq "setup.sh drops an unusable cache" "<unset>" "$(_run_setup "$_TMP/updro" '')"
fi

# The notice for a cache that could not be co-located has to name a remedy that exists. A
# caller's own UV_CACHE_DIR wins in _configure_uv_cache and returns before --isolated-uv-cache is
# read at all, so naming that flag to a custom-cache user sends them back for an identical run.
_NOTICE_FILE=$(mktemp)
# Anchored on `name() {` with anything allowed after the brace: _same_volume carries a trailing
# comment, and a pattern ending in `\{$` silently matched nothing. Under bash a missing function
# exits 127, which `&& return 0` reads as "different volume", so every case below passed for the
# wrong reason. Each extraction is therefore checked, not assumed.
_slice_fn() {  # name
    awk -v fn="$1" 'index($0, fn "() {") == 1 { grab = 1 } grab { print } grab && /^\}$/ { grab = 0 }' \
        "$INSTALL_SH"
}
: > "$_NOTICE_FILE"
for _fn in _uv_no_cache_requested _same_volume _path_device_id _warn_if_uv_cache_is_off_volume; do
    _slice_fn "$_fn" >> "$_NOTICE_FILE"
    if ! grep -q "^$_fn() {" "$_NOTICE_FILE"; then
        echo "FAIL: could not extract $_fn from install.sh"
        exit 1
    fi
done
_run_notice() {  # mode
    _rn_mode="$1"
    {
        printf '%s\n' 'step() { printf "STEP %s\n" "$2"; }'
        printf '%s\n' 'C_WARN=""'
        # A path that cannot exist on any mounted filesystem, so the device comparison answers
        # "different" without needing a second real volume here.
        printf '%s\n' 'UV_CACHE_DIR="/proc/self/uv-cache-elsewhere"'
        printf "STUDIO_HOME='%s'\n" "$_TMP"
        printf "_UV_CACHE_MODE='%s'\n" "$_rn_mode"
        cat "$_NOTICE_FILE"
        printf '%s\n' '_warn_if_uv_cache_is_off_volume'
        printf '%s\n' 'printf "OFF_VOLUME=%s\n" "${_ROLLBACK_COSTS_FULL_SIZE:-false}"'
    } | "$_SH" 2>&1
}
# UV_NO_CACHE: uv takes a temporary cache it discards when the command ends, so nothing the old
# environment shares its blocks with survives the install and keeping that tree costs its full
# size. The rollback warning must fire; the cross-volume notice must not, since where the cache
# sits is not the problem.
_run_no_cache() {  # same_volume_cache
    {
        printf '%s\n' 'step() { printf "STEP %s\n" "$2"; }'
        printf '%s\n' 'C_WARN=""'
        printf '%s\n' 'UV_NO_CACHE=1'
        printf "UV_CACHE_DIR='%s/cache/uv'\n" "$_TMP"
        printf "STUDIO_HOME='%s'\n" "$_TMP"
        cat "$_NOTICE_FILE"
        printf '%s\n' '_warn_if_uv_cache_is_off_volume'
        printf '%s\n' 'printf "COSTS_FULL_SIZE=%s\n" "${_ROLLBACK_COSTS_FULL_SIZE:-false}"'
    } | "$_SH" 2>&1
}
_NOTICE_NO_CACHE=$(_run_no_cache)
case "$_NOTICE_NO_CACHE" in
    *"COSTS_FULL_SIZE=true"*) ok "UV_NO_CACHE still costs a full second environment" ;;
    *) bad "UV_NO_CACHE was treated as if the old environment were shared: $_NOTICE_NO_CACHE" ;;
esac
case "$_NOTICE_NO_CACHE" in
    *"different filesystem"*) bad "UV_NO_CACHE printed a cross-volume notice about a co-located cache" ;;
    *) ok "UV_NO_CACHE does not blame the cache location" ;;
esac

# st_dev, not the df source: the question is whether uv can hardlink, and two btrfs subvolumes or
# two mounts of one device share a df source while `ln` between them fails EXDEV. Driven with a
# stubbed device id rather than a second filesystem, which this host cannot create.
_run_devids() {  # id_a  id_b
    {
        printf '%s\n' 'step() { printf "STEP %s\n" "$2"; }'
        printf '%s\n' 'C_WARN=""'
        printf "UV_CACHE_DIR='%s'\n" "$_TMP"
        printf "STUDIO_HOME='%s'\n" "$_TMP"
        printf '%s\n' '_UV_CACHE_MODE=studio'
        cat "$_NOTICE_FILE"
        printf '_path_device_id() { case "$1" in *cache*) echo %s ;; *) echo %s ;; esac; }\n' "$1" "$2"
        printf '%s\n' 'UV_CACHE_DIR="$STUDIO_HOME/cache"'
        printf '%s\n' 'mkdir -p "$UV_CACHE_DIR"'
        printf '%s\n' '_warn_if_uv_cache_is_off_volume'
        printf '%s\n' 'printf "COSTS_FULL_SIZE=%s\n" "${_ROLLBACK_COSTS_FULL_SIZE:-false}"'
    } | "$_SH" 2>&1
}
case "$(_run_devids 41 41)" in
    *"COSTS_FULL_SIZE=false"*) ok "one device id is one volume" ;;
    *) bad "a co-located cache was reported as off-volume" ;;
esac
case "$(_run_devids 41 42)" in
    *"COSTS_FULL_SIZE=true"*) ok "two device ids are two volumes, whatever df says" ;;
    *) bad "a cache on another filesystem was reported as co-located" ;;
esac

_NOTICE_CUSTOM=$(_run_notice custom)
_NOTICE_STUDIO=$(_run_notice studio)
case "$_NOTICE_CUSTOM" in
    *"unset UV_CACHE_DIR"*) ok "a custom cache is told to unset or move UV_CACHE_DIR" ;;
    *) bad "a custom cache was not told what would actually change the answer: $_NOTICE_CUSTOM" ;;
esac
case "$_NOTICE_CUSTOM" in
    *"--isolated-uv-cache"*) bad "a custom cache was sent back for an identical run" ;;
    *) ok "a custom cache is not sent back for an identical run" ;;
esac
case "$_NOTICE_STUDIO" in
    *"--isolated-uv-cache"*) ok "a selected cache still names the flag that would move it" ;;
    *) bad "a selected cache lost the --isolated-uv-cache remedy: $_NOTICE_STUDIO" ;;
esac
# The rollback free-space warning reads this, and only warns across the boundary.
case "$_NOTICE_STUDIO" in
    *"OFF_VOLUME=true"*) ok "the off-volume finding is recorded for the rollback warning" ;;
    *) bad "the off-volume finding was not recorded" ;;
esac
rm -f "$_NOTICE_FILE"

echo ""
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
