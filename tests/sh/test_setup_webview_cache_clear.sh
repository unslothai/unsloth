#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit tests for _clear_webview_caches() from studio/setup.sh.
#
# WebView caches keyed by the bundle id (ai.unsloth.studio) hold copies of the
# previous frontend, so an install/update must clear them or the app keeps
# rendering old styles. Cache-only: LocalStorage, IndexedDB, app data and
# unrelated apps' caches stay intact.
#
# Follows the extract-via-sed pattern of test_uninstall_shared_icon.sh; uname
# is overridden per test with a shell function to select the OS branch.
# shellcheck disable=SC2329  # uname stubs are invoked inside the extracted function
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
BID="ai.unsloth.studio"
PASS=0
FAIL=0

_TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$_TMP_ROOT"' EXIT

# -e || -L, matching the production guard: -e alone is false for a dangling
# symlink, so assert_gone would pass on one that was never removed.
assert_gone()    { _l="$1"; if [ -e "$2" ] || [ -L "$2" ]; then echo "  FAIL: $_l (still present: $2)"; FAIL=$((FAIL+1)); else echo "  PASS: $_l"; PASS=$((PASS+1)); fi; }
assert_present() { _l="$1"; if [ -e "$2" ] || [ -L "$2" ]; then echo "  PASS: $_l"; PASS=$((PASS+1)); else echo "  FAIL: $_l (missing: $2)"; FAIL=$((FAIL+1)); fi; }

# Explicit template: -p is GNU-only (BSD got it in macOS 14) and a bare
# mktemp -d implies -t, landing outside _TMP_ROOT on macOS.
new_home() { mktemp -d "$_TMP_ROOT/home.XXXXXX"; }

# Extract just the function definition (top-level, closes at column 0).
FUNC_FILE=$(mktemp "$_TMP_ROOT/fn.XXXXXX")
sed -n '/^_clear_webview_caches() {/,/^}/p' "$SETUP_SH" > "$FUNC_FILE"
# shellcheck disable=SC1090
. "$FUNC_FILE"
substep() { :; }  # stub the setup.sh logger

# ── 1. macOS: cache-typed stores hang off Library/Caches/<bid>/WebKit;
# Library/WebKit/<bid> is user storage only ──
H=$(new_home)
mkdir -p "$H/Library/Caches/$BID/WebKit/NetworkCache/Version 17" \
         "$H/Library/Caches/$BID/WebKit/CacheStorage" \
         "$H/Library/Caches/$BID/WebKit/ServiceWorkers" \
         "$H/Library/WebKit/$BID/WebsiteData/Default" \
         "$H/Library/WebKit/$BID/WebsiteData/LocalStorage" \
         "$H/Library/WebKit/$BID/WebsiteData/IndexedDB" \
         "$H/Library/Application Support/$BID" \
         "$H/Library/Caches/com.other.app"
uname() { echo Darwin; }
HOME="$H" _clear_webview_caches
assert_gone    "macOS: Caches/$BID removed"              "$H/Library/Caches/$BID"
assert_present "macOS: origin-keyed storage kept"        "$H/Library/WebKit/$BID/WebsiteData/Default"
assert_present "macOS: LocalStorage kept"                "$H/Library/WebKit/$BID/WebsiteData/LocalStorage"
assert_present "macOS: IndexedDB kept"                   "$H/Library/WebKit/$BID/WebsiteData/IndexedDB"
assert_present "macOS: Application Support kept"         "$H/Library/Application Support/$BID"
assert_present "macOS: unrelated app cache kept"         "$H/Library/Caches/com.other.app"

# ── 2. Linux: wry points the base-cache dir at the app data dir, so caches
# sit beside user storage under ~/.local/share/<bid> ──
H=$(new_home)
D="$H/.local/share/$BID"
mkdir -p "$D/WebKitCache" "$D/CacheStorage" "$D/serviceworkers" \
         "$D/localstorage" "$D/databases/indexeddb" "$H/.config/$BID" \
         "$H/.local/share/${BID}2/WebKitCache" "$H/.cache/other.app"
: > "$D/cookies"
uname() { echo Linux; }
HOME="$H" XDG_CACHE_HOME="" XDG_DATA_HOME="" _clear_webview_caches
assert_gone    "linux: data-dir WebKitCache removed"   "$D/WebKitCache"
assert_gone    "linux: data-dir CacheStorage removed"  "$D/CacheStorage"
assert_gone    "linux: data-dir serviceworkers removed" "$D/serviceworkers"
assert_present "linux: localstorage kept"              "$D/localstorage"
assert_present "linux: indexeddb kept"                 "$D/databases/indexeddb"
assert_present "linux: cookies kept"                   "$D/cookies"
assert_present "linux: ~/.config/$BID kept"            "$H/.config/$BID"
assert_present "linux: prefix-decoy bundle id kept"    "$H/.local/share/${BID}2/WebKitCache"
assert_present "linux: unrelated app cache kept"       "$H/.cache/other.app"

# ── 3. Linux: XDG_DATA_HOME override honored ──
H=$(new_home)
XDG=$(mktemp -d "$_TMP_ROOT/xdg.XXXXXX")
mkdir -p "$XDG/data/$BID/WebKitCache" "$XDG/data/$BID/localstorage" "$H/.local/share/$BID/WebKitCache"
HOME="$H" XDG_CACHE_HOME="" XDG_DATA_HOME="$XDG/data" _clear_webview_caches
assert_gone    "linux: XDG_DATA_HOME WebKitCache removed"    "$XDG/data/$BID/WebKitCache"
assert_present "linux: XDG_DATA_HOME localstorage kept"      "$XDG/data/$BID/localstorage"
assert_present "linux: default data dir kept under override" "$H/.local/share/$BID/WebKitCache"

# ── 3a. A relative XDG_DATA_HOME is invalid per the XDG spec, so dirs (and Tauri)
# ignores it. Following it would rm -rf under the installer's cwd and leave the
# cache Tauri actually uses in place ──
H=$(new_home)
W=$(mktemp -d "$_TMP_ROOT/work.XXXXXX")
mkdir -p "$W/reldata/$BID/WebKitCache" "$H/.local/share/$BID/WebKitCache"
uname() { echo Linux; }
( cd "$W" && HOME="$H" XDG_CACHE_HOME="" XDG_DATA_HOME="reldata" _clear_webview_caches )
assert_present "linux: relative XDG_DATA_HOME not followed under cwd" "$W/reldata/$BID/WebKitCache"
assert_gone    "linux: relative XDG_DATA_HOME uses the default"       "$H/.local/share/$BID/WebKitCache"

# ── 3b. A dangling cache symlink still occupies the path, so it must go ──
H=$(new_home)
mkdir -p "$H/.local/share/$BID"
ln -s "$_TMP_ROOT/does-not-exist" "$H/.local/share/$BID/WebKitCache"
HOME="$H" XDG_CACHE_HOME="" XDG_DATA_HOME="" _clear_webview_caches
assert_gone    "linux: dangling cache symlink removed"       "$H/.local/share/$BID/WebKitCache"

# ── 4. Nothing to clear is a clean no-op ──
H=$(new_home)
uname() { echo Darwin; }
if HOME="$H" _clear_webview_caches; then
    echo "  PASS: empty HOME -> no-op exit 0"; PASS=$((PASS+1))
else
    echo "  FAIL: empty HOME -> nonzero exit"; FAIL=$((FAIL+1))
fi

# ── 5. Unknown OS is a no-op ──
H=$(new_home)
mkdir -p "$H/Library/Caches/$BID"
uname() { echo SunOS; }
HOME="$H" _clear_webview_caches
assert_present "unknown OS: nothing removed" "$H/Library/Caches/$BID"

# ── 6. No HOME is a no-op, not a delete under /Library or /.cache. Unset, not empty:
# `set -u` fires on unset only, and the empty case passes even without the guard ──
uname() { echo Darwin; }
_wvc_rc=0
# `|| _wvc_rc=$?` keeps the abort we are testing for off `set -e`'s exit path.
_wvc_out=$( set -u; unset HOME; _clear_webview_caches 2>&1 ) || _wvc_rc=$?
if [ "$_wvc_rc" = 0 ] && [ -z "$_wvc_out" ]; then
    echo "  PASS: unset HOME -> silent no-op"; PASS=$((PASS+1))
else
    echo "  FAIL: unset HOME -> rc=$_wvc_rc out=$_wvc_out"; FAIL=$((FAIL+1))
fi

# ── 6b. The clear must invalidate the app's version stamp. ──
# The app skips its own clear whenever .webview-cache-cleared matches the running
# version (main.rs), and that clear is the retry for an rm here that failed. A repair
# or a local rebuild leaves the version unchanged, so a surviving stamp suppresses the
# retry and the cache we could not remove stays forever.
_st_home=$(new_home)
_st_data="$_st_home/.local/share/$BID"
mkdir -p "$_st_data/WebKitCache"
: > "$_st_data/WebKitCache/asset.js"
printf '2026.4.8' > "$_st_data/.webview-cache-cleared"
uname() { echo Linux; }
( HOME="$_st_home" _clear_webview_caches >/dev/null 2>&1 )
assert_gone "linux: version stamp invalidated so the app retries" \
    "$_st_data/.webview-cache-cleared"

# The stamp must go even when nothing was removable, which is the case that matters:
# an unremovable cache plus a surviving stamp is what makes the staleness permanent.
_st_home=$(new_home)
_st_data="$_st_home/.local/share/$BID"
mkdir -p "$_st_data"
printf '2026.4.8' > "$_st_data/.webview-cache-cleared"
( HOME="$_st_home" _clear_webview_caches >/dev/null 2>&1 )
assert_gone "linux: stamp dropped even with no cache dirs present" \
    "$_st_data/.webview-cache-cleared"

# macOS keeps the stamp under Application Support, not Caches, so the two paths differ.
_st_home=$(new_home)
mkdir -p "$_st_home/Library/Caches/$BID" "$_st_home/Library/Application Support/$BID"
: > "$_st_home/Library/Caches/$BID/stale.js"
printf '2026.4.8' > "$_st_home/Library/Application Support/$BID/.webview-cache-cleared"
uname() { echo Darwin; }
( HOME="$_st_home" _clear_webview_caches >/dev/null 2>&1 )
assert_gone "macos: stamp under Application Support invalidated" \
    "$_st_home/Library/Application Support/$BID/.webview-cache-cleared"
assert_gone "macos: Caches/<bid> still cleared" "$_st_home/Library/Caches/$BID/stale.js"
unset -f uname

# ── 7. setup.sh still calls it (the sed extract above cannot prove that) ──
if grep -qE '^[[:space:]]*_clear_webview_caches[[:space:]]*$' "$SETUP_SH"; then
    echo "  PASS: setup.sh invokes _clear_webview_caches"; PASS=$((PASS+1))
else
    echo "  FAIL: setup.sh never calls _clear_webview_caches"; FAIL=$((FAIL+1))
fi

# ── 7b. setup.ps1 is not driven here, so assert on source order: the call must come
# after the override validation, or a mistyped override wipes the cache and then aborts ──
SETUP_PS1="$SCRIPT_DIR/../../studio/setup.ps1"
if [ ! -f "$SETUP_PS1" ]; then
    echo "  FAIL: setup.ps1 not found"; FAIL=$((FAIL+1))
else
    _ps_call=$(grep -n '^[[:space:]]*Clear-WebViewCaches[[:space:]]*$' "$SETUP_PS1" | head -n1 | cut -d: -f1)
    _ps_validate=$(grep -n 'does not exist"$' "$SETUP_PS1" | tail -n1 | cut -d: -f1)
    if [ -z "$_ps_call" ]; then
        echo "  FAIL: setup.ps1 never calls Clear-WebViewCaches"; FAIL=$((FAIL+1))
    elif [ -z "$_ps_validate" ]; then
        echo "  FAIL: could not find the setup.ps1 override validation"; FAIL=$((FAIL+1))
    elif [ "$_ps_call" -gt "$_ps_validate" ]; then
        echo "  PASS: setup.ps1 clears after validating the override"; PASS=$((PASS+1))
    else
        echo "  FAIL: setup.ps1 clears at line $_ps_call, before validation at $_ps_validate"
        FAIL=$((FAIL+1))
    fi
fi

# ── 7c. An override that exists and is writable but holds no Studio install: setup
# aborts later at the venv check, so clearing first would cost the cache for nothing ──
_novenv_home=$(new_home)
_novenv_root="$_TMP_ROOT/exists-but-empty.$$"
mkdir -p "$_novenv_root"
mkdir -p "$_novenv_home/.local/share/$BID/WebKitCache"
: > "$_novenv_home/.local/share/$BID/WebKitCache/asset.js"
# Stub node/npm as failing and opt out of the isolated install so setup.sh reaches the
# venv check without provisioning Node or building the frontend.
_novenv_bin="$_TMP_ROOT/no-node.$$"
mkdir -p "$_novenv_bin"
for _stub in node npm; do
    printf '#!/bin/sh\nexit 1\n' > "$_novenv_bin/$_stub"
    chmod +x "$_novenv_bin/$_stub"
done
_novenv_out=$(
    cd "$(dirname "$SETUP_SH")" &&
    env -u XDG_DATA_HOME -u STUDIO_HOME \
        HOME="$_novenv_home" UNSLOTH_STUDIO_HOME="$_novenv_root" \
        PATH="$_novenv_bin:$PATH" UNSLOTH_SKIP_NODE_INSTALL=1 SKIP_STUDIO_FRONTEND=1 \
        UNSLOTH_TAURI_MODE=0 bash "$SETUP_SH" 2>&1
) && _novenv_rc=0 || _novenv_rc=$?
if [ "$_novenv_rc" = 0 ]; then
    echo "  FAIL: override without a venv did not abort"; FAIL=$((FAIL+1))
else
    echo "  PASS: override without a venv aborts (rc=$_novenv_rc)"; PASS=$((PASS+1))
fi
# Or the cache survived for an unrelated reason.
case "$_novenv_out" in
    *"venv not found at"*) echo "  PASS: the abort is the venv check"; PASS=$((PASS+1)) ;;
    *) echo "  FAIL: aborted before the venv check"; FAIL=$((FAIL+1)) ;;
esac
case "$_novenv_out" in
    *"no suitable Node"*) echo "  PASS: reached it without provisioning Node"; PASS=$((PASS+1)) ;;
    *) echo "  FAIL: the fixture is not hermetic (Node/npm ran)"; FAIL=$((FAIL+1)) ;;
esac
assert_present "existing-but-empty override leaves the cache alone" \
    "$_novenv_home/.local/share/$BID/WebKitCache/asset.js"

# ── 8. a bad override must not cost the user their cache ──
# The cases above extract the function, so none sees where the call sits; driven in situ.
# Under `set -euo pipefail` with no trap, clearing before the UNSLOTH_STUDIO_HOME
# validation turns a typo into cache loss plus the same abort.
_ord_home=$(new_home)
mkdir -p "$_ord_home/.local/share/$BID/WebKitCache" "$_ord_home/.local/share/$BID/CacheStorage"
: > "$_ord_home/.local/share/$BID/WebKitCache/asset.js"
_ord_out=$(
    cd "$(dirname "$SETUP_SH")" &&
    env -u XDG_DATA_HOME -u STUDIO_HOME \
        HOME="$_ord_home" UNSLOTH_STUDIO_HOME="$_ord_home/typo-not-a-real-dir" \
        UNSLOTH_TAURI_MODE=0 bash "$SETUP_SH" 2>&1
) && _ord_rc=0 || _ord_rc=$?
if [ "$_ord_rc" = 0 ]; then
    echo "  FAIL: bad override did not abort (rc=0)"; FAIL=$((FAIL+1))
else
    echo "  PASS: bad override still aborts (rc=$_ord_rc)"; PASS=$((PASS+1))
fi
case "$_ord_out" in
    *"does not exist"*) echo "  PASS: abort names the bad override"; PASS=$((PASS+1)) ;;
    *) echo "  FAIL: abort message missing: $_ord_out"; FAIL=$((FAIL+1)) ;;
esac
assert_present "bad override leaves the cache alone" \
    "$_ord_home/.local/share/$BID/WebKitCache/asset.js"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
