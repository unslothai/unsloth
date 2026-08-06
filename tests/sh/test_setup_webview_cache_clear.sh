#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit tests for _clear_webview_caches() from studio/setup.sh.
#
# The WebView caches keyed by the app bundle id (ai.unsloth.studio) hold copies
# of the previous frontend, so an install/update must clear them or the app can
# keep rendering old styles. Clearing must be cache-only: LocalStorage,
# IndexedDB, app data, and unrelated apps' caches stay intact.
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

assert_gone()    { _l="$1"; if [ -e "$2" ]; then echo "  FAIL: $_l (still present: $2)"; FAIL=$((FAIL+1)); else echo "  PASS: $_l"; PASS=$((PASS+1)); fi; }
assert_present() { _l="$1"; if [ -e "$2" ]; then echo "  PASS: $_l"; PASS=$((PASS+1)); else echo "  FAIL: $_l (missing: $2)"; FAIL=$((FAIL+1)); fi; }

# Explicit mktemp templates: -p is GNU-only (BSD mktemp gained it in macOS 14),
# and a bare mktemp -d implies -t and would land outside _TMP_ROOT on macOS.
new_home() { mktemp -d "$_TMP_ROOT/home.XXXXXX"; }

# Extract just the function definition (top-level, closes at column 0).
FUNC_FILE=$(mktemp "$_TMP_ROOT/fn.XXXXXX")
sed -n '/^_clear_webview_caches() {/,/^}/p' "$SETUP_SH" > "$FUNC_FILE"
# shellcheck disable=SC1090
. "$FUNC_FILE"
substep() { :; }  # stub the setup.sh logger

# ── 1. macOS: the real WKWebView layout. Every cache-typed store hangs off
# Library/Caches/<bid>/WebKit; Library/WebKit/<bid> holds only user storage. ──
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

# ── 2. Linux: wry points WebKitGTK's base-cache dir at the app data dir, so
# the caches sit beside the user storage under ~/.local/share/<bid>. ──
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

# ── 6. No HOME is a no-op, not a delete under /Library or /.cache ──
uname() { echo Darwin; }
if ( set -u; HOME="" _clear_webview_caches ); then
    echo "  PASS: empty HOME -> no-op exit 0"; PASS=$((PASS+1))
else
    echo "  FAIL: empty HOME -> nonzero exit"; FAIL=$((FAIL+1))
fi

# ── 7. setup.sh still calls the function (the extract above cannot prove it) ──
if grep -qE '^[[:space:]]*_clear_webview_caches[[:space:]]*$' "$SETUP_SH"; then
    echo "  PASS: setup.sh invokes _clear_webview_caches"; PASS=$((PASS+1))
else
    echo "  FAIL: setup.sh never calls _clear_webview_caches"; FAIL=$((FAIL+1))
fi

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
