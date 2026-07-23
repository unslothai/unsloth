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

# Extract just the function definition (top-level, closes at column 0).
FUNC_FILE=$(mktemp -p "$_TMP_ROOT")
sed -n '/^_clear_webview_caches() {/,/^}/p' "$SETUP_SH" > "$FUNC_FILE"
# shellcheck disable=SC1090
. "$FUNC_FILE"
substep() { :; }  # stub the setup.sh logger

# ── 1. macOS: cache paths removed, user-facing storage kept ──
H=$(mktemp -d -p "$_TMP_ROOT")
mkdir -p "$H/Library/Caches/$BID/WebKit/NetworkCache" \
         "$H/Library/WebKit/$BID/WebsiteData/CacheStorage" \
         "$H/Library/WebKit/$BID/WebsiteData/ServiceWorkers" \
         "$H/Library/WebKit/$BID/WebsiteData/DiskCache" \
         "$H/Library/WebKit/$BID/WebsiteData/LocalStorage" \
         "$H/Library/WebKit/$BID/WebsiteData/IndexedDB" \
         "$H/Library/Application Support/$BID" \
         "$H/Library/Caches/com.other.app"
uname() { echo Darwin; }
HOME="$H" _clear_webview_caches
assert_gone    "macOS: Caches/$BID removed"              "$H/Library/Caches/$BID"
assert_gone    "macOS: WebsiteData/CacheStorage removed" "$H/Library/WebKit/$BID/WebsiteData/CacheStorage"
assert_gone    "macOS: WebsiteData/ServiceWorkers removed" "$H/Library/WebKit/$BID/WebsiteData/ServiceWorkers"
assert_gone    "macOS: WebsiteData/DiskCache removed"    "$H/Library/WebKit/$BID/WebsiteData/DiskCache"
assert_present "macOS: LocalStorage kept"                "$H/Library/WebKit/$BID/WebsiteData/LocalStorage"
assert_present "macOS: IndexedDB kept"                   "$H/Library/WebKit/$BID/WebsiteData/IndexedDB"
assert_present "macOS: Application Support kept"         "$H/Library/Application Support/$BID"
assert_present "macOS: unrelated app cache kept"         "$H/Library/Caches/com.other.app"

# ── 2. Linux: cache dir removed, data/config kept ──
H=$(mktemp -d -p "$_TMP_ROOT")
mkdir -p "$H/.cache/$BID" "$H/.local/share/$BID" "$H/.config/$BID" "$H/.cache/other.app"
uname() { echo Linux; }
HOME="$H" XDG_CACHE_HOME="" _clear_webview_caches
assert_gone    "linux: ~/.cache/$BID removed"     "$H/.cache/$BID"
assert_present "linux: ~/.local/share/$BID kept"  "$H/.local/share/$BID"
assert_present "linux: ~/.config/$BID kept"       "$H/.config/$BID"
assert_present "linux: unrelated app cache kept"  "$H/.cache/other.app"

# ── 3. Linux: XDG_CACHE_HOME override honored ──
H=$(mktemp -d -p "$_TMP_ROOT")
XDG=$(mktemp -d -p "$_TMP_ROOT")
mkdir -p "$XDG/$BID" "$H/.cache/$BID"
HOME="$H" XDG_CACHE_HOME="$XDG" _clear_webview_caches
assert_gone    "linux: XDG_CACHE_HOME/$BID removed"        "$XDG/$BID"
assert_present "linux: ~/.cache/$BID kept under override"  "$H/.cache/$BID"

# ── 4. Nothing to clear is a clean no-op ──
H=$(mktemp -d -p "$_TMP_ROOT")
uname() { echo Darwin; }
if HOME="$H" _clear_webview_caches; then
    echo "  PASS: empty HOME -> no-op exit 0"; PASS=$((PASS+1))
else
    echo "  FAIL: empty HOME -> nonzero exit"; FAIL=$((FAIL+1))
fi

# ── 5. Unknown OS is a no-op ──
H=$(mktemp -d -p "$_TMP_ROOT")
mkdir -p "$H/Library/Caches/$BID"
uname() { echo SunOS; }
HOME="$H" _clear_webview_caches
assert_present "unknown OS: nothing removed" "$H/Library/Caches/$BID"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
