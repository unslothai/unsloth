#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Regression tests for WebView runtime-data cleanup in scripts/uninstall.sh.
#
# WKWebView (macOS) and webkit2gtk (Linux) key their data by bundle id and create it at first
# launch, not at install time, so the uninstaller used to miss it and a leftover cache served a
# stale frontend to the next install. Runs the full script against a fixture HOME with the OS
# branch and the tools it calls stubbed via PATH, asserting bundle-id paths go and others stay.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UNINSTALL_SH="$SCRIPT_DIR/../../scripts/uninstall.sh"
BID="ai.unsloth.studio"
PASS=0
FAIL=0

_TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$_TMP_ROOT"' EXIT

# The script sweeps $XDG_RUNTIME_DIR/unsloth-studio-launcher-<uid>*.lock, so an unsandboxed
# runtime dir would cost the real launcher its locks.
XDG_RUNTIME_DIR="$_TMP_ROOT/run"
export XDG_RUNTIME_DIR
mkdir -p "$XDG_RUNTIME_DIR"

# Explicit template: -p is GNU-only (BSD got it in macOS 14) and a bare mktemp -d implies -t,
# landing outside _TMP_ROOT on macOS.
new_home() { mktemp -d "$_TMP_ROOT/home.XXXXXX"; }

assert_gone()    { _l="$1"; if [ -e "$2" ]; then echo "  FAIL: $_l (still present: $2)"; FAIL=$((FAIL+1)); else echo "  PASS: $_l"; PASS=$((PASS+1)); fi; }
assert_present() { _l="$1"; if [ -e "$2" ]; then echo "  PASS: $_l"; PASS=$((PASS+1)); else echo "  FAIL: $_l (missing: $2)"; FAIL=$((FAIL+1)); fi; }

# Kill and macOS pref tools, stubbed so the script leaves the real system alone.
STUB_BIN="$_TMP_ROOT/stubbin"
mkdir -p "$STUB_BIN"
for _tool in pkill defaults; do
    printf '#!/bin/sh\nexit 0\n' > "$STUB_BIN/$_tool"
    chmod +x "$STUB_BIN/$_tool"
done
# On a WSL host the script's `grep -qi microsoft /proc/version` probe fires even with uname
# stubbed to Linux, and the real WSL cleanup would touch the host's /mnt/* shortcuts and /etc
# profile. Fail that one probe, delegate the rest. REAL_GREP must be absolute or the stub
# execs itself forever.
REAL_GREP=$(command -v grep)
case "$REAL_GREP" in /*) ;; *) REAL_GREP=/usr/bin/grep ;; esac
cat > "$STUB_BIN/grep" <<EOF
#!/bin/sh
for _a in "\$@"; do
    [ "\$_a" = "/proc/version" ] && exit 1
done
exec "$REAL_GREP" "\$@"
EOF
chmod +x "$STUB_BIN/grep"
# Belt and braces if the WSL branch is entered anyway: powershell.exe exiting 0 takes the no-op
# path (skipping the /mnt/* drvfs fallback), and sudo exiting 0 without its argv keeps /etc clean.
for _tool in powershell.exe sudo; do
    printf '#!/bin/sh\nexit 0\n' > "$STUB_BIN/$_tool"
    chmod +x "$STUB_BIN/$_tool"
done

# run_uninstall <home> <uname_output> : run the full script with a stubbed OS.
run_uninstall() {
    printf '#!/bin/sh\necho %s\n' "$2" > "$STUB_BIN/uname"
    chmod +x "$STUB_BIN/uname"
    env -u UNSLOTH_STUDIO_HOME -u STUDIO_HOME \
        -u XDG_CACHE_HOME -u XDG_DATA_HOME -u XDG_CONFIG_HOME -u XDG_STATE_HOME \
        HOME="$1" PATH="$STUB_BIN:$PATH" sh "$UNINSTALL_SH" >/dev/null 2>&1
}

# ── 1. macOS: every bundle-id-keyed ~/Library path is removed ──
H=$(new_home)
mkdir -p "$H/Library/Caches/$BID/WebKit/NetworkCache" \
         "$H/Library/WebKit/$BID/WebsiteData/CacheStorage" \
         "$H/Library/Application Support/$BID" \
         "$H/Library/HTTPStorages/$BID" \
         "$H/Library/Saved Application State/$BID.savedState" \
         "$H/Library/Preferences" \
         "$H/Library/Cookies" \
         "$H/Library/Caches/com.other.app"
: > "$H/Library/HTTPStorages/$BID.binarycookies"
: > "$H/Library/Cookies/$BID.binarycookies"
: > "$H/Library/Preferences/$BID.plist"
: > "$H/Library/Caches/$BID/stale-frontend.js"
: > "$H/Library/Caches/com.other.app/keepme"
run_uninstall "$H" Darwin
assert_gone "macOS: Caches/$BID removed"                     "$H/Library/Caches/$BID"
assert_gone "macOS: WebKit/$BID removed"                     "$H/Library/WebKit/$BID"
assert_gone "macOS: Application Support/$BID removed"        "$H/Library/Application Support/$BID"
assert_gone "macOS: HTTPStorages/$BID removed"               "$H/Library/HTTPStorages/$BID"
assert_gone "macOS: HTTPStorages/$BID.binarycookies removed" "$H/Library/HTTPStorages/$BID.binarycookies"
assert_gone "macOS: Cookies/$BID.binarycookies removed"      "$H/Library/Cookies/$BID.binarycookies"
assert_gone "macOS: Saved Application State removed"         "$H/Library/Saved Application State/$BID.savedState"
assert_gone "macOS: Preferences/$BID.plist removed"          "$H/Library/Preferences/$BID.plist"
assert_present "macOS: unrelated app cache kept"             "$H/Library/Caches/com.other.app/keepme"

# ── 2. Linux: bundle-id-keyed XDG default paths are removed ──
H=$(new_home)
mkdir -p "$H/.cache/$BID" "$H/.local/share/$BID" "$H/.config/$BID" \
         "$H/.local/state/$BID" "$H/.cache/other.app"
: > "$XDG_RUNTIME_DIR/unsloth-studio-launcher-$(id -u).lock"
run_uninstall "$H" Linux
# Proves the lock sweep hit the fixture runtime dir, not the real one.
assert_gone "linux: fixture launcher lock removed" \
    "$XDG_RUNTIME_DIR/unsloth-studio-launcher-$(id -u).lock"
assert_gone "linux: ~/.cache/$BID removed"       "$H/.cache/$BID"
assert_gone "linux: ~/.local/share/$BID removed" "$H/.local/share/$BID"
assert_gone "linux: ~/.config/$BID removed"      "$H/.config/$BID"
assert_gone "linux: ~/.local/state/$BID removed" "$H/.local/state/$BID"
assert_present "linux: unrelated app cache kept" "$H/.cache/other.app"

# ── 3. Linux: XDG_*_HOME overrides are honored ──
H=$(new_home)
XDG=$(mktemp -d "$_TMP_ROOT/xdg.XXXXXX")
mkdir -p "$XDG/cache/$BID" "$XDG/data/$BID" "$XDG/config/$BID" "$XDG/state/$BID"
printf '#!/bin/sh\necho Linux\n' > "$STUB_BIN/uname"
chmod +x "$STUB_BIN/uname"
env -u UNSLOTH_STUDIO_HOME -u STUDIO_HOME \
    XDG_CACHE_HOME="$XDG/cache" XDG_DATA_HOME="$XDG/data" \
    XDG_CONFIG_HOME="$XDG/config" XDG_STATE_HOME="$XDG/state" \
    HOME="$H" PATH="$STUB_BIN:$PATH" sh "$UNINSTALL_SH" >/dev/null 2>&1
assert_gone "linux: XDG_CACHE_HOME override honored"  "$XDG/cache/$BID"
assert_gone "linux: XDG_DATA_HOME override honored"   "$XDG/data/$BID"
assert_gone "linux: XDG_CONFIG_HOME override honored" "$XDG/config/$BID"
assert_gone "linux: XDG_STATE_HOME override honored"  "$XDG/state/$BID"

# ── 3b. Relative XDG overrides are invalid per the spec and dropped by the resolver Tauri uses,
# so the $HOME default goes and the same-named dir under the caller's cwd is left alone. ──
H=$(new_home)
CWD=$(mktemp -d "$_TMP_ROOT/cwd.XXXXXX")
mkdir -p "$H/.local/share/$BID" "$H/.cache/$BID" "$H/.config/$BID" "$H/.local/state/$BID" \
         "$CWD/reldata/$BID" "$CWD/relcache/$BID"
( cd "$CWD" && env -u UNSLOTH_STUDIO_HOME -u STUDIO_HOME \
    XDG_DATA_HOME="reldata" XDG_CACHE_HOME="relcache" \
    XDG_CONFIG_HOME="relconfig" XDG_STATE_HOME="relstate" \
    HOME="$H" PATH="$STUB_BIN:$PATH" sh "$UNINSTALL_SH" >/dev/null 2>&1 )
assert_gone    "linux: relative XDG_DATA_HOME falls back to HOME"   "$H/.local/share/$BID"
assert_gone    "linux: relative XDG_CACHE_HOME falls back to HOME"  "$H/.cache/$BID"
assert_gone    "linux: relative XDG_CONFIG_HOME falls back to HOME" "$H/.config/$BID"
assert_gone    "linux: relative XDG_STATE_HOME falls back to HOME"  "$H/.local/state/$BID"
assert_present "linux: relative XDG left cwd/reldata alone"         "$CWD/reldata/$BID"
assert_present "linux: relative XDG left cwd/relcache alone"        "$CWD/relcache/$BID"

# ── 4. Nothing to remove is a clean no-op (fresh HOME, exit 0) ──
H=$(new_home)
if run_uninstall "$H" Darwin; then
    echo "  PASS: empty HOME -> no-op exit 0"; PASS=$((PASS+1))
else
    echo "  FAIL: empty HOME -> nonzero exit"; FAIL=$((FAIL+1))
fi

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
