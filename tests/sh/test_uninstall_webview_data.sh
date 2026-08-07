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
printf '#!/bin/sh\nexit 0\n' > "$STUB_BIN/defaults"
chmod +x "$STUB_BIN/defaults"
# pkill records its argv so the app kill can be asserted; real pkill exits 1 on no match,
# so mimic that rather than a blanket 0, which would hide a missing `|| true`.
PKILL_LOG="$_TMP_ROOT/pkill.args"
cat > "$STUB_BIN/pkill" <<EOF
#!/bin/sh
printf '%s\n' "\$*" >> "$PKILL_LOG"
exit 1
EOF
chmod +x "$STUB_BIN/pkill"
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

# Same, but capture stdout so the closing summary can be asserted.
run_uninstall_out() {
    printf '#!/bin/sh\necho %s\n' "$2" > "$STUB_BIN/uname"
    chmod +x "$STUB_BIN/uname"
    env -u UNSLOTH_STUDIO_HOME -u STUDIO_HOME \
        -u XDG_CACHE_HOME -u XDG_DATA_HOME -u XDG_CONFIG_HOME -u XDG_STATE_HOME \
        HOME="$1" PATH="$STUB_BIN:$PATH" sh "$UNINSTALL_SH" 2>/dev/null
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
         "$H/.local/state/$BID" "$H/.cache/other.app" \
         "$H/.local/share/applications"
: > "$H/.local/share/applications/unsloth-studio-handler.desktop"
: > "$H/.local/share/applications/other-app.desktop"
: > "$XDG_RUNTIME_DIR/unsloth-studio-launcher-$(id -u).lock"
# Truncate first: every fixture home has this user's uid, so the Darwin run above logged the
# same argv and the assertion below would pass on it even if Linux emitted no kill at all.
: > "$PKILL_LOG"
run_uninstall "$H" Linux
# Proves the lock sweep hit the fixture runtime dir, not the real one.
assert_gone "linux: fixture launcher lock removed" \
    "$XDG_RUNTIME_DIR/unsloth-studio-launcher-$(id -u).lock"
assert_gone "linux: ~/.cache/$BID removed"       "$H/.cache/$BID"
assert_gone "linux: ~/.local/share/$BID removed" "$H/.local/share/$BID"
assert_gone "linux: ~/.config/$BID removed"      "$H/.config/$BID"
assert_gone "linux: ~/.local/state/$BID removed" "$H/.local/state/$BID"
assert_present "linux: unrelated app cache kept" "$H/.cache/other.app"
# tauri-plugin-deep-link rewrites <exe>-handler.desktop on every launch to claim
# the unsloth:// scheme, so it is present on any machine the app has ever run on.
# Leaving it points the desktop's scheme handler at a binary that no longer exists.
assert_gone "linux: deep-link handler .desktop removed" \
    "$H/.local/share/applications/unsloth-studio-handler.desktop"
assert_present "linux: another app's .desktop kept" \
    "$H/.local/share/applications/other-app.desktop"
# The app kill must be scoped: unscoped, a root-run uninstall signals every user's
# unsloth-studio. -u takes the owner of the $HOME being cleared, not just `id -u`.
_want_uid=$(stat -c %u "$H" 2>/dev/null || stat -f %u "$H")
if grep -q -- "-x -u $_want_uid unsloth-studio" "$PKILL_LOG" 2>/dev/null; then
    echo "  PASS: linux: app kill scoped to the \$HOME owner"; PASS=$((PASS+1))
else
    echo "  FAIL: linux: app kill not scoped (want '-x -u $_want_uid unsloth-studio')"; FAIL=$((FAIL+1))
fi

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

# ── 3d. A symlinked $HOME still resolves an owner and still clears the target. stat lstats by
# default, so without -L a root-owned link to a user-owned home would resolve to uid 0. Owner
# and target owner match here, so this guards the symlink path, not the differing-owner case. ──
H=$(new_home)
_HL="$_TMP_ROOT/homelink.$$"
ln -s "$H" "$_HL"
mkdir -p "$H/.cache/$BID" "$H/.local/share/$BID"
: > "$PKILL_LOG"
run_uninstall "$_HL" Linux
_want_uid=$(stat -L -c %u "$_HL" 2>/dev/null || stat -L -f %u "$_HL")
if grep -q -- "-x -u $_want_uid unsloth-studio" "$PKILL_LOG" 2>/dev/null; then
    echo "  PASS: linux: symlinked HOME still scopes the kill"; PASS=$((PASS+1))
else
    echo "  FAIL: linux: symlinked HOME lost the kill scope"; FAIL=$((FAIL+1))
fi
assert_gone "linux: symlinked HOME cleared through the link" "$H/.cache/$BID"
rm -f "$_HL"

# ── 3c. A path that cannot be removed must not be reported as gone. The summary names the
# session, API keys and chat history, so claiming that after a failed rm is a false all-clear. ──
H=$(new_home)
mkdir -p "$H/.cache/$BID"
# An rm stub that refuses this one path, not chmod: root ignores mode bits, so a permission
# fixture deletes the dir anyway and the cleanup then aborts the suite under set -e.
REAL_RM=$(command -v rm)
case "$REAL_RM" in /*) ;; *) REAL_RM=/bin/rm ;; esac
cat > "$STUB_BIN/rm" <<EOF
#!/bin/sh
for _a in "\$@"; do
    [ "\$_a" = "$H/.cache/$BID" ] && exit 1
done
exec "$REAL_RM" "\$@"
EOF
chmod +x "$STUB_BIN/rm"
_out=$(run_uninstall_out "$H" Linux)
rm -f "$STUB_BIN/rm"
assert_present "linux: the refused path really did survive" "$H/.cache/$BID"
case "$_out" in
    *"may"*"still be on disk"*) echo "  PASS: linux: failed removal is not reported as gone"; PASS=$((PASS+1)) ;;
    *) echo "  FAIL: linux: failed removal still claimed the data is gone"; FAIL=$((FAIL+1)) ;;
esac
case "$_out" in
    *"are gone."*) echo "  FAIL: linux: summary still asserts the data is gone"; FAIL=$((FAIL+1)) ;;
    *) echo "  PASS: linux: summary drops the 'are gone' claim"; PASS=$((PASS+1)) ;;
esac

# ── 3e. Same, for a custom root. That removal runs inside a pipeline subshell, so a shell
# variable set there never reaches the summary; custom roots hold studio.db and the auth data. ──
H=$(new_home)
CUSTOM="$_TMP_ROOT/customroot.$$"
mkdir -p "$CUSTOM/share"
: > "$CUSTOM/share/studio.conf"          # what _is_studio_root accepts as ownership
cat > "$STUB_BIN/rm" <<EOF
#!/bin/sh
for _a in "\$@"; do
    [ "\$_a" = "$CUSTOM" ] && exit 1
done
exec "$REAL_RM" "\$@"
EOF
chmod +x "$STUB_BIN/rm"
printf '#!/bin/sh\necho Linux\n' > "$STUB_BIN/uname"; chmod +x "$STUB_BIN/uname"
# Every -u before the first assignment: env stops parsing options at the first VAR=VALUE.
_out=$(env -u STUDIO_HOME -u XDG_CACHE_HOME -u XDG_DATA_HOME -u XDG_CONFIG_HOME -u XDG_STATE_HOME \
    UNSLOTH_STUDIO_HOME="$CUSTOM" HOME="$H" PATH="$STUB_BIN:$PATH" sh "$UNINSTALL_SH" 2>/dev/null)
rm -f "$STUB_BIN/rm"
assert_present "linux: the refused custom root really did survive" "$CUSTOM"
case "$_out" in
    *"are gone."*) echo "  FAIL: linux: custom-root failure still claimed the data is gone"; FAIL=$((FAIL+1)) ;;
    *) echo "  PASS: linux: custom-root failure reaches the summary"; PASS=$((PASS+1)) ;;
esac

# ── 3f. A custom root the deny list refuses is also incomplete removal. install.sh accepts any
# writable root (mkdir -p + -w, no deny list), so /var/tmp/studio installs without elevation and
# then survives uninstall untouched. Uses the fixture HOME, which _is_unsafe_root also refuses. ──
H=$(new_home)
mkdir -p "$H/share"
: > "$H/share/studio.conf"
printf '#!/bin/sh\necho Linux\n' > "$STUB_BIN/uname"; chmod +x "$STUB_BIN/uname"
_out=$(env -u STUDIO_HOME -u XDG_CACHE_HOME -u XDG_DATA_HOME -u XDG_CONFIG_HOME -u XDG_STATE_HOME \
    UNSLOTH_STUDIO_HOME="$H" HOME="$H" PATH="$STUB_BIN:$PATH" sh "$UNINSTALL_SH" 2>/dev/null)
assert_present "linux: the deny-listed root was left alone" "$H/share/studio.conf"
case "$_out" in
    *"are gone."*) echo "  FAIL: linux: deny-listed root still claimed the data is gone"; FAIL=$((FAIL+1)) ;;
    *) echo "  PASS: linux: deny-listed root counts as incomplete removal"; PASS=$((PASS+1)) ;;
esac

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
