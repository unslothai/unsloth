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

# ── 3g. Env-mode install, bare uninstall. An env-mode install writes studio.conf into
# $STUDIO_HOME/share (install.sh:567), not $HOME, so nothing in $HOME points at the custom
# root. Running the documented bare uninstaller leaves studio.db (chat_threads,
# chat_messages, provider keys) untouched, so the summary must not say it is gone. ──
H=$(new_home)
CUSTOM2="$_TMP_ROOT/envroot.$$"
mkdir -p "$CUSTOM2/share"
: > "$CUSTOM2/share/studio.conf"
: > "$CUSTOM2/studio.db"
printf '#!/bin/sh\necho Linux\n' > "$STUB_BIN/uname"; chmod +x "$STUB_BIN/uname"
_out=$(env -u UNSLOTH_STUDIO_HOME -u STUDIO_HOME \
    -u XDG_CACHE_HOME -u XDG_DATA_HOME -u XDG_CONFIG_HOME -u XDG_STATE_HOME \
    HOME="$H" PATH="$STUB_BIN:$PATH" sh "$UNINSTALL_SH" 2>/dev/null)
assert_present "linux: undiscovered env-mode studio.db survives" "$CUSTOM2/studio.db"
case "$_out" in
    *"studio.db it found"*)
        echo "  FAIL: linux: claimed chat history is gone with studio.db still on disk"
        FAIL=$((FAIL+1)) ;;
    *) echo "  PASS: linux: no studio.db removed -> no claim that the history is gone"
        PASS=$((PASS+1)) ;;
esac

# ── 3h. The other side of 3g: when a studio.db IS removed, the full claim must return,
# otherwise the softened wording above would just always fire and assert nothing. ──
H=$(new_home)
mkdir -p "$H/.unsloth/studio/unsloth_studio"
: > "$H/.unsloth/studio/unsloth_studio/.unsloth-studio-owned"
: > "$H/.unsloth/studio/studio.db"
_out=$(run_uninstall_out "$H" Linux)
assert_gone "linux: default-mode studio.db removed" "$H/.unsloth/studio/studio.db"
case "$_out" in
    *"studio.db it found"*)
        echo "  PASS: linux: studio.db removed -> summary states the history is gone"
        PASS=$((PASS+1)) ;;
    *) echo "  FAIL: linux: studio.db was removed but the summary never says so"
        FAIL=$((FAIL+1)) ;;
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

# ── 3i. An unusable TMPDIR must not abort the run. The summary markers are written with
# `printf`, not `: >`: `:` is a POSIX special builtin, so a redirection error on it kills a
# non-interactive shell outright (dash exits 2, busybox ash 1) and `2>/dev/null || true`
# does not stop it. The uninstall would then stop partway with most of the tree still there. ──
H=$(new_home)
mkdir -p "$H/.cache/$BID" "$H/.unsloth/studio/unsloth_studio"
: > "$H/.unsloth/studio/unsloth_studio/.unsloth-studio-owned"
: > "$H/.unsloth/studio/studio.db"
printf '#!/bin/sh\necho Linux\n' > "$STUB_BIN/uname"; chmod +x "$STUB_BIN/uname"
for _sh in sh dash busybox; do
    command -v "$_sh" >/dev/null 2>&1 || continue
    _H2=$(new_home)
    mkdir -p "$_H2/.cache/$BID" "$_H2/.unsloth/studio/unsloth_studio"
    : > "$_H2/.unsloth/studio/unsloth_studio/.unsloth-studio-owned"
    _rc=0
    # busybox needs its applet name as a separate argument; spelled out per branch
    # rather than word-splitting one variable.
    if [ "$_sh" = busybox ]; then
        env -u UNSLOTH_STUDIO_HOME -u STUDIO_HOME \
            -u XDG_CACHE_HOME -u XDG_DATA_HOME -u XDG_CONFIG_HOME -u XDG_STATE_HOME \
            TMPDIR="$_TMP_ROOT/no-such-tmpdir" HOME="$_H2" PATH="$STUB_BIN:$PATH" \
            busybox sh "$UNINSTALL_SH" >/dev/null 2>&1 || _rc=$?
    else
        env -u UNSLOTH_STUDIO_HOME -u STUDIO_HOME \
            -u XDG_CACHE_HOME -u XDG_DATA_HOME -u XDG_CONFIG_HOME -u XDG_STATE_HOME \
            TMPDIR="$_TMP_ROOT/no-such-tmpdir" HOME="$_H2" PATH="$STUB_BIN:$PATH" \
            "$_sh" "$UNINSTALL_SH" >/dev/null 2>&1 || _rc=$?
    fi
    if [ "$_rc" = 0 ]; then
        echo "  PASS: $_sh: unusable TMPDIR did not abort the run"; PASS=$((PASS+1))
    else
        echo "  FAIL: $_sh: unusable TMPDIR aborted with rc=$_rc"; FAIL=$((FAIL+1))
    fi
    assert_gone "$_sh: cleanup still completed with an unusable TMPDIR" "$_H2/.cache/$BID"
done

# ── 3j. The marker directory is private and removed on the way out. TMPDIR points at a
# fixture-owned directory so the check sees only what this run created: counting entries
# in the shared temp dir would call a concurrent process's mktemp a leak, and an unrelated
# directory disappearing at the same time would hide a real one. ──
H=$(new_home)
_MK_TMP=$(mktemp -d "$_TMP_ROOT/markertmp.XXXXXX")
printf '#!/bin/sh\necho Linux\n' > "$STUB_BIN/uname"; chmod +x "$STUB_BIN/uname"
env -u UNSLOTH_STUDIO_HOME -u STUDIO_HOME \
    -u XDG_CACHE_HOME -u XDG_DATA_HOME -u XDG_CONFIG_HOME -u XDG_STATE_HOME \
    TMPDIR="$_MK_TMP" HOME="$H" PATH="$STUB_BIN:$PATH" sh "$UNINSTALL_SH" >/dev/null 2>&1
_left=$(find "$_MK_TMP" -mindepth 1 2>/dev/null | wc -l)
if [ "$_left" = 0 ]; then
    echo "  PASS: linux: marker directory cleaned up on exit"; PASS=$((PASS+1))
else
    echo "  FAIL: linux: left $_left entries in TMPDIR"; FAIL=$((FAIL+1))
    find "$_MK_TMP" -mindepth 1 | sed 's/^/         /'
fi

# ── 3j2. With no marker storage at all there is no record of what failed, so the summary
# must take the cautious branch rather than reporting the session gone. mktemp is stubbed
# to fail, which is what an unwritable or missing TMPDIR produces. ──
H=$(new_home)
mkdir -p "$H/.unsloth/studio/unsloth_studio"
: > "$H/.unsloth/studio/unsloth_studio/.unsloth-studio-owned"
: > "$H/.unsloth/studio/studio.db"
printf '#!/bin/sh\nexit 1\n' > "$STUB_BIN/mktemp"; chmod +x "$STUB_BIN/mktemp"
_out=$(run_uninstall_out "$H" Linux)
rm -f "$STUB_BIN/mktemp"
case "$_out" in
    *"may"*"still be on disk"*)
        echo "  PASS: linux: no marker storage -> cautious summary"; PASS=$((PASS+1)) ;;
    *)
        echo "  FAIL: linux: no marker storage but the summary still reported success"
        FAIL=$((FAIL+1)) ;;
esac

# ── 3j3. A relocated install: ~/.unsloth/studio is a symlink to another disk. rm -rf on a
# symlink unlinks the link and leaves the target, so the database survives and the summary
# must not report it gone. Reachable whenever a user moves Studio off the system disk. ──
H=$(new_home)
_ELSEWHERE=$(mktemp -d "$_TMP_ROOT/otherdisk.XXXXXX")
mkdir -p "$_ELSEWHERE/unsloth_studio" "$H/.unsloth"
: > "$_ELSEWHERE/unsloth_studio/.unsloth-studio-owned"
: > "$_ELSEWHERE/studio.db"
ln -s "$_ELSEWHERE" "$H/.unsloth/studio"
_out=$(run_uninstall_out "$H" Linux)
assert_present "linux: relocated studio.db survived the symlink removal" "$_ELSEWHERE/studio.db"
case "$_out" in
    *"studio.db it found"*)
        echo "  FAIL: linux: claimed the database is gone but it is on the other disk"
        FAIL=$((FAIL+1)) ;;
    *)
        echo "  PASS: linux: relocated install -> no claim that the database is gone"
        PASS=$((PASS+1)) ;;
esac
case "$_out" in
    *"may"*"still be on disk"*)
        echo "  PASS: linux: relocated install reported as incomplete"; PASS=$((PASS+1)) ;;
    *)
        echo "  FAIL: linux: relocated install not reported as incomplete"; FAIL=$((FAIL+1)) ;;
esac

# ── 3j4. Marker storage that vanishes mid-run. The pathname still looks fine, so a
# predicate that only tests for emptiness keeps reporting success while every failure
# _set_marker tried to record was silently dropped. ──
H=$(new_home)
mkdir -p "$H/.local/share/$BID"
_VAN=$(mktemp -d "$_TMP_ROOT/vanish.XXXXXX")
# mktemp -d names a directory it does not leave behind. Deterministic, rather than racing
# a background delete: $_MARKER_DIR is non-empty so the pathname still looks usable, which
# is the state the predicate has to catch.
cat > "$STUB_BIN/mktemp" <<EOF
#!/bin/sh
printf '%s\n' "$_VAN/marker.gone"
exit 0
EOF
chmod +x "$STUB_BIN/mktemp"
_out=$(run_uninstall_out "$H" Linux)
rm -f "$STUB_BIN/mktemp"
case "$_out" in
    *"may"*"still be on disk"*)
        echo "  PASS: linux: vanished marker dir -> cautious summary"; PASS=$((PASS+1)) ;;
    *)
        echo "  FAIL: linux: vanished marker dir still reported success"; FAIL=$((FAIL+1)) ;;
esac

# ── 3j5. studio.db itself a symlink out of the tree. -f follows it, rm -rf unlinks only
# the link, so the database survives and the summary must not report it gone. Same root
# cause as the relocated-install case, different shape. ──
H=$(new_home)
_DBTARGET=$(mktemp -d "$_TMP_ROOT/dbtarget.XXXXXX")
mkdir -p "$H/.unsloth/studio/unsloth_studio"
: > "$H/.unsloth/studio/unsloth_studio/.unsloth-studio-owned"
: > "$_DBTARGET/studio.db"
ln -s "$_DBTARGET/studio.db" "$H/.unsloth/studio/studio.db"
_out=$(run_uninstall_out "$H" Linux)
assert_present "linux: symlinked studio.db target survived" "$_DBTARGET/studio.db"
case "$_out" in
    *"studio.db it found"*)
        echo "  FAIL: linux: claimed the database is gone but the target survived"
        FAIL=$((FAIL+1)) ;;
    *)  echo "  PASS: linux: symlinked studio.db -> no claim that it is gone"
        PASS=$((PASS+1)) ;;
esac

# ── 3j5b. Same, with `readlink -f` unavailable and a RELATIVE link target. BSD readlink
# only gained -f in macOS 12.3, so the GNU form fails on older macOS; the resolution has
# to work from the raw link text. Stub rejects -f the way BSD readlink does. ──
H=$(new_home)
mkdir -p "$H/.unsloth/studio/unsloth_studio" "$H/dbdir"
: > "$H/.unsloth/studio/unsloth_studio/.unsloth-studio-owned"
: > "$H/dbdir/studio.db"
ln -s "../../dbdir/studio.db" "$H/.unsloth/studio/studio.db"
_REAL_READLINK=$(command -v readlink)
cat > "$STUB_BIN/readlink" <<EOF
#!/bin/sh
for _a in "\$@"; do
    [ "\$_a" = "-f" ] && { echo "readlink: illegal option -- f" >&2; exit 1; }
done
exec "$_REAL_READLINK" "\$@"
EOF
chmod +x "$STUB_BIN/readlink"
_out=$(run_uninstall_out "$H" Linux)
rm -f "$STUB_BIN/readlink"
assert_present "linux: relative symlinked studio.db survived without readlink -f" \
    "$H/dbdir/studio.db"
case "$_out" in
    *"studio.db it found"*)
        echo "  FAIL: linux: claimed the database is gone (no readlink -f, relative link)"
        FAIL=$((FAIL+1)) ;;
    *)  echo "  PASS: linux: resolves a relative link without readlink -f"
        PASS=$((PASS+1)) ;;
esac

# ── 3j6. Provider API keys are NOT in studio.db. providers_db.py: "API keys are NOT
# stored here: they live only in the browser (localStorage) and are sent encrypted
# per-request." install.sh runs with TAURI_MODE=false, so the default install serves the
# UI to a normal browser and the keys sit in that browser's profile, which this script
# never touches. No branch may claim they were removed. ──
for _case in dbremoved nodb; do
    H=$(new_home)
    mkdir -p "$H/.unsloth/studio/unsloth_studio"
    : > "$H/.unsloth/studio/unsloth_studio/.unsloth-studio-owned"
    [ "$_case" = dbremoved ] && : > "$H/.unsloth/studio/studio.db"
    _out=$(run_uninstall_out "$H" Linux)
    case "$_out" in
        *"saved provider API keys"*|*"API keys and chat history"*|*"API keys and local chat"*)
            echo "  FAIL: $_case: claimed the provider API keys were removed"; FAIL=$((FAIL+1)) ;;
        *)  echo "  PASS: $_case: no claim that the provider API keys were removed"
            PASS=$((PASS+1)) ;;
    esac
    case "$_out" in
        *"localStorage, not in studio.db"*)
            echo "  PASS: $_case: says where the keys actually are"; PASS=$((PASS+1)) ;;
        *)  echo "  FAIL: $_case: never says where the keys actually are"; FAIL=$((FAIL+1)) ;;
    esac
    # Removing the WebView profile clears the desktop app's session only. A browser
    # session keeps its tokens in localStorage (frontend/src/features/auth/session.ts),
    # which this script never touches, so an unqualified claim is wrong there too.
    case "$_out" in
        *"the signed-in session is gone"*)
            echo "  FAIL: $_case: unqualified signed-out claim"; FAIL=$((FAIL+1)) ;;
        *)  echo "  PASS: $_case: signed-out claim scoped to the desktop app"
            PASS=$((PASS+1)) ;;
    esac
done

# ── 3k. _set_marker itself must survive a write it cannot perform. 3i only proves the
# mktemp -d guard holds, since an unusable TMPDIR leaves the marker path empty and the
# redirection never runs. This drives the redirection directly: the marker directory
# exists at startup and is gone by the time the write happens, which is what an operator
# clearing /tmp mid-run looks like. With `: >` the shell dies here and takes the rest of
# the uninstall with it. ──
_SM_FILE=$(mktemp "$_TMP_ROOT/setmarker.XXXXXX")
sed -n '/^_set_marker() {/,/^}/p' "$UNINSTALL_SH" > "$_SM_FILE"
if [ ! -s "$_SM_FILE" ]; then
    echo "  FAIL: could not extract _set_marker"; FAIL=$((FAIL+1))
else
    for _sh in dash busybox sh; do
        command -v "$_sh" >/dev/null 2>&1 || continue
        _probe=$(mktemp "$_TMP_ROOT/probe.XXXXXX")
        {
            cat "$_SM_FILE"
            echo '_set_marker "'"$_TMP_ROOT"'/gone-dir/marker"'
            echo 'echo SURVIVED'
        } > "$_probe"
        if [ "$_sh" = busybox ]; then
            _got=$(busybox sh "$_probe" 2>/dev/null)
        else
            _got=$("$_sh" "$_probe" 2>/dev/null)
        fi
        if [ "$_got" = "SURVIVED" ]; then
            echo "  PASS: $_sh: _set_marker survives an impossible write"; PASS=$((PASS+1))
        else
            echo "  FAIL: $_sh: _set_marker killed the shell (special-builtin redirection)"
            FAIL=$((FAIL+1))
        fi
    done
fi

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
