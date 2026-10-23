#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# WebView data is created at first launch, keyed by bundle id; uninstall must remove it.
# Runs the full script against a fixture HOME with OS tools stubbed via PATH.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UNINSTALL_SH="$SCRIPT_DIR/../../scripts/uninstall.sh"
BID="ai.unsloth.studio"
PASS=0
FAIL=0

_TMP_ROOT=$(mktemp -d)
trap 'rm -rf "$_TMP_ROOT"' EXIT

# The script sweeps $XDG_RUNTIME_DIR launcher locks, so sandbox it or the real launcher loses its own.
XDG_RUNTIME_DIR="$_TMP_ROOT/run"
export XDG_RUNTIME_DIR
mkdir -p "$XDG_RUNTIME_DIR"

new_home() { mktemp -d "$_TMP_ROOT/home.XXXXXX"; }

assert_gone()    { _l="$1"; if [ -e "$2" ]; then echo "  FAIL: $_l (still present: $2)"; FAIL=$((FAIL+1)); else echo "  PASS: $_l"; PASS=$((PASS+1)); fi; }
assert_present() { _l="$1"; if [ -e "$2" ]; then echo "  PASS: $_l"; PASS=$((PASS+1)); else echo "  FAIL: $_l (missing: $2)"; FAIL=$((FAIL+1)); fi; }

STUB_BIN="$_TMP_ROOT/stubbin"
mkdir -p "$STUB_BIN"
printf '#!/bin/sh\nexit 0\n' > "$STUB_BIN/defaults"
chmod +x "$STUB_BIN/defaults"
# Exits 1 like real pkill on no match: a blanket 0 would hide a missing `|| true`.
PKILL_LOG="$_TMP_ROOT/pkill.args"
cat > "$STUB_BIN/pkill" <<EOF
#!/bin/sh
printf '%s\n' "\$*" >> "$PKILL_LOG"
exit 1
EOF
chmod +x "$STUB_BIN/pkill"
# On WSL hosts the /proc/version probe still fires; fail it. REAL_GREP must be absolute.
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
for _tool in powershell.exe sudo; do
    printf '#!/bin/sh\nexit 0\n' > "$STUB_BIN/$_tool"
    chmod +x "$STUB_BIN/$_tool"
done

# Empty apps dir so a real /Applications/Unsloth.app does not change the result.
APPS_DIR="$_TMP_ROOT/Applications"
mkdir -p "$APPS_DIR"

make_app() {
    mkdir -p "$1/Contents/MacOS"
    printf '#!/bin/sh\nexit 0\n' > "$1/Contents/MacOS/$2"
    chmod +x "$1/Contents/MacOS/$2"
    cat > "$1/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>$BID</string>
    <key>CFBundleExecutable</key>
    <string>$2</string>
</dict>
</plist>
EOF
}

run_uninstall() {
    printf '#!/bin/sh\necho %s\n' "$2" > "$STUB_BIN/uname"
    chmod +x "$STUB_BIN/uname"
    env -u UNSLOTH_STUDIO_HOME -u STUDIO_HOME \
        -u XDG_CACHE_HOME -u XDG_DATA_HOME -u XDG_CONFIG_HOME -u XDG_STATE_HOME \
        UNSLOTH_APPLICATIONS_DIR="${3:-$APPS_DIR}" \
        HOME="$1" PATH="$STUB_BIN:$PATH" sh "$UNINSTALL_SH" >/dev/null 2>&1
}

run_uninstall_out() {
    printf '#!/bin/sh\necho %s\n' "$2" > "$STUB_BIN/uname"
    chmod +x "$STUB_BIN/uname"
    env -u UNSLOTH_STUDIO_HOME -u STUDIO_HOME \
        -u XDG_CACHE_HOME -u XDG_DATA_HOME -u XDG_CONFIG_HOME -u XDG_STATE_HOME \
        UNSLOTH_APPLICATIONS_DIR="${3:-$APPS_DIR}" \
        HOME="$1" PATH="$STUB_BIN:$PATH" sh "$UNINSTALL_SH" 2>/dev/null
}

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

# ── 1b. macOS: the packaged desktop app owns this bundle id, so its data survives ──
OWNED_APPS="$_TMP_ROOT/Applications-owned"
make_app "$OWNED_APPS/Unsloth.app" unsloth-studio
H=$(new_home)
seed_app_data() {
    mkdir -p "$1/Library/Caches/$BID" "$1/Library/WebKit/$BID" \
             "$1/Library/Application Support/$BID" "$1/Library/HTTPStorages" \
             "$1/Library/Saved Application State/$BID.savedState" \
             "$1/Library/Preferences" "$1/Library/Cookies"
    : > "$1/Library/HTTPStorages/$BID.binarycookies"
    : > "$1/Library/Cookies/$BID.binarycookies"
    : > "$1/Library/Preferences/$BID.plist"
    make_app "$1/Applications/Unsloth Studio.app" launch-studio
}
seed_app_data "$H"
run_uninstall "$H" Darwin "$OWNED_APPS"
assert_present "macOS: Caches/$BID kept when the app owns it"              "$H/Library/Caches/$BID"
assert_present "macOS: WebKit/$BID kept when the app owns it"              "$H/Library/WebKit/$BID"
assert_present "macOS: Application Support/$BID kept when the app owns it" "$H/Library/Application Support/$BID"
assert_present "macOS: Saved Application State kept when the app owns it"  "$H/Library/Saved Application State/$BID.savedState"
assert_present "macOS: Preferences/$BID.plist kept when the app owns it"   "$H/Library/Preferences/$BID.plist"
assert_present "macOS: Cookies kept when the app owns it"                  "$H/Library/Cookies/$BID.binarycookies"
assert_gone    "macOS: shell launcher bundle still removed"                "$H/Applications/Unsloth Studio.app"

# ── 1c. macOS: the app is found by bundle id at any depth, not by path ──
for _case in "Unsloth Studio Beta.app" "AI & ML/Unsloth.app" "Development/AI/Local/Unsloth.app"; do
    MOVED_APPS="$_TMP_ROOT/Applications-moved"
    rm -rf "$MOVED_APPS"
    make_app "$MOVED_APPS/$_case" unsloth-studio
    H=$(new_home)
    seed_app_data "$H"
    run_uninstall "$H" Darwin "$MOVED_APPS"
    assert_present "macOS: app data kept for $_case"         "$H/Library/Caches/$BID"
    assert_present "macOS: prefs kept for $_case"            "$H/Library/Preferences/$BID.plist"
done

# ── 1d. macOS: install.sh's launcher bundle shares the id but is not the owner ──
LAUNCHER_APPS="$_TMP_ROOT/Applications-launcher"
make_app "$LAUNCHER_APPS/Unsloth Studio.app" launch-studio
H=$(new_home)
seed_app_data "$H"
run_uninstall "$H" Darwin "$LAUNCHER_APPS"
assert_gone "macOS: Caches/$BID removed when only a launcher is present" "$H/Library/Caches/$BID"
assert_gone "macOS: prefs removed when only a launcher is present"       "$H/Library/Preferences/$BID.plist"

H=$(new_home)
mkdir -p "$H/.cache/$BID" "$H/.local/share/$BID" "$H/.config/$BID" \
         "$H/.local/state/$BID" "$H/.cache/other.app" \
         "$H/.local/share/applications"
: > "$H/.local/share/applications/unsloth-studio-handler.desktop"
: > "$H/.local/share/applications/other-app.desktop"
: > "$XDG_RUNTIME_DIR/unsloth-studio-launcher-$(id -u).lock"
# Truncate first: the Darwin run logged the same argv.
: > "$PKILL_LOG"
run_uninstall "$H" Linux
assert_gone "linux: fixture launcher lock removed" \
    "$XDG_RUNTIME_DIR/unsloth-studio-launcher-$(id -u).lock"
assert_gone "linux: ~/.cache/$BID removed"       "$H/.cache/$BID"
assert_gone "linux: ~/.local/share/$BID removed" "$H/.local/share/$BID"
assert_gone "linux: ~/.config/$BID removed"      "$H/.config/$BID"
assert_gone "linux: ~/.local/state/$BID removed" "$H/.local/state/$BID"
assert_present "linux: unrelated app cache kept" "$H/.cache/other.app"
# tauri-plugin-deep-link rewrites <exe>-handler.desktop on every launch.
assert_gone "linux: deep-link handler .desktop removed" \
    "$H/.local/share/applications/unsloth-studio-handler.desktop"
assert_present "linux: another app's .desktop kept" \
    "$H/.local/share/applications/other-app.desktop"
# Unscoped, a root-run uninstall signals every user's app; -u is the $HOME owner.
_want_uid=$(stat -c %u "$H" 2>/dev/null || stat -f %u "$H")
if grep -q -- "-x -u $_want_uid unsloth-studio" "$PKILL_LOG" 2>/dev/null; then
    echo "  PASS: linux: app kill scoped to the \$HOME owner"; PASS=$((PASS+1))
else
    echo "  FAIL: linux: app kill not scoped (want '-x -u $_want_uid unsloth-studio')"; FAIL=$((FAIL+1))
fi

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

# ── 3b. Relative XDG overrides are invalid per spec and ignored by Tauri ──
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

# ── 3d. Symlinked $HOME: stat needs -L or a root-owned link resolves to uid 0 ──
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

# ── 3c. A path that cannot be removed must not be reported as gone ──
H=$(new_home)
mkdir -p "$H/.cache/$BID"
# An rm stub, not chmod: root ignores mode bits.
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

# ── 3e. Same for a custom root, whose removal runs in a pipeline subshell ──
H=$(new_home)
CUSTOM="$_TMP_ROOT/customroot.$$"
mkdir -p "$CUSTOM/share"
: > "$CUSTOM/share/studio.conf"
cat > "$STUB_BIN/rm" <<EOF
#!/bin/sh
for _a in "\$@"; do
    [ "\$_a" = "$CUSTOM" ] && exit 1
done
exec "$REAL_RM" "\$@"
EOF
chmod +x "$STUB_BIN/rm"
printf '#!/bin/sh\necho Linux\n' > "$STUB_BIN/uname"; chmod +x "$STUB_BIN/uname"
# env stops parsing options at the first VAR=VALUE.
_out=$(env -u STUDIO_HOME -u XDG_CACHE_HOME -u XDG_DATA_HOME -u XDG_CONFIG_HOME -u XDG_STATE_HOME \
    UNSLOTH_STUDIO_HOME="$CUSTOM" HOME="$H" PATH="$STUB_BIN:$PATH" sh "$UNINSTALL_SH" 2>/dev/null)
rm -f "$STUB_BIN/rm"
assert_present "linux: the refused custom root really did survive" "$CUSTOM"
case "$_out" in
    *"are gone."*) echo "  FAIL: linux: custom-root failure still claimed the data is gone"; FAIL=$((FAIL+1)) ;;
    *) echo "  PASS: linux: custom-root failure reaches the summary"; PASS=$((PASS+1)) ;;
esac

# ── 3g. Env-mode install, bare uninstall: studio.db is not found, so do not claim it gone ──
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

# ── 3h. When studio.db IS removed the full claim must return ──
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

# ── 3f. A deny-listed custom root is incomplete removal too ──
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

# ── 3i. An unusable TMPDIR must not abort: `:` is a special builtin, so `: >` errors kill sh ──
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
    # busybox needs its applet name as a separate argument, so spell out both branches.
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

# ── 3j. The marker directory is removed on exit (fixture TMPDIR) ──
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

# ── 3j2. No marker storage: the summary must take the cautious branch ──
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

# ── 3j3. ~/.unsloth/studio symlinked elsewhere: rm unlinks only the link ──
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

# ── 3j4. Marker storage vanishing mid-run must not read as success ──
H=$(new_home)
mkdir -p "$H/.local/share/$BID"
_VAN=$(mktemp -d "$_TMP_ROOT/vanish.XXXXXX")
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

# ── 3j5. studio.db symlinked out of the tree: rm unlinks only the link ──
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

# ── 3j5b. Same without `readlink -f` (BSD before macOS 12.3) and a relative target ──
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

# ── 3j6. Provider API keys live in browser localStorage, never removed here ──
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
    # Browser sessions keep tokens in localStorage, so the claim must be qualified.
    case "$_out" in
        *"the signed-in session is gone"*)
            echo "  FAIL: $_case: unqualified signed-out claim"; FAIL=$((FAIL+1)) ;;
        *)  echo "  PASS: $_case: signed-out claim scoped to the desktop app"
            PASS=$((PASS+1)) ;;
    esac
done

# ── 3j7. A refused default root still holds studio.db; the summary must not deny it ──
H=$(new_home)
mkdir -p "$H/.unsloth/studio"
: > "$H/.unsloth/studio/studio.db"
_out=$(run_uninstall_out "$H" Linux)
_err=$(printf '#!/bin/sh\necho Linux\n' > "$STUB_BIN/uname"; chmod +x "$STUB_BIN/uname";
       env -u UNSLOTH_STUDIO_HOME -u STUDIO_HOME UNSLOTH_APPLICATIONS_DIR="$APPS_DIR" \
           HOME="$H" PATH="$STUB_BIN:$PATH" sh "$UNINSTALL_SH" 2>&1 >/dev/null)
case "$_err" in
    *"refusing to remove non-Unsloth path"*)
        echo "  PASS: refused default root: says it refused"; PASS=$((PASS+1)) ;;
    *)  echo "  FAIL: refused default root: never says it refused"; FAIL=$((FAIL+1)) ;;
esac
if [ -f "$H/.unsloth/studio/studio.db" ]; then
    echo "  PASS: refused default root: the database survived"; PASS=$((PASS+1))
else
    echo "  FAIL: refused default root: the database was removed"; FAIL=$((FAIL+1))
fi
case "$_out" in
    *"No studio.db was found"*)
        echo "  FAIL: refused default root: claimed no studio.db was found"; FAIL=$((FAIL+1)) ;;
    *)  echo "  PASS: refused default root: no claim that none was found"; PASS=$((PASS+1)) ;;
esac
case "$_out" in
    *"carries no Unsloth install marker"*)
        echo "  PASS: refused default root: names the kept directory"; PASS=$((PASS+1)) ;;
    *)  echo "  FAIL: refused default root: never names the kept directory"; FAIL=$((FAIL+1)) ;;
esac
# Refused because it is not ours, so "remove by hand" advice would defeat the gate.
case "$_out" in
    *"those paths by hand"*)
        echo "  FAIL: refused default root: told the reader to delete a foreign directory"
        FAIL=$((FAIL+1)) ;;
    *)  echo "  PASS: refused default root: no advice to delete it"; PASS=$((PASS+1)) ;;
esac

# ── 3j8. A dangling symlink at ~/.unsloth/studio still counts as present ──
H=$(new_home)
mkdir -p "$H/.unsloth"
ln -s "$H/.unsloth/nowhere" "$H/.unsloth/studio"
run_uninstall "$H" Linux
if [ -L "$H/.unsloth/studio" ]; then
    echo "  PASS: a dangling symlink at the default root is refused, not unlinked"; PASS=$((PASS+1))
else
    echo "  FAIL: a dangling symlink at the default root was removed"; FAIL=$((FAIL+1))
fi

# ── 3k. _set_marker must survive a write to a marker dir removed mid-run ──
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

# ── 3m. uninstall.ps1 must also stop the legacy Unsloth.exe, or it recreates the profile ──
UNINSTALL_PS1="$SCRIPT_DIR/../../scripts/uninstall.ps1"
if [ ! -f "$UNINSTALL_PS1" ]; then
    echo "  FAIL: uninstall.ps1 not found"; FAIL=$((FAIL+1))
else
    _ps_filter=$(grep -n "Name = 'unsloth-studio.exe'" "$UNINSTALL_PS1" | head -n1 | cut -d: -f2-)
    case "$_ps_filter" in
        *"Unsloth.exe"*)
            echo "  PASS: windows: process query covers the legacy binary name"; PASS=$((PASS+1)) ;;
        *)
            echo "  FAIL: windows: process query misses the legacy Unsloth.exe"; FAIL=$((FAIL+1)) ;;
    esac
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
