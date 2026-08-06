#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# The generated desktop launchers must support suppressing the automatic
# default-browser open after the server becomes healthy (PWA users run Studio
# in a browser-app window, not the OS default browser). Covered surface:
#   - launch-studio.sh: --no-browser flag, UNSLOTH_STUDIO_NO_BROWSER env var,
#     persisted STUDIO_OPEN_BROWSER from studio.conf; prints the URL when off.
#   - install.sh: --no-browser flag, interactive prompt, studio.conf persistence
#     that survives the --shortcuts-only refresh run by `unsloth studio update`.
#   - install.ps1: same feature baked into launch-studio.ps1 (Open-StudioUrl).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
INSTALL_PS1="$SCRIPT_DIR/../../install.ps1"
PASS=0
FAIL=0

# Here-strings, not `echo | grep -q`: grep exiting on first match SIGPIPEs
# the echo on large haystacks, spamming "write error: Broken pipe" in CI logs.
assert_contains() {
    _label="$1"; _haystack="$2"; _needle="$3"
    if grep -qF -- "$_needle" <<< "$_haystack"; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected to find '$_needle')"
        FAIL=$((FAIL + 1))
    fi
}

assert_not_contains() {
    _label="$1"; _haystack="$2"; _needle="$3"
    if grep -qF -- "$_needle" <<< "$_haystack"; then
        echo "  FAIL: $_label (found '$_needle' but should not)"
        FAIL=$((FAIL + 1))
    else
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    fi
}

echo ""
echo "=== install.sh launcher template ==="

# Extract the heredoc that generates ~/.local/share/unsloth/launch-studio.sh.
_launcher=$(awk '/cat > "\$_css_launcher"/{found=1} found{print} /^LAUNCHER_EOF$/{found=0}' "$INSTALL_SH")
assert_contains \
    "launcher template: --no-browser argument handled" \
    "$_launcher" "--no-browser) OPEN_BROWSER=0"
assert_contains \
    "launcher template: UNSLOTH_STUDIO_NO_BROWSER env var handled" \
    "$_launcher" "UNSLOTH_STUDIO_NO_BROWSER"
# Mixed-case falsy values (False, Off) must not disable, matching the
# PowerShell launcher's case-insensitive -notin.
assert_contains \
    "launcher template: env var check is case-insensitive" \
    "$_launcher" "tr '[:upper:]' '[:lower:]'"
assert_contains \
    "launcher template: studio.conf preference is the default" \
    "$_launcher" 'OPEN_BROWSER="${STUDIO_OPEN_BROWSER:-1}"'
assert_contains \
    "launcher template: _open_browser is gated on OPEN_BROWSER" \
    "$_launcher" '[ "$OPEN_BROWSER" = "0" ]'
assert_contains \
    "launcher template: URL still printed when auto-open is off" \
    "$_launcher" "Unsloth Studio is running at:"

echo ""
echo "=== install.sh installer plumbing ==="

_installer=$(cat "$INSTALL_SH")
assert_contains \
    "install.sh: --no-browser flag parsed" \
    "$_installer" "--no-browser) _STUDIO_OPEN_BROWSER=0"
assert_contains \
    "install.sh: preference persisted into studio.conf" \
    "$_installer" "STUDIO_OPEN_BROWSER='\$_css_open_browser'"
assert_contains \
    "install.sh: existing studio.conf choice preserved on refresh" \
    "$_installer" "s/^STUDIO_OPEN_BROWSER="
assert_contains \
    "install.sh: interactive prompt asks about browser auto-open" \
    "$_installer" "Open Unsloth Studio in your default browser after launch?"
# A reinstall that accepts the prompt default must keep the saved choice.
assert_contains \
    "install.sh: prompt Enter keeps the persisted preference" \
    "$_installer" '*) _STUDIO_OPEN_BROWSER="${_existing_open_browser:-1}"'
# UNSLOTH_SKIP_AUTOSTART is documented for automated installs, so it must gate
# this prompt too: under `curl ... | sh` only stdin is the pipe, [ -t 1 ] is
# still true, so an ungated prompt blocks an install that used to run unattended.
assert_contains \
    "install.sh: browser prompt respects UNSLOTH_SKIP_AUTOSTART" \
    "$_installer" 'if [ "$_SKIP_AUTOSTART" != true ] && [ -z "$_STUDIO_OPEN_BROWSER" ]'
# The WSL Strix Halo reroute must forward an explicit browser choice.
assert_contains \
    "install.sh: reroute forwards --no-browser" \
    "$_installer" '[ "$_STUDIO_OPEN_BROWSER" = "0" ] && _rr_args="$_rr_args --no-browser"'
# The post-install foreground launch honors the preference too.
assert_contains \
    "install.sh: post-install launch opens browser via gated watcher" \
    "$_installer" '_post_install_browser_watch "$_post_install_port"'
assert_contains \
    "install.sh: post-install launch selects a free port" \
    "$_installer" "_find_post_install_port()"
assert_contains \
    "install.sh: server uses the watcher-selected port" \
    "$_installer" 'studio -p "$_post_install_port"'
# The --shortcuts-only early exit must not EPIPE a curl | sh pipeline.
assert_contains \
    "install.sh: shortcuts-only exit drains piped stdin" \
    "$_installer" '[ ! -t 0 ] && cat > /dev/null'

echo ""
echo "=== install.sh _open_browser gating (functional) ==="

# Extract the _open_browser function from the (column-0) launcher heredoc and
# drive it with stubbed browser openers on PATH.
_fn=$(printf '%s\n' "$_launcher" | awk '/^_open_browser\(\) \{/{found=1} found{print} found && /^\}/{exit}')
if [ -z "$_fn" ]; then
    echo "  FAIL: could not extract _open_browser from launcher template"
    FAIL=$((FAIL + 1))
else
    _tmp=$(mktemp -d)
    trap 'rm -rf "$_tmp"' EXIT
    # Stub EVERY opener _open_browser can reach, not just open/xdg-open: on WSL
    # it prefers powershell.exe then cmd.exe, so on a WSL dev box an unstubbed
    # run escapes to the real Windows browser and opens a window for real.
    for _stub in open xdg-open powershell.exe cmd.exe; do
        printf '#!/bin/sh\necho "BROWSER_OPENED:$*" >> "$RECORD"\n' > "$_tmp/$_stub"
        chmod +x "$_tmp/$_stub"
    done
    # Pin PATH to the stubs plus coreutils so no host opener can leak in, and
    # force the OS branch rather than inheriting the runner's.
    _stub_path="$_tmp:/usr/bin:/bin"
    printf '#!/bin/sh\necho Linux\n' > "$_tmp/uname"
    chmod +x "$_tmp/uname"
    # Fixture /proc/version: "microsoft" selects the WSL branch, anything else
    # the plain-Linux branch. Lets one Linux runner cover both.
    printf 'Linux version 5.15.0-microsoft-standard-WSL2\n' > "$_tmp/procversion_wsl"
    printf 'Linux version 6.1.0-generic\n' > "$_tmp/procversion_linux"
    _fn_linux=$(printf '%s\n' "$_fn" | sed "s#/proc/version#$_tmp/procversion_linux#")

    # Off: no browser process, URL echoed instead.
    _out=$(RECORD="$_tmp/record_off" PATH="$_stub_path" bash -c \
        "OPEN_BROWSER=0; $_fn; _open_browser http://localhost:9999")
    assert_contains \
        "OPEN_BROWSER=0 prints the URL" \
        "$_out" "Unsloth Studio is running at: http://localhost:9999"
    if [ -f "$_tmp/record_off" ]; then
        echo "  FAIL: OPEN_BROWSER=0 still invoked a browser opener"
        FAIL=$((FAIL + 1))
    else
        echo "  PASS: OPEN_BROWSER=0 does not invoke a browser opener"
        PASS=$((PASS + 1))
    fi

    # On (default): browser opener invoked with the URL.
    RECORD="$_tmp/record_on" PATH="$_stub_path" bash -c \
        "OPEN_BROWSER=1; $_fn_linux; _open_browser http://localhost:9999" > /dev/null
    # xdg-open is backgrounded inside _open_browser; give the stub a moment.
    _wait_for_record() {
        _i=0
        while [ ! -s "$1" ] && [ "$_i" -lt 20 ]; do
            sleep 0.1
            _i=$((_i + 1))
        done
    }
    _wait_for_record "$_tmp/record_on"
    assert_contains \
        "OPEN_BROWSER=1 invokes a browser opener with the URL" \
        "$(cat "$_tmp/record_on" 2>/dev/null)" "BROWSER_OPENED:http://localhost:9999"

    # Each rung of the WSL ladder, by removing the one above it. Without this,
    # the WSL branch is never exercised on a Linux CI runner at all.
    _rung=1
    for _drop in "" "powershell.exe" "powershell.exe cmd.exe"; do
        _rung_dir="$_tmp/rung$_rung"
        mkdir -p "$_rung_dir"
        for _s in open xdg-open powershell.exe cmd.exe uname; do
            cp "$_tmp/$_s" "$_rung_dir/$_s"
        done
        for _s in $_drop; do rm -f "$_rung_dir/$_s"; done
        case "$_drop" in
            "") _want="powershell.exe" ;;
            "powershell.exe") _want="cmd.exe" ;;
            *) _want="xdg-open" ;;
        esac
        _rec="$_tmp/record_rung$_rung"
        _fn_r=$(printf '%s\n' "$_fn" | sed "s#/proc/version#$_tmp/procversion_wsl#")
        RECORD="$_rec" PATH="$_rung_dir:/usr/bin:/bin" bash -c \
            "OPEN_BROWSER=1; $_fn_r; _open_browser http://localhost:9999" > /dev/null 2>&1
        _wait_for_record "$_rec"
        # The recorder prints the whole arg list, so the URL shows up for
        # `powershell.exe -NoProfile -Command "Start-Process '<url>'"` too.
        assert_contains \
            "WSL rung: ${_want} receives the URL when it is the first available opener" \
            "$(cat "$_rec" 2>/dev/null)" "http://localhost:9999"
        _rung=$((_rung + 1))
    done

    # And with no opener at all, nothing is spawned and a hint goes to stderr.
    # PATH must contain ONLY this dir: leaving /usr/bin on it would find the
    # real xdg-open and the "no opener" case would never be reached.
    _none_dir="$_tmp/rung_none"
    mkdir -p "$_none_dir"
    cp "$_tmp/uname" "$_none_dir/uname"
    cp "$(command -v grep)" "$_none_dir/grep"
    # Absolute path for bash: the PATH assignment below also governs how bash
    # itself is looked up, and this PATH deliberately holds almost nothing.
    _bash=$(command -v bash)
    _err=$(RECORD="$_tmp/record_none" PATH="$_none_dir" "$_bash" -c \
        "OPEN_BROWSER=1; $_fn_linux; _open_browser http://localhost:9999" 2>&1 >/dev/null)
    assert_contains \
        "no opener available: prints a manual hint instead" \
        "$_err" "Open in your browser: http://localhost:9999"
fi

echo ""
echo "=== install.ps1 launcher template ==="

# grep the file directly: piping the whole installer into grep -q trips
# SIGPIPE noise from echo once grep exits on first match.
assert_file_contains() {
    _label="$1"; _file="$2"; _needle="$3"
    if grep -qF -- "$_needle" "$_file"; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected to find '$_needle')"
        FAIL=$((FAIL + 1))
    fi
}

assert_file_contains \
    "install.ps1: --no-browser flag parsed" \
    "$INSTALL_PS1" "\"--no-browser\" { \$OpenBrowserPref = '0' }"
assert_file_contains \
    "install.ps1: preference baked into launch-studio.ps1" \
    "$INSTALL_PS1" "openBrowserDefault = '\$_openBrowser'"
assert_file_contains \
    "install.ps1: launcher honors UNSLOTH_STUDIO_NO_BROWSER" \
    "$INSTALL_PS1" "UNSLOTH_STUDIO_NO_BROWSER"
assert_file_contains \
    "install.ps1: gated helper defined" \
    "$INSTALL_PS1" "function Open-StudioUrl {"
assert_file_contains \
    "install.ps1: interactive prompt asks about browser auto-open" \
    "$INSTALL_PS1" "Open Unsloth Studio in your default browser after launch?"
assert_file_contains \
    "install.ps1: prompt Enter keeps the baked preference" \
    "$INSTALL_PS1" 'elseif ($_existingPref) { $_existingPref }'
assert_file_contains \
    "install.ps1: browser prompt respects UNSLOTH_SKIP_AUTOSTART" \
    "$INSTALL_PS1" '$_browserPromptOk = (-not $SkipAutostart) -and [Environment]::UserInteractive -and'
assert_file_contains \
    "install.ps1: post-install launch opens browser via gated watcher" \
    "$INSTALL_PS1" 'Start-Job -ScriptBlock $_browserWatch'
assert_file_contains \
    "install.ps1: post-install launch selects a free port" \
    "$INSTALL_PS1" 'function Find-PostInstallStudioPort {'
assert_file_contains \
    "install.ps1: watcher receives the selected port" \
    "$INSTALL_PS1" 'ArgumentList @($_watchRootId, $_launchPort)'
assert_file_contains \
    "install.ps1: server uses the watcher-selected port" \
    "$INSTALL_PS1" '& $UnslothExe studio -p $_launchPort'
# All launcher URL opens must route through the gated helper. The one
# allowed direct call is inside the post-install $_browserWatch scriptblock,
# whose Start-Job call site is itself gated on the preference.
_ps1_direct_open=$(grep -cE 'Start-Process "http://localhost:' "$INSTALL_PS1" || true)
_ps1_watch_open=$(awk '/\$_browserWatch = \{/{f=1} f && /^    \}$/{exit} f' "$INSTALL_PS1" \
    | grep -cE 'Start-Process "http://localhost:' || true)
if [ "$_ps1_direct_open" -eq 1 ] && [ "$_ps1_watch_open" -eq 1 ]; then
    echo "  PASS: only the gated browser watcher opens a URL directly"
    PASS=$((PASS + 1))
else
    echo "  FAIL: found $_ps1_direct_open direct URL opens ($_ps1_watch_open in the watcher); all others must route through Open-StudioUrl"
    FAIL=$((FAIL + 1))
fi

echo ""
echo "=== Results ==="
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo "FAILED"
    exit 1
fi
echo "ALL PASSED"
