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
    for _stub in open xdg-open; do
        printf '#!/bin/sh\necho "BROWSER_OPENED:$1" >> "$RECORD"\n' > "$_tmp/$_stub"
        chmod +x "$_tmp/$_stub"
    done

    # Off: no browser process, URL echoed instead.
    _out=$(RECORD="$_tmp/record_off" PATH="$_tmp:$PATH" bash -c \
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
    RECORD="$_tmp/record_on" PATH="$_tmp:$PATH" bash -c \
        "OPEN_BROWSER=1; $_fn; _open_browser http://localhost:9999" > /dev/null
    # xdg-open is backgrounded inside _open_browser; give the stub a moment.
    _i=0
    while [ ! -s "$_tmp/record_on" ] && [ "$_i" -lt 20 ]; do
        sleep 0.1
        _i=$((_i + 1))
    done
    assert_contains \
        "OPEN_BROWSER=1 invokes a browser opener with the URL" \
        "$(cat "$_tmp/record_on" 2>/dev/null)" "BROWSER_OPENED:http://localhost:9999"
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
# All launcher URL opens must route through the gated helper.
_ps1_direct_open=$(grep -cF 'Start-Process "http://localhost:' "$INSTALL_PS1" || true)
if [ "$_ps1_direct_open" -eq 0 ]; then
    echo "  PASS: no ungated Start-Process http://localhost calls remain"
    PASS=$((PASS + 1))
else
    echo "  FAIL: $_ps1_direct_open ungated Start-Process http://localhost call(s) remain"
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
