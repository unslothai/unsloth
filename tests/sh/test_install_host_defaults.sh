#!/bin/bash
# Static analysis: installer scripts and README must not hard-code 0.0.0.0 as default launch command.
# TDD: fails against current code, passes after host-default changes are applied.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
INSTALL_PS1="$SCRIPT_DIR/../../install.ps1"
README="$SCRIPT_DIR/../../README.md"
PASS=0
FAIL=0

assert_contains() {
    _label="$1"; _haystack="$2"; _needle="$3"
    if echo "$_haystack" | grep -qF -- "$_needle"; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected to find '$_needle')"
        FAIL=$((FAIL + 1))
    fi
}

assert_not_contains() {
    _label="$1"; _haystack="$2"; _needle="$3"
    if echo "$_haystack" | grep -qF -- "$_needle"; then
        echo "  FAIL: $_label (found '$_needle' but should not)"
        FAIL=$((FAIL + 1))
    else
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    fi
}

echo ""
echo "=== install.sh launcher template ==="

# Extract the heredoc block that generates launch-studio.sh.
# The block runs from the line containing 'cat > "$_css_launcher"'
# to the LAUNCHER_EOF terminator.
_launcher=$(awk '/cat > "\$_css_launcher"/{found=1} found{print} /^LAUNCHER_EOF$/{found=0}' "$INSTALL_SH")
# Verify the extraction worked (block must contain the shebang line)
assert_contains \
    "launcher template: extraction found the heredoc content" \
    "$_launcher" "#!/usr/bin/env bash"
# The generated launcher is only useful for local (desktop) use, so it must
# not hard-code 0.0.0.0 — the default 127.0.0.1 should be used instead.
assert_not_contains \
    "launcher template: no hardcoded 'studio -H 0.0.0.0'" \
    "$_launcher" "studio -H 0.0.0.0"

echo ""
echo "=== install.sh end-of-install block ==="

_end=$(tail -40 "$INSTALL_SH")
# The interactive path must prompt the user before starting Studio.
assert_contains \
    "install.sh: interactive block prompts user (bash read)" \
    "$_end" "read"
# The actual launch command and manual-instructions hint must not include
# 'studio -H 0.0.0.0' as the primary invocation.
assert_not_contains \
    "install.sh: no 'studio -H 0.0.0.0' in end-of-install commands" \
    "$_end" "studio -H 0.0.0.0"

echo ""
echo "=== install.ps1 end-of-install block ==="

_ps1_end=$(tail -20 "$INSTALL_PS1")
assert_contains \
    "install.ps1: interactive block prompts user (Read-Host)" \
    "$_ps1_end" "Read-Host"
assert_not_contains \
    "install.ps1: no 'studio -H 0.0.0.0' in end-of-install commands" \
    "$_ps1_end" "studio -H 0.0.0.0"

echo ""
echo "=== install.ps1 Windows launcher (New-StudioShortcuts) ==="

# The $studioCommand built inside the Windows launcher here-string must not
# hard-code -H 0.0.0.0 — desktop shortcuts are local-only and should use
# the loopback default.
_ps1_launcher=$(grep -n 'studioCommand' "$INSTALL_PS1")
assert_not_contains \
    "install.ps1: Windows launcher studioCommand has no -H 0.0.0.0" \
    "$_ps1_launcher" "studio -H 0.0.0.0"

echo ""
echo "=== README.md Launch section ==="

# The primary Launch example must show 'unsloth studio' without -H 0.0.0.0
# (0.0.0.0 may appear later as an opt-in note, but not in the code block).
_readme_launch=$(awk '/#### Launch/{found=1} found{print} /^####/{if(!/#### Launch/)found=0}' "$README" | head -10)
assert_contains \
    "README: Launch section exists" \
    "$_readme_launch" "unsloth studio"
assert_not_contains \
    "README: Launch section primary command has no -H 0.0.0.0" \
    "$_readme_launch" "studio -H 0.0.0.0"

echo ""
echo "=== Results ==="
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo "FAILED"
    exit 1
fi
echo "ALL PASSED"
