#!/bin/bash
# Static analysis: installer scripts and README must not hard-code 0.0.0.0
# in any user-visible default launch command. The dynamic-port launcher
# templates and post-install hints should rely on the new loopback default.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
INSTALL_PS1="$SCRIPT_DIR/../../install.ps1"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
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

# Extract the heredoc that generates ~/.local/share/unsloth/launch-studio.sh.
_launcher=$(awk '/cat > "\$_css_launcher"/{found=1} found{print} /^LAUNCHER_EOF$/{found=0}' "$INSTALL_SH")
assert_contains \
    "launcher template: extraction found the heredoc content" \
    "$_launcher" "#!/usr/bin/env bash"
# The desktop launcher should rely on the new 127.0.0.1 default.
assert_not_contains \
    "launcher template: no hardcoded 'studio -H 0.0.0.0'" \
    "$_launcher" "studio -H 0.0.0.0"

echo ""
echo "=== install.sh end-of-install block ==="

_end=$(tail -50 "$INSTALL_SH")
assert_contains \
    "install.sh: interactive block prompts user (read)" \
    "$_end" "read"
assert_not_contains \
    "install.sh: no 'studio -H 0.0.0.0' in end-of-install commands" \
    "$_end" "studio -H 0.0.0.0"

echo ""
echo "=== install.ps1 end-of-install block ==="

_ps1_end=$(tail -25 "$INSTALL_PS1")
assert_contains \
    "install.ps1: interactive block prompts user (Read-Host)" \
    "$_ps1_end" "Read-Host"
assert_not_contains \
    "install.ps1: no 'studio -H 0.0.0.0' in end-of-install commands" \
    "$_ps1_end" "studio -H 0.0.0.0"

echo ""
echo "=== studio/setup.sh launch hint ==="

_setup_tail=$(tail -30 "$SETUP_SH")
assert_not_contains \
    "studio/setup.sh: launch hint has no '-H 0.0.0.0'" \
    "$_setup_tail" "studio -H 0.0.0.0"

echo ""
echo "=== README.md Launch section ==="

# The primary Launch example must not include -H 0.0.0.0; the LAN/cloud
# note appears as an opt-in line outside the code block.
_readme_launch=$(awk '/^#### Launch$/{found=1} found{print} /^#### Update$/{found=0}' "$README")
assert_contains \
    "README: Launch section exists" \
    "$_readme_launch" "unsloth studio"
assert_not_contains \
    "README: Launch section primary command has no -H 0.0.0.0" \
    "$_readme_launch" "studio -H 0.0.0.0"
assert_contains \
    "README: Launch section documents -H 0.0.0.0 opt-in" \
    "$_readme_launch" "0.0.0.0"

echo ""
echo "=== Results ==="
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo "FAILED"
    exit 1
fi
echo "ALL PASSED"
