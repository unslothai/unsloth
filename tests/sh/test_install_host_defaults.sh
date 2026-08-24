#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
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

studio_commands() {
    printf '%s\n' "$1" \
        | awk '/^[[:space:]]*(\$[[:space:]]*)?unsloth[[:space:]]+studio([[:space:]]|$)/'
}

assert_studio_wildcard_host_state() {
    _label="$1"; _haystack="$2"; _expected="$3"
    _commands=$(studio_commands "$_haystack")
    if printf '%s\n' "$_commands" \
        | grep -Eq '(^|[[:space:]])(-H|--host)(=|[[:space:]]+)0\.0\.0\.0([[:space:]]|$)'; then
        _actual="present"
    else
        _actual="absent"
    fi
    if [ "$_actual" = "$_expected" ]; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected wildcard host $_expected, found $_actual)"
        FAIL=$((FAIL + 1))
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
# Anchored on content, not a line count: both installers outgrew their tail windows.
echo "=== install.sh end-of-install block ==="

_end=$(awk '/In interactive terminals/{found=1} found{print}' "$INSTALL_SH")
# "read" alone also matches "readable" and "_can_read_tty", so pin the full prompt.
assert_contains \
    "install.sh: interactive block prompts user (read)" \
    "$_end" "read -r _reply"
assert_not_contains \
    "install.sh: no 'studio -H 0.0.0.0' in end-of-install commands" \
    "$_end" "studio -H 0.0.0.0"

echo ""
echo "=== install.ps1 end-of-install block ==="

_ps1_end=$(awk '/In interactive terminals/{found=1} found{print}' "$INSTALL_PS1")
assert_contains \
    "install.ps1: interactive block prompts user (Read-Host)" \
    "$_ps1_end" "Read-Host"
assert_not_contains \
    "install.ps1: no 'studio -H 0.0.0.0' in end-of-install commands" \
    "$_ps1_end" "studio -H 0.0.0.0"

echo ""
echo "=== studio/setup.sh launch hint ==="

_setup_tail=$(awk '/"launch"/{found=1} found{print}' "$SETUP_SH")
# Canary: an empty window would let the negative assertion below pass vacuously.
assert_contains \
    "studio/setup.sh: extraction found the launch hint" \
    "$_setup_tail" "unsloth studio -p 8888"
assert_not_contains \
    "studio/setup.sh: launch hint has no '-H 0.0.0.0'" \
    "$_setup_tail" "studio -H 0.0.0.0"

echo ""
echo "=== README.md Launch section ==="

# The primary Launch example must not include -H 0.0.0.0. Stop at the next heading:
# '#### Update' was later deleted, silently extending the section to the file end.
_readme_launch=$(awk '/^#### Launch$/{found=1; print; next} found && /^#{1,6} /{exit} found{print}' "$README")
# The opt-in belongs anywhere in the Studio quickstart, including a sibling subsection.
_readme_studio=$(awk '/^### Unsloth Studio \(web UI\)$/{found=1; print; next} found && /^#{1,3} /{exit} found{print}' "$README")
assert_contains \
    "README: Launch section exists" \
    "$_readme_launch" "unsloth studio"
assert_studio_wildcard_host_state \
    "README: Launch section primary command has no wildcard host" \
    "$_readme_launch" "absent"
assert_studio_wildcard_host_state \
    "README: Studio quickstart documents a wildcard-host opt-in" \
    "$_readme_studio" "present"

echo ""
echo "=== Results ==="
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo "FAILED"
    exit 1
fi
echo "ALL PASSED"
