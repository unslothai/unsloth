#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Tests that setup.sh's fastpath escapes when UNSLOTH_DESKTOP_BACKEND_VERSION
# requires a backend upgrade even if INSTALLED_VER == LATEST_VER.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# Extract the fast-path check block from setup.sh
awk '/if \[ -n "\$INSTALLED_VER" \] && \[ -n "\$LATEST_VER" \] && \[ "\$INSTALLED_VER" = "\$LATEST_VER" \]/ {on=1}
     on && /^        _setup_pin=/ {exit}
     on {print}' "$SETUP_SH" > "$WORK/fastpath_blk.sh"
echo "fi" >> "$WORK/fastpath_blk.sh"

[ -s "$WORK/fastpath_blk.sh" ] || { echo "FATAL: fastpath block not found in $SETUP_SH" >&2; exit 1; }

PASS=0
FAIL=0

check() {
    local label="$1"
    local got="$2"
    local want="$3"
    if [ "$got" = "$want" ]; then
        echo "  PASS: $label (got=$got)"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $label (got=$got, want=$want)"
        FAIL=$((FAIL + 1))
    fi
}

# Create a mock venv that runs Python without site-packages, exercising setup's fallback parser.
VENV_DIR="$WORK/mock_venv"
mkdir -p "$VENV_DIR/bin"
cat << 'EOF' > "$VENV_DIR/bin/python"
#!/bin/sh
exec python3 -S "$@"
EOF
chmod +x "$VENV_DIR/bin/python"

# Mock install_manifest to return ok: True so manifest check passes
printf 'def verify_install():\n    return {"ok": True}\n' > "$WORK/install_manifest.py"

eval_fastpath() {
    local installed_ver="$1"
    local latest_ver="$2"
    local desktop_ver="${3:-}"
    (
        INSTALLED_VER="$installed_ver"
        LATEST_VER="$latest_ver"
        UNSLOTH_DESKTOP_BACKEND_VERSION="$desktop_ver"
        _PKG_NAME="unsloth"
        SCRIPT_DIR="$WORK"
        _SKIP_PYTHON_DEPS=false
        step() { :; }
        substep() { :; }
        
        # Execute extracted block
        # shellcheck disable=SC1090
        . "$WORK/fastpath_blk.sh"
        echo "$_SKIP_PYTHON_DEPS"
    )
}

echo "Testing UNSLOTH_DESKTOP_BACKEND_VERSION fastpath escape in setup.sh:"

# 1. When versions match and no desktop version required -> skips python deps
check "matching versions, no desktop requirement" \
    "$(eval_fastpath '2026.8.15' '2026.8.15' '')" "true"

# 2. When installed version satisfies desktop requirement -> skips python deps
check "installed satisfies desktop requirement" \
    "$(eval_fastpath '2026.8.15' '2026.8.15' '2026.8.15')" "true"

check "installed exceeds desktop requirement" \
    "$(eval_fastpath '2026.8.16' '2026.8.16' '2026.8.15')" "true"

# 3. When installed version is older than desktop requirement -> escapes fastpath (_SKIP_PYTHON_DEPS=false)
check "installed older than desktop requirement (2026.8.4 < 2026.8.15)" \
    "$(eval_fastpath '2026.8.4' '2026.8.4' '2026.8.15')" "false"

check "installed older than desktop requirement (2026.8.14 < 2026.8.15)" \
    "$(eval_fastpath '2026.8.14' '2026.8.14' '2026.8.15')" "false"

# 4. Without packaging, a suffix cannot be ordered safely, so force the dependency pass.
check "post-release requirement forces dependency pass without packaging" \
    "$(eval_fastpath '2026.8.15' '2026.8.15' '2026.8.15.post1')" "false"

echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
