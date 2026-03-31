#!/usr/bin/env bash
# SPDX-License-Identifier: Apache 2.0
#
# Clean rebuild of Unsloth Studio (frontend + backend)
#
# Removes frontend build artifacts (dist/, node_modules/),
# deletes the Python virtual environment, and runs a fresh
# local install. Useful for developers working on Studio code.
#
# Usage:
#   ./rebuild.sh           # Normal output
#   ./rebuild.sh -v        # Verbose output

set -euo pipefail

# Parse args
VERBOSE=0
for arg in "$@"; do
    if [[ "$arg" == "-v" || "$arg" == "--verbose" ]]; then
        VERBOSE=1
        export UNSLOTH_VERBOSE=1
    fi
done

echo ""
echo "=== Unsloth Studio Clean Rebuild ==="
echo ""

# ── Clean Frontend ──
echo "[1/3] Cleaning frontend build artifacts..."

FRONTEND_DIST="studio/frontend/dist"
FRONTEND_NODE_MODULES="studio/frontend/node_modules"

if [ -d "$FRONTEND_DIST" ]; then
    echo "  → Removing $FRONTEND_DIST"
    rm -rf "$FRONTEND_DIST"
    echo "  ✓ Deleted frontend dist"
else
    echo "  → $FRONTEND_DIST not found (already clean)"
fi

if [ -d "$FRONTEND_NODE_MODULES" ]; then
    echo "  → Removing $FRONTEND_NODE_MODULES (this may take a moment...)"
    rm -rf "$FRONTEND_NODE_MODULES"
    echo "  ✓ Deleted node_modules"
else
    echo "  → node_modules not found (already clean)"
fi

# ── Clean Python Environment ──
echo ""
echo "[2/3] Cleaning Python virtual environment..."

VENV_PATH="$HOME/.unsloth/studio/unsloth_studio"

if [ -d "$VENV_PATH" ]; then
    echo "  → Removing $VENV_PATH"
    rm -rf "$VENV_PATH"
    echo "  ✓ Deleted venv successfully"
else
    echo "  → Venv not found (already clean)"
fi

# ── Rebuild Everything ──
echo ""
echo "[3/3] Running fresh local install..."
echo ""

bash ./install.sh --local

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Rebuild Complete! ==="
else
    echo ""
    echo "=== Rebuild Failed ==="
    exit 1
fi
