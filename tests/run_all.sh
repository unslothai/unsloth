#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
set -e

TESTS_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Bash tests ==="
# Discovered, not listed: a hand-maintained list drifts.
# tests/studio/test_ci_shell_suite_coverage.py keeps this in step with Backend CI.
SH_SKIP=""
for _t in "$TESTS_DIR"/sh/test_*.sh; do
    case " $SH_SKIP " in
        *" $(basename "$_t") "*) echo "skipping $(basename "$_t")"; continue ;;
    esac
    # bash, not sh: several tests/sh files use bashisms that fail under dash (/bin/sh on Debian/Ubuntu).
    bash "$_t"
done

echo ""
echo "=== Python tests ==="
python -m pytest "$TESTS_DIR/python/test_install_python_stack.py" -v
python -m pytest "$TESTS_DIR/python/test_cross_platform_parity.py" -v
python -m pytest "$TESTS_DIR/python/test_no_torch_filtering.py" -v
python -m pytest "$TESTS_DIR/python/test_studio_import_no_torch.py" -v
python -m pytest "$TESTS_DIR/python/test_tokenizers_and_torch_constraint.py" -v -k "not e2e"

echo ""
echo "All tests passed."
