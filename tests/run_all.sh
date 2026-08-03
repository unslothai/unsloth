#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Run all installer tests.
set -e

TESTS_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Bash tests ==="
# Discovered, not listed: a hand-maintained list drifts (this one had fallen
# eight files behind sh/, and the Backend CI copy of it had fallen seven).
# Backend CI discovers the same directory and skips the same files:
# test_install_rollback_lifecycle.sh, which cross-platform-parity-ci.yml already
# runs on both platforms. tests/studio/test_ci_shell_suite_coverage.py fails if
# either side stops discovering, or skips something undocumented.
SH_SKIP=""
for _t in "$TESTS_DIR"/sh/test_*.sh; do
    case " $SH_SKIP " in
        *" $(basename "$_t") "*) echo "skipping $(basename "$_t")"; continue ;;
    esac
    # bash, not sh: every file under sh/ declares a bash shebang, and three of
    # them fail on bashisms under dash, which is /bin/sh on Debian and Ubuntu
    # (test_apt_distro_prompt, test_studio_home_node_dir, and
    # test_with_llama_cpp_dir_link_behavior). The old hand-written list happened
    # to name only dash-clean files, so discovering the directory is what
    # exposed this. Backend CI already invokes them with bash.
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
