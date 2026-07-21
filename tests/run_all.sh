#!/bin/sh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Run all installer tests.
set -e

TESTS_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Bash tests ==="
sh "$TESTS_DIR/sh/test_get_torch_index_url.sh"
sh "$TESTS_DIR/sh/test_mac_intel_compat.sh"
sh "$TESTS_DIR/sh/test_torch_constraint.sh"
sh "$TESTS_DIR/sh/test_nvcc_meets_llama_minimum.sh"
sh "$TESTS_DIR/sh/test_resolve_cuda_archs.sh"
sh "$TESTS_DIR/sh/test_select_cuda_jit_tools.sh"
sh "$TESTS_DIR/sh/test_strixhalo_wsl_reroute.sh"
sh "$TESTS_DIR/sh/test_uninstall_shared_icon.sh"
sh "$TESTS_DIR/sh/test_torch_flavor.sh"
sh "$TESTS_DIR/sh/test_redact_install_output.sh"
sh "$TESTS_DIR/sh/test_install_uv_override_space.sh"

echo ""
echo "=== Python tests ==="
python -m pytest "$TESTS_DIR/python/test_install_python_stack.py" -v
python -m pytest "$TESTS_DIR/python/test_cross_platform_parity.py" -v
python -m pytest "$TESTS_DIR/python/test_no_torch_filtering.py" -v
python -m pytest "$TESTS_DIR/python/test_studio_import_no_torch.py" -v
python -m pytest "$TESTS_DIR/python/test_tokenizers_and_torch_constraint.py" -v -k "not e2e"

echo ""
echo "All tests passed."
