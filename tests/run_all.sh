#!/bin/sh
# Run all installer tests.
set -e

TESTS_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Bash tests ==="
sh "$TESTS_DIR/sh/test_get_torch_index_url.sh"
sh "$TESTS_DIR/sh/test_mac_intel_compat.sh"
sh "$TESTS_DIR/sh/test_torch_constraint.sh"

echo ""
echo "=== Python tests ==="
python -m pytest "$TESTS_DIR/python/test_install_python_stack.py" -v
python -m pytest "$TESTS_DIR/python/test_cross_platform_parity.py" -v
python -m pytest "$TESTS_DIR/python/test_no_torch_filtering.py" -v
python -m pytest "$TESTS_DIR/python/test_studio_import_no_torch.py" -v
python -m pytest "$TESTS_DIR/python/test_tokenizers_and_torch_constraint.py" -v -k "not e2e"

echo ""
echo "All tests passed."
