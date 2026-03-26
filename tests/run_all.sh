#!/bin/sh
# Run all installer tests.
set -e

TESTS_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Bash tests ==="
sh "$TESTS_DIR/sh/test_get_torch_index_url.sh"

echo ""
echo "=== Python tests ==="
python -m pytest "$TESTS_DIR/python/test_install_python_stack.py" -v
python -m pytest "$TESTS_DIR/python/test_cross_platform_parity.py" -v

echo ""
echo "All tests passed."
