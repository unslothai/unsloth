#!/usr/bin/env bash
# Run all examples and report a summary.
# Usage:
#   bash examples/run_all.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Auto-discover example scripts. New examples added will be picked up automatically.
mapfile -t EXAMPLES < <(
    find "$SCRIPT_DIR" -maxdepth 1 -name "*.py" \
        ! -name "_*" \
        ! -name "run_all*" \
    | sort | xargs -n1 basename
)

PASSED=()
FAILED=()
SKIPPED=()

run_example() {
    local script="$1"
    local path="$SCRIPT_DIR/$script"
    echo
    echo "════════════════════════════════════════"
    echo " Running $script"
    echo "════════════════════════════════════════"
    # Capture output so we can detect [SKIP] without re-running the script.
    local output
    output=$(python "$path" 2>&1)
    local rc=$?
    echo "$output"
    # Read the last [PASS/SKIP/FAIL] tag from output.
    last_tag=$(echo "$output" | grep -oE '\[(PASS|SKIP|FAIL)\]' | tail -1)
    _LAST_TAG="$last_tag"
    return $rc
}

for script in "${EXAMPLES[@]}"; do
    _LAST_TAG=""
    run_example "$script"
    rc=$?
    # Classify outcome based on the last tag the script printed.
    # A crash also counts as a failure.
    if [[ $rc -ne 0 ]]; then
        FAILED+=("$script")
    else
        case "$_LAST_TAG" in
            "[PASS]") PASSED+=("$script")  ;;
            "[SKIP]") SKIPPED+=("$script") ;;
            *)        FAILED+=("$script")  ;;
        esac
    fi
done

echo
echo "════════════════════════════════════════"
echo " Summary"
echo "════════════════════════════════════════"
for s in "${PASSED[@]+"${PASSED[@]}"}";  do echo "  [PASSED] $s"; done
for s in "${SKIPPED[@]+"${SKIPPED[@]}"}"; do echo "  [SKIPPED] $s"; done
for s in "${FAILED[@]+"${FAILED[@]}"}";  do echo "  [FAILED] $s"; done

echo
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "${#FAILED[@]} example(s) failed."
else
    echo "All examples passed or skipped."
fi
