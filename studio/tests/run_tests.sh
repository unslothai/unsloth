#!/usr/bin/env bash
# tests/run_tests.sh

set -o errexit
trap 'echo "Aborting due to errexit on line $LINENO. Exit code: $?" >&2' ERR
set -o errtrace
set -e -o pipefail
set -x

# Colors
C_OK='\033[0;32m'
C_ERR='\033[0;31m'
C_DIM='\033[0;90m'
C_RST='\033[0m'

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$TEST_DIR/docker"
REPO_ROOT="$(cd "$TEST_DIR/../.." && pwd)"
RESULTS_LOG="$TEST_DIR/test_results.log"
REBUILD_DOCKER_IMAGES="${REBUILD_DOCKER_IMAGES:-true}"
CLEAN_MODE="${CLEAN_MODE:-false}"

# --- CACHE CONFIGURATION ---
# We create a persistent cache directory on the host to speed up bun/npm/llama
CACHE_DIR="$TEST_DIR/permanent_cache"


# Test Case Definition: "ID|Dockerfile|Expected_Exit_Code|Description"
# UPDATED: All IDs are now lowercase to comply with Docker tag requirements
TEST_CASES=(
    "fresh_linux|test_fresh_linux.Dockerfile|0|Clean installation"
    "prebuilt_linux|test_prebuilt_linux.Dockerfile|0|Existing prebuilt validation"
    "colab|test_colab.Dockerfile|0|Colab environment (dependency stripping)"
    "wsl|test_wsl.Dockerfile|0|WSL environment (dependency installation)"
    "macos_mock|test_macos_mock.Dockerfile|0|macOS Darwin emulation (Metal flags)"
    "llama_only|test_llama_only.Dockerfile|0|Llama-only mode (skip frontend)"
    "force_compile|test_force_compile.Dockerfile|0|Force source build (skip prebuilt)"
)

prepare_environment() {
    if [[ "${CLEAN_MODE}" == "true" ]]; then
        rm -rf "${CACHE_DIR}"
    fi
    mkdir -p "$CACHE_DIR/.bun" "$CACHE_DIR/.npm" "$CACHE_DIR/llama_cpp"
    # Clear previous results
    > "$RESULTS_LOG"
}

run_test_case() {
    local IFS='|'
    read -r ID DOCKER_FILE EXPECTED_EXIT DESC <<< "$1"

    echo -e "${C_DIM}Running Test: $ID...${C_RST}"
    echo -e "  Description: $DESC"

    # 1. Build the specific test container
    # Using lowercase ID ensures the tag is valid for Docker
    if ! docker build -t "test_suite_$ID" -f "$DOCKER_DIR/$DOCKER_FILE" "$REPO_ROOT"; then
        echo -e "  [${C_ERR}FAIL${C_RST}] Docker Build Failed" | tee -a "$RESULTS_LOG"
        return 1
    fi

    # 2. Execute the script inside the container
    # We capture the exit code of the setup.sh execution
    set +e
    # We run the script. We use 'tester' user to ensure permissions are correct.
    docker run --rm \
        -v "$CACHE_DIR/.bun:/home/tester/.bun:rw" \
        -v "$CACHE_DIR/.npm:/home/tester/.npm:rw" \
        "test_suite_$ID" /bin/bash -c "/home/tester/studio/setup.sh"
    ACTUAL_EXIT=$?
    set -e

    # 3. Validate Results
    if [ "$ACTUAL_EXIT" -eq "$EXPECTED_EXIT" ]; then
        echo -e "  [${C_OK}PASS${C_RST}] Exit Code $ACTUAL_EXIT" | tee -a "$RESULTS_LOG"
    else
        echo -e "  [${C_ERR}FAIL${C_RST}] Expected $EXPECTED_EXIT, but got $ACTUAL_EXIT" | tee -a "$RESULTS_LOG"
        return 1
    fi
}

print_help() {
    exit_code=$1
    echo -e "Usage: ${BASH_SOURCE[0]} -l|--local -h|--help
    -l|--local -- uses local unsloth repository for build"
    exit "$exit_code"
}

parse_arguments() {
    # ── Parse flags ──
    # --local: install from the local repo checkout (overlays unsloth as editable
    # and unsloth-zoo from git main). Mirrors install.sh --local for the Colab
    # path that runs setup.sh directly without going through install.sh.
    if [ "$#" -gt 0 ]; then
        for _arg in "$@"; do
            case "$_arg" in
                -c|--clean)
                    echo "Clean enabled: removing nmp bun and llama_cpp caches"
                    CLEAN_MODE='true'
                -s|--speed)
                    REBUILD_DOCKER_IMAGES='false'
                    ;;
                -h|--help)
                    print_help 0
                    ;;
                *)
                    step  "error" "Unknown argument" "$C_ERR" >&2
                    print_help 1
                    ;;
            esac
        done
    fi
}

execute_tests() {
    # --- Main Execution ---
    echo -e "${C_DIM}=======================================${C_RST}"
    echo -e "${C_DIM}   UNSLOTH SETUP TEST SUITE           ${C_RST}"
    echo -e "${C_DIM}=======================================${C_RST}\n"

    TOTAL=0
    PASSED=0

    if [ "${REBUILD_DOCKER_IMAGES}" == 'true' ]; then
        docker build -t "unsloth_studio_base_test_img" \
            -f "$DOCKER_DIR/unsloth_studio_base_test_img.Dockerfile" "$REPO_ROOT"
    fi

    for case in "${TEST_CASES[@]}"; do
        TOTAL=$((TOTAL + 1))
        if run_test_case "$case"; then
            PASSED=$((PASSED + 1))
        fi
        echo ""
    done

    # Final Summary
    echo -e "${C_DIM}---------------------------------------${C_RST}"
    if [ "$PASSED" -eq "$TOTAL" ]; then
        echo -e "${C_OK}SUMMARY: ALL $TOTAL TESTS PASSED${C_RST}"
        exit 0
    else
        echo -e "${C_ERR}SUMMARY: $TOTAL TESTS COMPLETED ($PASSED/$TOTAL) PASSED${C_RST}"
        exit 1
    fi
}


parse_arguments "$@"
prepare_environment
execute_tests
