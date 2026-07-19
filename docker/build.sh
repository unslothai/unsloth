#!/usr/bin/env bash
# Build the unsloth-blackwell image on this B200 host (or any Linux host with Docker).
# The build host's GPU is NOT used -- nvcc cross-compiles for sm_100 + sm_120.
#
# Usage:
#   ./build.sh                 # builds unsloth-blackwell:latest pinned to unsloth main
#   TAG=2026.05.1 ./build.sh   # custom tag
#   UNSLOTH_REF=v2026.5.6 UNSLOTH_ZOO_REF=v2026.5.4 ./build.sh   # pin git refs
set -euo pipefail

cd "$(dirname "$0")"

IMAGE_NAME="${IMAGE_NAME:-unsloth-blackwell}"
TAG="${TAG:-latest}"
CUDA_VERSION="${CUDA_VERSION:-12.8.1}"
UBUNTU_VERSION="${UBUNTU_VERSION:-24.04}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
UNSLOTH_REF="${UNSLOTH_REF:-main}"
UNSLOTH_ZOO_REF="${UNSLOTH_ZOO_REF:-main}"

# llama.cpp prebuilt: default to the newest release, resolved here to a concrete
# tag so the build-arg changes only on a new release (correct layer caching).
# Pin for a frozen build: LLAMA_PREBUILT_TAG=b9596-mix-e6f2453 ./build.sh
resolve_latest_llama_tag() {
    curl -fsSL -o /dev/null -w '%{url_effective}' \
        "https://github.com/unslothai/llama.cpp/releases/latest" 2>/dev/null \
        | sed -n 's#.*/releases/tag/##p'
}
if [ -z "${LLAMA_PREBUILT_TAG:-}" ]; then
    LLAMA_PREBUILT_TAG="$(resolve_latest_llama_tag || true)"
    if [ -n "$LLAMA_PREBUILT_TAG" ]; then
        echo "Resolved latest llama.cpp release: ${LLAMA_PREBUILT_TAG}"
    else
        LLAMA_PREBUILT_TAG="latest"
        echo "Could not resolve latest llama.cpp tag here; passing 'latest' (resolved inside the build)"
    fi
fi

echo "Building ${IMAGE_NAME}:${TAG}"
echo "  CUDA           ${CUDA_VERSION}  Ubuntu ${UBUNTU_VERSION}  Python ${PYTHON_VERSION}"
echo "  unsloth        @${UNSLOTH_REF}"
echo "  unsloth-zoo    @${UNSLOTH_ZOO_REF}"
echo "  llama.cpp      ${LLAMA_PREBUILT_TAG}"
echo "  arch list      8.0;8.6;8.9;9.0;10.0;12.0+PTX"
echo

DOCKER_BUILDKIT=1 docker build \
    --progress=plain \
    --build-arg CUDA_VERSION="${CUDA_VERSION}" \
    --build-arg UBUNTU_VERSION="${UBUNTU_VERSION}" \
    --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
    --build-arg UNSLOTH_REF="${UNSLOTH_REF}" \
    --build-arg UNSLOTH_ZOO_REF="${UNSLOTH_ZOO_REF}" \
    --build-arg LLAMA_PREBUILT_TAG="${LLAMA_PREBUILT_TAG}" \
    -t "${IMAGE_NAME}:${TAG}" \
    .

echo
echo "Built ${IMAGE_NAME}:${TAG}"
echo
echo "Smoke test on this host (B200, sm_100):"
echo "  docker run --rm --gpus all ${IMAGE_NAME}:${TAG} python /workspace/smoke_test.py"
echo
echo "Smoke test on an RTX 5090 host (sm_120):"
echo "  docker pull ${IMAGE_NAME}:${TAG}   # or load .tar"
echo "  docker run --rm --gpus all ${IMAGE_NAME}:${TAG} python /workspace/smoke_test.py"
