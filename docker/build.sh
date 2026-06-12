#!/usr/bin/env bash
# Build an Unsloth Docker image (NVIDIA CUDA or AMD ROCm).
# The build host's GPU is NOT used at build time.
#
# Usage:
#   ./build.sh                          # CUDA/Blackwell (default)
#   ./build.sh --rocm                   # AMD ROCm
#   TAG=2026.05.1 ./build.sh            # custom tag
#   UNSLOTH_REF=v2026.5.6 UNSLOTH_ZOO_REF=v2026.5.4 ./build.sh
#
# ROCm: for RDNA4 / Strix Halo (gfx1150/1151/1200/1201) override:
#   ROCM_VERSION=7.x.x TORCH_INDEX_URL=https://download.pytorch.org/whl/rocm7.2 \
#     ./build.sh --rocm
set -euo pipefail

cd "$(dirname "$0")"

ROCM=0
for arg in "$@"; do
    [[ "$arg" == "--rocm" ]] && ROCM=1
done

UNSLOTH_REF="${UNSLOTH_REF:-main}"
UNSLOTH_ZOO_REF="${UNSLOTH_ZOO_REF:-main}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TAG="${TAG:-latest}"

if [[ $ROCM -eq 1 ]]; then
    IMAGE_NAME="${IMAGE_NAME:-unsloth-rocm}"
    ROCM_VERSION="${ROCM_VERSION:-6.2.4}"
    TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/rocm6.2}"

    echo "Building ${IMAGE_NAME}:${TAG}  [AMD ROCm]"
    echo "  ROCm           ${ROCM_VERSION}  Python ${PYTHON_VERSION}"
    echo "  torch index    ${TORCH_INDEX_URL}"
    echo "  unsloth        @${UNSLOTH_REF}"
    echo "  unsloth-zoo    @${UNSLOTH_ZOO_REF}"
    echo

    DOCKER_BUILDKIT=1 docker build \
        --progress=plain \
        -f Dockerfile.rocm \
        --build-arg ROCM_VERSION="${ROCM_VERSION}" \
        --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
        --build-arg TORCH_INDEX_URL="${TORCH_INDEX_URL}" \
        --build-arg UNSLOTH_REF="${UNSLOTH_REF}" \
        --build-arg UNSLOTH_ZOO_REF="${UNSLOTH_ZOO_REF}" \
        -t "${IMAGE_NAME}:${TAG}" \
        .

    echo
    echo "Built ${IMAGE_NAME}:${TAG}"
    echo
    echo "Smoke test (requires AMD GPU passthrough):"
    echo "  bash run.sh --rocm python /workspace/smoke_test_rocm.py"
else
    IMAGE_NAME="${IMAGE_NAME:-unsloth-blackwell}"
    CUDA_VERSION="${CUDA_VERSION:-12.8.1}"
    UBUNTU_VERSION="${UBUNTU_VERSION:-24.04}"

    echo "Building ${IMAGE_NAME}:${TAG}  [NVIDIA CUDA]"
    echo "  CUDA           ${CUDA_VERSION}  Ubuntu ${UBUNTU_VERSION}  Python ${PYTHON_VERSION}"
    echo "  unsloth        @${UNSLOTH_REF}"
    echo "  unsloth-zoo    @${UNSLOTH_ZOO_REF}"
    echo "  arch list      8.0;8.6;8.9;9.0;10.0;12.0+PTX"
    echo

    DOCKER_BUILDKIT=1 docker build \
        --progress=plain \
        --build-arg CUDA_VERSION="${CUDA_VERSION}" \
        --build-arg UBUNTU_VERSION="${UBUNTU_VERSION}" \
        --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
        --build-arg UNSLOTH_REF="${UNSLOTH_REF}" \
        --build-arg UNSLOTH_ZOO_REF="${UNSLOTH_ZOO_REF}" \
        -t "${IMAGE_NAME}:${TAG}" \
        .

    echo
    echo "Built ${IMAGE_NAME}:${TAG}"
    echo
    echo "Smoke test on this host (B200, sm_100):"
    echo "  bash run.sh python /workspace/smoke_test.py"
    echo
    echo "Smoke test on an RTX 5090 host (sm_120):"
    echo "  docker pull ${IMAGE_NAME}:${TAG}   # or load .tar"
    echo "  bash run.sh python /workspace/smoke_test.py"
fi
