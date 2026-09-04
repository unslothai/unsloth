#!/usr/bin/env bash
# Build the unsloth-blackwell image on any Linux host with Docker. The build host's
# GPU is NOT used: nvcc cross-compiles.
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

# Frozen to a commit here, for the same reason LLAMA_PREBUILT_TAG is resolved below and
# the publish workflow freezes both refs with git ls-remote: docker matches a RUN layer
# on the COMMAND STRING alone, so with the default "main" the pip-install layer is a
# cache HIT on every rebuild and the image keeps the commits of the first build while
# reporting success. A sha changes the build arg exactly when the branch moves.
resolve_git_ref() {
    local repo="$1" ref="$2" ls_out sha
    printf '%s' "$ref" | grep -Eq '^[0-9a-f]{40}$' && { printf '%s' "$ref"; return 0; }
    command -v git >/dev/null 2>&1 || { printf '%s' "$ref"; return 0; }
    # ls-remote exits 0 whether or not a ref matched, so a non-zero exit means the
    # remote was never reached; an offline build cannot install from git anyway, so
    # warn and pass the name through rather than fail before docker has even started.
    if ! ls_out="$(git ls-remote "$repo" "$ref" 2>/dev/null)"; then
        echo "warning: ${repo} unreachable; passing mutable ref '${ref}' (docker may reuse a cached layer)" >&2
        printf '%s' "$ref"
        return 0
    fi
    sha="$(printf '%s\n' "$ls_out" | awk 'NR==1{print $1}')"
    [ -n "$sha" ] || sha="$ref"     # not a branch/tag: a short sha or already-gone ref
    printf '%s' "$sha"
}
UNSLOTH_REF="$(resolve_git_ref https://github.com/unslothai/unsloth "$UNSLOTH_REF")"
UNSLOTH_ZOO_REF="$(resolve_git_ref https://github.com/unslothai/unsloth-zoo "$UNSLOTH_ZOO_REF")"

# Resolved to a concrete tag here, so the build-arg changes only on a new release and
# layer caching stays correct. Pin with LLAMA_PREBUILT_TAG=... for a frozen build.
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
# Read the arch list out of the Dockerfile rather than repeating it: the hand-copied
# banner had already drifted. Bare filename because the script cd'd to its own dir.
ARCH_LIST="$(sed -n 's/^[[:space:]]*TORCH_CUDA_ARCH_LIST="\([^"]*\)".*/\1/p' \
             Dockerfile | head -n1)"
echo "  arch list      ${ARCH_LIST:-unknown}"
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
