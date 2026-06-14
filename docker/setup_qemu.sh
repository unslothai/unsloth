#!/usr/bin/env bash
# One-time host setup: register QEMU binfmt handlers so `docker buildx` can
# build images for foreign architectures (e.g. linux/arm64 on an x86_64 host).
#
# After this runs once per host reboot you can do:
#
#   docker buildx build --platform linux/arm64 -t unsloth-blackwell:arm64 .
#   docker buildx build --platform linux/amd64,linux/arm64 --push -t YOU/img:tag .
#
# Important: QEMU is used at BUILD time only. The resulting arm64 image must
# be RUN on an aarch64 host (e.g. DGX Spark / GB10) -- CUDA does not work under
# runtime emulation. To smoke-test the arm64 image you need an actual arm64
# GPU machine.
#
# Usage:
#   bash docker/setup_qemu.sh
#
# Requires: docker (28+ recommended), docker buildx plugin, root via sudo or
# membership in the `docker` group. No network access to NVIDIA registries
# is needed for this step.
set -euo pipefail

command -v docker >/dev/null || { echo "ERROR: docker not on PATH"; exit 1; }
docker buildx version >/dev/null 2>&1 || {
    echo "ERROR: 'docker buildx' missing. Install:" >&2
    echo "  Ubuntu/Debian:  sudo apt-get install -y docker-buildx" >&2
    echo "  RHEL/Fedora:    sudo dnf install -y docker-buildx-plugin" >&2
    exit 1
}

ARCH="$(uname -m)"
echo ">> host arch: ${ARCH}"

# `tonistiigi/binfmt --install all` registers handlers for every supported
# foreign arch; harmless if some are already registered. This is the canonical
# upstream Docker recipe; see https://docs.docker.com/build/building/multi-platform/
echo ">> registering QEMU binfmt handlers via tonistiigi/binfmt..."
docker run --privileged --rm tonistiigi/binfmt --install all

# Ensure we have a buildx builder that can target multiple platforms.
# The default 'docker' driver builder is single-platform; we create (or
# reuse) a 'unsloth-multiarch' container-driver builder which is multi-arch.
BUILDER="unsloth-multiarch"
if docker buildx inspect "${BUILDER}" >/dev/null 2>&1; then
    echo ">> buildx builder '${BUILDER}' already exists"
else
    echo ">> creating buildx builder '${BUILDER}'"
    docker buildx create --name "${BUILDER}" --driver docker-container --use
fi
docker buildx use "${BUILDER}"
docker buildx inspect --bootstrap "${BUILDER}" | sed -n '1,12p'

echo
echo ">> done. Verify with:"
echo "     docker buildx ls"
echo "     docker buildx inspect ${BUILDER}"
echo
echo ">> cross-arch build example:"
echo "     docker buildx build --platform linux/arm64 -t unsloth-blackwell:arm64 docker/"
