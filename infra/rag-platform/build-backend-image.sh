#!/usr/bin/env bash

# Rebuilds the Faz 0 backend alias with the Go API/admin executable that the
# published v0.26.4 image omits. The backend checkout itself remains untouched:
# a clean archive of the verified release tag is used as disposable context,
# plus the owned no-CGO compatibility stub copied explicitly by the Dockerfile.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile.backend-with-go"
SOURCE_REF="v0.26.4"
EXPECTED_COMMIT="cb93883f3f8c975eecb2fed81210effeb3bdb06f"
TARGET_IMAGE="rag-platform-backend:0.26.4"
BUILD_PROVENANCE="${RAG_PLATFORM_BUILD_PROVENANCE:-false}"

if [[ -z "${RAG_PLATFORM_BACKEND_DIR:-}" ]]; then
  RAG_PLATFORM_BACKEND_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)/rag-backend"
fi

if [[ ! -d "${RAG_PLATFORM_BACKEND_DIR}/.git" ]]; then
  echo "error: backend repository not found: ${RAG_PLATFORM_BACKEND_DIR}" >&2
  exit 1
fi

ACTUAL_COMMIT="$(git -C "${RAG_PLATFORM_BACKEND_DIR}" rev-parse "${SOURCE_REF}^{commit}")"
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "error: ${SOURCE_REF} resolved to ${ACTUAL_COMMIT}, expected ${EXPECTED_COMMIT}" >&2
  exit 1
fi

BUILD_CONTEXT="$(mktemp -d "${TMPDIR:-/tmp}/rag-platform-backend-build.XXXXXX")"
cleanup() {
  rm -rf -- "${BUILD_CONTEXT}"
}
trap cleanup EXIT

git -C "${RAG_PLATFORM_BACKEND_DIR}" archive "${SOURCE_REF}" | tar -x -C "${BUILD_CONTEXT}"

docker build \
  --platform linux/amd64 \
  --provenance="${BUILD_PROVENANCE}" \
  --build-context "rag-platform-assets=${SCRIPT_DIR}" \
  --file "${DOCKERFILE}" \
  --tag "${TARGET_IMAGE}" \
  --label "org.opencontainers.image.source.commit=${EXPECTED_COMMIT}" \
  "${BUILD_CONTEXT}"

docker run --rm --platform linux/amd64 --entrypoint /bin/bash \
  "${TARGET_IMAGE}" -lc 'test -x /ragflow/bin/ragflow_server'

echo "built ${TARGET_IMAGE} from ${SOURCE_REF}@${EXPECTED_COMMIT} (provenance=${BUILD_PROVENANCE})"
