#!/usr/bin/env bash

# Rebuilds the owned backend alias with the Go API/admin executable and the
# Phase 14 Python authorization routes. The backend checkout itself remains
# untouched: a clean archive of the verified backend authority commit is used
# as disposable context and only the reviewed Phase 14 worktree files are
# overlaid explicitly.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile.backend-with-go"
SOURCE_REF="a0e091e75051f278ab21e7e1c2ce3d1fcccbd5a2"
EXPECTED_COMMIT="a0e091e75051f278ab21e7e1c2ce3d1fcccbd5a2"
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

# Overlay only Faz 14-owned files. This deliberately excludes every unrelated
# dirty worktree file, so local user changes cannot leak into the runtime image.
PHASE14_FILES=(
  internal/admin/handler.go
  internal/admin/handler_ingestor_test.go
  internal/engine/nats/ingestor_control.go
  internal/engine/nats/ingestor_control_test.go
  internal/handler/auth.go
  internal/handler/auth_test.go
  internal/handler/tenant.go
  internal/ingestion/service/ingestion_service.go
  internal/router/router.go
  internal/admin/service.go
  internal/dao/database.go
  internal/service/tenant.go
  internal/service/tenant_test.go
)
for relative_path in "${PHASE14_FILES[@]}"; do
  source_path="${RAG_PLATFORM_BACKEND_DIR}/${relative_path}"
  if [[ ! -f "${source_path}" ]]; then
    echo "error: missing Phase 14 overlay: ${source_path}" >&2
    exit 1
  fi
  mkdir -p "${BUILD_CONTEXT}/$(dirname -- "${relative_path}")"
  cp -- "${source_path}" "${BUILD_CONTEXT}/${relative_path}"
done

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

echo "built ${TARGET_IMAGE} from ${SOURCE_REF}@${EXPECTED_COMMIT} with reviewed Phase 14 overlays (provenance=${BUILD_PROVENANCE})"
