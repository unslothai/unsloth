#!/usr/bin/env bash

# Rebuilds the owned backend alias from a clean, protected backend commit. The
# backend checkout itself remains untouched and uncommitted worktree overlays
# are deliberately rejected so a developer desktop snapshot cannot become a
# production release source.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile.backend-with-go"
SOURCE_REF="${RAG_PLATFORM_BACKEND_RELEASE_REF:-HEAD}"
EXPECTED_COMMIT="${RAG_PLATFORM_BACKEND_EXPECTED_COMMIT:-}"
LOCAL_SMOKE="${RAG_PLATFORM_LOCAL_SMOKE:-false}"
TARGET_IMAGE="rag-platform-backend:0.26.4"
if [[ "${LOCAL_SMOKE}" == "true" ]]; then
  TARGET_IMAGE="rag-platform-backend:0.26.4-phase15-local-smoke"
fi
BUILD_PROVENANCE="${RAG_PLATFORM_BUILD_PROVENANCE:-false}"
FRONTEND_ROOT="$(cd -- "${SCRIPT_DIR}/../../studio/frontend" && pwd)"
FRONTEND_DIST="${FRONTEND_ROOT}/dist"
FRONTEND_COMMIT="$(git -C "${FRONTEND_ROOT}" rev-parse HEAD)"

if [[ -z "${RAG_PLATFORM_BACKEND_DIR:-}" ]]; then
  RAG_PLATFORM_BACKEND_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)/rag-backend"
fi

if [[ ! -d "${RAG_PLATFORM_BACKEND_DIR}/.git" ]]; then
  echo "error: backend repository not found: ${RAG_PLATFORM_BACKEND_DIR}" >&2
  exit 1
fi

if [[ ! -s "${FRONTEND_DIST}/index.html" ]]; then
  echo "error: frontend production build not found: ${FRONTEND_DIST}/index.html" >&2
  echo "       run npm ci && npm run build in ${FRONTEND_ROOT} first" >&2
  exit 1
fi

ACTUAL_COMMIT="$(git -C "${RAG_PLATFORM_BACKEND_DIR}" rev-parse "${SOURCE_REF}^{commit}")"
BACKEND_DIRTY="false"
if [[ -n "$(git -C "${RAG_PLATFORM_BACKEND_DIR}" status --porcelain)" ]]; then
  BACKEND_DIRTY="true"
fi
if [[ -n "${EXPECTED_COMMIT}" && "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "error: ${SOURCE_REF} resolved to ${ACTUAL_COMMIT}, expected ${EXPECTED_COMMIT}" >&2
  exit 1
fi

if [[ "${LOCAL_SMOKE}" != "true" ]] && [[ "${BACKEND_DIRTY}" == "true" ]]; then
  echo "error: backend worktree is dirty; release images require a clean protected commit" >&2
  exit 1
fi

if [[ "${LOCAL_SMOKE}" != "true" ]] && git -C "${RAG_PLATFORM_BACKEND_DIR}" show-ref --verify --quiet refs/remotes/origin/main; then
  if ! git -C "${RAG_PLATFORM_BACKEND_DIR}" merge-base --is-ancestor \
      "${ACTUAL_COMMIT}" refs/remotes/origin/main; then
    if ! git -C "${RAG_PLATFORM_BACKEND_DIR}" tag --points-at "${ACTUAL_COMMIT}" | grep -q .; then
      echo "error: backend release commit is neither on origin/main nor tagged" >&2
      exit 1
    fi
  fi
fi

BUILD_CONTEXT="$(mktemp -d "${TMPDIR:-/tmp}/rag-platform-backend-build.XXXXXX")"
cleanup() {
  rm -rf -- "${BUILD_CONTEXT}"
}
trap cleanup EXIT

git -C "${RAG_PLATFORM_BACKEND_DIR}" archive "${SOURCE_REF}" | tar -x -C "${BUILD_CONTEXT}"

if [[ "${LOCAL_SMOKE}" == "true" ]]; then
  echo "local-smoke: validating tracked and untracked non-ignored content before overlay"
  RAG_PLATFORM_BACKEND_DIR="${RAG_PLATFORM_BACKEND_DIR}" \
    node "${SCRIPT_DIR}/../../scripts/rag-platform/secret-scan.mjs"

  while IFS= read -r -d '' relative_path; do
    source_path="${RAG_PLATFORM_BACKEND_DIR}/${relative_path}"
    if [[ -f "${source_path}" ]]; then
      mkdir -p "${BUILD_CONTEXT}/$(dirname -- "${relative_path}")"
      cp -p -- "${source_path}" "${BUILD_CONTEXT}/${relative_path}"
    fi
  done < <(
    {
      git -C "${RAG_PLATFORM_BACKEND_DIR}" diff --name-only --diff-filter=ACMRTUXB -z "${SOURCE_REF}" --
      git -C "${RAG_PLATFORM_BACKEND_DIR}" ls-files --others --exclude-standard -z
    }
  )

  while IFS= read -r -d '' relative_path; do
    rm -f -- "${BUILD_CONTEXT}/${relative_path}"
  done < <(
    git -C "${RAG_PLATFORM_BACKEND_DIR}" diff --name-only --diff-filter=D -z "${SOURCE_REF}" --
  )
fi

docker build \
  --platform linux/amd64 \
  --provenance="${BUILD_PROVENANCE}" \
  --build-context "rag-platform-assets=${SCRIPT_DIR}" \
  --build-context "rag-platform-frontend-dist=${FRONTEND_DIST}" \
  --file "${DOCKERFILE}" \
  --tag "${TARGET_IMAGE}" \
  --build-arg "RAG_PLATFORM_FRONTEND_COMMIT=${FRONTEND_COMMIT}" \
  --build-arg "RAG_PLATFORM_BACKEND_COMMIT=${ACTUAL_COMMIT}" \
  --label "org.opencontainers.image.source.commit=${ACTUAL_COMMIT}" \
  --label "io.rag-platform.local-smoke.dirty=${BACKEND_DIRTY}" \
  --label "io.rag-platform.release-profile=$([[ "${LOCAL_SMOKE}" == "true" ]] && echo phase15-local-smoke || echo phase15-protected-release)" \
  "${BUILD_CONTEXT}"

docker run --rm --platform linux/amd64 --entrypoint /bin/bash \
  "${TARGET_IMAGE}" -lc 'test -x /ragflow/bin/ragflow_server && test -x /usr/local/bin/rag-platform-readiness && test -s /ragflow/web/dist/index.html'

if [[ "${LOCAL_SMOKE}" == "true" ]]; then
  echo "built non-release ${TARGET_IMAGE} from ${SOURCE_REF}@${ACTUAL_COMMIT} plus explicit local worktree overlay"
else
  echo "built ${TARGET_IMAGE} from clean ${SOURCE_REF}@${ACTUAL_COMMIT} and frontend ${FRONTEND_COMMIT} (provenance=${BUILD_PROVENANCE})"
fi
