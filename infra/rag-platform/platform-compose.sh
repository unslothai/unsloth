#!/usr/bin/env bash
#
# Rag Platform — Docker Compose entry point
#
# Wraps `docker compose` so every invocation carries the owned project
# name, the owned compose file and both env files in the required order.
# Running plain `docker compose` against the compose file directly works
# too, but only if all of that is passed by hand; this script is the
# supported path and the one the docs and CI use.
#
# Usage:
#   ./platform-compose.sh config
#   ./platform-compose.sh up -d
#   ./platform-compose.sh ps
#   ./platform-compose.sh logs -f platform-backend-cpu
#   ./platform-compose.sh down
#
#   ./platform-compose.sh --gpu config      # validate the GPU profile
#   ./platform-compose.sh --check-profiles  # assert CPU+GPU is rejected
#
# Environment:
#   RAG_PLATFORM_BACKEND_DIR  Backend checkout. Defaults to the sibling
#                             ../../../rag-backend of this directory.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose.rag-platform.yml"
OWNED_ENV="${SCRIPT_DIR}/.env.rag-platform"
PROJECT_NAME="rag-platform"

# Resolve the backend checkout. The compose file also guards this with
# ${RAG_PLATFORM_BACKEND_DIR:?...}, but failing here gives a message that
# names the expected layout instead of an interpolation error.
if [[ -z "${RAG_PLATFORM_BACKEND_DIR:-}" ]]; then
  RAG_PLATFORM_BACKEND_DIR="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)/rag-backend"
fi

if [[ ! -d "${RAG_PLATFORM_BACKEND_DIR}" ]]; then
  echo "error: backend checkout not found: ${RAG_PLATFORM_BACKEND_DIR}" >&2
  echo "       set RAG_PLATFORM_BACKEND_DIR to the rag-backend directory" >&2
  exit 1
fi

# Confirm it really is a backend checkout and not just any directory, so a
# typo cannot silently produce a half-configured stack. Only the two files
# the owned layer actually consumes are required: the base compose file it
# includes and the upstream env file it layers over. The checkout's
# entrypoint.sh and service_conf.yaml.template are deliberately NOT used —
# see the volumes comment in docker-compose.rag-platform.yml.
for required in \
  "docker/docker-compose-base.yml" \
  "docker/.env"; do
  if [[ ! -f "${RAG_PLATFORM_BACKEND_DIR}/${required}" ]]; then
    echo "error: ${RAG_PLATFORM_BACKEND_DIR} is missing ${required}" >&2
    echo "       RAG_PLATFORM_BACKEND_DIR does not look like a backend checkout" >&2
    exit 1
  fi
done

export RAG_PLATFORM_BACKEND_DIR
UPSTREAM_ENV="${RAG_PLATFORM_BACKEND_DIR}/docker/.env"

# --env-file order matters and is later-wins, matching the env_file order
# inside the compose file. The upstream file supplies credentials and
# unowned defaults; the owned file overrides only Rag Platform's own
# naming and the proxy scheme.
COMPOSE_ARGS=(
  --project-name "${PROJECT_NAME}"
  --env-file "${UPSTREAM_ENV}"
  --env-file "${OWNED_ENV}"
  --file "${COMPOSE_FILE}"
)

DEVICE_PROFILE=""
case "${1:-}" in
  --gpu)
    DEVICE_PROFILE="gpu"
    shift
    ;;
  --cpu)
    DEVICE_PROFILE="cpu"
    shift
    ;;
  --check-profiles)
    # The two owned backend services deliberately share the container name
    # rag-platform-backend, so Compose itself refuses the combination. This
    # asserts that guard still holds rather than trusting a comment.
    echo "checking that platform-cpu + platform-gpu is rejected..."
    if COMPOSE_PROFILES="elasticsearch,ragflow-go,platform-cpu,platform-gpu" \
        docker compose "${COMPOSE_ARGS[@]}" config > /dev/null 2>/tmp/rag-platform-profile-check.err; then
      echo "FAIL: CPU and GPU profiles were accepted together" >&2
      exit 1
    fi
    if ! grep -q 'container name "rag-platform-backend" is already in use' \
        /tmp/rag-platform-profile-check.err; then
      echo "FAIL: rejected, but not by the container-name collision:" >&2
      cat /tmp/rag-platform-profile-check.err >&2
      exit 1
    fi
    echo "OK: rejected with container name \"rag-platform-backend\" is already in use"
    exit 0
    ;;
esac

# --gpu / --cpu override the profile set for this one invocation; without
# them COMPOSE_PROFILES from the env files decides.
if [[ -n "${DEVICE_PROFILE}" ]]; then
  export COMPOSE_PROFILES="elasticsearch,ragflow-go,platform-${DEVICE_PROFILE}"
fi

if [[ $# -eq 0 ]]; then
  echo "usage: $(basename "$0") [--cpu|--gpu|--check-profiles] <docker compose args...>" >&2
  exit 1
fi

exec docker compose "${COMPOSE_ARGS[@]}" "$@"
