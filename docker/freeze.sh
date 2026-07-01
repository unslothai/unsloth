#!/usr/bin/env bash
# Pull the lockfile out of a built image so the next rebuild can be byte-identical.
#
#   ./freeze.sh                      # extracts to requirements.lock.txt next to Dockerfile
#   ./freeze.sh some-tag-or-digest   # custom source
#
# To rebuild against the frozen lockfile later, replace the `pip install` lines
# in the Dockerfile with `pip install -r /tmp/requirements.lock.txt --no-deps`
# (mounted via `docker build --build-context lock=./requirements.lock.txt`).
set -euo pipefail

cd "$(dirname "$0")"

SRC="${1:-unsloth-blackwell:latest}"
DEST="${2:-./requirements.lock.txt}"

CID=$(docker create "${SRC}")
trap 'docker rm -f "${CID}" >/dev/null' EXIT

docker cp "${CID}:/opt/unsloth-venv/requirements.lock.txt" "${DEST}"
echo "Wrote ${DEST}"
echo
echo "Top of lockfile:"
head -20 "${DEST}"
echo
echo "Lines: $(wc -l < "${DEST}")"
