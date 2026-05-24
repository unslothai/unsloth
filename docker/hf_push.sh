#!/usr/bin/env bash
# Simulate `docker push <image>` against a Hugging Face Hub model repo.
#
# HF Hub doesn't act as an OCI registry for arbitrary images (only Spaces have
# that). So we approximate the push by:
#   1. docker save | pigz  -> single tar.gz blob
#   2. huggingface-cli upload to <repo>/{tag}.tar.gz
#
# This is good for cross-host testing where you want one canonical place to
# pull from. For the real release, use Docker Hub or GHCR with `docker push`,
# which gives you layer dedup, manifest negotiation, and standard `docker pull`
# UX -- see .github/workflows/docker-publish.yml in this repo.
#
# Usage:
#   bash docker/hf_push.sh <image:tag> <hf_repo>
#   bash docker/hf_push.sh unsloth-blackwell:test danielhanchen/unsloth-blackwell-docker
#
# Requires: docker, pigz (or gzip), huggingface-cli logged in with a write token.
set -euo pipefail

IMAGE="${1:?usage: hf_push.sh <image:tag> <hf_repo>}"
REPO="${2:?usage: hf_push.sh <image:tag> <hf_repo>}"
TAG="${IMAGE##*:}"
NAME="${IMAGE%:*}"
NAME="${NAME##*/}"
BLOB="${NAME}-${TAG}.tar.gz"
WORK="${HF_PUSH_TMP:-/tmp}"

command -v docker >/dev/null              || { echo "ERROR: docker not on PATH"; exit 1; }
command -v huggingface-cli >/dev/null     || { echo "ERROR: huggingface-cli not on PATH (pip install -U huggingface_hub)"; exit 1; }
COMPRESSOR=$(command -v pigz || command -v gzip) || { echo "ERROR: need pigz or gzip"; exit 1; }

OUT="${WORK}/${BLOB}"
echo ">> saving ${IMAGE} -> ${OUT} (using ${COMPRESSOR##*/})"
docker save "${IMAGE}" | "${COMPRESSOR}" > "${OUT}"
ls -lh "${OUT}"

echo ">> uploading to https://huggingface.co/${REPO}/blob/main/${BLOB}"
huggingface-cli upload "${REPO}" "${OUT}" "${BLOB}" \
    --repo-type=model \
    --commit-message="push ${IMAGE} ($(docker inspect --format '{{.Id}}' "${IMAGE}" | cut -c8-19))"

echo ">> pushed."
echo "On the consumer side, run:"
echo "  bash docker/hf_pull.sh ${REPO} ${BLOB} ${IMAGE}"
