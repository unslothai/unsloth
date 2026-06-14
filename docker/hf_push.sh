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
# Requires: docker, pigz (or gzip), hf (or huggingface-cli) authenticated
# with a WRITE-scoped token: `hf auth login`.
set -euo pipefail

IMAGE="${1:?usage: hf_push.sh <image:tag> <hf_repo>}"
REPO="${2:?usage: hf_push.sh <image:tag> <hf_repo>}"
TAG="${IMAGE##*:}"
NAME="${IMAGE%:*}"
NAME="${NAME##*/}"
BLOB="${NAME}-${TAG}.tar.gz"
WORK="${HF_PUSH_TMP:-/tmp}"

command -v docker >/dev/null || { echo "ERROR: docker not on PATH"; exit 1; }

# Prefer the new `hf` CLI. The old `huggingface-cli` was deprecated in
# huggingface_hub >= 0.27 and silently exits with a deprecation notice
# instead of doing the upload.
if command -v hf >/dev/null 2>&1; then
    HF_CMD=(hf upload)
elif command -v huggingface-cli >/dev/null 2>&1; then
    echo "WARN: 'hf' not found, falling back to 'huggingface-cli' (deprecated)" >&2
    HF_CMD=(huggingface-cli upload)
else
    echo "ERROR: need 'hf' (pip install -U huggingface_hub) or 'huggingface-cli'"; exit 1
fi
COMPRESSOR=$(command -v pigz || command -v gzip) || { echo "ERROR: need pigz or gzip"; exit 1; }

OUT="${WORK}/${BLOB}"
echo ">> saving ${IMAGE} -> ${OUT} (using ${COMPRESSOR##*/})"
docker save "${IMAGE}" | "${COMPRESSOR}" > "${OUT}"
ls -lh "${OUT}"

echo ">> uploading to https://huggingface.co/${REPO}/blob/main/${BLOB}  (via: ${HF_CMD[*]})"
"${HF_CMD[@]}" "${REPO}" "${OUT}" "${BLOB}" \
    --repo-type=model \
    --commit-message="push ${IMAGE} ($(docker inspect --format '{{.Id}}' "${IMAGE}" | cut -c8-19))"

echo ">> pushed."
echo "On the consumer side, run:"
echo "  bash docker/hf_pull.sh ${REPO} ${BLOB} ${IMAGE}"
