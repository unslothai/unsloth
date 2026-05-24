#!/usr/bin/env bash
# Simulate `docker pull <image>` against a Hugging Face Hub model repo.
#
# Counterpart to docker/hf_push.sh -- downloads the tar.gz blob from the HF
# repo and `docker load`s it.
#
# Usage:
#   bash docker/hf_pull.sh <hf_repo> [<blob>] [<verify_tag>]
#   bash docker/hf_pull.sh danielhanchen/unsloth-blackwell-docker unsloth-blackwell-test.tar.gz unsloth-blackwell:test
#
# Requires: docker, pigz (or gzip), huggingface-cli logged in (read scope is
# sufficient for public repos).
set -euo pipefail

REPO="${1:?usage: hf_pull.sh <hf_repo> [<blob>] [<verify_tag>]}"
BLOB="${2:-unsloth-blackwell.tar.gz}"
VERIFY="${3:-}"
WORK="${HF_PULL_TMP:-/tmp}"

command -v docker >/dev/null          || { echo "ERROR: docker not on PATH"; exit 1; }
command -v huggingface-cli >/dev/null || { echo "ERROR: huggingface-cli not on PATH (pip install -U huggingface_hub)"; exit 1; }
DECOMPRESSOR=$(command -v pigz || command -v gzip) || { echo "ERROR: need pigz or gzip"; exit 1; }

DEST="${WORK}/$(basename "${BLOB}")"
echo ">> downloading ${REPO}/${BLOB} -> ${DEST}"
huggingface-cli download "${REPO}" "${BLOB}" --repo-type=model --local-dir "${WORK}"
ls -lh "${DEST}"

echo ">> loading into docker (using ${DECOMPRESSOR##*/})"
"${DECOMPRESSOR}" -d -c "${DEST}" | docker load

if [[ -n "${VERIFY}" ]]; then
    if docker image inspect "${VERIFY}" >/dev/null 2>&1; then
        echo ">> verified: ${VERIFY} is loaded"
        docker image inspect --format 'image_id={{.Id}} size={{.Size}}' "${VERIFY}"
    else
        echo "WARN: expected tag ${VERIFY} not found after load. docker images:"
        docker images
        exit 1
    fi
fi
