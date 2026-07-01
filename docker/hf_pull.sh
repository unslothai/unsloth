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
# Requires: docker, pigz (or gzip), hf (or huggingface-cli) authenticated
# (read scope is sufficient for public repos: `hf auth login`).
set -euo pipefail

REPO="${1:?usage: hf_pull.sh <hf_repo> [<blob>] [<verify_tag>]}"
BLOB="${2:-unsloth-blackwell.tar.gz}"
VERIFY="${3:-}"
WORK="${HF_PULL_TMP:-/tmp}"

command -v docker >/dev/null || { echo "ERROR: docker not on PATH"; exit 1; }

# Prefer the new `hf` CLI. The old `huggingface-cli` was deprecated in
# huggingface_hub >= 0.27 and silently exits with a deprecation notice
# instead of doing the download, so we treat its presence as a fallback
# only and warn if it's all we have.
if command -v hf >/dev/null 2>&1; then
    HF_CMD=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
    echo "WARN: 'hf' not found, falling back to 'huggingface-cli' (deprecated)" >&2
    HF_CMD=(huggingface-cli download)
else
    echo "ERROR: need 'hf' (pip install -U huggingface_hub) or 'huggingface-cli'"; exit 1
fi
DECOMPRESSOR=$(command -v pigz || command -v gzip) || { echo "ERROR: need pigz or gzip"; exit 1; }

DEST="${WORK}/$(basename "${BLOB}")"
echo ">> downloading ${REPO}/${BLOB} -> ${DEST}  (via: ${HF_CMD[*]})"
"${HF_CMD[@]}" "${REPO}" "${BLOB}" --repo-type=model --local-dir "${WORK}"
test -s "${DEST}" || { echo "ERROR: download produced no file at ${DEST}"; exit 1; }
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
