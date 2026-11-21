#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Download one file from a Hugging Face repo, killing and retrying a stalled hf-xet transfer.
# Usage: hf-download-with-retry.sh REPO FILE LOCAL_DIR
# Retries are unbounded: give every calling step a timeout-minutes.

set -uo pipefail

REPO="${1:?usage: hf-download-with-retry.sh REPO FILE [LOCAL_DIR]}"
FILE="${2:?usage: hf-download-with-retry.sh REPO FILE [LOCAL_DIR]}"
# Empty LOCAL_DIR falls back to HF_HUB_CACHE, which callers relying on HF_HOME want.
LOCAL_DIR="${3:-}"

STALL_S="${HF_DOWNLOAD_STALL_SECONDS:-180}"

# HF_HUB_ENABLE_HF_TRANSFER is a no-op on huggingface_hub>=1.15, so it is not set.
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export HF_XET_NUM_CONCURRENT_RANGE_GETS=64
export HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=0
export HF_XET_CLIENT_READ_TIMEOUT=500

if [ -n "$LOCAL_DIR" ]; then
  mkdir -p "$LOCAL_DIR"
fi

attempt=1
while : ; do
  log="$(mktemp -t hf-download.XXXXXX)"
  echo "[hf-download] $FILE attempt $attempt (stall threshold ${STALL_S}s, log=$log)"

  if [ -n "$LOCAL_DIR" ]; then
    hf download "$REPO" "$FILE" --local-dir "$LOCAL_DIR" > "$log" 2>&1 &
  else
    hf download "$REPO" "$FILE" > "$log" 2>&1 &
  fi
  pid=$!

  elapsed=0
  while kill -0 "$pid" 2>/dev/null && [ "$elapsed" -lt "$STALL_S" ]; do
    sleep 5
    elapsed=$((elapsed + 5))
  done

  if kill -0 "$pid" 2>/dev/null; then
    echo "[hf-download] $FILE attempt $attempt exceeded ${STALL_S}s -- killing PID $pid and retrying"
    kill -TERM "$pid" 2>/dev/null || true
    sleep 2
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    echo "[hf-download] $FILE attempt $attempt log tail (last 40 lines):"
    tail -40 "$log" || true
    attempt=$((attempt + 1))
    continue
  fi

  if wait "$pid"; then
    rc=0
  else
    rc=$?
  fi

  if [ "$rc" -eq 0 ]; then
    echo "[hf-download] $FILE attempt $attempt succeeded"
    tail -20 "$log" || true
    exit 0
  fi

  echo "[hf-download] $FILE attempt $attempt failed (exit $rc) -- retrying"
  tail -40 "$log" || true
  attempt=$((attempt + 1))
done
