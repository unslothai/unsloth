#!/usr/bin/env bash
#
# Download a single file from a Hugging Face repo with a stall-retry
# watchdog. Used by the Studio CI workflows so a hung hf-xet transfer
# kills + retries instead of silently consuming the job's timeout.
#
# Usage: hf-download-with-retry.sh REPO FILE LOCAL_DIR
#
# Why this exists
# ---------------
# huggingface_hub 1.15+ deprecated `hf_transfer` and routes every
# transfer through the `hf-xet` binary package. In CI we observed
# `hf download` on a 3 GB GGUF (gemma-4-E2B-it-UD-Q4_K_XL) progress
# to ~46% via Xet, then go completely silent for the remainder of
# the 30-min job timeout -- no progress bytes, no error, no exit.
# A sibling 940 MB mmproj on the same step downloaded in ~21s
# moments earlier, so the hang is per-file inside hf-xet rather
# than a network outage. The Xet env-vars below put hf-xet into
# its highest-throughput mode and force a 500 s client-read
# timeout; the watchdog loop ensures a stall does not eat the
# whole job: if the hf process has not exited after STALL_S
# seconds (default 180 = 3 min), we SIGTERM, then SIGKILL, then
# start a fresh attempt. Retries are unbounded -- the enclosing
# GitHub Actions job's `timeout-minutes` is the real bound.
#
# See https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables
# for the HF_XET_* documentation, and npm/cli#7308's pattern (silent
# CI hang with no error) for prior art on this class of failure.

set -uo pipefail

REPO="${1:?usage: hf-download-with-retry.sh REPO FILE [LOCAL_DIR]}"
FILE="${2:?usage: hf-download-with-retry.sh REPO FILE [LOCAL_DIR]}"
# LOCAL_DIR is optional. If empty, hf falls back to HF_HUB_CACHE
# (~/.cache/huggingface/hub) which is the desired path for callers
# that populate HF_HOME for a downstream Studio model load.
LOCAL_DIR="${3:-}"

# Stall threshold per attempt, in seconds. Override with
# HF_DOWNLOAD_STALL_SECONDS in the workflow env if 3 min is too tight
# for a specific runner / file. The script keeps retrying past this
# until the job timeout fires.
STALL_S="${HF_DOWNLOAD_STALL_SECONDS:-180}"

# hf-xet tuning. HF_HUB_ENABLE_HF_TRANSFER is deliberately NOT set --
# it is a no-op on huggingface_hub>=1.15 and only emits a deprecation
# FutureWarning. The five HF_XET_* knobs below mirror the settings
# Daniel asked for: max bandwidth + 64 parallel range gets, no chunk
# cache (download-once usage pattern), parallel disk writes (SSD/NVMe
# runners), and a generous 500 s read timeout so individual chunk
# requests fail loudly instead of stalling forever.
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
