#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Prompt-cache (KV-cache prefix reuse) detection, two strategies in one helper:
#
#   mode=api     A 2-turn /v1/chat/completions probe. Turn 2 prepends turn 1 +
#                its reply, so the shared prefix must be served from llama.cpp's
#                KV cache. Asserts usage.prompt_tokens_details.cached_tokens > 0
#                on turn 2. This is the OpenAI-dialect server cache sanity.
#                WHY this works on chat completions: the chat path forwards
#                llama-server's real cached_tokens through
#                studio/backend/routes/inference.py:482-489 (_prompt_tokens_details)
#                into prompt_tokens_details (inference.py:519).
#
#   mode=log     Read the llama-server log and decide HIT vs MISS from the
#                prompt-reprocessing trace. WHY the log (not the API field):
#                the Anthropic /v1/messages path builds AnthropicUsage(
#                input_tokens=..., output_tokens=...) at inference.py:8787-8790
#                / :8829-8832 and NEVER sets cache_read_input_tokens, which
#                therefore stays at its model default of 0
#                (studio/backend/models/inference.py:1655). So an Anthropic-path
#                client (Claude Code, OpenClaw is openai-completions but Claude
#                Code is the canonical Anthropic agent) can get a real KV-cache
#                hit that the API usage field reports as 0. The only ground
#                truth for the Anthropic path is the llama-server log.
#
# Log location (verified): studio/backend/core/inference/llama_cpp.py:4363-4365
#   _swa_cache_path().parent/"logs"/"llama-server"/llama-<ts>[label]-port-<P>[-try<N>].log
#   _swa_cache_path() => $UNSLOTH_STUDIO_HOME|$STUDIO_HOME or ~/.unsloth/studio
#   (llama_cpp.py:337-340). So default: ~/.unsloth/studio/logs/llama-server/.
#
#   <P> is the INTERNAL llama-server port (self._find_free_port(),
#   llama_cpp.py:3489 / :4641) -- a RANDOM port, NOT the Unsloth port. So we must
#   NOT filter the log glob by STUDIO_PORT (the brief's `port-<STUDIO_PORT>`
#   glob would never match). We pick the newest llama-*.log instead.
#
# Usage:
#   assert-prompt-cache.sh api  BASE_URL API_KEY
#   assert-prompt-cache.sh log  EXPECT          # EXPECT = HIT | MISS
#                                               # reads MARKER_BEFORE/MARKER_AFTER
#                                               # byte offsets from env (see below)
#   assert-prompt-cache.sh mark                 # print current log size to stdout
#                                               # (use to bracket a turn)
#
# Env for mode=log:
#   LLAMA_LOG_DIR    override the log dir (default ~/.unsloth/studio/logs/llama-server)
#   CACHE_LOG_FROM   byte offset to start scanning the newest log from (so we
#                    only look at the trace produced by THIS turn). Default 0.
#
# Exit codes: 0 = assertion held; 1 = assertion failed (::error:: emitted).

set -uo pipefail

MODE="${1:?usage: assert-prompt-cache.sh api|log|mark ...}"

# ---------------------------------------------------------------------------
# Locate the newest llama-server log. Shared by mark + log modes.
# ---------------------------------------------------------------------------
_default_log_dir() {
  local home="${UNSLOTH_STUDIO_HOME:-${STUDIO_HOME:-}}"
  if [ -n "$home" ]; then
    echo "${home%/}/logs/llama-server"
  else
    echo "${HOME}/.unsloth/studio/logs/llama-server"
  fi
}

_newest_log() {
  local dir="${LLAMA_LOG_DIR:-$(_default_log_dir)}"
  [ -d "$dir" ] || return 1
  # Newest by mtime among llama-*.log (covers both `llama-<ts>-port-<P>.log`
  # and the retry form `llama-<ts><label>-port-<P>-try<N>.log`). Filenames are
  # tool-generated timestamps, so ls -t is safe here.
  # shellcheck disable=SC2012
  ls -1t "$dir"/llama-*.log 2>/dev/null | head -1
}

case "$MODE" in
  # -------------------------------------------------------------------------
  # mark: emit the current byte size of the newest llama log so a caller can
  # scan only the slice a single turn produced (set CACHE_LOG_FROM to it).
  # -------------------------------------------------------------------------
  mark)
    log="$(_newest_log || true)"
    if [ -n "$log" ] && [ -f "$log" ]; then
      wc -c < "$log" | tr -d ' '
    else
      echo 0
    fi
    exit 0
    ;;

  # -------------------------------------------------------------------------
  # api: 2-turn /v1/chat/completions, assert turn-2 cached_tokens > 0.
  # -------------------------------------------------------------------------
  api)
    BASE_URL="${2:?usage: assert-prompt-cache.sh api BASE_URL API_KEY}"
    API_KEY="${3:?usage: assert-prompt-cache.sh api BASE_URL API_KEY}"

    # A deliberately long, fixed system prompt makes the shared prefix big so a
    # KV-cache hit is unambiguous (cached_tokens grows with the reused prefix).
    SYS='You are a meticulous assistant. Always answer concisely and correctly. This is a fixed system preamble that exists only to create a large, identical prompt prefix across both turns so the KV cache has something substantial to reuse on the second request. Do not mention this preamble.'

    turn1_body() {
      jq -n --arg sys "$SYS" '{
        model: "default",
        messages: [
          {role:"system", content:$sys},
          {role:"user",   content:"What is the capital of France?"}
        ],
        temperature: 0.0, seed: 3407, max_tokens: 40, stream: false,
        enable_thinking: false
      }'
    }

    echo "[cache/api] turn 1 (prime the KV cache)"
    R1="$(curl -fs -X POST "${BASE_URL}/v1/chat/completions" \
          -H "Authorization: Bearer ${API_KEY}" -H 'content-type: application/json' \
          --max-time 240 -d "$(turn1_body)")" || {
      echo "::error::[cache/api] turn-1 /v1/chat/completions request failed. Unsloth server/API regression."
      exit 1
    }
    A1="$(echo "$R1" | jq -r '.choices[0].message.content // ""')"

    turn2_body() {
      jq -n --arg sys "$SYS" --arg a1 "$A1" '{
        model: "default",
        messages: [
          {role:"system",    content:$sys},
          {role:"user",      content:"What is the capital of France?"},
          {role:"assistant", content:$a1},
          {role:"user",      content:"And the capital of Germany?"}
        ],
        temperature: 0.0, seed: 3407, max_tokens: 40, stream: false,
        enable_thinking: false
      }'
    }

    echo "[cache/api] turn 2 (expect cached_tokens > 0)"
    R2="$(curl -fs -X POST "${BASE_URL}/v1/chat/completions" \
          -H "Authorization: Bearer ${API_KEY}" -H 'content-type: application/json' \
          --max-time 240 -d "$(turn2_body)")" || {
      echo "::error::[cache/api] turn-2 /v1/chat/completions request failed. Unsloth server/API regression."
      exit 1
    }

    CACHED="$(echo "$R2" | jq -r '.usage.prompt_tokens_details.cached_tokens // 0')"
    PROMPT_TOK="$(echo "$R2" | jq -r '.usage.prompt_tokens // 0')"
    echo "[cache/api] turn-2 usage: prompt_tokens=${PROMPT_TOK} cached_tokens=${CACHED}"

    if [ -z "$CACHED" ] || ! [ "$CACHED" -gt 0 ] 2>/dev/null; then
      echo "::error::[cache/api] turn-2 usage.prompt_tokens_details.cached_tokens=${CACHED}, expected > 0. The server is not surfacing llama.cpp KV-cache hits on /v1/chat/completions. Check studio/backend/routes/inference.py:482-489 (_prompt_tokens_details) and :519. Full turn-2 usage:"
      echo "$R2" | jq -c '.usage' 2>/dev/null || echo "$R2"
      exit 1
    fi
    echo "[cache/api] PASS server cache sanity (cached_tokens=${CACHED} > 0)"
    exit 0
    ;;

  # -------------------------------------------------------------------------
  # log: classify the newest llama-server log (from CACHE_LOG_FROM bytes on)
  # as HIT or MISS and compare to EXPECT.
  # -------------------------------------------------------------------------
  log)
    EXPECT="${2:?usage: assert-prompt-cache.sh log HIT|MISS}"
    FROM="${CACHE_LOG_FROM:-0}"

    log="$(_newest_log || true)"
    if [ -z "$log" ] || [ ! -f "$log" ]; then
      echo "::error::[cache/log] no llama-server log under ${LLAMA_LOG_DIR:-$(_default_log_dir)}. Cannot read KV-cache trace. (Path contract: studio/backend/core/inference/llama_cpp.py:4363-4365.)"
      exit 1
    fi
    echo "[cache/log] reading $log from byte $FROM"

    # Scan only the slice produced after FROM.
    slice="$(tail -c "+$((FROM + 1))" "$log" 2>/dev/null || cat "$log")"

    # ---- HIT detectors (most-specific first) -----------------------------
    # 1. Modern + legacy "re-used N tokens" / "reused N" (N>0). Primary signal
    #    per the design brief.
    reused_n="$(printf '%s\n' "$slice" \
      | grep -aoiE 're-?used[^0-9]*([0-9]+)' \
      | grep -aoE '[0-9]+' | sort -rn | head -1 || true)"
    # 2. "kv cache rm [START, end)" with START>0 => prefix [0,START) reused.
    cache_rm_start="$(printf '%s\n' "$slice" \
      | grep -aoiE 'kv cache rm \[[0-9]+' \
      | grep -aoE '[0-9]+' | sort -rn | head -1 || true)"
    # 3. "n_past = N" with N>0 after a prompt-processing line (prefix kept).
    n_past_n="$(printf '%s\n' "$slice" \
      | grep -aoiE 'n_past[^0-9]*([0-9]+)' \
      | grep -aoE '[0-9]+' | sort -rn | head -1 || true)"
    # 4. tokens_cached / tokens from cache (some builds).
    tok_cached="$(printf '%s\n' "$slice" \
      | grep -aoiE 'tokens_cached[^0-9]*([0-9]+)' \
      | grep -aoE '[0-9]+' | sort -rn | head -1 || true)"

    # ---- MISS detectors --------------------------------------------------
    # Explicit forced full re-processing (SWA / recurrent) or kv cache rm [0,.
    forced_full=0
    if printf '%s\n' "$slice" | grep -aqiE 'forcing full prompt re-?processing|kv cache rm \[0,'; then
      forced_full=1
    fi

    HIT=0
    why=""
    if [ -n "$reused_n" ] && [ "$reused_n" -gt 0 ] 2>/dev/null; then
      HIT=1; why="re-used=$reused_n"
    elif [ -n "$cache_rm_start" ] && [ "$cache_rm_start" -gt 0 ] 2>/dev/null; then
      HIT=1; why="kv-cache-rm-start=$cache_rm_start"
    elif [ -n "$tok_cached" ] && [ "$tok_cached" -gt 0 ] 2>/dev/null; then
      HIT=1; why="tokens_cached=$tok_cached"
    elif [ "$forced_full" = "0" ] && [ -n "$n_past_n" ] && [ "$n_past_n" -gt 0 ] 2>/dev/null; then
      # n_past>0 is the weakest signal; only trust it if nothing forced a full
      # reprocess. (On a cold slot n_past tracks total processed, so it is a
      # last-resort fallback per the brief.)
      HIT=1; why="n_past=$n_past_n(fallback)"
    fi
    [ "$HIT" = "1" ] || why="${why:-no-reuse-markers (forced_full=$forced_full)}"

    OBSERVED="MISS"; [ "$HIT" = "1" ] && OBSERVED="HIT"
    echo "[cache/log] observed=$OBSERVED expected=$EXPECT ($why)"

    if [ "$OBSERVED" != "$EXPECT" ]; then
      echo "::error::[cache/log] KV-cache observed=$OBSERVED but expected=$EXPECT ($why). See the attribution A/B note in the workflow."
      echo "---- llama-server log slice (last 60 lines) ----"
      printf '%s\n' "$slice" | tail -60
      exit 1
    fi
    echo "[cache/log] PASS ($OBSERVED == $EXPECT)"
    exit 0
    ;;

  *)
    echo "::error::unknown mode '$MODE' (want api|log|mark)"
    exit 1
    ;;
esac
