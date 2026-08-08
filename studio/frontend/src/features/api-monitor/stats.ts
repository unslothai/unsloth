// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Split out of use-api-monitor.ts so the arithmetic runs under `node --test`: the hook
// imports chat-api through the "@/" alias, which node cannot resolve.

import type { ApiMonitorEntry } from "../chat/types/api";

export interface MonitorStats {
  active: number;
  total: number;
  completed: number;
  errors: number;
  cancelled: number;
  /** Mean duration over finished requests, or null when none have finished. */
  avgDurationMs: number | null;
  /** Slowest finished request, for spotting a pathological call. */
  maxDurationMs: number | null;
  totalTokens: number;
  /** Share of finished requests that failed, 0-1. Null when nothing finished. */
  errorRate: number | null;
  /** Mean completion tokens per second over requests that reported both. */
  tokensPerSecond: number | null;
}

function isTerminal(entry: ApiMonitorEntry): boolean {
  return entry.status !== "running";
}

function completionTokens(entry: ApiMonitorEntry): number | null {
  if (entry.completion_tokens != null) {
    return entry.completion_tokens;
  }
  // Some providers report only a total, so subtract the prompt to estimate generated.
  if (entry.total_tokens != null && entry.prompt_tokens != null) {
    return Math.max(0, entry.total_tokens - entry.prompt_tokens);
  }
  return null;
}

function entryTokens(entry: ApiMonitorEntry): number {
  if (entry.total_tokens != null) {
    return entry.total_tokens;
  }
  return (entry.prompt_tokens ?? 0) + (entry.completion_tokens ?? 0);
}

export function computeStats(entries: ApiMonitorEntry[]): MonitorStats {
  let active = 0;
  let completed = 0;
  let errors = 0;
  let cancelled = 0;
  let totalTokens = 0;
  let durationSum = 0;
  let durationCount = 0;
  let maxDurationMs: number | null = null;
  // Total tokens over total time: averaging rates lets one tiny request outweigh a long one.
  let generatedTokens = 0;
  let generatedDurationMs = 0;

  let requests = 0;

  for (const entry of entries) {
    // A load, unload or download is not an HTTP call: it reads as "running" throughout, so
    // counting it invents an in-flight request. The backend leaves these out of active_count.
    if (entry.kind === "lifecycle") {
      continue;
    }
    requests += 1;
    totalTokens += entryTokens(entry);
    if (entry.status === "running") {
      active += 1;
    } else if (entry.status === "error") {
      errors += 1;
    } else if (entry.status === "cancelled") {
      cancelled += 1;
    } else {
      completed += 1;
    }
    const duration = entry.duration_ms;
    if (duration != null && isTerminal(entry)) {
      durationSum += duration;
      durationCount += 1;
      maxDurationMs =
        maxDurationMs == null ? duration : Math.max(maxDurationMs, duration);
      const generated = completionTokens(entry);
      // Rate the decode window, not the whole request: duration_ms carries queue wait
      // and prefill, which reports a model generating at 50 tok/s as 5 behind a busy
      // slot. predicted_ms covers every predicted token, the streamed window one fewer.
      // A reply with neither window is left out rather than dragged in at the old rate.
      const decodeMs = entry.decode_ms;
      if (decodeMs != null && decodeMs > 0 && generated != null) {
        const decoded = entry.decode_ms_authoritative ? generated : generated - 1;
        if (decoded > 0) {
          generatedTokens += decoded;
          generatedDurationMs += decodeMs;
        }
      }
    }
  }

  const finished = completed + errors + cancelled;
  return {
    active,
    total: requests,
    completed,
    errors,
    cancelled,
    avgDurationMs: durationCount > 0 ? durationSum / durationCount : null,
    maxDurationMs,
    totalTokens,
    errorRate: finished > 0 ? errors / finished : null,
    tokensPerSecond:
      generatedDurationMs > 0
        ? generatedTokens / (generatedDurationMs / 1000)
        : null,
  };
}
