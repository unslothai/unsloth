// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LiveGenerationProgress } from "../stores/chat-runtime-store";

export type GenerationStatsUpdates = "settled" | "realtime";

export const GENERATION_STATS_UPDATES: readonly GenerationStatsUpdates[] = [
  "settled",
  "realtime",
];

export const DEFAULT_GENERATION_STATS_UPDATES: GenerationStatsUpdates =
  "settled";

// Under the server's 250 ms decode cadence, so throttling never drops a frame it sends, only bursts.
export const LIVE_STATS_INTERVAL_MS = 200;

export function normaliseGenerationStatsUpdates(
  value: unknown,
): GenerationStatsUpdates {
  return value === "realtime" ? "realtime" : DEFAULT_GENERATION_STATS_UPDATES;
}

export type ThrottleClock = {
  now: () => number;
  setTimeout: (run: () => void, ms: number) => unknown;
  clearTimeout: (handle: unknown) => void;
};

const defaultClock: ThrottleClock = {
  now: () => Date.now(),
  setTimeout: (run, ms) => globalThis.setTimeout(run, ms),
  clearTimeout: (handle) =>
    globalThis.clearTimeout(handle as ReturnType<typeof setTimeout>),
};

/** Leading and trailing: the last change always lands. */
export function createTrailingThrottle(
  intervalMs: number,
  run: () => void,
  clock: ThrottleClock = defaultClock,
): { schedule: () => void; cancel: () => void } {
  let lastRunAt = Number.NEGATIVE_INFINITY;
  let pending: unknown = null;
  const fire = () => {
    pending = null;
    lastRunAt = clock.now();
    run();
  };
  return {
    schedule: () => {
      if (pending !== null) return;
      const wait = lastRunAt + intervalMs - clock.now();
      if (wait <= 0) fire();
      else pending = clock.setTimeout(fire, wait);
    },
    cancel: () => {
      if (pending !== null) clock.clearTimeout(pending);
      pending = null;
    },
  };
}

export type LiveContextUsage = {
  used: number;
  promptTokens: number;
  completionTokens: number;
  cached: number;
};

/** A new generation pass starts when live progress appears, or when prefill follows a decoded pass. Read on
 *  every store change, before throttling, so a gap shorter than the throttle still counts. */
export function nextPassNumber(
  pass: number,
  previous: LiveGenerationProgress | undefined,
  next: LiveGenerationProgress | undefined,
): number {
  if (!next?.prefill) return pass;
  if (!previous?.prefill) return pass + 1;
  return previous.decode && !next.decode ? pass + 1 : pass;
}

/** True once this pass's own context_usage landed (the last decode frame trails it). Keyed on the snapshot
 *  object, not its numbers, so a regenerate with the same prompt still counts live. */
export function createPassSettledTracker<S>(): (
  pass: number,
  snapshot: S,
) => boolean {
  let baseline: S | undefined;
  let seenPass: number | undefined;
  return (pass, snapshot) => {
    if (pass !== seenPass) {
      baseline = snapshot;
      seenPass = pass;
      return false;
    }
    return snapshot !== baseline;
  };
}

/** Null when the stream has not said, so the meter keeps the last settled count. */
export function liveContextUsage(
  live: LiveGenerationProgress | undefined,
  passSettled: boolean,
): LiveContextUsage | null {
  const prefill = live?.prefill;
  if (!prefill || live?.compacting || passSettled) return null;
  const cached = prefill.cached ?? 0;
  const decode = live.decode;
  if (decode) {
    if (!prefill.total) return null;
    return {
      used: prefill.total + decode.tokens,
      promptTokens: prefill.total,
      completionTokens: decode.tokens,
      cached,
    };
  }
  return {
    used: prefill.processed,
    promptTokens: prefill.processed,
    completionTokens: 0,
    cached,
  };
}
