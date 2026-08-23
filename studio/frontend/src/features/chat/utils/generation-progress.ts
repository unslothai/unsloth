// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface ChatGenerationProgress {
  runId: string;
  phase: "preparing" | "prefill" | "waiting" | "reconnecting";
  runStartedAt: number;
  startedAt: number;
  estimatedPromptTokens?: number;
  previousPromptTokensPerSecond?: number;
  promptProgress?: PromptProgressSample;
}

export interface PromptProgressSample {
  total: number;
  processed: number;
  cache: number;
  timeMs: number;
}

export function promptProgressMetrics(progress: PromptProgressSample): {
  percentage: number;
  tokensPerSecond?: number;
  etaMs?: number;
} {
  const percentage =
    progress.total > 0
      ? Math.min(100, Math.max(0, (progress.processed / progress.total) * 100))
      : 0;
  const evaluatedTokens = Math.max(0, progress.processed - progress.cache);
  const tokensPerSecond =
    evaluatedTokens > 0 && progress.timeMs > 0
      ? evaluatedTokens / (progress.timeMs / 1000)
      : undefined;
  const remainingTokens = Math.max(0, progress.total - progress.processed);
  return {
    percentage,
    tokensPerSecond,
    etaMs:
      tokensPerSecond !== undefined && remainingTokens > 0
        ? (remainingTokens / tokensPerSecond) * 1000
        : undefined,
  };
}

const ENCODED_MEDIA_KEYS = new Set([
  "audio",
  "data",
  "file_data",
  "image_url",
  "input_audio",
  "video_url",
]);

function promptCharacterCount(
  value: unknown,
  key: string | undefined,
  seen: WeakSet<object>,
): number {
  if (typeof value === "string") {
    if (ENCODED_MEDIA_KEYS.has(key ?? "") || value.startsWith("data:")) {
      return 0;
    }
    return value.length;
  }
  if (typeof value !== "object" || value === null) {
    return 0;
  }
  if (seen.has(value)) {
    return 0;
  }
  seen.add(value);

  if (Array.isArray(value)) {
    return value.reduce(
      (total, item) => total + promptCharacterCount(item, key, seen),
      0,
    );
  }

  let total = 0;
  for (const [entryKey, entryValue] of Object.entries(value)) {
    if (ENCODED_MEDIA_KEYS.has(entryKey)) {
      continue;
    }
    // Tool definitions and structured content include their field names in the
    // rendered prompt, so count that small amount of syntax as well.
    total += entryKey.length + promptCharacterCount(entryValue, entryKey, seen);
  }
  return total;
}

/** Approximate request size without walking large base64 media blobs. */
export function estimatePromptTokens(value: unknown): number | undefined {
  const characters = promptCharacterCount(value, undefined, new WeakSet());
  return characters > 0 ? Math.max(1, Math.round(characters / 4)) : undefined;
}

/** Last exact prompt rate reported by an engine, used only as a labelled baseline. */
export function lastMeasuredPromptRate(
  messages: readonly unknown[],
): number | undefined {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index] as
      | {
          metadata?: {
            custom?: {
              serverTimings?: Record<string, unknown>;
            };
          };
        }
      | undefined;
    const timings = message?.metadata?.custom?.serverTimings;
    const rate = Reflect.get(timings ?? {}, "prompt_per_second");
    const promptTokens = Reflect.get(timings ?? {}, "prompt_n");
    const promptMs = Reflect.get(timings ?? {}, "prompt_ms");
    if (
      typeof rate === "number" &&
      Number.isFinite(rate) &&
      rate > 0 &&
      typeof promptTokens === "number" &&
      promptTokens > 0 &&
      typeof promptMs === "number" &&
      promptMs > 0
    ) {
      return rate;
    }
  }
  return undefined;
}
