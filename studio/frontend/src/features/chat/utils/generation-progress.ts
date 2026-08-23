// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface PromptProgressSample {
  total: number;
  processed: number;
  cache: number;
  timeMs: number;
}

type PromptProgressPoint = Pick<
  PromptProgressSample,
  "processed" | "cache" | "timeMs"
>;

export interface ChatGenerationProgress extends PromptProgressSample {
  runId: string;
  history: PromptProgressPoint[];
}

const MAX_PROGRESS_HISTORY = 32;

export function recordPromptProgress(
  runId: string,
  sample: PromptProgressSample,
  previous?: ChatGenerationProgress,
): ChatGenerationProgress {
  const history =
    previous?.runId === runId
      ? [
          ...previous.history,
          {
            processed: previous.processed,
            cache: previous.cache,
            timeMs: previous.timeMs,
          },
        ].slice(-MAX_PROGRESS_HISTORY)
      : [];
  return { runId, ...sample, history };
}

type PromptBatch = {
  tokens: number;
  msPerToken: number;
};

function promptBatches(
  progress: PromptProgressSample,
  history: readonly PromptProgressPoint[],
): PromptBatch[] {
  const samples = [...history, progress];
  const batches: PromptBatch[] = [];
  for (let index = 1; index < samples.length; index += 1) {
    const previous = samples[index - 1];
    const current = samples[index];
    const tokens =
      current.processed - current.cache - (previous.processed - previous.cache);
    const elapsedMs = current.timeMs - previous.timeMs;
    if (tokens <= 0 || elapsedMs <= 0) {
      continue;
    }
    batches.push({
      tokens,
      msPerToken: elapsedMs / tokens,
    });
  }
  return batches;
}

function averageBatchSlowdown(batches: readonly PromptBatch[]): number {
  if (batches.length < 2) {
    return 1;
  }
  const factors = batches
    .slice(1)
    .map((batch, index) => batch.msPerToken / batches[index].msPerToken)
    .filter((factor) => Number.isFinite(factor) && factor > 0)
    .map((factor) => Math.min(2, Math.max(0.5, factor)))
    .sort((a, b) => a - b);
  if (factors.length === 0) {
    return 1;
  }
  // Once enough evidence exists, discard one fastest and slowest transition so
  // a warm-up or scheduling spike cannot dominate every remaining batch.
  const measured = factors.length >= 5 ? factors.slice(1, -1) : factors;
  const observed = Math.exp(
    measured.reduce((sum, factor) => sum + Math.log(factor), 0) /
      measured.length,
  );
  const evidence = Math.min(1, factors.length / 5);
  return Math.min(1.25, 1 + Math.max(0, observed - 1) * evidence);
}

function forecastPromptEtaMs(
  progress: PromptProgressSample,
  batches: readonly PromptBatch[],
): number | undefined {
  const remainingTokens = Math.max(0, progress.total - progress.processed);
  if (remainingTokens <= 0 || batches.length === 0) {
    return undefined;
  }
  const latest = batches[batches.length - 1];
  const slowdown = averageBatchSlowdown(batches);
  const orderedSizes = batches
    .map((batch) => batch.tokens)
    .sort((a, b) => a - b);
  const batchSize = orderedSizes[Math.floor(orderedSizes.length / 2)];
  let cursor = progress.processed;
  let etaMs = 0;
  let predictedMsPerToken = latest.msPerToken;
  while (cursor < progress.total) {
    const tokens = Math.min(batchSize, progress.total - cursor);
    predictedMsPerToken *= slowdown;
    etaMs += tokens * predictedMsPerToken;
    cursor += tokens;
  }
  return etaMs;
}

export function promptProgressMetrics(
  progress: PromptProgressSample,
  history: readonly PromptProgressPoint[] = [],
): {
  percentage: number;
  tokensPerSecond?: number;
  etaMs?: number;
} {
  const percentage =
    progress.total > 0
      ? Math.min(100, Math.max(0, (progress.processed / progress.total) * 100))
      : 0;
  const evaluatedTokens = Math.max(0, progress.processed - progress.cache);
  const cumulativeRate =
    evaluatedTokens > 0 && progress.timeMs > 0
      ? evaluatedTokens / (progress.timeMs / 1000)
      : undefined;
  const batches = promptBatches(progress, history);
  const latestBatch = batches.at(-1);
  const tokensPerSecond = latestBatch
    ? 1000 / latestBatch.msPerToken
    : cumulativeRate;
  return {
    percentage,
    tokensPerSecond,
    etaMs:
      forecastPromptEtaMs(progress, batches) ??
      (tokensPerSecond !== undefined && progress.total > progress.processed
        ? ((progress.total - progress.processed) / tokensPerSecond) * 1000
        : undefined),
  };
}
