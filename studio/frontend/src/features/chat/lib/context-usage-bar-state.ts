// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Plain module so the node suite can import it.

export const formatTokenCount = (n: number): string => {
  if (n >= 999_950) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
};

export const formatTokenCountFull = (n: number): string => {
  return n.toLocaleString();
};

export type ContextUsageBarInput = {
  used?: number | null;
  total?: number | null;
  cached?: number;
  cacheWrites?: number;
  compactionCount?: number;
  promptTokens?: number;
  completionTokens?: number;
  // MLX keeps generating past the window, so it needs different advice than llama.cpp.
  isMlx?: boolean;
  contextUnboundedWhenBatched?: boolean;
  parallelSlots?: number | null;
  contextEnforced?: boolean | null;
  contextBudget?: number | null;
  estimated?: boolean;
  /** A cloud model: the window is the provider's, so the local limit advice does not apply. */
  cloud?: boolean;
  /** Tokens at which this chat auto-compacts; marked on the ring and named in the tooltip. */
  compactAt?: number | null;
};

export type ContextUsageRows = Pick<
  ContextUsageBarInput,
  "promptTokens" | "completionTokens" | "cached" | "cacheWrites"
>;

export function contextUsageRows(
  usage:
    | {
        promptTokens: number;
        completionTokens: number;
        contextTokens?: number;
        cachedTokens?: number;
        cacheWriteTokens?: number;
      }
    | null
    | undefined,
  summedPrompt: boolean,
): ContextUsageRows {
  if (!usage) return {};
  const context = usage.contextTokens;
  if (context === undefined) {
    return {
      promptTokens: usage.promptTokens,
      completionTokens: usage.completionTokens,
      cached: usage.cachedTokens,
      cacheWrites: usage.cacheWriteTokens,
    };
  }
  const describesContext = summedPrompt
    ? context === usage.promptTokens + usage.completionTokens
    : context >= usage.promptTokens;
  if (!describesContext) return {};
  return {
    promptTokens: usage.promptTokens,
    completionTokens: context - usage.promptTokens,
    cached: usage.cachedTokens,
    cacheWrites: usage.cacheWriteTokens,
  };
}

/**
 * Which limit warning the tooltip carries: llama.cpp stops at the window, so its advice is to raise
 * the limit first; MLX runs past it, so no stop is promised, and an over-window conversation gets
 * other wording; read the unclamped ratio, since the reported percent caps at 100%.
 */
export type ContextLimitAdvice =
  | "none"
  | "stops-at-limit"
  | "mlx-near-limit"
  | "mlx-past-limit"
  | "mlx-refuses-past-limit"
  | "unenforced-limit";

function contextLimitAdvice(
  used: number,
  total: number,
  isMlx: boolean | undefined,
  enforced: boolean | null | undefined,
  unboundedWhenBatched: boolean | undefined,
  slots: number | null | undefined,
  budget: number | null | undefined,
): ContextLimitAdvice {
  if ((used / total) * 100 <= 85) return "none";
  // Budget wins over contextEnforced: the cache is unbounded but requests are refused.
  if (budget) return "mlx-refuses-past-limit";
  // An unbounded (or unjudged MLX) cache is not a limit: nothing rotates or stops.
  if (
    enforced === false ||
    (isMlx && enforced == null) ||
    (unboundedWhenBatched && (slots ?? 1) > 1)
  ) {
    return "unenforced-limit";
  }
  if (!isMlx) return "stops-at-limit";
  return used > total ? "mlx-past-limit" : "mlx-near-limit";
}

export type ContextUsageBarState = {
  face: string;
  compactFace: string | null;
  label: string;
  totalRowName: string;
  totalRowValue: string;
  percent: number | null;
  hasUsageDetails: boolean;
  advice: ContextLimitAdvice;
  /** Where on the ring compaction happens, as a share of the window; null when not drawn. */
  compactMarkPercent: number | null;
};

/** The compaction point as a share of the window, when both are known and it falls inside. */
export function compactMarkPercent(
  compactAt: number | null | undefined,
  total: number | null | undefined,
): number | null {
  if (typeof compactAt !== "number" || !(compactAt > 0)) return null;
  if (typeof total !== "number" || !(total > 0) || compactAt >= total) return null;
  return (compactAt / total) * 100;
}

// An unmeasured prompt must not read as 0% of the window.
export function deriveContextUsageBar({
  used,
  total,
  cached,
  cacheWrites,
  promptTokens,
  completionTokens,
  isMlx,
  contextEnforced,
  contextUnboundedWhenBatched,
  parallelSlots,
  contextBudget,
  estimated,
  cloud,
  compactAt,
}: ContextUsageBarInput): ContextUsageBarState | null {
  const state = deriveUsageFace({
    used,
    total,
    cached,
    cacheWrites,
    promptTokens,
    completionTokens,
    isMlx,
    contextEnforced,
    contextUnboundedWhenBatched,
    parallelSlots,
    contextBudget,
    estimated,
  });
  if (!state) return null;
  return {
    ...state,
    // A provider's window is not a local setting: nothing here stops at it or can raise it.
    advice: cloud ? "none" : state.advice,
    compactMarkPercent: compactMarkPercent(compactAt, total),
  };
}

function deriveUsageFace({
  used,
  total,
  cached,
  cacheWrites,
  promptTokens,
  completionTokens,
  isMlx,
  contextEnforced,
  contextUnboundedWhenBatched,
  parallelSlots,
  contextBudget,
  estimated,
}: ContextUsageBarInput): Omit<ContextUsageBarState, "compactMarkPercent"> | null {
  const limit = typeof total === "number" && total > 0 ? total : null;
  const usedTokens =
    typeof used === "number" && Number.isFinite(used) && used >= 0 ? used : null;

  if (estimated && usedTokens !== null && usedTokens > 0) {
    const approx = `~${formatTokenCount(usedTokens)}`;
    const approxFull = `~${formatTokenCountFull(usedTokens)}`;
    if (limit === null) {
      return {
        face: `${approx} tokens`,
        compactFace: approx,
        label: `Estimated context usage: ${approx} tokens`,
        totalRowName: "Estimated tokens",
        totalRowValue: approxFull,
        percent: null,
        hasUsageDetails: false,
        advice: "none",
      };
    }
    return {
      face: `${approx} / ${formatTokenCount(limit)}`,
      compactFace: null,
      label: `Estimated context usage: ${approx} of ${formatTokenCount(limit)} tokens`,
      totalRowName: "Estimated total",
      totalRowValue: `${approxFull} / ${formatTokenCountFull(limit)}`,
      percent: Math.min((usedTokens / limit) * 100, 100),
      hasUsageDetails: false,
      advice: "none",
    };
  }
  const hasUsageDetails =
    promptTokens !== undefined ||
    completionTokens !== undefined ||
    (cached !== undefined && cached > 0) ||
    (cacheWrites !== undefined && cacheWrites > 0);

  if (limit === null) {
    if (usedTokens === null) return null;
    if (usedTokens <= 0 && !hasUsageDetails) return null;
    return {
      face: `${formatTokenCount(usedTokens)} tokens`,
      compactFace: formatTokenCount(usedTokens),
      label: `Token usage: ${formatTokenCount(usedTokens)} tokens`,
      totalRowName: "Total tokens",
      totalRowValue: formatTokenCountFull(usedTokens),
      percent: null,
      hasUsageDetails,
      advice: "none",
    };
  }

  if (usedTokens === null) {
    return {
      face: `— / ${formatTokenCount(limit)}`,
      compactFace: null,
      label: `Context window: ${formatTokenCount(limit)} tokens, usage not counted yet`,
      totalRowName: "Context window",
      totalRowValue: formatTokenCountFull(limit),
      percent: null,
      hasUsageDetails,
      advice: "none",
    };
  }

  const percent = Math.min((usedTokens / limit) * 100, 100);
  return {
    face: `${formatTokenCount(usedTokens)} / ${formatTokenCount(limit)}`,
    compactFace: null,
    label: `Context usage: ${formatTokenCount(usedTokens)} of ${formatTokenCount(limit)} tokens`,
    totalRowName: "Total",
    totalRowValue: `${formatTokenCountFull(usedTokens)} / ${formatTokenCountFull(limit)}`,
    percent,
    hasUsageDetails,
    advice: contextLimitAdvice(
      usedTokens,
      limit,
      isMlx,
      contextEnforced,
      contextUnboundedWhenBatched,
      parallelSlots,
      contextBudget,
    ),
  };
}
