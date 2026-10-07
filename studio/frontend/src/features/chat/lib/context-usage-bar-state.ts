// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Plain module so the node suite can import it.

export const formatTokenCount = (n: number): string => {
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
  promptTokens?: number;
  completionTokens?: number;
  // MLX keeps generating past the window, so it needs different advice than llama.cpp.
  isMlx?: boolean;
  contextUnboundedWhenBatched?: boolean;
  parallelSlots?: number | null;
  contextEnforced?: boolean | null;
  contextBudget?: number | null;
  estimated?: boolean;
};

/** Limit warning for the tooltip; read from the unclamped ratio since percent caps at 100. */
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
};

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
}: ContextUsageBarInput): ContextUsageBarState | null {
  const limit = typeof total === "number" && total > 0 ? total : null;
  const usedTokens =
    typeof used === "number" && Number.isFinite(used) ? used : null;

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
