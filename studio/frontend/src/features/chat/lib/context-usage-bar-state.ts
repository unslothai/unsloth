// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// its own plain module so the node suite can drive it: the component is JSX the runner cannot import

export const formatTokenCount = (n: number): string => {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
};

export const formatTokenCountFull = (n: number): string => {
  return n.toLocaleString();
};

export type ContextUsageBarInput = {
  // null when nothing has been counted yet
  used?: number | null;
  // null on external providers, whose context window is unknown
  total?: number | null;
  cached?: number;
  // anthropic-only (billed at the write premium)
  cacheWrites?: number;
  promptTokens?: number;
  completionTokens?: number;
  // MLX keeps generating past the window instead of stopping there, so it needs the
  // opposite advice from llama.cpp once a conversation outgrows the limit.
  isMlx?: boolean;
  /** context_length_enforced as the load reported it; null where it does not answer. */
  contextEnforced?: boolean | null;
};

/**
 * Which limit warning the tooltip carries, if any.
 *
 * llama.cpp stops generating at the window, so its advice is to raise the limit before
 * hitting it. MLX generates straight past instead, so the same wording would promise a
 * stop that never comes -- and once a conversation is over the window there is something
 * different to say about it. Read from the unclamped ratio: the reported percent caps at
 * 100%, which is exactly the state being reported on.
 */
export type ContextLimitAdvice =
  | "none"
  | "stops-at-limit"
  | "mlx-near-limit"
  | "mlx-past-limit"
  | "unenforced-limit";

function contextLimitAdvice(
  used: number,
  total: number,
  isMlx: boolean | undefined,
  enforced: boolean | null | undefined,
): ContextLimitAdvice {
  if ((used / total) * 100 <= 85) return "none";
  // A window the backend confirmed does not bound the cache is not a limit at all:
  // nothing rotates and nothing stops, so neither of the other two is true of it. An
  // unjudged MLX window says the same thing operationally: the probe could not build a
  // cache, so none was bounded and it grows exactly as a confirmed false one does.
  if (enforced === false || (isMlx && enforced == null)) return "unenforced-limit";
  if (!isMlx) return "stops-at-limit";
  return used > total ? "mlx-past-limit" : "mlx-near-limit";
}

export type ContextUsageBarState = {
  face: string;
  label: string;
  totalRowName: string;
  totalRowValue: string;
  // null whenever no ratio can be stated, which also withholds the fill and the 85% warning
  percent: number | null;
  // whether any per-turn row renders, so the tooltip rule never floats above nothing
  hasUsageDetails: boolean;
  advice: ContextLimitAdvice;
};

// a counted zero and an uncounted chat differ: an unmeasured prompt must not read as 0% of the window
export function deriveContextUsageBar({
  used,
  total,
  cached,
  cacheWrites,
  promptTokens,
  completionTokens,
  isMlx,
  contextEnforced,
}: ContextUsageBarInput): ContextUsageBarState | null {
  const limit = typeof total === "number" && total > 0 ? total : null;
  const usedTokens =
    typeof used === "number" && Number.isFinite(used) ? used : null;
  const hasUsageDetails =
    promptTokens !== undefined ||
    completionTokens !== undefined ||
    (cached !== undefined && cached > 0) ||
    (cacheWrites !== undefined && cacheWrites > 0);

  if (limit === null) {
    // nothing to show: no window to name, and no counted usage to report against one
    if (usedTokens === null) return null;
    if (usedTokens <= 0 && !hasUsageDetails) return null;
    return {
      face: `${formatTokenCount(usedTokens)} tokens`,
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
      label: `Context window: ${formatTokenCount(limit)} tokens, usage not counted yet`,
      totalRowName: "Context window",
      totalRowValue: formatTokenCountFull(limit),
      percent: null,
      hasUsageDetails,
      advice: "none",
    };
  }

  return {
    face: `${formatTokenCount(usedTokens)} / ${formatTokenCount(limit)}`,
    label: `Context usage: ${formatTokenCount(usedTokens)} of ${formatTokenCount(limit)} tokens`,
    totalRowName: "Total",
    totalRowValue: `${formatTokenCountFull(usedTokens)} / ${formatTokenCountFull(limit)}`,
    percent: Math.min((usedTokens / limit) * 100, 100),
    hasUsageDetails,
    advice: contextLimitAdvice(usedTokens, limit, isMlx, contextEnforced),
  };
}
