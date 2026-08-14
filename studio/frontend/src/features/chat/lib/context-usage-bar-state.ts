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
};

export type ContextUsageBarState = {
  face: string;
  label: string;
  totalRowName: string;
  totalRowValue: string;
  // null whenever no ratio can be stated, which also withholds the fill and the 85% warning
  percent: number | null;
  // whether any per-turn row renders, so the tooltip rule never floats above nothing
  hasUsageDetails: boolean;
};

// a counted zero and an uncounted chat differ: an unmeasured prompt must not read as 0% of the window
export function deriveContextUsageBar({
  used,
  total,
  cached,
  cacheWrites,
  promptTokens,
  completionTokens,
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
    };
  }

  return {
    face: `${formatTokenCount(usedTokens)} / ${formatTokenCount(limit)}`,
    label: `Context usage: ${formatTokenCount(usedTokens)} of ${formatTokenCount(limit)} tokens`,
    totalRowName: "Total",
    totalRowValue: `${formatTokenCountFull(usedTokens)} / ${formatTokenCountFull(limit)}`,
    percent: Math.min((usedTokens / limit) * 100, 100),
    hasUsageDetails,
  };
}
