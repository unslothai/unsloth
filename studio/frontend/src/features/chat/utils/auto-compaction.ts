// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Local GGUF and MLX auto-compaction. On or off; how it compacts is the server's call.
 *
 *  Studio used to offer the policy as a setting, but the choice needed the reader to know what a
 *  checkpoint epoch and a rolling window were before it meant anything, and both sides of it were
 *  already the server's to configure. It now always follows the server, which is what the setting
 *  shipped as anyway (UNSLOTH_CONTEXT_POLICY, default "checkpoint"). */

export const DEFAULT_AUTO_COMPACT_ENABLED = true;

export function ggufCompactionRequestFields(options: {
  isGguf: boolean;
  isMlx?: boolean;
  autoCompactEnabled: boolean;
}): {
  context_overflow?: "error" | "truncate_oldest";
} {
  if (!options.isGguf && !options.isMlx) return {};
  if (!options.autoCompactEnabled) {
    // An omitted field falls back to UNSLOTH_CONTEXT_OVERFLOW, which may still compact. "error" is an
    // explicit refusal of that fallback.
    return { context_overflow: "error" };
  }
  // No context_policy: the server applies UNSLOTH_CONTEXT_POLICY.
  return { context_overflow: "truncate_oldest" };
}

// The server's default ROLLING_COMPACTION_HEADROOM_RATIO.
const API_COMPACTION_HEADROOM = 0.25;
// The request schema's ceiling on compaction_threshold.
const API_COMPACTION_THRESHOLD_MAX = 2_000_000;

export function apiCompactionRequestFields(options: {
  autoCompactEnabled: boolean;
  contextLength: number | null | undefined;
}): {
  context_overflow?: "truncate_oldest";
  compaction_threshold?: number;
  context_window?: number;
} {
  const { autoCompactEnabled, contextLength } = options;
  if (!autoCompactEnabled || !contextLength || contextLength <= 0) return {};
  return {
    context_overflow: "truncate_oldest",
    compaction_threshold: Math.min(
      API_COMPACTION_THRESHOLD_MAX,
      Math.floor(contextLength * (1 - API_COMPACTION_HEADROOM)),
    ),
    context_window: contextLength,
  };
}
