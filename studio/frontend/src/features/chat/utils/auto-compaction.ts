// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Local GGUF, MLX and managed engine auto-compaction is on or off only; the server chooses how it
 * compacts (UNSLOTH_CONTEXT_POLICY, default "checkpoint").
 */

export const DEFAULT_AUTO_COMPACT_ENABLED = true;

export function ggufCompactionRequestFields(options: {
  isGguf: boolean;
  isMlx?: boolean;
  isManagedEngine?: boolean;
  autoCompactEnabled: boolean;
  // /compact: asked for by the user, so it applies with auto-compaction off too
  compactNow?: boolean;
}): {
  context_overflow?: "error" | "truncate_oldest";
  compact_context?: true;
} {
  if (!options.isGguf && !options.isMlx && !options.isManagedEngine) return {};
  if (options.compactNow) {
    return { context_overflow: "truncate_oldest", compact_context: true };
  }
  if (!options.autoCompactEnabled) {
    // explicit error avoids the UNSLOTH_CONTEXT_OVERFLOW fallback, which may compact
    return { context_overflow: "error" };
  }
  // omit context_policy so the server applies UNSLOTH_CONTEXT_POLICY
  return { context_overflow: "truncate_oldest" };
}

// leaves 25% of the window free, matching _EXTERNAL_COMPACTION_HEADROOM for self-hosted servers
const API_COMPACTION_HEADROOM = 0.25;
// request schema ceiling for compaction_threshold
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
  if (!autoCompactEnabled) return {};
  // without a catalogued window, the backend reads the self-hosted server's reported window
  if (!contextLength || contextLength <= 0) return { context_overflow: "truncate_oldest" };
  return {
    context_overflow: "truncate_oldest",
    compaction_threshold: Math.min(
      API_COMPACTION_THRESHOLD_MAX,
      Math.floor(contextLength * (1 - API_COMPACTION_HEADROOM)),
    ),
    context_window: contextLength,
  };
}
