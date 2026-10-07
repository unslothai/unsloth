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
    // explicit error avoids the UNSLOTH_CONTEXT_OVERFLOW fallback, which may compact
    return { context_overflow: "error" };
  }
  // omit context_policy so the server applies UNSLOTH_CONTEXT_POLICY
  return { context_overflow: "truncate_oldest" };
}

// Share of the window left free when compaction starts. Mirrored by _EXTERNAL_COMPACTION_HEADROOM in
// routes/inference.py, which derives the threshold from a self-hosted server's reported window.
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
  // No catalogued window: a self-hosted server reports the one it runs with, so the backend reads it there.
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
