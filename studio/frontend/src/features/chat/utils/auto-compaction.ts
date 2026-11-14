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

/** Bounds of a connection's "Auto-compact after" setting, matching the backend's. */
export const AUTO_COMPACT_TOKENS_MIN = 1_000;
export const AUTO_COMPACT_TOKENS_MAX = API_COMPACTION_THRESHOLD_MAX;

// Server-side compaction limits, mirrored from external_provider.py so the meter marks the real trigger.
const NATIVE_COMPACTION_DEFAULT_MAX = 200_000;
const ANTHROPIC_COMPACTION_MIN = 50_000;

export function normalizeAutoCompactTokens(value: unknown): number | undefined {
  if (
    typeof value !== "number" ||
    !Number.isSafeInteger(value) ||
    value < AUTO_COMPACT_TOKENS_MIN ||
    value > AUTO_COMPACT_TOKENS_MAX
  ) {
    return undefined;
  }
  return value;
}

/** Blank clears the setting; anything else must be a whole number of tokens in range. */
export function parseAutoCompactTokens(input: string): number | null {
  const trimmed = input.trim().replace(/[,_\s]/g, "");
  if (!trimmed) return null;
  if (!/^\d+$/.test(trimmed)) {
    throw new Error("Auto-compact after must be a whole number of tokens.");
  }
  const value = Number(trimmed);
  if (value < AUTO_COMPACT_TOKENS_MIN || value > AUTO_COMPACT_TOKENS_MAX) {
    throw new Error(
      `Auto-compact after must be between ${AUTO_COMPACT_TOKENS_MIN.toLocaleString()} and ${AUTO_COMPACT_TOKENS_MAX.toLocaleString()} tokens.`,
    );
  }
  return value;
}

/** The threshold a request carries: the connection's setting, else 75% of the catalogued window. */
function requestedCompactionThreshold(
  contextLength: number | null | undefined,
  autoCompactTokens: number | null | undefined,
): { threshold: number; uncapped: boolean } | null {
  const window = contextLength && contextLength > 0 ? contextLength : null;
  const override = normalizeAutoCompactTokens(autoCompactTokens);
  if (override !== undefined) {
    return { threshold: window ? Math.min(override, window) : override, uncapped: true };
  }
  if (!window) return null;
  return {
    threshold: Math.min(
      API_COMPACTION_THRESHOLD_MAX,
      Math.floor(window * (1 - API_COMPACTION_HEADROOM)),
    ),
    uncapped: false,
  };
}

/**
 * Where a cloud chat compacts, in tokens, for the context meter. Anthropic and OpenAI compact
 * server-side: 200K at most unless the connection sets its own trigger, and never below
 * Anthropic's 50K floor.
 */
export function cloudCompactionTrigger(options: {
  providerType: string | null | undefined;
  contextLength: number | null | undefined;
  autoCompactTokens?: number | null;
}): number | null {
  const requested = requestedCompactionThreshold(
    options.contextLength,
    options.autoCompactTokens,
  );
  if (!requested) return null;
  let trigger = requested.threshold;
  const native =
    options.providerType === "anthropic" || options.providerType === "openai";
  if (native && !requested.uncapped) {
    trigger = Math.min(trigger, NATIVE_COMPACTION_DEFAULT_MAX);
  }
  if (options.providerType === "anthropic") {
    trigger = Math.max(trigger, ANTHROPIC_COMPACTION_MIN);
  }
  return trigger;
}

export function apiCompactionRequestFields(options: {
  autoCompactEnabled: boolean;
  contextLength: number | null | undefined;
  /** The connection's "Auto-compact after", which replaces the 75% default. */
  autoCompactTokens?: number | null;
  // /compact: asked for by the user, so it applies with auto-compaction off too
  compactNow?: boolean;
}): {
  context_overflow?: "truncate_oldest";
  compaction_threshold?: number;
  compaction_uncapped?: true;
  context_window?: number;
  compact_context?: true;
} {
  const { autoCompactEnabled, contextLength, compactNow } = options;
  if (!autoCompactEnabled && !compactNow) return {};
  const requested = requestedCompactionThreshold(
    contextLength,
    options.autoCompactTokens,
  );
  return {
    context_overflow: "truncate_oldest",
    // without a catalogued window, the backend reads the self-hosted server's reported window
    ...(requested ? { compaction_threshold: requested.threshold } : {}),
    ...(requested?.uncapped ? { compaction_uncapped: true as const } : {}),
    ...(contextLength && contextLength > 0 ? { context_window: contextLength } : {}),
    ...(compactNow ? { compact_context: true as const } : {}),
  };
}
