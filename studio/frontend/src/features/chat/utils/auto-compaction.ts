// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Local GGUF, MLX, NPU and managed engine auto-compaction is on or off only; the server chooses how it
 * compacts (UNSLOTH_CONTEXT_POLICY, default "checkpoint").
 */

export const DEFAULT_AUTO_COMPACT_ENABLED = true;

export function ggufCompactionRequestFields(options: {
  isGguf: boolean;
  isMlx?: boolean;
  isManagedEngine?: boolean;
  isNpu?: boolean;
  autoCompactEnabled: boolean;
  // /compact: asked for by the user, so it applies with auto-compaction off too
  compactNow?: boolean;
}): {
  context_overflow?: "error" | "truncate_oldest";
  compact_context?: true;
} {
  if (
    !options.isGguf &&
    !options.isMlx &&
    !options.isManagedEngine &&
    !options.isNpu
  ) {
    return {};
  }
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
// No widely used model has a window past 1M; the request schema itself allows more.
export const AUTO_COMPACT_TOKENS_MAX = 1_200_000;

// Server-side compaction limits, mirrored from external_provider.py so the meter marks the real trigger.
const NATIVE_COMPACTION_DEFAULT_MAX = 200_000;
const ANTHROPIC_COMPACTION_MIN = 50_000;

export function normalizeAutoCompactTokens(value: unknown): number | undefined {
  if (
    typeof value !== "number" ||
    !Number.isSafeInteger(value) ||
    value < AUTO_COMPACT_TOKENS_MIN
  ) {
    return undefined;
  }
  // A value saved before the ceiling dropped from 2M keeps the closest limit it can.
  return Math.min(value, AUTO_COMPACT_TOKENS_MAX);
}

/** Model ID to tokens; drops entries out of range, and is undefined when none are left. */
export function normalizeAutoCompactTokensByModel(
  value: unknown,
): Record<string, number> | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }
  const limits: Record<string, number> = {};
  for (const [model, tokens] of Object.entries(value)) {
    const normalized = normalizeAutoCompactTokens(tokens);
    if (model.trim() && normalized !== undefined) limits[model.trim()] = normalized;
  }
  return Object.keys(limits).length > 0 ? limits : undefined;
}

export function autoCompactTokensForModel(
  provider:
    | {
        autoCompactTokens?: number;
        autoCompactTokensByModel?: Record<string, number>;
      }
    | null
    | undefined,
  modelId: string | null | undefined,
): number | undefined {
  const own =
    modelId == null ? undefined : provider?.autoCompactTokensByModel?.[modelId];
  return own ?? provider?.autoCompactTokens;
}

/** Unset is "summary"; "rolling" and "checkpoint" drop whole turns into the archive, no summary call. */
export type ProviderCompactionMode = "summary" | "rolling" | "checkpoint";

export function normalizeCompactionMode(
  value: unknown,
): ProviderCompactionMode | undefined {
  return value === "summary" || value === "rolling" || value === "checkpoint"
    ? value
    : undefined;
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

// Slider notches; index 0 is Default (blank). The box beside it takes any value in range.
const AUTO_COMPACT_SLIDER_TOKENS = [
  8_000, 16_000, 32_000, 64_000, 100_000, 128_000, 200_000, 256_000, 400_000,
  512_000, 1_000_000, 1_200_000,
];

/** A known window ends the slider there: past it the limit is capped anyway. */
export function autoCompactSliderStops(
  minimum = AUTO_COMPACT_TOKENS_MIN,
  window?: number | null,
): number[] {
  const top =
    window && window >= minimum
      ? Math.min(window, AUTO_COMPACT_TOKENS_MAX)
      : AUTO_COMPACT_TOKENS_MAX;
  const stops = AUTO_COMPACT_SLIDER_TOKENS.filter(
    (tokens) => tokens >= minimum && tokens < top,
  );
  return [0, ...stops, top];
}

/** The notch nearest the typed value (log scale); blank or unparsable sits on Default. */
export function autoCompactSliderIndex(input: string, stops: number[]): number {
  let value: number | null;
  try {
    value = parseAutoCompactTokens(input);
  } catch {
    return 0;
  }
  if (value === null) return 0;
  const distance = (index: number) => Math.abs(Math.log(stops[index] / value));
  let best = 1;
  for (let index = 2; index < stops.length; index += 1) {
    if (distance(index) < distance(best)) best = index;
  }
  return best;
}

/** What the slider writes into the box: en-US grouping, which parseAutoCompactTokens strips in any locale. */
export function formatAutoCompactDraft(tokens: number): string {
  return tokens.toLocaleString("en-US");
}

export function autoCompactSliderLabel(
  tokens: number,
  unset = "Default",
): string {
  if (tokens === 0) return unset;
  return `${tokens.toLocaleString()} tokens`;
}

/** Saves the per-model drafts of the listed models; blank ones fall back to the connection. */
export function parseAutoCompactTokensByModel(
  drafts: Record<string, string>,
  models: readonly string[],
): Record<string, number> {
  const limits: Record<string, number> = {};
  for (const model of models) {
    let value: number | null;
    try {
      value = parseAutoCompactTokens(drafts[model] ?? "");
    } catch (error) {
      throw new Error(`${model}: ${(error as Error).message}`);
    }
    if (value !== null) limits[model] = value;
  }
  return limits;
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
  compactionMode?: ProviderCompactionMode | null;
  // /compact: asked for by the user, so it applies with auto-compaction off too
  compactNow?: boolean;
}): {
  context_overflow?: "truncate_oldest";
  compaction_threshold?: number;
  compaction_uncapped?: true;
  context_window?: number;
  compact_context?: true;
  compaction_mode?: "rolling" | "checkpoint";
} {
  const { autoCompactEnabled, contextLength, compactNow } = options;
  const mode = options.compactionMode;
  const evict =
    mode === "rolling" || mode === "checkpoint" ? { compaction_mode: mode } : {};
  // Still sent with auto-compact off: the backend replays a /compact epoch only for "checkpoint".
  if (!autoCompactEnabled && !compactNow) return evict;
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
    ...evict,
  };
}
