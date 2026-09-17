// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Local GGUF auto-compaction preferences. Default follows the server policy. */

export const DEFAULT_AUTO_COMPACT_ENABLED = true;
export const DEFAULT_CONTEXT_POLICY = "inherit" as const;
export const DEFAULT_COMPACTION_HEADROOM_RATIO = 0.25;

export type LocalContextPolicy = "inherit" | "checkpoint" | "rolling";

export const COMPACTION_HEADROOM_CHOICES = [0.25, 0.1, 0.05, 0] as const;

export type CompactionStyleValue =
  | "inherit"
  | "checkpoint"
  | "rolling:0.25"
  | "rolling:0.1"
  | "rolling:0.05"
  | "rolling:0";

export function sanitizeContextPolicy(
  value: unknown,
): LocalContextPolicy | undefined {
  return value === "inherit" || value === "checkpoint" || value === "rolling"
    ? value
    : undefined;
}

export function sanitizeCompactionHeadroomRatio(
  value: unknown,
): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value)) return undefined;
  const clamped = Math.max(0, Math.min(0.9, Math.round(value * 100) / 100));
  let nearest: (typeof COMPACTION_HEADROOM_CHOICES)[number] =
    COMPACTION_HEADROOM_CHOICES[0];
  let best = Math.abs(clamped - nearest);
  for (const choice of COMPACTION_HEADROOM_CHOICES) {
    const distance = Math.abs(clamped - choice);
    if (distance < best) {
      nearest = choice;
      best = distance;
    }
  }
  return nearest;
}

export function compactionStyleValue(
  policy: LocalContextPolicy,
  ratio: number,
): CompactionStyleValue {
  if (policy === "inherit") return "inherit";
  if (policy !== "rolling") return "checkpoint";
  const rounded =
    sanitizeCompactionHeadroomRatio(ratio) ?? DEFAULT_COMPACTION_HEADROOM_RATIO;
  if (rounded === 0) return "rolling:0";
  if (rounded === 0.05) return "rolling:0.05";
  if (rounded === 0.1) return "rolling:0.1";
  return "rolling:0.25";
}

export function parseCompactionStyle(value: string): {
  contextPolicy: LocalContextPolicy;
  compactionHeadroomRatio: number;
} {
  switch (value) {
    case "inherit":
      return {
        contextPolicy: "inherit",
        compactionHeadroomRatio: DEFAULT_COMPACTION_HEADROOM_RATIO,
      };
    case "rolling:0":
      return { contextPolicy: "rolling", compactionHeadroomRatio: 0 };
    case "rolling:0.05":
      return { contextPolicy: "rolling", compactionHeadroomRatio: 0.05 };
    case "rolling:0.1":
      return { contextPolicy: "rolling", compactionHeadroomRatio: 0.1 };
    case "rolling:0.25":
      return { contextPolicy: "rolling", compactionHeadroomRatio: 0.25 };
    default:
      return {
        contextPolicy: "checkpoint",
        compactionHeadroomRatio: DEFAULT_COMPACTION_HEADROOM_RATIO,
      };
  }
}

export function ggufCompactionRequestFields(options: {
  isGguf: boolean;
  autoCompactEnabled: boolean;
  contextPolicy: LocalContextPolicy;
  compactionHeadroomRatio: number;
}): {
  context_overflow?: "error" | "truncate_oldest";
  context_policy?: Exclude<LocalContextPolicy, "inherit">;
  compaction_headroom_ratio?: number;
} {
  if (!options.isGguf) return {};
  if (!options.autoCompactEnabled) {
    // An omitted field falls back to UNSLOTH_CONTEXT_OVERFLOW, which may still compact. "error" is an
    // explicit refusal of that fallback.
    return { context_overflow: "error" };
  }
  if (options.contextPolicy === "rolling") {
    return {
      context_overflow: "truncate_oldest",
      context_policy: "rolling",
      compaction_headroom_ratio:
        sanitizeCompactionHeadroomRatio(options.compactionHeadroomRatio) ??
        DEFAULT_COMPACTION_HEADROOM_RATIO,
    };
  }
  if (options.contextPolicy === "inherit") {
    return { context_overflow: "truncate_oldest" };
  }
  return {
    context_overflow: "truncate_oldest",
    context_policy: "checkpoint",
  };
}
