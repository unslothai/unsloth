// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Local GGUF auto-compaction preferences. Default matches today's Studio behaviour. */

export const DEFAULT_AUTO_COMPACT_ENABLED = true;
export const DEFAULT_CONTEXT_POLICY = "checkpoint" as const;
export const DEFAULT_COMPACTION_HEADROOM_RATIO = 0.25;

export type LocalContextPolicy = "checkpoint" | "rolling";

export const COMPACTION_HEADROOM_CHOICES = [0.25, 0.1, 0.05, 0] as const;

export type CompactionStyleValue =
  | "checkpoint"
  | "rolling:0.25"
  | "rolling:0.1"
  | "rolling:0.05"
  | "rolling:0";

export function sanitizeContextPolicy(
  value: unknown,
): LocalContextPolicy | undefined {
  return value === "checkpoint" || value === "rolling" ? value : undefined;
}

export function sanitizeCompactionHeadroomRatio(
  value: unknown,
): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value)) return undefined;
  return Math.max(0, Math.min(0.9, Math.round(value * 100) / 100));
}

export function compactionStyleValue(
  policy: LocalContextPolicy,
  ratio: number,
): CompactionStyleValue {
  if (policy !== "rolling") return "checkpoint";
  const rounded = sanitizeCompactionHeadroomRatio(ratio) ?? DEFAULT_COMPACTION_HEADROOM_RATIO;
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
  context_overflow?: "truncate_oldest";
  context_policy?: LocalContextPolicy;
  compaction_headroom_ratio?: number;
} {
  if (!options.isGguf || !options.autoCompactEnabled) return {};
  if (options.contextPolicy === "rolling") {
    return {
      context_overflow: "truncate_oldest",
      context_policy: "rolling",
      compaction_headroom_ratio: options.compactionHeadroomRatio,
    };
  }
  // Checkpoint is the process default; omit context_policy so an older server
  // and today's payload stay identical.
  return { context_overflow: "truncate_oldest" };
}
