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

/** The self-note request fields, or {} when the request has nothing to say about them.
 *
 *  Omitted rather than defaulted when the setting is undefined, so the server`s
 *  UNSLOTH_SELF_NOTE stays in force and an install that never touched the toggle keeps
 *  exactly the behaviour it had. The reserve rides along only when the feature is ON:
 *  sending a budget for a disabled feature is noise.
 */
function selfNoteRequestFields(options: {
  selfNoteEnabled?: boolean;
  selfNoteReserveTokens?: number;
}): {
  self_note_enabled?: boolean;
  self_note_reserve_tokens?: number;
} {
  const enabled = sanitizeSelfNoteEnabled(options.selfNoteEnabled);
  if (enabled === undefined) return {};
  if (!enabled) return { self_note_enabled: false };
  const reserve = sanitizeSelfNoteReserveTokens(options.selfNoteReserveTokens);
  return {
    self_note_enabled: true,
    ...(reserve === undefined ? {} : { self_note_reserve_tokens: reserve }),
  };
}

export function ggufCompactionRequestFields(options: {
  isGguf: boolean;
  autoCompactEnabled: boolean;
  contextPolicy: LocalContextPolicy;
  compactionHeadroomRatio: number;
  selfNoteEnabled?: boolean;
  selfNoteReserveTokens?: number;
}): {
  context_overflow?: "error" | "truncate_oldest";
  context_policy?: Exclude<LocalContextPolicy, "inherit">;
  compaction_headroom_ratio?: number;
  self_note_enabled?: boolean;
  self_note_reserve_tokens?: number;
} {
  if (!options.isGguf) return {};
  // The note is carried by the compaction block, so it only means anything once
  // compaction can fire at all. Computed up front so every return below carries it.
  const selfNote = selfNoteRequestFields(options);
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
      ...selfNote,
    };
  }
  if (options.contextPolicy === "inherit") {
    return { context_overflow: "truncate_oldest", ...selfNote };
  }
  return {
    context_overflow: "truncate_oldest",
    context_policy: "checkpoint",
    ...selfNote,
  };
}

/** The model's self-note across a compaction. Off by default, like the server. */
export const DEFAULT_SELF_NOTE_ENABLED = false;
export const DEFAULT_SELF_NOTE_RESERVE_TOKENS = 256;
/** Mirrors the backend's ge/le, which 400s the whole save on one bad field. */
export const SELF_NOTE_RESERVE_MIN = 64;
export const SELF_NOTE_RESERVE_MAX = 4096;

export function sanitizeSelfNoteEnabled(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

export function sanitizeSelfNoteReserveTokens(
  value: unknown,
): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value)) return undefined;
  return Math.max(
    SELF_NOTE_RESERVE_MIN,
    Math.min(SELF_NOTE_RESERVE_MAX, Math.round(value)),
  );
}
