// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Local GGUF and MLX auto-compaction. On or off; how it compacts is the server's call.
 *
 *  Studio used to offer the policy as a setting, but the choice needed the reader to know what a
 *  checkpoint epoch and a rolling window were before it meant anything, and both sides of it were
 *  already the server's to configure. It now always follows the server, which is what the setting
 *  shipped as anyway (UNSLOTH_CONTEXT_POLICY, default "checkpoint"). */

export const DEFAULT_AUTO_COMPACT_ENABLED = true;

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
  isMlx?: boolean;
  autoCompactEnabled: boolean;
  selfNoteEnabled?: boolean;
  selfNoteReserveTokens?: number;
}): {
  context_overflow?: "error" | "truncate_oldest";
  self_note_enabled?: boolean;
  self_note_reserve_tokens?: number;
} {
  if (!options.isGguf && !options.isMlx) return {};
  if (!options.autoCompactEnabled) {
    // explicit error avoids the UNSLOTH_CONTEXT_OVERFLOW fallback, which may compact
    return { context_overflow: "error" };
  }
  // omit context_policy so the server applies UNSLOTH_CONTEXT_POLICY
  return { context_overflow: "truncate_oldest", ...selfNoteRequestFields(options) };
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

export const DEFAULT_SELF_NOTE_ENABLED = false;
export const DEFAULT_SELF_NOTE_RESERVE_TOKENS = 256;
export const SELF_NOTE_RESERVE_MIN = 64;
export const SELF_NOTE_RESERVE_MAX = 4096;
export const SELF_NOTE_RESERVE_CHOICES = [128, 256, 512, 1024] as const;

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
