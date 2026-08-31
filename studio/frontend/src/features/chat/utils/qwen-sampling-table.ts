// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The single Qwen3 sampling table, kept free of store imports.
 *
 * qwen-params.ts imports the runtime store, and the store imports the defaults
 * migration, so the migration cannot read the table from there without a cycle.
 * Keeping the pure resolver here lets model load, the Think toggle and the
 * upgrade migration all agree by construction: change a recommended value once
 * and every consumer follows, including rows already written to disk.
 */

export type QwenThinkingParams = {
  temperature: number;
  topP: number;
  topK: number;
  minP: number;
  presencePenalty?: number;
};

// Anchored at identifier boundaries rather than a bare substring: "Qwen3.80"
// and "Qwen3.8B" are a future family and a parameter count, not Qwen3.8, and a
// substring test would hand both the presence bump. Real ids keep matching
// because a family segment always ends the string or runs into -, _, / or \.
const PRESENCE_BUMP_QWEN = /(?:^|[^a-z0-9])qwen3\.(?:5|6|8)(?:$|[-_/\\])/;

/** Resolve the sampling table shared by model load and the Think toggle. */
export function resolveQwenThinkingParams(
  checkpoint: string,
  thinkingOn: boolean,
): QwenThinkingParams | null {
  const normalized = checkpoint.toLowerCase();
  if (!normalized.includes("qwen3")) {
    return null;
  }

  const needsPresencePenalty = PRESENCE_BUMP_QWEN.test(normalized);
  const base = thinkingOn
    ? { temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0 }
    : { temperature: 0.7, topP: 0.8, topK: 20, minP: 0.0 };
  return needsPresencePenalty ? { ...base, presencePenalty: 1.5 } : base;
}
