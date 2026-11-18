// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * After the prompt cache lapses, a large cloud thread is re-read at full price, so offer a summary.
 */

/** Below this a re-read is cheap and a summary loses more than it saves. */
export const COMPACT_ON_RETURN_MIN_TOKENS = 50_000;

const MINUTE_MS = 60_000;

/**
 * Anthropic keeps a prefix 5 minutes, or 1h with the 1h cache setting; OpenAI up to an hour; any
 * other provider is treated as the Anthropic default, the shortest of the three
 */
export function promptCacheLifetimeMs(options: {
  providerType: string | null | undefined;
  promptCacheTtl?: "5m" | "1h";
}): number {
  if (options.promptCacheTtl === "1h") return 60 * MINUTE_MS;
  if (options.providerType === "openai") return 60 * MINUTE_MS;
  return 5 * MINUTE_MS;
}

export function shouldOfferCompactOnReturn(options: {
  isCloudModel: boolean;
  /** Tokens the thread held after its last reply. */
  threadTokens: number | null | undefined;
  /** When the last reply was written, in epoch milliseconds. */
  lastReplyAtMs: number | null | undefined;
  nowMs: number;
  cacheLifetimeMs: number;
  /** A temporary chat keeps no summary, and a running reply has a live cache. */
  blocked?: boolean;
}): boolean {
  if (!options.isCloudModel || options.blocked) return false;
  const tokens = options.threadTokens;
  if (typeof tokens !== "number" || !(tokens >= COMPACT_ON_RETURN_MIN_TOKENS)) return false;
  const last = options.lastReplyAtMs;
  if (typeof last !== "number" || !Number.isFinite(last)) return false;
  return options.nowMs - last >= options.cacheLifetimeMs;
}
