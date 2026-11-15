// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Coming back to a large cloud thread after the provider's prompt cache has lapsed means the
 * whole history is read again at full price. The composer then offers to summarize it first.
 */

/** Below this a re-read is cheap and a summary loses more than it saves. */
export const COMPACT_ON_RETURN_MIN_TOKENS = 50_000;

const MINUTE_MS = 60_000;

/**
 * How long a provider keeps a prompt prefix cached after the last request. Anthropic: 5 minutes,
 * or an hour with the connection's 1h cache setting. OpenAI keeps prefixes for up to an hour.
 * Everyone else is treated as Anthropic's default, the shortest of the three.
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
