// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Pure helpers for /context: type-only imports, so the node tests load this file directly. */

import type { ChatCountTokensResponse } from "../api/chat-api";

const CONTEXT_COMMAND = "/context";

/** Exactly `/context`, ignoring surrounding whitespace and case. */
export function isContextCommand(text: string): boolean {
  return text.trim().toLowerCase() === CONTEXT_COMMAND;
}

type ContextSegmentKey = "system" | "tools" | "messages";

interface ContextSegment {
  key: ContextSegmentKey;
  tokens: number;
}

export interface ContextView {
  used: number;
  window: number;
  /** Of the window, capped at 100. */
  percent: number;
  segments: ContextSegment[];
  free: number;
  /** Prompt size at which the server compacts, and its share of the window; null when it would not. */
  compactAt: { tokens: number; share: number } | null;
  toolNames: string[];
  /** The rendered prompt, where the backend can render one. */
  prompt: string | null;
}

/** Share of the window in use, as a whole percent capped at 100. */
export function usagePercent(used: number, window: number): number {
  return Math.min(100, Math.round((used / window) * 100));
}

/** A breakdown count as /context lists it, in render order. */
export function deriveContextView(
  count: ChatCountTokensResponse,
  fallbackWindow: number | null,
): ContextView | null {
  const window = count.context_length || fallbackWindow;
  const parts = count.breakdown;
  if (!window || !parts) return null;
  const used = count.input_tokens;
  const budget = count.compaction_budget;
  return {
    used,
    window,
    percent: usagePercent(used, window),
    segments: (["system", "tools", "messages"] as const).map((key) => ({
      key,
      tokens: parts[key],
    })),
    free: Math.max(0, window - used),
    compactAt: budget
      ? { tokens: budget, share: Math.min(1, budget / window) }
      : null,
    toolNames: parts.tool_names,
    prompt: count.prompt ?? null,
  };
}
