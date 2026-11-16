// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Type-only imports: the node tests load this file directly.

import type { ChatCountTokensResponse } from "../api/chat-api";

const CONTEXT_COMMAND = "/context";

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
  percent: number;
  segments: ContextSegment[];
  free: number;
  compactAt: { tokens: number; share: number } | null;
  toolNames: string[];
  prompt: string | null;
}

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
    percent: Math.min(100, Math.round((used / window) * 100)),
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
