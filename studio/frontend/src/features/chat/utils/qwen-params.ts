// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "../stores/chat-runtime-store";

export type QwenThinkingParams = {
  temperature: number;
  topP: number;
  topK: number;
  minP: number;
  presencePenalty?: number;
};

/** Resolve the sampling table shared by model load and the Think toggle. */
export function resolveQwenThinkingParams(
  checkpoint: string,
  thinkingOn: boolean,
): QwenThinkingParams | null {
  const normalized = checkpoint.toLowerCase();
  if (!normalized.includes("qwen3")) {
    return null;
  }

  const needsPresencePenalty =
    normalized.includes("qwen3.5") ||
    normalized.includes("qwen3.6") ||
    normalized.includes("qwen3.8");
  const qwen38Thinking = thinkingOn && normalized.includes("qwen3.8");
  const base = thinkingOn
    ? {
        temperature: qwen38Thinking ? 1.0 : 0.6,
        topP: 0.95,
        topK: 20,
        minP: 0.0,
      }
    : { temperature: 0.7, topP: 0.8, topK: 20, minP: 0.0 };
  return needsPresencePenalty
    ? { ...base, presencePenalty: qwen38Thinking ? 0.0 : 1.5 }
    : base;
}

/**
 * Apply Qwen3-family recommended sampling parameters when the Think toggle
 * changes. Qwen3.8 uses its model-card thinking row; Qwen3.5, Qwen3.6 and the
 * non-thinking modes keep their existing presence-penalty recommendation.
 *
 * Used by both the thread assistant UI and the shared chat composer so the
 * two call sites stay in sync.
 */
export function applyQwenThinkingParams(thinkingOn: boolean): void {
  const store = useChatRuntimeStore.getState();
  const checkpoint = store.params.checkpoint?.toLowerCase() ?? "";
  const params = resolveQwenThinkingParams(checkpoint, thinkingOn);
  if (params === null || store.activePresetSource !== "builtin-default") {
    return;
  }
  // Deliberately unmarked, unlike the post-load path applying the same table: the
  // user asked for this mode here, so it must land even on a chat pinning sampling.
  store.setParams({ ...store.params, ...params });
}
