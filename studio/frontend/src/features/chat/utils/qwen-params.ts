// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "../stores/chat-runtime-store";

/**
 * Apply Qwen3-family recommended sampling parameters when the Think toggle
 * changes. Qwen3.5 and Qwen3.6 also need a presence_penalty bump on top of
 * the Qwen3 defaults.
 *
 * Used by both the thread assistant UI and the shared chat composer so the
 * two call sites stay in sync.
 */
export function applyQwenThinkingParams(thinkingOn: boolean): void {
  const store = useChatRuntimeStore.getState();
  const checkpoint = store.params.checkpoint?.toLowerCase() ?? "";
  if (!checkpoint.includes("qwen3")) {
    return;
  }
  const needsPresencePenalty =
    checkpoint.includes("qwen3.5") || checkpoint.includes("qwen3.6");
  const base = thinkingOn
    ? { temperature: 0.6, topP: 0.95, topK: 20, minP: 0.0 }
    : { temperature: 0.7, topP: 0.8, topK: 20, minP: 0.0 };
  const params = needsPresencePenalty
    ? { ...base, presencePenalty: 1.5 }
    : base;
  store.setParams({ ...store.params, ...params });
}
