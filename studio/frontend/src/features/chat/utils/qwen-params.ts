// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "../stores/chat-runtime-store";
// The table itself lives in qwen-sampling-table.ts, free of store imports, so
// the defaults migration can read it without a resolver -> store -> migration
// cycle. Import it from there directly rather than through this module.
import { resolveQwenThinkingParams } from "./qwen-sampling-table";

/**
 * Apply Qwen3-family recommended sampling when the Think toggle changes.
 * Shared by the thread assistant UI and the chat composer so both stay in sync.
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
