// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "../stores/chat-runtime-store";
// The table lives in qwen-sampling-table.ts to avoid a resolver -> store -> migration cycle.
import { resolveQwenThinkingParams } from "./qwen-sampling-table";

export function applyQwenThinkingParams(thinkingOn: boolean): void {
  const store = useChatRuntimeStore.getState();
  const checkpoint = store.params.checkpoint?.toLowerCase() ?? "";
  const params = resolveQwenThinkingParams(checkpoint, thinkingOn);
  if (params === null || store.activePresetSource !== "builtin-default") {
    return;
  }
  // Unmarked on purpose: the user asked for this mode, so it applies even when sampling is pinned.
  store.setParams({ ...store.params, ...params });
}
