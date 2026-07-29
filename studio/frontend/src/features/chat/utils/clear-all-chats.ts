// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { clearStoredChats, countStoredChats } from "./chat-history-storage";
import {
  getPreStreamRunThreadIds,
  requestPromptQueueStop,
} from "./prompt-queue-boundary";

export const countAllChats = countStoredChats;

export async function clearAllChats() {
  const { runningByThreadId, cancelByThreadId } =
    useChatRuntimeStore.getState();
  const activeThreadIds = new Set([
    ...Object.keys(runningByThreadId),
    ...getPreStreamRunThreadIds(),
    ...Object.keys(cancelByThreadId),
  ]);
  // Stop schedulers before cancelling active runs: their running-to-idle
  // transition is otherwise interpreted as permission to dispatch the next
  // queued prompt while storage is being cleared.
  requestPromptQueueStop();
  for (const threadId of activeThreadIds) {
    cancelByThreadId[threadId]?.();
  }
  return clearStoredChats();
}
