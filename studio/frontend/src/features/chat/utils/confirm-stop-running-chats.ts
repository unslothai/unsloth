// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { useStopRunningChatsDialogStore } from "../stores/stop-running-chats-dialog-store";
import { listStoredChatThreads } from "./chat-history-storage";
import { stopAllChatThreads } from "./stop-chat-thread";

export interface StopRunningChatsDecision {
  /** False when the user chose to keep generating; the caller must not load. */
  proceed: boolean;
  /**
   * Pass as `force_cancel_active` on the load/unload request. True only after an
   * explicit confirmation, so the backend's 409 still guards every other caller.
   */
  forceCancelActive: boolean;
}

/**
 * Gate a model load / reload on the chats still generating.
 *
 * They all decode on the same llama-server, so a reload ends every one: ask
 * first, then tell the backend it may cancel them. The local stop runs too, so
 * the UI settles at once instead of waiting for each socket to close.
 */
export async function confirmStopRunningChatsIfNeeded(
  action = "Loading a different model",
): Promise<StopRunningChatsDecision> {
  const running = Object.entries(
    useChatRuntimeStore.getState().runningByThreadId,
  )
    .filter(([, on]) => on)
    .map(([threadId]) => threadId);

  if (running.length === 0) {
    return { proceed: true, forceCancelActive: false };
  }

  let titles: string[] = [];
  try {
    const threads = await listStoredChatThreads();
    const byId = new Map(threads.map((t) => [t.id, t.title]));
    titles = running.map((id) => byId.get(id) || "Untitled chat");
  } catch {
    // Titles are decoration; the count alone is enough to make the choice.
    titles = [];
  }

  const confirmed = await useStopRunningChatsDialogStore
    .getState()
    .requestConfirm({ count: running.length, titles, action });

  if (!confirmed) {
    return { proceed: false, forceCancelActive: false };
  }

  stopAllChatThreads();
  return { proceed: true, forceCancelActive: true };
}
