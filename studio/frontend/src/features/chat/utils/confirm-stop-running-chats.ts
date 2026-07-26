// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getActiveGenerations } from "../api/chat-api";
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
 * Every chat on the local model decodes on the same llama-server, so a reload
 * ends all of them: ask first, then tell the backend it may cancel them. The
 * local stop runs too, so the UI settles at once instead of waiting for each
 * socket to close. External-provider chats are not on that server and are left
 * out of both the question and the stop.
 */
export async function confirmStopRunningChatsIfNeeded(
  action = "Loading a different model",
): Promise<StopRunningChatsDecision> {
  // Local runs only: an external-provider chat streams from that provider, so
  // swapping the local model neither stops it nor needs its consent. Counting
  // it would block a safe load behind a dialog whose only honest answer is
  // "stop a chat that never had to stop" -- the backend excludes those runs
  // from active_generations for exactly this reason.
  const { runningByThreadId, localRunByThreadId } =
    useChatRuntimeStore.getState();
  let running = Object.entries(runningByThreadId)
    .filter(([threadId, on]) => on && localRunByThreadId[threadId])
    .map(([threadId]) => threadId);
  let count = running.length;

  if (count === 0) {
    // runningByThreadId is this tab's in-memory state: empty after a reload and
    // blind to a second tab. Without this the backend gate would 409 a swap the
    // user was never given the chance to approve, with no way to retry.
    try {
      const active = await getActiveGenerations();
      count = Math.max(active.count ?? 0, active.thread_ids?.length ?? 0);
      running = active.thread_ids ?? [];
    } catch {
      // Backend unreachable / older build: fall back to the local map only.
    }
  }

  if (count === 0) {
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
    .requestConfirm({ count, titles, action });

  if (!confirmed) {
    return { proceed: false, forceCancelActive: false };
  }

  // Same scope as the count above: local runs only.
  stopAllChatThreads();
  return { proceed: true, forceCancelActive: true };
}
