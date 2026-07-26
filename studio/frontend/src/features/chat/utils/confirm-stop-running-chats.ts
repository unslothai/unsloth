// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getActiveGenerations } from "../api/chat-api";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { useStopRunningChatsDialogStore } from "../stores/stop-running-chats-dialog-store";
import { listStoredChatThreads } from "./chat-history-storage";

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
 * ends all of them: ask first, then tell the backend it may cancel them, and
 * let it do so once the load is past preflight. External-provider chats are not
 * on that server and are left out of both the question and the cancel.
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

  // Merge the backend snapshot in every time, not only when this tab looks
  // idle: runningByThreadId is this tab's in-memory state, so it is empty after
  // a reload and blind to a second tab, and confirming here sends
  // force_cancel_active, which cancels every backend run rather than only the
  // ones listed. Reconciling first means the dialog names what will actually
  // stop. External-provider runs are never registered in active_generations, so
  // the union stays local-only.
  try {
    const active = await getActiveGenerations();
    const merged = new Set(running);
    for (const threadId of active.thread_ids ?? []) {
      merged.add(threadId);
    }
    running = [...merged];
    // A first turn that started before its thread id was persisted is counted
    // but cannot be named, so never claim fewer chats than the backend reports.
    count = Math.max(active.count ?? 0, running.length);
  } catch {
    // Backend unreachable / older build: fall back to the local map only.
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

  // Deliberately no local stop here. The backend holds the cancel until the
  // load clears preflight, so stopping now would truncate every chat even when
  // the load then fails identifier resolution, GPU validation or the training
  // guard and leaves the resident model untouched. force_cancel_active lets the
  // backend end them at its own point of no return instead.
  return { proceed: true, forceCancelActive: true };
}
