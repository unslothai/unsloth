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
 * Gate a model load / reload on the chats still generating. Every chat on the
 * local model shares one llama-server, so a reload ends all of them: ask first,
 * then let the backend cancel them once the load is past preflight.
 * External-provider chats are elsewhere and are left out of both.
 */
export async function confirmStopRunningChatsIfNeeded(
  action = "Loading a different model",
): Promise<StopRunningChatsDecision> {
  // Local runs only: an external-provider chat is not stopped by the swap and
  // needs no consent, so counting it would block a safe load behind a dialog.
  // The backend excludes those runs from active_generations for the same reason.
  const { runningByThreadId, localRunByThreadId } =
    useChatRuntimeStore.getState();
  let running = Object.entries(runningByThreadId)
    .filter(([threadId, on]) => on && localRunByThreadId[threadId])
    .map(([threadId]) => threadId);
  let count = running.length;

  // Always merge the backend snapshot: runningByThreadId is this tab's memory,
  // empty after a reload and blind to a second tab, while force_cancel_active
  // cancels every backend run. Reconciling first means the dialog names what
  // will actually stop, and the union stays local-only since external-provider
  // runs are never in active_generations.
  try {
    const active = await getActiveGenerations();
    const merged = new Set(running);
    for (const threadId of active.thread_ids ?? []) {
      merged.add(threadId);
    }
    running = [...merged];
    // A first turn started before its id was persisted is counted but cannot be
    // named, so never claim fewer chats than the backend reports.
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

  // Deliberately no local stop: the backend holds the cancel until the load
  // clears preflight, so stopping now would truncate every chat even when the
  // load is then rejected and the resident model left untouched.
  return { proceed: true, forceCancelActive: true };
}
