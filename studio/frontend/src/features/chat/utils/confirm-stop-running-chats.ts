// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getActiveGenerations } from "../api/chat-api";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { useStopRunningChatsDialogStore } from "../stores/stop-running-chats-dialog-store";
import { listStoredChatThreads } from "./chat-history-storage";

export interface StopRunningChatsDecision {
  /** False when the user chose to keep generating; the caller must not load. */
  proceed: boolean;
  /** Pass as `force_cancel_active`. True only after an explicit confirmation, so the
   * backend's 409 still guards every other caller. */
  forceCancelActive: boolean;
}

/**
 * Gate a model load / reload on the chats still generating: they share one llama-server,
 * so a reload ends all of them. Ask first, then let the backend cancel them once the load
 * is past preflight. External-provider chats are left out of both.
 */
export async function confirmStopRunningChatsIfNeeded(
  action = "Loading a different model",
): Promise<StopRunningChatsDecision> {
  // Local runs only: an external-provider chat is not stopped by the swap, so counting it
  // would block a safe load behind a dialog. The backend excludes them for the same reason.
  const { runningByThreadId, localRunByThreadId } =
    useChatRuntimeStore.getState();
  let running = Object.entries(runningByThreadId)
    .filter(([threadId, on]) => on && localRunByThreadId[threadId])
    .map(([threadId]) => threadId);
  let count = running.length;
  let hasNonChat = false;

  // Always merge the backend snapshot: runningByThreadId is this tab's memory, empty after
  // a reload and blind to a second tab, while force_cancel_active cancels every backend
  // run. The union stays local-only, since external-provider runs are never in it.
  try {
    const active = await getActiveGenerations();
    const merged = new Set(running);
    for (const threadId of active.thread_ids ?? []) {
      merged.add(threadId);
    }
    running = [...merged];
    // A first turn started before its id was persisted is counted but not named, so
    // never claim fewer chats than the backend reports.
    count = Math.max(active.count ?? 0, running.length);
    // Embeddings / completions / audio share the model but are not conversations, so the
    // prompt must not offer to stop chats that do not exist.
    hasNonChat = (active.active ?? []).some(
      (entry) => (entry.kind ?? "chat") !== "chat",
    );
  } catch {
    // Backend unreachable / older build: fall back to the local map only.
  }

  if (count === 0) {
    return { proceed: true, forceCancelActive: false };
  }

  let titles: string[] = [];
  try {
    const threads = await listStoredChatThreads();
    const byId = new Map(threads.map((t) => [t.id, t]));
    // A compare conversation runs two pane threads, and the sidebar and the route both
    // treat it as one chat. Counting the raw ids asked to stop two and listed its title
    // twice. Fold panes onto their pairId, keeping the count the backend reported when
    // it is higher (a first turn it can see but cannot name).
    const seen = new Set<string>();
    for (const id of running) {
      const thread = byId.get(id);
      const key = thread?.pairId ?? id;
      if (seen.has(key)) continue;
      seen.add(key);
      titles.push(thread?.title || "Untitled chat");
    }
    count = Math.max(seen.size, count - (running.length - seen.size));
  } catch {
    // Titles are decoration; the count alone is enough to make the choice.
    titles = [];
  }

  const confirmed = await useStopRunningChatsDialogStore
    .getState()
    .requestConfirm({ count, titles, action, hasNonChat });

  if (!confirmed) {
    return { proceed: false, forceCancelActive: false };
  }

  // Deliberately no local stop: the backend holds the cancel until the load clears
  // preflight, so stopping now would truncate every chat even for a rejected load.
  return { proceed: true, forceCancelActive: true };
}
