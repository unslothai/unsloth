// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getActiveGenerations } from "../api/chat-api";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { usePromptQueueUI } from "../stores/prompt-queue-ui-store";
import {
  type StopRunningChatsEffect,
  useStopRunningChatsDialogStore,
} from "../stores/stop-running-chats-dialog-store";
import { listStoredChatThreads } from "./chat-history-storage";
import { listLocalPreStreamRunReservations } from "./pre-stream-run-reservation";

export interface StopRunningChatsDecision {
  /** False when the user chose to keep generating; the caller must not load. */
  proceed: boolean;
  /** Pass as `force_cancel_active`. True only after an explicit confirmation, so the backend's 409
   *  still guards every other caller. */
  forceCancelActive: boolean;
  /** Local-model prompt queues to stop immediately before the model-changing request. */
  promptQueueThreadIds: string[];
  /** Accepted local sends that have not reached the running map yet. */
  preStreamRunTokens: symbol[];
}

export function getLocalPromptQueueThreadIds(): string[] {
  return [
    ...new Set(
      Object.entries(usePromptQueueUI.getState().byThreadId)
        .filter(([, entry]) => entry.local)
        .map(([threadId]) => threadId),
    ),
  ];
}

/** Gate a model load or reload on local chats still generating or queued: they share one
 *  llama-server, so a reload ends all of them. Ask first, then let the backend cancel active
 *  runs and return the pending queue targets for the caller to stop after preflight.
 *  External-provider chats and queues are left out of both. */
export async function confirmStopRunningChatsIfNeeded(
  action = "Loading a different model",
  effect: StopRunningChatsEffect = "reload",
): Promise<StopRunningChatsDecision> {
  // Local runs only: an external-provider chat is not stopped by the swap, so counting it would
  // block a safe load behind a dialog. The backend excludes them for the same reason.
  const { runningByThreadId, localRunByThreadId } =
    useChatRuntimeStore.getState();
  let running = Object.entries(runningByThreadId)
    .filter(([threadId, on]) => on && localRunByThreadId[threadId])
    .map(([threadId]) => threadId);
  const preStreamRuns = listLocalPreStreamRunReservations();
  const preStreamRunTokens = preStreamRuns.map(({ token }) => token);
  let unnamedPreStreamRuns = 0;
  const runningIds = new Set(running);
  for (const { threadIds } of preStreamRuns) {
    if (threadIds.some((id) => runningIds.has(id))) {
      continue;
    }
    if (threadIds.length > 0) {
      running.push(threadIds[0]);
      for (const threadId of threadIds) {
        runningIds.add(threadId);
      }
    } else {
      unnamedPreStreamRuns += 1;
    }
  }
  const promptQueueThreadIds = getLocalPromptQueueThreadIds();
  const promptQueuesByThreadId = usePromptQueueUI.getState().byThreadId;
  const aliasesByQueuedRun = new Map<string, string[]>();
  for (const threadId of promptQueueThreadIds) {
    const entry = promptQueuesByThreadId[threadId];
    if (!entry) {
      continue;
    }
    const aliases = aliasesByQueuedRun.get(entry.runId) ?? [];
    aliases.push(threadId);
    aliasesByQueuedRun.set(entry.runId, aliases);
  }
  for (const aliases of aliasesByQueuedRun.values()) {
    if (aliases.some((threadId) => runningIds.has(threadId))) {
      continue;
    }
    const threadId = aliases[0];
    running.push(threadId);
    runningIds.add(threadId);
  }
  running = [...new Set(running)];
  let count = running.length + unnamedPreStreamRuns;
  let hasNonChat = false;

  // Always merge the backend snapshot: runningByThreadId is this tab's memory, empty after a
  // reload and blind to a second tab, while force_cancel_active cancels every backend run. The
  // union stays local-only, since external-provider runs are never in it.
  try {
    const active = await getActiveGenerations();
    const entries = active.active ?? [];
    const merged = new Set(running);
    for (const threadId of active.thread_ids ?? []) {
      merged.add(threadId);
    }
    running = [...merged];
    // Count conversations, not handles: one chat holds several at once while a tool continuation
    // registers its next leg before the previous unwinds, and active.count counts those
    // separately. A first turn started before its id was persisted has no id to merge, so add
    // those back or the prompt names fewer chats than will stop.
    const unnamed = entries.filter((entry) => !entry.thread_id).length;
    count = entries.length
      ? running.length + unnamed + unnamedPreStreamRuns
      : Math.max(active.count ?? 0, running.length) + unnamedPreStreamRuns;
    // Embeddings, completions and audio share the model but are not conversations, so the prompt
    // must not offer to stop chats that do not exist.
    hasNonChat = entries.some((entry) => (entry.kind ?? "chat") !== "chat");
  } catch {
    // Backend unreachable / older build: fall back to the local map only.
  }

  if (count === 0) {
    return {
      proceed: true,
      forceCancelActive: false,
      promptQueueThreadIds: [],
      preStreamRunTokens: [],
    };
  }

  let titles: string[] = [];
  try {
    const threads = await listStoredChatThreads();
    const byId = new Map(threads.map((t) => [t.id, t]));
    // A compare conversation runs two pane threads, and the sidebar and the route both treat it as
    // one chat. Counting the raw ids asked to stop two and listed its title twice. Fold panes onto
    // their pairId, keeping the backend's count when it is higher.
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
    .requestConfirm({ count, titles, action, hasNonChat, effect });

  if (!confirmed) {
    return {
      proceed: false,
      forceCancelActive: false,
      promptQueueThreadIds: [],
      preStreamRunTokens: [],
    };
  }

  // Deliberately no local stop: the backend holds the cancel until the load clears preflight, so
  // stopping now would truncate every chat even for a rejected load.
  return {
    proceed: true,
    forceCancelActive: true,
    promptQueueThreadIds,
    preStreamRunTokens,
  };
}
