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
  proceed: boolean;
  /** True only after explicit confirmation, so the backend 409 still guards other callers. */
  forceCancelActive: boolean;
  promptQueueThreadIds: string[];
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

/** Local chats share one llama-server, so a reload ends all of them: ask first.
 *  External-provider chats and queues are excluded. */
export async function confirmStopRunningChatsIfNeeded(
  action = "Loading a different model",
  effect: StopRunningChatsEffect = "reload",
  /** Filter by model on the backend: a tab cannot tell which model a local run is on. */
  model?: string,
): Promise<StopRunningChatsDecision> {
  // External-provider chats are not stopped by the swap, so they are not counted.
  const { runningByThreadId, localRunByThreadId } =
    useChatRuntimeStore.getState();
  let running = model
    ? []
    : Object.entries(runningByThreadId)
        .filter(([threadId, on]) => on && localRunByThreadId[threadId])
        .map(([threadId]) => threadId);
  const preStreamRuns = model ? [] : listLocalPreStreamRunReservations();
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
  let promptQueueThreadIds = model ? [] : getLocalPromptQueueThreadIds();
  const promptQueuesByThreadId = usePromptQueueUI.getState().byThreadId;
  const aliasesByQueuedRun = new Map<string, string[]>();
  for (const threadId of promptQueueThreadIds) {
    const entry = promptQueuesByThreadId[threadId];
    if (!entry || entry.paused) {
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

  // Always merge the backend snapshot: the local map is empty after reload and blind to other tabs.
  try {
    const active = await getActiveGenerations(model);
    const entries = active.active ?? [];
    const merged = new Set(running);
    for (const threadId of active.thread_ids ?? []) {
      merged.add(threadId);
    }
    running = [...merged];
    if (model) {
      promptQueueThreadIds = running;
    }
    // Count conversations, not handles; add back unnamed first turns that have no id yet.
    const unnamed = entries.filter((entry) => !entry.thread_id).length;
    count = entries.length
      ? running.length + unnamed + unnamedPreStreamRuns
      : Math.max(active.count ?? 0, running.length) + unnamedPreStreamRuns;
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
    // A compare conversation runs two pane threads; fold them onto their pairId.
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

  // No local stop: the backend holds the cancel until the load passes preflight.
  return {
    proceed: true,
    forceCancelActive: true,
    promptQueueThreadIds,
    preStreamRunTokens,
  };
}
