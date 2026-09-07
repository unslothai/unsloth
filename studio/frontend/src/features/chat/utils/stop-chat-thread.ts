// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  cancelChatGenerationRun,
  getActiveChatGenerationRuns,
} from "../api/chat-generation-api";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";

/**
 * Ask the server to stop whatever it still has running for this thread.
 *
 * The registries below are module-scoped React state, so a page reload empties them, while
 * a durable run is server-owned and outlives the tab that started it: the UI says a reply
 * is generating and every stop path says there is nothing to stop. The server's active-run
 * list still knows, so a thread with no local handle falls back to it. Fire and forget;
 * a thread with no run just reads an empty list.
 */
function stopServerRunsForThread(threadId: string): void {
  void getActiveChatGenerationRuns(threadId)
    .then((runs) => {
      for (const run of runs) {
        void cancelChatGenerationRun(run.id).catch(() => {});
      }
    })
    .catch(() => {
      // No durable-run endpoint, or the backend is gone. Either way there is nothing
      // further this client can do about a run it holds no handle for.
    });
}

/**
 * Stop one conversation's generation, visible or not. Returns true if a stop was dispatched.
 *
 * `cancelByThreadId` is assistant-ui's `cancelRun()`, registered only for the thread on screen;
 * `serverCancelByThreadId` is registered for every run and POSTs that run's own `cancel_id`, so
 * it is the only handle a background conversation has. Both are per-run. Runs with an unresolved
 * thread id share the "__default" key, so stop every handle filed under it. With none of the
 * three, the server is asked directly rather than the call reporting nothing to do, so the
 * return means "a stop was sent", not "a local handle existed".
 */
export function stopChatThread(threadId: string | null | undefined): boolean {
  if (!threadId) return false;
  const { runningByThreadId, cancelByThreadId, serverCancelByThreadId } =
    useChatRuntimeStore.getState();
  const cancel = cancelByThreadId[threadId];
  const serverCancels = serverCancelByThreadId[threadId] ?? [];
  if (!runningByThreadId[threadId] && !cancel && serverCancels.length === 0) {
    stopServerRunsForThread(threadId);
    return true;
  }
  let stopped = false;
  try {
    if (cancel) {
      cancel();
      stopped = true;
    }
  } catch {
    // The run may have ended between the read above and this call.
  }
  // Also after cancelRun(): a proxy that swallows the fetch abort leaves the backend decoding.
  for (const serverCancel of serverCancels) {
    try {
      serverCancel();
      stopped = true;
    } catch {
      // Same as above.
    }
  }
  // A thread flagged running with no handle at all is the reload case again, one run later.
  if (!stopped) {
    stopServerRunsForThread(threadId);
    stopped = true;
  }
  return stopped;
}
