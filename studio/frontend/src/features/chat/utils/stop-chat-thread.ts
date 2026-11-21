// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  cancelChatGenerationRun,
  getActiveChatGenerationRuns,
} from "../api/chat-generation-api";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";

/** Local registries are lost on reload while durable runs outlive the tab, so fall back
 *  to the server's active-run list. Fire and forget. */
function stopServerRunsForThread(threadId: string): void {
  void getActiveChatGenerationRuns(threadId)
    .then((runs) => {
      for (const run of runs) {
        void cancelChatGenerationRun(run.id).catch(() => {});
      }
    })
    .catch(() => {
      // No durable-run endpoint, or the backend is gone. Either way there is nothing further this
      // client can do about a run it holds no handle for.
    });
}

/** Stop one conversation's generation; true means a stop was sent. Runs with unresolved ids
 *  share the "__default" key, so every handle under it is stopped. */
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
  // Also after cancelRun(): a proxy may swallow the fetch abort and leave the backend decoding.
  for (const serverCancel of serverCancels) {
    try {
      serverCancel();
      stopped = true;
    } catch {
      // Same as above.
    }
  }
  if (!stopped) {
    stopServerRunsForThread(threadId);
    stopped = true;
  }
  return stopped;
}
