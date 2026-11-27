// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  cancelChatGenerationRun,
  getActiveChatGenerationRuns,
} from "../api/chat-generation-api";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";

/** Registries die on reload but durable runs outlive the tab; ask the server, fire and forget. */
function stopServerRunsForThread(threadId: string): void {
  void getActiveChatGenerationRuns(threadId)
    .then((runs) => {
      for (const run of runs) {
        void cancelChatGenerationRun(run.id).catch(() => {});
      }
    })
    .catch(() => {
      // No durable-run endpoint or backend gone: nothing to do for a run with no handle.
    });
}

/** Returns true if a stop was sent; unresolved-id runs share "__default" and are all stopped. */
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
    }
  }
  if (!stopped) {
    stopServerRunsForThread(threadId);
    stopped = true;
  }
  return stopped;
}
