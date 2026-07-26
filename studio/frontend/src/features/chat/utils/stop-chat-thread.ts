// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "../stores/chat-runtime-store";

/**
 * Stop one conversation's generation, visible or not. Returns true if a stop
 * was dispatched.
 *
 * `cancelByThreadId` is assistant-ui's `cancelRun()`, registered only for the
 * thread on screen. `serverCancelByThreadId` is registered for every run and
 * POSTs that run's own `cancel_id`, so it is the only handle a background
 * conversation has. Both are per-run: neither can touch a sibling.
 */
export function stopChatThread(threadId: string | null | undefined): boolean {
  if (!threadId) return false;
  const { runningByThreadId, cancelByThreadId, serverCancelByThreadId } =
    useChatRuntimeStore.getState();
  if (!runningByThreadId[threadId]) return false;
  let stopped = false;
  try {
    const cancel = cancelByThreadId[threadId];
    if (cancel) {
      cancel();
      stopped = true;
    }
  } catch {
    // The run may have ended between the read above and this call.
  }
  try {
    const serverCancel = serverCancelByThreadId[threadId];
    if (serverCancel) {
      // Also after cancelRun(): a proxy that swallows the fetch abort would
      // leave the backend decoding.
      serverCancel();
      stopped = true;
    }
  } catch {
    // Same as above.
  }
  return stopped;
}
