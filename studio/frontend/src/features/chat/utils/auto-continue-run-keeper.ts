// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Wires the automatic-continuation lease keeper to the runs actually in flight.
 *
 * The keeper itself is in `continuation.ts` and knows nothing about this app; all it asks
 * for is "is thread X generating right now", which `runningByThreadId` answers per thread
 * and regardless of what is on screen. The same store field is what the prompt queue uses
 * for completion detection, for the same reason: it "tracks the actual thread (not
 * aui.thread()), so detection survives navigation".
 */

import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import {
  AUTO_CONTINUE_LEASE_RENEW_MS,
  createAutoContinueLeaseKeeper,
} from "./continuation";

const keeper = createAutoContinueLeaseKeeper({
  signal: {
    isRunning: (threadId) =>
      Boolean(useChatRuntimeStore.getState().runningByThreadId[threadId]),
    subscribe: (onChange) => useChatRuntimeStore.subscribe(onChange),
  },
});

let timer: ReturnType<typeof setInterval> | null = null;

function tick(): void {
  keeper.tick();
  if (keeper.held() === 0 && timer !== null) {
    clearInterval(timer);
    timer = null;
  }
}

/**
 * Hold `messageId`'s lease for as long as `threadId`'s own run is generating.
 *
 * Called by the continuation bar at the moment it starts a run, because that is the last
 * point at which anything knows both ids: the bar unmounts as soon as the continuation's
 * sibling becomes the selected branch, and the thread it ran on may not be the one on
 * screen a second later.
 *
 * `threadId` is the key the run files itself under, which is `threadListItem.remoteId` --
 * the same value assistant-ui passes the adapter as `unstable_threadId`. A thread with no
 * remote id yet (the first turn of a New Chat) files its run under a shared placeholder
 * that is not safe to watch, so nothing is held for it and its lease simply runs out its
 * TTL. That is the same outcome as a tab closing mid-run, and it is never an early release.
 */
export function holdAutoContinueRun(
  messageId: string,
  threadId: string | undefined,
): void {
  if (!threadId) {
    return;
  }
  keeper.hold(messageId, threadId);
  timer ??= setInterval(tick, AUTO_CONTINUE_LEASE_RENEW_MS);
}
