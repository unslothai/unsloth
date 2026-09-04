// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Wires the automatic-continuation lease keeper to the runs actually in flight. The keeper itself
 *  is in `continuation.ts` and knows nothing about this app; all it asks is "is thread X
 *  generating right now", which `runningByThreadId` answers per thread and regardless of what is
 *  on screen. The prompt queue uses the same field for completion detection, so detection
 *  survives navigation. */

import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import {
  AUTO_CONTINUE_LEASE_RENEW_MS,
  createAutoContinueLeaseKeeper,
} from "./continuation";
import { isImageGateRunOnly } from "./image-input-support";
import {
  PROMPT_QUEUE_RUN_FAILED_EVENT,
  type PromptQueueRunFailedEventDetail,
} from "./prompt-queue-boundary";

const keeper = createAutoContinueLeaseKeeper({
  signal: {
    /** Generating, and not the image gate settling its waiters. A continuation carrying an image the
     *  loaded model cannot read is refused before any request is made, and the gate pulses this
     *  field true and then false so compare mode's `waitForRunEnd` resolves. To a hold waiting for
     *  its own run to start, that pair reads as the run starting and finishing: it arms on the
     *  first and is released on the second, with the `done` marker that tells every other tab for a
     *  day that the message HAS been continued, which the failure that follows cannot undo. So the
     *  pulse is not counted as a run. Only when the gate is ALL that holds the flag: a real run
     *  sharing the key still answers yes. */
    isRunning: (threadId) => {
      const state = useChatRuntimeStore.getState();
      return (
        Boolean(state.runningByThreadId[threadId]) &&
        !isImageGateRunOnly(state.runOwnerByThreadId[threadId])
      );
    },
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

/** A run that failed before it ever reached `runningByThreadId`. The adapter wrapper catches
 *  everything `adapter.run` throws and announces it per thread, which is the only signal
 *  separating a preflight that FAILED from one that is merely long. Nothing here reads a clock:
 *  a slow run emits no event and keeps its hold and its renewals. */
function onRunFailed(event: Event): void {
  const threadId = (event as CustomEvent<PromptQueueRunFailedEventDetail>)
    .detail?.threadId;
  if (!threadId) {
    return;
  }
  keeper.failed(threadId);
  if (keeper.held() === 0 && timer !== null) {
    clearInterval(timer);
    timer = null;
  }
}

let listening = false;

/** Hold `messageId`'s lease for as long as `threadId`'s own run is generating. Called by the
 *  continuation bar at the moment it starts a run, the last point at which anything knows both
 *  ids: the bar unmounts as soon as the continuation's sibling becomes selected. `threadId` is
 *  the key the run files itself under, `threadListItem.remoteId`. A thread with no remote id yet
 *  files under a shared placeholder that is not safe to watch, so nothing is held and its lease
 *  runs out its TTL, the same outcome as a tab closing mid-run. */
export function holdAutoContinueRun(
  messageId: string,
  threadId: string | undefined,
): void {
  if (!threadId) {
    return;
  }
  keeper.hold(messageId, threadId);
  // Registered with the first hold rather than at import, so a module nobody continues in adds no
  // listener. Once, and never removed: it is one handler for the tab.
  if (!listening && typeof window !== "undefined") {
    listening = true;
    window.addEventListener(PROMPT_QUEUE_RUN_FAILED_EVENT, onRunFailed);
  }
  timer ??= setInterval(tick, AUTO_CONTINUE_LEASE_RENEW_MS);
}
