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
import { issuedRunFrom } from "./auto-continue-issued-run";
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
    /**
     * Generating, and not the image gate settling its waiters.
     *
     * A continuation carrying an image that the loaded model cannot read -- a chat with a
     * picture in it, reopened on a truncated reply with a text-only model loaded, or with
     * Vision switched off since -- is refused before any request is made, and the gate
     * pulses this field true and then false so compare mode's `waitForRunEnd` resolves. To
     * a hold that is waiting for its own run to start, that pair reads as the run starting
     * and finishing: it arms on the first and is released on the second, with the `done`
     * marker that tells every other tab for a day that the message HAS been continued. The
     * failure that follows cannot undo it, because a released hold is gone. So the pulse is
     * not counted as a run, the hold never arms, and the failure discards it as it does any
     * other run that died before it started.
     *
     * Only when the gate is ALL that holds the flag. A real run sharing the key -- the
     * other compare pane, or a sibling under "__default" -- still answers yes.
     */
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

/**
 * A run that failed before it ever reached `runningByThreadId`.
 *
 * The adapter wrapper catches everything `adapter.run` throws -- this chat's settings
 * pairing running out, a model that will not load, a connection it refuses -- and announces
 * it per thread, which is the only signal that separates a preflight that FAILED from one
 * that is merely long. Nothing here reads a clock: a slow run emits no event and keeps its
 * hold and its renewals, which is what the absence of an arming deadline is for.
 */
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
  // Registered with the first hold rather than at import, so a module nobody continues in
  // adds no listener. Once, and never removed: it is one handler for the tab.
  if (!listening && typeof window !== "undefined") {
    listening = true;
    window.addEventListener(PROMPT_QUEUE_RUN_FAILED_EVENT, onRunFailed);
  }
  timer ??= setInterval(tick, AUTO_CONTINUE_LEASE_RENEW_MS);
}

/**
 * Tie the hold just taken for `messageId` to the run the bar has just issued for it.
 *
 * `started` is whatever `startRun` handed back, passed through untyped: the runtime declares
 * `void` and returns the roundtrip's promise, and `issuedRunFrom` is what decides which of
 * those it actually got.
 *
 * This is the only thing that ends a hold whose preflight the user STOPPED. That run raises no
 * failure -- the abort is what was asked for, so the adapter wrapper skips its notice on
 * purpose -- and it never reached the stream flag either, so without this the hold renewed its
 * lease until the tab closed and every other tab refused the message for just as long.
 *
 * Nothing here reads a clock. A preflight that is merely long leaves the promise pending, and
 * a pending promise settles nothing, which is what the absence of an arming deadline is for.
 */
export function watchAutoContinueRun(
  messageId: string,
  threadId: string | undefined,
  started: unknown,
): void {
  if (!threadId) {
    return;
  }
  keeper.settleOn(messageId, threadId, issuedRunFrom(started));
}
