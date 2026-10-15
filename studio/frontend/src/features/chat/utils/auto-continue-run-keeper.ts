// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
    /** Ignores the image gate's true/false pulse, which would otherwise mark the message continued.
     *  A real run sharing the key still counts. */
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

/** The adapter wrapper's per-thread failure event is the only way to tell failed from slow. */
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

/** Called when the bar starts a run (it unmounts right after). Threads without a remote id
 *  share a placeholder, so nothing is held and the lease runs out its TTL. */
export function holdAutoContinueRun(
  messageId: string,
  threadId: string | undefined,
): void {
  if (!threadId) {
    return;
  }
  keeper.hold(messageId, threadId);
  if (!listening && typeof window !== "undefined") {
    listening = true;
    window.addEventListener(PROMPT_QUEUE_RUN_FAILED_EVENT, onRunFailed);
  }
  timer ??= setInterval(tick, AUTO_CONTINUE_LEASE_RENEW_MS);
}

/** Ends holds whose preflight the user stopped; those raise no failure and never stream. */
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
