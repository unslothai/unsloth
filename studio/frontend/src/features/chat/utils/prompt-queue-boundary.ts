import { usePromptQueueUI } from "../stores/prompt-queue-ui-store";
import {
  PROMPT_QUEUE_RUN_FAILED_EVENT,
  PROMPT_QUEUE_STOP_EVENT,
} from "./prompt-queue-events";
import { localPromptQueueModelBoundary } from "./prompt-queue-model-boundary";

// Re-exported so existing importers and the barrel keep their current paths.
// A module-scope reader should import from ./prompt-queue-events directly: this
// module has dependencies, so it can be caught mid-initialization by the cycle.
export {
  PROMPT_QUEUE_RUN_FAILED_EVENT,
  PROMPT_QUEUE_STOP_EVENT,
} from "./prompt-queue-events";
export type {
  PromptQueueRunFailedEventDetail,
  PromptQueueStopEventDetail,
} from "./prompt-queue-events";

import type {
  PromptQueueRunFailedEventDetail,
  PromptQueueStopEventDetail,
} from "./prompt-queue-events";

export function requestPromptQueueStop(threadIds?: string[]) {
  if (
    typeof window === "undefined" ||
    (threadIds !== undefined && threadIds.length === 0)
  ) {
    return;
  }
  window.dispatchEvent(
    new CustomEvent<PromptQueueStopEventDetail>(PROMPT_QUEUE_STOP_EVENT, {
      detail: threadIds ? { threadIds } : undefined,
    }),
  );
}

/**
 * Stop every materialized local queue and invalidate local queue factories
 * that are still waiting for settings hydration.
 */
export function requestLocalPromptQueueStop(
  additionalThreadIds: string[] = [],
) {
  localPromptQueueModelBoundary.advance();
  const threadIds = [
    ...new Set([
      ...additionalThreadIds,
      ...Object.entries(usePromptQueueUI.getState().byThreadId)
        .filter(([, entry]) => entry.local)
        .map(([threadId]) => threadId),
    ]),
  ];
  if (typeof window === "undefined" || threadIds.length === 0) {
    return;
  }
  window.dispatchEvent(
    new CustomEvent<PromptQueueStopEventDetail>(PROMPT_QUEUE_STOP_EVENT, {
      detail: { threadIds, localOnly: true },
    }),
  );
}

export function requestTemporaryPromptQueueStop() {
  const threadIds = [
    ...new Set(
      Object.entries(usePromptQueueUI.getState().byThreadId)
        .filter(([, entry]) => entry.temporary)
        .map(([threadId]) => threadId),
    ),
  ];
  if (typeof window !== "undefined") {
    window.dispatchEvent(
      new CustomEvent<PromptQueueStopEventDetail>(PROMPT_QUEUE_STOP_EVENT, {
        detail: { threadIds, temporaryOnly: true },
      }),
    );
  }
}

export function notifyPromptQueueRunFailed(threadId?: string | null) {
  if (typeof window === "undefined") {
    return;
  }
  window.dispatchEvent(
    new CustomEvent<PromptQueueRunFailedEventDetail>(
      PROMPT_QUEUE_RUN_FAILED_EVENT,
      { detail: { threadId } },
    ),
  );
}
