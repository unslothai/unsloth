import { usePromptQueueUI } from "../stores/prompt-queue-ui-store";
import {
  PROMPT_QUEUE_RUN_FAILED_EVENT,
  PROMPT_QUEUE_STOP_EVENT,
} from "./prompt-queue-events";
import { localPromptQueueModelBoundary } from "./prompt-queue-model-boundary";
import {
  chatModelLifecycleGate,
  type ModelLifecycleLease,
} from "./model-lifecycle-gate";

// Module-scope readers should import ./prompt-queue-events directly to avoid the import cycle.
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

/** The same local stop for the chats on one model of several: holds their unsent prompts without
 *  advancing the boundary, which would also void queue starts on the models left loaded. */
export function requestScopedLocalPromptQueueStop(threadIds: string[]) {
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

export function notifyLocalPromptQueueLoadFailed(
  lease: ModelLifecycleLease | null,
) {
  if (lease === null || !chatModelLifecycleGate.markFailed(lease)) return;
  localPromptQueueModelBoundary.advance();
  if (typeof window === "undefined") return;
  window.dispatchEvent(
    new CustomEvent<PromptQueueRunFailedEventDetail>(
      PROMPT_QUEUE_RUN_FAILED_EVENT,
      {
        detail: { localOnly: true },
      },
    ),
  );
}
