import { usePromptQueueUI } from "../stores/prompt-queue-ui-store";
import { localPromptQueueModelBoundary } from "./prompt-queue-model-boundary";

export const PROMPT_QUEUE_STOP_EVENT = "unsloth:prompt-queue-stop";
export const PROMPT_QUEUE_RUN_FAILED_EVENT = "unsloth:prompt-queue-run-failed";

export type PromptQueueStopEventDetail = {
  threadIds?: string[];
  temporaryOnly?: boolean;
  localOnly?: boolean;
};

export type PromptQueueRunFailedEventDetail = {
  threadId?: string | null;
};

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
