export const PROMPT_QUEUE_STOP_EVENT = "unsloth:prompt-queue-stop";
export const PROMPT_QUEUE_RUN_FAILED_EVENT = "unsloth:prompt-queue-run-failed";

export type PromptQueueStopEventDetail = {
  threadIds?: string[];
};

export type PromptQueueRunFailedEventDetail = {
  threadId?: string | null;
};

export function requestPromptQueueStop(threadIds?: string[]) {
  if (typeof window === "undefined") {
    return;
  }
  window.dispatchEvent(
    new CustomEvent<PromptQueueStopEventDetail>(PROMPT_QUEUE_STOP_EVENT, {
      detail: threadIds && threadIds.length > 0 ? { threadIds } : undefined,
    }),
  );
}

export function notifyPromptQueueRunFailed(threadId?: string | null) {
  if (typeof window === "undefined") {
    return;
  }
  window.dispatchEvent(
    new CustomEvent<PromptQueueRunFailedEventDetail>(
      PROMPT_QUEUE_RUN_FAILED_EVENT,
      {
        detail: { threadId },
      },
    ),
  );
}
