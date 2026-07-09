export const PROMPT_QUEUE_STOP_EVENT = "unsloth:prompt-queue-stop";

export type PromptQueueStopEventDetail = {
  threadIds?: string[];
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
