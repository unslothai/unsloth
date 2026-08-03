export const PROMPT_QUEUE_STOP_EVENT = "unsloth:prompt-queue-stop";

export interface PromptQueueStopOptions {
  /** Also cancel the prompt the queue already dispatched. Navigation passes `false` to
   * leave it generating; an explicit stop passes `true` (the default). */
  cancelActiveRun?: boolean;
}

export function requestPromptQueueStop(options: PromptQueueStopOptions = {}) {
  if (typeof window === "undefined") {
    return;
  }
  const { cancelActiveRun = true } = options;
  window.dispatchEvent(
    new CustomEvent(PROMPT_QUEUE_STOP_EVENT, { detail: { cancelActiveRun } }),
  );
}
