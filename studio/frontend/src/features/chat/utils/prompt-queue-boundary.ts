export const PROMPT_QUEUE_STOP_EVENT = "unsloth:prompt-queue-stop";

export interface PromptQueueStopOptions {
  /**
   * Whether to also cancel the prompt the queue already dispatched.
   *
   * Navigation stops the queue feeding more prompts but leaves the one in
   * flight generating like any other chat, so it passes `false`. An explicit
   * stop passes `true` (the default).
   */
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
