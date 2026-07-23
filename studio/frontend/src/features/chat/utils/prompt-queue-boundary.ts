export const PROMPT_QUEUE_STOP_EVENT = "unsloth:prompt-queue-stop";
export const PROMPT_QUEUE_RUN_FAILED_EVENT = "unsloth:prompt-queue-run-failed";
export const PRE_STREAM_RUN_FAILED_EVENT = "unsloth:pre-stream-run-failed";

let preStreamRunReservations = 0;
const preStreamRunThreadIds = new Set<string>();

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

export function tryReservePreStreamRun(): boolean {
  if (preStreamRunReservations > 0) {
    return false;
  }
  preStreamRunReservations += 1;
  return true;
}

export function releasePreStreamRunReservation(): void {
  preStreamRunReservations = Math.max(0, preStreamRunReservations - 1);
}

export function registerPreStreamRun(threadId?: string | null): void {
  if (threadId) {
    preStreamRunThreadIds.add(threadId);
  }
}

export function releasePreStreamRunForThread(threadId?: string | null): void {
  if (threadId) {
    preStreamRunThreadIds.delete(threadId);
  }
  releasePreStreamRunReservation();
}

export function isPreStreamRunActive(threadId: string): boolean {
  return preStreamRunThreadIds.has(threadId);
}

export function getPreStreamRunThreadIds(): string[] {
  return Array.from(preStreamRunThreadIds);
}

export function getPreStreamRunReservationCount(): number {
  return preStreamRunReservations;
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

export function notifyPreStreamRunFailed(threadId?: string | null) {
  releasePreStreamRunForThread(threadId);
  notifyPromptQueueRunFailed(threadId);
  if (typeof window === "undefined") {
    return;
  }
  window.dispatchEvent(
    new CustomEvent<PromptQueueRunFailedEventDetail>(
      PRE_STREAM_RUN_FAILED_EVENT,
      {
        detail: { threadId },
      },
    ),
  );
}
