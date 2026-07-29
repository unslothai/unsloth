export const PROMPT_QUEUE_STOP_EVENT = "unsloth:prompt-queue-stop";
export const PROMPT_QUEUE_RUN_FAILED_EVENT = "unsloth:prompt-queue-run-failed";
export const PRE_STREAM_RUN_FAILED_EVENT = "unsloth:pre-stream-run-failed";

let preStreamRunReservations = 0;
let preStreamRunReservationToken: symbol | null = null;
const preStreamRunThreadIds = new Set<string>();

export interface PromptQueueStopOptions {
  threadIds?: string[];
  /** Also cancel prompts the selected queues already dispatched. Navigation
   * leaves independent background queues alone; explicit stop defaults true. */
  cancelActiveRun?: boolean;
}

export type PromptQueueRunFailedEventDetail = {
  threadId?: string | null;
};

export function requestPromptQueueStop(
  options: PromptQueueStopOptions | string[] = {},
) {
  if (typeof window === "undefined") {
    return;
  }
  const detail = Array.isArray(options)
    ? { threadIds: options, cancelActiveRun: true }
    : { cancelActiveRun: true, ...options };
  window.dispatchEvent(
    new CustomEvent<PromptQueueStopOptions>(PROMPT_QUEUE_STOP_EVENT, {
      detail,
    }),
  );
}

export function tryReservePreStreamRun(): boolean {
  if (preStreamRunReservations > 0) {
    return false;
  }
  preStreamRunReservations += 1;
  preStreamRunReservationToken = Symbol("pre-stream-run");
  return true;
}

export function getPreStreamRunReservationToken(): symbol | null {
  return preStreamRunReservationToken;
}

export function releasePreStreamRunReservation(
  expectedToken?: symbol | null,
): boolean {
  if (
    expectedToken !== undefined &&
    expectedToken !== preStreamRunReservationToken
  ) {
    return false;
  }
  preStreamRunReservations = Math.max(0, preStreamRunReservations - 1);
  if (preStreamRunReservations === 0) {
    preStreamRunReservationToken = null;
  }
  return true;
}

export function registerPreStreamRun(threadId?: string | null): void {
  if (threadId) {
    preStreamRunThreadIds.add(threadId);
  }
}

export function releasePreStreamRunForThread(
  threadId?: string | null,
  expectedToken?: symbol | null,
): boolean {
  if (!releasePreStreamRunReservation(expectedToken)) {
    return false;
  }
  if (threadId) {
    preStreamRunThreadIds.delete(threadId);
  }
  return true;
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

export function notifyPreStreamRunFailed(
  threadId?: string | null,
  expectedToken?: symbol | null,
): boolean {
  if (!releasePreStreamRunForThread(threadId, expectedToken)) {
    return false;
  }
  notifyPromptQueueRunFailed(threadId);
  if (typeof window === "undefined") {
    return true;
  }
  window.dispatchEvent(
    new CustomEvent<PromptQueueRunFailedEventDetail>(
      PRE_STREAM_RUN_FAILED_EVENT,
      {
        detail: { threadId },
      },
    ),
  );
  return true;
}
