// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The queue and pause signals the local llama-server path sends as SSE comments.
 *
 * N chats share one `--kv-unified` cache while each is told it has all of it, so a chat waits for
 * room and can be paused mid-answer, both invisible on the wire: a 200 that produces nothing for a
 * while. Sent as SSE *comments*, so every reader that predates them ignores them for free.
 */

/** Queued: the request is admitted to the queue but holds no slot yet. */
export const ADMISSION_COMMENT_WAIT = "admission-wait";

/** The slot is ours. Paired with the above; a suspended client clock starts here. */
export const ADMISSION_COMMENT_DONE = "admission-done";

/** Paused mid-answer so another chat can finish. The text already shown is kept. */
export const ADMISSION_COMMENT_PAUSED = "preempt-paused";

/** The upstream request has been re-opened and tokens are flowing again. */
export const ADMISSION_COMMENT_RESUMED = "preempt-resumed";

/** What the stream last said about this run's access to the model. `waiting` and `paused` are
 *  deliberately distinct: queued-before-start promises nothing, while paused-mid-answer has
 *  visible text on screen that the user needs told is not lost. */
export type AdmissionStatus = "waiting" | "admitted" | "paused" | "resumed";

const BY_COMMENT: Record<string, AdmissionStatus> = {
  [ADMISSION_COMMENT_WAIT]: "waiting",
  [ADMISSION_COMMENT_DONE]: "admitted",
  [ADMISSION_COMMENT_PAUSED]: "paused",
  [ADMISSION_COMMENT_RESUMED]: "resumed",
};

/** Read one raw SSE line as an admission signal, or null for anything else. Matched on the
 *  payload after an optional single space, the SSE grammar allowing `:comment` and `: comment`
 *  to mean the same thing. */
export function readAdmissionComment(line: string): AdmissionStatus | null {
  if (!line.startsWith(":")) {
    return null;
  }
  const body = line.slice(1).trim();
  return BY_COMMENT[body] ?? null;
}

/** The line shown while a run is not generating, or null once it is. "Waiting" alone reads as a
 *  stall; naming the cause tells the user the wait is bounded by the other chats. */
export function admissionStatusLabel(status: AdmissionStatus): string | null {
  switch (status) {
    case "waiting":
      return "Waiting for a free slot";
    case "paused":
      return "Paused while another chat finishes";
    default:
      return null;
  }
}
