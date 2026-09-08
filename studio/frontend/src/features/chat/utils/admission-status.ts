// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The queue and pause signals the local llama-server path sends as SSE comments.
 *
 * N chats share one KV cache, so a chat spends real time waiting for room and a chat that
 * started can be paused so another finishes. Both are otherwise invisible on the wire: a 200
 * that produces nothing for a while, indistinguishable from a wedged backend.
 *
 * Sent as SSE *comments* rather than data events, so no chunk schema changes and a reader
 * that predates them sees the silence it saw before.
 *
 * A plain `.ts` because the test runner is `node --experimental-strip-types`, which does NOT
 * transform JSX: nothing reachable only from a `.tsx` can be unit-tested.
 */

/** Queued: the request is admitted to the queue but holds no slot yet. */
export const ADMISSION_COMMENT_WAIT = "admission-wait";

/** The slot is ours. Paired with the above; a suspended client clock starts here. */
export const ADMISSION_COMMENT_DONE = "admission-done";

/** Paused mid-answer so another chat can finish. The text already shown is kept. */
export const ADMISSION_COMMENT_PAUSED = "preempt-paused";

/** The upstream request has been re-opened and tokens are flowing again. */
export const ADMISSION_COMMENT_RESUMED = "preempt-resumed";

/**
 * What the stream last said about this run's access to the model.
 *
 * `waiting` and `paused` are deliberately distinct: queued-before-start has produced nothing,
 * while paused-mid-answer has visible text on screen, and collapsing them would put "waiting
 * for a free slot" under a half-written answer.
 */
export type AdmissionStatus = "waiting" | "admitted" | "paused" | "resumed";

const BY_COMMENT: Record<string, AdmissionStatus> = {
  [ADMISSION_COMMENT_WAIT]: "waiting",
  [ADMISSION_COMMENT_DONE]: "admitted",
  [ADMISSION_COMMENT_PAUSED]: "paused",
  [ADMISSION_COMMENT_RESUMED]: "resumed",
};

/**
 * Read one raw SSE line as an admission signal, or null for anything else.
 *
 * Matched after an optional single space rather than on the whole line: SSE treats
 * `:comment` and `: comment` alike and an intermediary may rewrite that space. Unknown
 * comments return null and are left to whoever else is reading them.
 */
export function readAdmissionComment(line: string): AdmissionStatus | null {
  if (!line.startsWith(":")) {
    return null;
  }
  const body = line.slice(1).trim();
  return BY_COMMENT[body] ?? null;
}

/**
 * The line shown while a run is not generating, or null once it is.
 *
 * "Waiting" alone reads as a stall, so both name the cause, and neither uses failure
 * vocabulary: neither state is an error.
 */
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
