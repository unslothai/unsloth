// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The queue and pause signals the local llama-server path sends as SSE comments.
 *
 * One llama-server holds one KV cache, and Studio runs it with `--parallel N
 * --kv-unified`, so N chats share N cells while each is told it has all of them. A chat
 * therefore spends real time waiting for room, and a chat that started can be paused so
 * another can finish. Both are invisible on the wire: the response is a 200 that simply
 * produces nothing for a while, which is indistinguishable from a wedged backend.
 *
 * Sent as SSE *comments* rather than data events, so every reader that predates them
 * ignores them for free -- no chunk schema changes, and an old client sees exactly the
 * silence it saw before. `chat-api.ts` is the reader that opts in.
 *
 * Split out of the adapter into a plain `.ts` because the test runner is
 * `node --experimental-strip-types`, which strips types but does NOT transform JSX:
 * nothing reachable only from a `.tsx` can be unit-tested.
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
 * `waiting` and `paused` are deliberately distinct even though both mean "no tokens right
 * now". Queued-before-start has produced nothing and promises nothing; paused-mid-answer
 * has visible text on screen, and the one thing that user needs told is that it is not
 * lost. Collapsing them would put "waiting for a free slot" under a half-written answer.
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
 * Matched on the payload after an optional single space rather than on the whole line:
 * the SSE grammar allows `:comment` and `: comment` to mean the same thing, and an
 * intermediary is free to rewrite that space. Trailing whitespace is trimmed for the same
 * reason. Unknown comments -- `: keep-alive` above all -- return null and are left to
 * whoever else is reading them.
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
 * Plain and specific. "Waiting" alone reads as a stall; naming the cause is what tells the
 * user the app is working as intended and that the wait is bounded by the other chats
 * rather than by a fault. No failure vocabulary in either: neither state is an error.
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
