// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The queue and pause signals the local llama-server path sends as SSE comments.
 *
 * N chats share one `--kv-unified` cache while each is told it has all of it, so a chat waits for
 * room and can be paused mid-answer, both invisible on the wire: a 200 that produces nothing for a
 * while. Sent as SSE *comments*, so every reader that predates them ignores them for free.
 */

export const ADMISSION_COMMENT_WAIT = "admission-wait";

export const ADMISSION_COMMENT_DONE = "admission-done";

export const ADMISSION_COMMENT_PAUSED = "preempt-paused";

export const ADMISSION_COMMENT_RESUMED = "preempt-resumed";

export const ADMISSION_COMMENT_RECOMPUTED = "preempt-recomputed";

/** `waiting` promises nothing; `paused` has text on screen the user must be told is kept.
 *  `recomputed` qualifies the preceding resume rather than being a state. */
export type AdmissionStatus =
  | "waiting"
  | "admitted"
  | "paused"
  | "resumed"
  | "recomputed";

const BY_COMMENT: Record<string, AdmissionStatus> = {
  [ADMISSION_COMMENT_WAIT]: "waiting",
  [ADMISSION_COMMENT_DONE]: "admitted",
  [ADMISSION_COMMENT_PAUSED]: "paused",
  [ADMISSION_COMMENT_RESUMED]: "resumed",
  [ADMISSION_COMMENT_RECOMPUTED]: "recomputed",
};

export function readAdmissionComment(line: string): AdmissionStatus | null {
  if (!line.startsWith(":")) {
    return null;
  }
  const body = line.slice(1).trim();
  return BY_COMMENT[body] ?? null;
}

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
