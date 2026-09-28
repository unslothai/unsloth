// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Attributing a retained generation failure to the attempt that is settling. */

/** Just the two fields of a progress read this decision needs. */
export interface RetainedGenerationFailure {
  error?: string | null;
  generation_attempt?: string | null;
  error_logged?: boolean | null;
}

/** The failure reason that belongs to THIS attempt, if any. */
export function generationFailureForAttempt(
  progress: RetainedGenerationFailure,
  attemptId: string | null,
): string | null {
  if (!progress.error) return null;
  if (!progress.generation_attempt || !attemptId) return null;
  return progress.generation_attempt === attemptId ? progress.error : null;
}

/** An id for one generation attempt: opaque, per POST, within the backend's own pattern
 * and length bound since it comes back out on a response. `randomUUID` needs a secure
 * context, which a LAN Studio over plain http is not, hence the fallback. */
export function newGenerationAttemptId(): string {
  const uuid = globalThis.crypto?.randomUUID;
  if (typeof uuid === "function") return globalThis.crypto.randomUUID();
  const bytes = new Uint8Array(16);
  if (typeof globalThis.crypto?.getRandomValues === "function")
    globalThis.crypto.getRandomValues(bytes);
  else
    for (let i = 0; i < bytes.length; i++) bytes[i] = (Math.random() * 256) | 0;
  return Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
}

/** The prefix every reason the backend CLASSIFIED and LOGGED carries. */
export const GENERATE_FAILURE_LOGGED_PREFIX = "Image generation failed.";

/** Logged failures that are NOT classified generation failures, so they carry no prefix. */
export const GENERATE_FAILURE_LOGGED_MESSAGES = [
  "Failed to save the generated image.",
];

/** Whether Settings > Logs can actually hold this failure. */
export function generationFailureWasLogged(message: string): boolean {
  if (GENERATE_FAILURE_LOGGED_MESSAGES.includes(message)) return true;
  return message.startsWith(GENERATE_FAILURE_LOGGED_PREFIX);
}

/** The same question for a RETAINED failure, where the message cannot answer it. */
export function retainedFailureWasLogged(
  progress: RetainedGenerationFailure,
): boolean {
  return progress.error_logged !== false;
}
