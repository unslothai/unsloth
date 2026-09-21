// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Attributing a retained generation failure to the attempt that is settling.
 *
 * Its own module, with no imports: the decision is the interesting part and the rest of the
 * images API pulls in enough of the app that it cannot be loaded on its own to test.
 */

/** Just the two fields of a progress read this decision needs. */
export interface RetainedGenerationFailure {
  error?: string | null;
  generation_attempt?: string | null;
}

/** The failure reason that belongs to THIS attempt, if any.
 *
 * A retained reason outlives the run it came from, because it is what a caller settling a
 * lost POST has to read. That is exactly why it has to be identified rather than merely
 * dated. "A run started after my post" is not the same claim as "my post started a run":
 * the post may never have reached the backend, in which case nothing ran for it and the
 * reason is someone else's; and a concurrent client, another tab or a second browser on the
 * same account, can start and fail its own run in the same window. Attributing either one
 * reports a failure that did not happen to this attempt, and skips the gallery probe that
 * would have said what did.
 *
 * So the test is an exact match on the attempt's own id. A reason with no id (an older
 * backend, or a request that carried none) is not used, and the pre-existing probe settles
 * the attempt as it did before this field existed.
 */
export function generationFailureForAttempt(
  progress: RetainedGenerationFailure,
  attemptId: string | null,
): string | null {
  if (!progress.error) return null;
  if (!progress.generation_attempt || !attemptId) return null;
  return progress.generation_attempt === attemptId ? progress.error : null;
}

/** An id for one generation attempt.
 *
 * Opaque and per POST: it exists only so a retained failure can be matched back to the
 * request that caused it. Kept to characters the backend's own pattern accepts, and short
 * enough for its length bound, since it comes back out on a response. `randomUUID` needs a
 * secure context, which a LAN Studio over plain http is not, so there is a fallback.
 */
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
