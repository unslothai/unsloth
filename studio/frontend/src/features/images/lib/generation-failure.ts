// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Attributing a retained generation failure to the attempt that is settling.
 *
 * Its own module, with no imports: the images API pulls in enough of the app that it
 * cannot be loaded on its own to test.
 */

/** Just the two fields of a progress read this decision needs. */
export interface RetainedGenerationFailure {
  error?: string | null;
  generation_attempt?: string | null;
  error_logged?: boolean | null;
}

/** The failure reason that belongs to THIS attempt, if any.
 *
 * The reason has to outlive its run, since a caller whose POST was lost has nothing else
 * to read, so it must be identified rather than merely dated: "a run started after my
 * post" includes a concurrent client's run, while a post that never arrived started none.
 * Hence an exact match on the attempt's own id. A reason with no id, from an older backend
 * or a request that carried none, is left to the gallery probe as before.
 */
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

/** Whether Settings > Logs can actually hold this failure.
 *
 * Only the 500 paths log: they call logger.error and answer with a classified reason, which
 * always opens with the prefix above. A request rejected at validation (400) is answered
 * without logging, and "did not reach the server" never left the browser, so offering the
 * action there opens an unrelated current log and points at a false diagnosis.
 */
export function generationFailureWasLogged(message: string): boolean {
  return message.startsWith(GENERATE_FAILURE_LOGGED_PREFIX);
}

/** The same question for a RETAINED failure, where the message cannot answer it.
 *
 * A caller settling a lost POST reads the reason off the progress poll, and by then it has
 * been classified: a client-input failure the route answered WITHOUT logging carries the
 * same prefix as an internal one. So the backend says which it was, and only an explicit
 * false withholds the action -- an older backend sends nothing, and the reasons it retains
 * are the logged ones.
 */
export function retainedFailureWasLogged(
  progress: RetainedGenerationFailure,
): boolean {
  return progress.error_logged !== false;
}
