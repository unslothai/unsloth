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
  generation_seq?: number | null;
}

/** The failure reason that belongs to THIS attempt, if any.
 *
 * A retained reason outlives the run it came from, because it is what a caller settling a
 * lost POST has to read. So it has to be dated: if that POST never reached the backend then
 * no run started, the counter has not moved since `seqBeforePost`, and the reason is a
 * previous run's. Attributing it would also skip the gallery probe that reports the request
 * never arrived, which is the more accurate answer.
 *
 * A backend older than `generation_seq` sends a reason that cannot be dated, so it is not
 * used: the pre-existing probe still settles those, as it did before this field existed.
 */
export function generationFailureForAttempt(
  progress: RetainedGenerationFailure,
  seqBeforePost: number,
): string | null {
  if (!progress.error) return null;
  if (typeof progress.generation_seq !== "number") return null;
  return progress.generation_seq > seqBeforePost ? progress.error : null;
}
