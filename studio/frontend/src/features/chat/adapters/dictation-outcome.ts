// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Whether the dictation that just ended lost transcript to an error.
 *
 * Both engines can end holding a partial transcript: the browser one keeps
 * finalized chunks through a recognition error, the local model one keeps the
 * segments that did transcribe. That text still belongs in the composer, but
 * the recording bar's send must not submit it on its own.
 *
 * Module state, not session state, because the recording bar reads it after
 * the session object is gone.
 */
let failed = false;

/** Start a session. Clears the previous session's result. */
export function resetDictationFailure(): void {
  failed = false;
}

/** A transcript chunk, or the session itself, failed. */
export function markDictationFailed(): void {
  failed = true;
}

/** Whether the last dictation reported a failure. */
export function dictationFailed(): boolean {
  return failed;
}
