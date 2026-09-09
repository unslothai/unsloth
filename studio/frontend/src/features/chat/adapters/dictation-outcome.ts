// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** What the dictation that just ended actually produced. The recording bar's send needs both
 *  answers and cannot get them from the composer: text can change while recording (the plus menu
 *  can insert a saved prompt), and a partial transcript still lands there. Module state, not
 *  session state, because the bar reads it once the session is gone. */
let producedTranscript = false;
let failed = false;

/** Start a session. Clears the previous session's result. */
export function beginDictationSession(): void {
  producedTranscript = false;
  failed = false;
}

/** A final transcript was published to the composer. */
export function markDictationTranscript(): void {
  producedTranscript = true;
}

/** Whether the last dictation published any transcript. */
export function dictationProducedTranscript(): boolean {
  return producedTranscript;
}

/** A transcript chunk, or the session itself, failed. Both engines can still publish what did
 *  transcribe, so the text is partial rather than absent. */
export function markDictationFailed(): void {
  failed = true;
}

/** Whether the last dictation reported a failure. */
export function dictationFailed(): boolean {
  return failed;
}
