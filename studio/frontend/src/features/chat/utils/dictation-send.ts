// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Whether the recording bar's send should submit once dictation ends. Only when this recording
 *  actually added text: silence and failed transcription leave a pre-recording draft in place
 *  rather than sending it half-finished. @param before composer text when send was pressed
 *  @param after composer text once the session ended */
export function dictationProducedText(before: string, after: string): boolean {
  const trimmed = after.trim();
  return trimmed.length > 0 && trimmed !== before.trim();
}

/** Every state the composer's submit rejects. One definition for both jobs: greying out the
 *  recording bar's send, and deciding whether a pending send waits rather than being spent on a
 *  submit that would bounce. Text presence is not here, since the transcript supplies it after
 *  the button is pressed. */
export function dictationSendBlocked(state: {
  /** The composer itself is unavailable. */
  composerDisabled: boolean;
  /** An attachment is still uploading. */
  uploading: boolean;
  /** A deep research run owns the composer. */
  researchActive: boolean;
  /** A response is streaming or the prompt queue is going. */
  runActive: boolean;
  /** This composer never queues, it asks the user to wait. */
  queueDisabled: boolean;
  /** An image edit overlay is open. */
  hasOverlay: boolean;
  /** Only text can be queued, so these block while a run is active. */
  hasAttachments: boolean;
  hasPendingAudio: boolean;
}): boolean {
  if (state.composerDisabled || state.uploading || state.researchActive) {
    return true;
  }
  if (!state.runActive) return false;
  return (
    state.queueDisabled ||
    state.hasOverlay ||
    state.hasAttachments ||
    state.hasPendingAudio
  );
}

/** Whether a pending dictation send may submit now. Three ways it must not. The composer is reused
 *  across thread switches, so a send pressed in one thread can land after a move to another,
 *  where it would submit that thread's draft. The plus menu stays open while recording and can
 *  insert a saved prompt, changing the text without any speech. And silence leaves what was there. */
export function shouldSubmitDictation(input: {
  /** Composer identity when send was pressed. */
  originComposer: string;
  /** Composer identity now that the session has ended. */
  currentComposer: string;
  /** The engine published a final transcript. */
  producedTranscript: boolean;
  /** Composer text at session start. */
  baseText: string;
  /** Composer text now. */
  text: string;
}): boolean {
  if (input.originComposer !== input.currentComposer) return false;
  if (!input.producedTranscript) return false;
  return dictationProducedText(input.baseText, input.text);
}
