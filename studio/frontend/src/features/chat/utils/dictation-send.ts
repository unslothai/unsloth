// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Only when the recording added text, so silence never sends a half-finished draft. */
export function dictationProducedText(before: string, after: string): boolean {
  const trimmed = after.trim();
  return trimmed.length > 0 && trimmed !== before.trim();
}

export function dictationSendBlocked(state: {
  composerDisabled: boolean;
  uploading: boolean;
  researchActive: boolean;
  runActive: boolean;
  queueDisabled: boolean;
  hasOverlay: boolean;
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

/** Refuse on a thread switch, a text change not from speech, or silence. */
export function shouldSubmitDictation(input: {
  originComposer: string;
  currentComposer: string;
  producedTranscript: boolean;
  baseText: string;
  text: string;
}): boolean {
  if (input.originComposer !== input.currentComposer) return false;
  if (!input.producedTranscript) return false;
  return dictationProducedText(input.baseText, input.text);
}
