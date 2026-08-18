// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * #9210: queue-button eligibility, extracted from the composer tree.
 *
 * The queue itself only carries text — a queued prompt replays as a string.
 * An attachment (image for a vision run, audio for a transcription) with no
 * text therefore cannot ride the queue; the queue button for that shape
 * submits the attachment directly through the same path the form submit
 * takes. These gates decide when that shape applies.
 */

export interface QueueAttachmentEligibilityInput {
  hasAttachments: boolean;
  hasPendingAudio: boolean;
  hasPendingAudioUpload?: boolean;
  isComposing: boolean;
  hasPendingAttachments: boolean;
  hasMaterializingImageAttachments: boolean;
  hasMaterializingAudioAttachments: boolean;
  disabled: boolean;
  overlay: boolean;
}

/**
 * True when the composer holds an attachment (or pending audio) that send()
 * can submit on its own. Mirrors the `composerAcceptsQueueing` constraints
 * the text legs already enforce: no uploads in flight, no IME composition,
 * composer enabled, no modal overlay.
 */
export function canQueueAttachmentOnlyPrompt(
  input: QueueAttachmentEligibilityInput,
): boolean {
  const hasSendableAttachment =
    input.hasAttachments || (input.hasPendingAudio && !input.hasPendingAudioUpload);
  if (!hasSendableAttachment) return false;

  return (
    !input.isComposing &&
    !input.hasPendingAttachments &&
    !input.hasMaterializingImageAttachments &&
    !input.hasMaterializingAudioAttachments &&
    !input.disabled &&
    !input.overlay
  );
}

/**
 * The combined gate the queue button consumes. Text prompts keep their own
 * legs; the attachment leg only fires when there is something send() can
 * submit without text.
 */
export function queueAttachmentEligibility(
  input: QueueAttachmentEligibilityInput & {
    composerText: string;
    attachmentsAreAllPastedText: boolean;
  },
): {
  canQueueCurrentPrompt: boolean;
  canQueuePastedTextPrompt: boolean;
  canQueueAttachmentPrompt: boolean;
} {
  const composerAcceptsQueueing =
    !input.hasPendingAudio &&
    !input.isComposing &&
    !input.hasPendingAttachments &&
    !input.hasMaterializingImageAttachments &&
    !input.hasMaterializingAudioAttachments &&
    !input.disabled &&
    !input.overlay;

  return {
    canQueueCurrentPrompt:
      input.composerText.trim().length > 0 && !input.hasAttachments && composerAcceptsQueueing,
    canQueuePastedTextPrompt: input.attachmentsAreAllPastedText && composerAcceptsQueueing,
    canQueueAttachmentPrompt: canQueueAttachmentOnlyPrompt(input),
  };
}
