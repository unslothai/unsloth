// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// #9210: the queue button stayed disabled when an attachment was staged with
// no text, because the eligibility gates only knew about text prompts and
// pasted-text chips. The decision logic is extracted from thread.tsx into
// queueAttachmentEligibility so it can be pinned without mounting the
// composer tree.

import assert from "node:assert/strict";
import test from "node:test";

import {
  canQueueAttachmentOnlyPrompt,
  queueAttachmentEligibility,
} from "../src/features/chat/utils/queue-attachment-eligibility.ts";

const base = {
  hasAttachments: false,
  hasPendingAudio: false,
  hasPendingAudioUpload: false,
  isComposing: false,
  hasPendingAttachments: false,
  hasMaterializingImageAttachments: false,
  hasMaterializingAudioAttachments: false,
  disabled: false,
  overlay: false,
};

test("an attachment with no text makes the queue button eligible (#9210)", () => {
  assert.equal(
    canQueueAttachmentOnlyPrompt({ ...base, hasAttachments: true }),
    true,
  );
  assert.equal(
    canQueueAttachmentOnlyPrompt({ ...base, hasPendingAudio: true }),
    true,
  );
});

test("text-only and empty composers stay ineligible for the attachment leg", () => {
  // Empty: nothing to send at all.
  assert.equal(canQueueAttachmentOnlyPrompt(base), false);
});

test("uploads in flight or a disabled composer block the attachment leg", () => {
  assert.equal(
    canQueueAttachmentOnlyPrompt({
      ...base,
      hasAttachments: true,
      hasPendingAttachments: true,
    }),
    false,
  );
  assert.equal(
    canQueueAttachmentOnlyPrompt({
      ...base,
      hasPendingAudio: true,
      hasMaterializingAudioAttachments: true,
    }),
    false,
  );
  assert.equal(
    canQueueAttachmentOnlyPrompt({ ...base, hasAttachments: true, disabled: true }),
    false,
  );
  assert.equal(
    canQueueAttachmentOnlyPrompt({ ...base, hasAttachments: true, isComposing: true }),
    false,
  );
  assert.equal(
    canQueueAttachmentOnlyPrompt({ ...base, hasAttachments: true, overlay: true }),
    false,
  );
});

test("the combined gate keeps the text and pasted-text legs intact", () => {
  const combined = queueAttachmentEligibility({
    ...base,
    hasAttachments: true,
    composerText: "hello",
  });
  // Text present with a real attachment: the text leg is not usable (it
  // requires !hasAttachments) and pasted-text requires ALL attachments to
  // be pasted text; the attachment leg carries it.
  assert.equal(combined.canQueueAttachmentPrompt, true);

  const pastedOnly = queueAttachmentEligibility({
    ...base,
    hasAttachments: true,
    composerText: "",
    attachmentsAreAllPastedText: true,
  });
  assert.equal(pastedOnly.canQueuePastedTextPrompt, true);
  assert.equal(pastedOnly.canQueueAttachmentPrompt, true);
});
