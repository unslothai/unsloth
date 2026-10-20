// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  clearPendingVoiceNotes,
  consumePendingFiller,
  consumePendingReplace,
  setPendingFiller,
  setPendingReplace,
  voiceNotesEpoch,
} from "../src/features/chat/voice/thinking-filler.ts";

// What the chat adapter attaches to the next outgoing run.
function nextRunNotes(): string[] {
  const notes: string[] = [];
  const filler = consumePendingFiller();
  if (filler) notes.push(`filler:${filler.phrase}`);
  if (consumePendingReplace()) notes.push("replace");
  return notes;
}

test("letting go of the voice turn drops a pending replace and filler", () => {
  setPendingReplace(true);
  setPendingFiller({ say: true, cls: "lookup", phrase: "Let me check that." } as never);
  clearPendingVoiceNotes();
  assert.deepEqual(nextRunNotes(), []);
});

test("a filler verdict from before the reset is recognised as stale", () => {
  const atSend = voiceNotesEpoch();
  clearPendingVoiceNotes();
  assert.notEqual(voiceNotesEpoch(), atSend);
  assert.equal(voiceNotesEpoch(), voiceNotesEpoch());
});

test("notes set for the current turn still reach its run", () => {
  clearPendingVoiceNotes();
  setPendingReplace(true);
  setPendingFiller({ say: true, cls: "lookup", phrase: "One moment." } as never);
  assert.deepEqual(nextRunNotes(), ["filler:One moment.", "replace"]);
  assert.deepEqual(nextRunNotes(), []);
});
