// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type AudioSourceSelection,
  EDIT_SOURCE_MAX_SECONDS,
} from "./audio-run-request";
import {
  EDIT_TOO_LONG,
  type EditDelivery,
  type EditMode,
} from "./edit-adapters";
import { countChanges } from "./edit-diff";

export const EDIT_COPY = {
  recordingLabel: "Recording",
  recordingHint: "Up to 30 s of one voice.",
  transcriptLabel: "① Check the transcript",
  transcriptHint:
    "Fix any word the recognizer got wrong, so it matches the audio.",
  changesLabel: "② Make your changes",
  changesHint: "Change, add or remove words. Each change is marked below.",
  resetLabel: "Reset to transcript",
  wordsTab: "Words",
  deliveryTab: "Delivery",
  speedLabel: "Speed",
  pitchLabel: "Raise pitch",
  deliveryLabel: "② Change the delivery",
  pitchHint:
    "Below 1× is slower, above is faster. FireRedAudio can raise pitch, not lower it, and makes one pass per change.",
  howItEditsTitle: "How edits apply",
  vevo2SwitchNote:
    "Vevo2 reloads for editing when it was last used on Clone, about 1–9 s.",
  emptyText:
    "Edited speech lands here. Add a recording, fix its transcript, change the words, and press Generate.",
} as const;

export const EDIT_NO_SOURCE = "Add a recording to edit.";
export const EDIT_SOURCE_BUSY = "Waiting for the recording…";
export const EDIT_SOURCE_EXPIRED = "This recording expired. Add it again.";
export const EDIT_SOURCE_TOO_LONG = `Edit works on recordings up to ${EDIT_SOURCE_MAX_SECONDS} s. Record or upload a shorter take.`;
export const EDIT_TRANSCRIBING = "Checking the transcript…";
export const EDIT_TRANSCRIPT_EMPTY = "Type what's said in the recording.";
export const EDIT_NO_CHANGES = "Change at least one word in ②.";
export const EDIT_CHANGES_EMPTY = "Keep at least one word in ②.";
export const EDIT_DELIVERY_EMPTY = "Pick a speed or a pitch change.";

export type EditBlockerActionId =
  | "add-recording"
  | "choose-recording"
  | "transcribe"
  | "type-transcript"
  | "focus-changes";

interface EditBlockerAction {
  id: EditBlockerActionId;
  label: string;
}

interface EditBlocker {
  kind: string;
  reason: string;
  actions: readonly EditBlockerAction[];
}

interface EditBlockerInput {
  source: AudioSourceSelection | null;
  sourceBusy: boolean;
  sourceExpired: boolean;
  sourceError?: string | null;
  sourceDurationS: number | null;
  transcribing: boolean;
  transcript: string;
  edited: string;
  mode: EditMode;
  delivery: EditDelivery;
  panelError: string | null;
}

const blocker = (
  kind: string,
  reason: string,
  actions: EditBlockerAction[] = [],
): EditBlocker => ({ kind, reason, actions });

/** The first missing page input, in rail order. Model blockers come from the host first;
 *  Delivery reads no words, so the transcript steps apply to Words only. */
export function editBlocker(input: EditBlockerInput): EditBlocker | null {
  if (!input.source && !input.sourceBusy) {
    return blocker("source", input.sourceError || EDIT_NO_SOURCE, [
      { id: "add-recording", label: "Add recording" },
    ]);
  }
  if (input.sourceBusy) return blocker("source-busy", EDIT_SOURCE_BUSY);
  if (input.sourceExpired) {
    return blocker("source-expired", EDIT_SOURCE_EXPIRED, [
      { id: "add-recording", label: "Add it again" },
    ]);
  }
  if (
    typeof input.sourceDurationS === "number" &&
    input.sourceDurationS > EDIT_SOURCE_MAX_SECONDS
  ) {
    return blocker("source-too-long", EDIT_SOURCE_TOO_LONG, [
      { id: "choose-recording", label: "Choose another" },
    ]);
  }
  if (input.mode === "words") {
    if (input.transcribing) return blocker("transcribing", EDIT_TRANSCRIBING);
    if (!input.transcript.trim()) {
      return blocker("transcript", EDIT_TRANSCRIPT_EMPTY, [
        { id: "transcribe", label: "Transcribe it" },
        { id: "type-transcript", label: "Type it" },
      ]);
    }
    if (!input.edited.trim()) {
      return blocker("changes-empty", EDIT_CHANGES_EMPTY, [
        { id: "focus-changes", label: "Go to ②" },
      ]);
    }
    const changes = countChanges(input.transcript, input.edited);
    if (changes === null) return blocker("too-long", EDIT_TOO_LONG);
    if (changes === 0) {
      return blocker("changes", EDIT_NO_CHANGES, [
        { id: "focus-changes", label: "Go to ②" },
      ]);
    }
  }
  if (input.panelError) return blocker("panel", input.panelError);
  if (
    input.mode === "delivery" &&
    Math.abs(input.delivery.speed - 1) < 1e-6 &&
    !(input.delivery.pitchSteps > 0)
  ) {
    return blocker("delivery", EDIT_DELIVERY_EMPTY);
  }
  return null;
}
