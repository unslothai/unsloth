// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const {
  EDIT_CHANGES_EMPTY,
  EDIT_DELIVERY_EMPTY,
  EDIT_NO_CHANGES,
  EDIT_NO_SOURCE,
  EDIT_SOURCE_BUSY,
  EDIT_SOURCE_EXPIRED,
  EDIT_SOURCE_TOO_LONG,
  EDIT_TRANSCRIBING,
  EDIT_TRANSCRIPT_EMPTY,
  editBlocker,
} = await import("../src/features/audio/edit-policy.ts");

const EDIT_EXAMPLE = {
  original: "Okay, I'm Cemo and what you just heard wasn't a human voice.",
  edited: "Okay, I'm Cemo and what you just heard wasn't a robot voice.",
};
const { EDIT_TOO_LONG, FIRERED_TOO_MANY_CHANGES } = await import(
  "../src/features/audio/edit-adapters.ts"
);

type Input = Parameters<typeof editBlocker>[0];

const recording = {
  kind: "input" as const,
  id: "in_1",
  name: "take.wav",
  durationS: 4.7,
};

const ready: Input = {
  source: recording,
  sourceBusy: false,
  sourceExpired: false,
  sourceError: null,
  sourceDurationS: 4.7,
  transcribing: false,
  transcript: EDIT_EXAMPLE.original,
  edited: EDIT_EXAMPLE.edited,
  mode: "words",
  delivery: { speed: 1, pitchSteps: 0 },
  panelError: null,
};

const actionIds = (input: Input) =>
  editBlocker(input)?.actions.map((action) => action.id) ?? [];

test("a recording with a checked transcript and one change can run", () => {
  assert.equal(editBlocker(ready), null);
  // An unknown length does not block; the server checks it again.
  assert.equal(editBlocker({ ...ready, sourceDurationS: null }), null);
});

test("Edit's blockers come in rail order, each with its action", () => {
  const empty: Input = {
    ...ready,
    source: null,
    sourceDurationS: null,
    transcript: "",
    edited: "",
    transcribing: true,
    panelError: "panel says no",
  };
  assert.equal(editBlocker(empty)?.reason, EDIT_NO_SOURCE);
  assert.deepEqual(actionIds(empty), ["add-recording"]);
  assert.equal(
    editBlocker({ ...empty, sourceError: "That file is not audio." })?.reason,
    "That file is not audio.",
  );
  // Uploading or recording: wait, rather than ask for a recording.
  assert.equal(
    editBlocker({ ...empty, sourceBusy: true })?.reason,
    EDIT_SOURCE_BUSY,
  );
  const picked = { ...empty, source: recording, sourceDurationS: 31 };
  assert.equal(
    editBlocker({ ...picked, sourceExpired: true })?.reason,
    EDIT_SOURCE_EXPIRED,
  );
  assert.deepEqual(editBlocker({ ...picked, sourceExpired: true })?.actions, [
    { id: "add-recording", label: "Add it again" },
  ]);
  assert.equal(editBlocker(picked)?.reason, EDIT_SOURCE_TOO_LONG);
  assert.deepEqual(actionIds(picked), ["choose-recording"]);
  const fits = { ...picked, sourceDurationS: 30 };
  assert.equal(editBlocker(fits)?.reason, EDIT_TRANSCRIBING);
  const transcribed = { ...fits, transcribing: false };
  assert.equal(editBlocker(transcribed)?.reason, EDIT_TRANSCRIPT_EMPTY);
  assert.deepEqual(actionIds(transcribed), ["transcribe", "type-transcript"]);
  const typed = {
    ...transcribed,
    transcript: EDIT_EXAMPLE.original,
    edited: `  ${EDIT_EXAMPLE.original} `,
  };
  assert.equal(editBlocker(typed)?.reason, EDIT_NO_CHANGES);
  // Deleting every word would send an empty text the run route refuses with a 422.
  assert.equal(editBlocker({ ...typed, edited: "  " })?.reason, EDIT_CHANGES_EMPTY);
  assert.deepEqual(editBlocker(typed)?.actions, [
    { id: "focus-changes", label: "Go to ②" },
  ]);
  const changed = { ...typed, edited: EDIT_EXAMPLE.edited };
  assert.equal(editBlocker(changed)?.kind, "panel");
  assert.equal(
    editBlocker({ ...changed, panelError: FIRERED_TOO_MANY_CHANGES })?.reason,
    FIRERED_TOO_MANY_CHANGES,
  );
  assert.equal(editBlocker({ ...changed, panelError: null }), null);
});

test("past the word cap Words is held back", () => {
  const long = Array.from({ length: 401 }, (_, i) => `w${i}`).join(" ");
  assert.equal(
    editBlocker({ ...ready, transcript: long, edited: `${long} x` })?.reason,
    EDIT_TOO_LONG,
  );
});

test("Delivery needs a speed or a pitch change, but no transcript", () => {
  const delivery: Input = {
    ...ready,
    mode: "delivery",
    transcript: "",
    edited: "",
    transcribing: true,
  };
  assert.equal(editBlocker(delivery)?.reason, EDIT_DELIVERY_EMPTY);
  for (const change of [
    { speed: 1.5, pitchSteps: 0 },
    { speed: 1, pitchSteps: 3 },
  ]) {
    assert.equal(editBlocker({ ...delivery, delivery: change }), null);
  }
  // A panel error (Delivery on a model without it) comes first.
  assert.equal(
    editBlocker({
      ...delivery,
      panelError: "Delivery changes need FireRedAudio.",
    })?.reason,
    "Delivery changes need FireRedAudio.",
  );
  // The recording steps still apply.
  assert.equal(
    editBlocker({ ...delivery, source: null, sourceDurationS: null })?.reason,
    EDIT_NO_SOURCE,
  );
});
