// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A generation ends in three stages the page sees as separate states: the denoise stops, the
// gallery record persists (progress still reports active, now with no preview and step back at
// 0), then the finished PNG loads. Reading the preview straight off live progress blanked the
// viewer for the last two -- the user watched a near-complete image vanish into a spinner -- and
// the step-0 report reset the bar to "Preparing" after the final step, reading as a restart.
//
// These replay that whole sequence and assert the viewer never goes backwards.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  nextProgress,
  previewFrame,
} from "../src/features/images/lib/generation-preview.ts";

const FRAME_A = "data:image/jpeg;base64,AAAA";
const FRAME_B = "data:image/jpeg;base64,BBBB";

test("nothing shows before the first frame arrives", () => {
  assert.equal(
    previewFrame({
      held: null,
      generating: true,
      hasSelection: false,
      finishedLoaded: false,
    }),
    null,
  );
});

test("the latest frame shows while denoising", () => {
  assert.equal(
    previewFrame({
      held: FRAME_B,
      generating: true,
      hasSelection: false,
      finishedLoaded: false,
    }),
    FRAME_B,
  );
});

test("the frame survives the persist window, where progress reports no preview", () => {
  assert.equal(
    previewFrame({
      held: FRAME_B,
      generating: true,
      hasSelection: true,
      finishedLoaded: false,
    }),
    FRAME_B,
  );
});

test("the frame survives the blob load, after generating has already gone false", () => {
  assert.equal(
    previewFrame({
      held: FRAME_B,
      generating: false,
      hasSelection: true,
      finishedLoaded: false,
    }),
    FRAME_B,
  );
});

test("the finished image takes over once its blob is ready", () => {
  assert.equal(
    previewFrame({
      held: FRAME_B,
      generating: false,
      hasSelection: true,
      finishedLoaded: true,
    }),
    null,
  );
});

test("a cancelled run with nothing selected does not strand its last frame", () => {
  assert.equal(
    previewFrame({
      held: FRAME_B,
      generating: false,
      hasSelection: false,
      finishedLoaded: false,
    }),
    null,
  );
});

test("the viewer never blanks across a whole run", () => {
  // The recorded sequence: warmup, two denoise frames, persist, blob load, done.
  const run = [
    {
      held: null,
      generating: true,
      hasSelection: false,
      finishedLoaded: false,
    },
    {
      held: FRAME_A,
      generating: true,
      hasSelection: false,
      finishedLoaded: false,
    },
    {
      held: FRAME_B,
      generating: true,
      hasSelection: false,
      finishedLoaded: false,
    },
    {
      held: FRAME_B,
      generating: true,
      hasSelection: true,
      finishedLoaded: false,
    },
    {
      held: FRAME_B,
      generating: false,
      hasSelection: true,
      finishedLoaded: false,
    },
    {
      held: FRAME_B,
      generating: false,
      hasSelection: true,
      finishedLoaded: true,
    },
  ];
  const shown = run.map(previewFrame);
  assert.deepEqual(shown, [null, FRAME_A, FRAME_B, FRAME_B, FRAME_B, null]);
  // Once a frame is up it stays up until the finished image replaces it: no null in between.
  const first = shown.findIndex((frame) => frame !== null);
  const last = shown.length - 1;
  assert.ok(shown.slice(first, last).every((frame) => frame !== null));
  assert.equal(
    shown[last],
    null,
    "the finished image, not a frame, ends the run",
  );
});

test("the persist window's step-0 report does not rewind the bar", () => {
  const atLastStep = { step: 9, total_steps: 9 };
  assert.equal(
    nextProgress(atLastStep, { step: 0, total_steps: 0 }),
    atLastStep,
  );
});

test("step 0 still shows before the run has ticked", () => {
  const warmup = { step: 0, total_steps: 9 };
  assert.equal(nextProgress(null, warmup), warmup);
  assert.equal(nextProgress({ step: 0, total_steps: 9 }, warmup), warmup);
});

test("real progress still advances the bar", () => {
  const advanced = { step: 5, total_steps: 9 };
  assert.equal(nextProgress({ step: 4, total_steps: 9 }, advanced), advanced);
});

// The page polls progress from two independent places: the resume path, and the loop
// handleGenerate starts when you press Generate. Wiring the preview into only one of them
// left the button path with no frames at all, which is invisible to the unit tests above
// because the helper was fine and simply never received anything.
test("every progress poll that drives the bar also drives the preview", () => {
  const page = readFileSync(
    fileURLToPath(
      new URL("../src/features/images/images-page.tsx", import.meta.url),
    ),
    "utf8",
  );
  // Scoped to polls that render progress: settleLostGeneration also polls, but it waits out a
  // lost POST from module scope and has no viewer state to set.
  const driving = [...page.matchAll(/await getGenerateProgress\(\)/g)].filter(
    (poll) => page.slice(poll.index, poll.index + 1400).includes("setGenStep("),
  );
  assert.ok(
    driving.length >= 2,
    "expected both the resume and the Generate poll",
  );
  for (const poll of driving) {
    assert.match(
      page.slice(poll.index, poll.index + 1400),
      /setHeldPreview\(/,
      "a poll drives the progress bar but hands the viewer no preview",
    );
  }
});
