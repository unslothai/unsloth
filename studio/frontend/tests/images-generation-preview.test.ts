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
  releaseHeldPreview,
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
      browsing: false,
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
      browsing: false,
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
      browsing: false,
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
      browsing: false,
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
      browsing: false,
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
      browsing: false,
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
      browsing: false,
    },
    {
      held: FRAME_A,
      generating: true,
      hasSelection: false,
      finishedLoaded: false,
      browsing: false,
    },
    {
      held: FRAME_B,
      generating: true,
      hasSelection: false,
      finishedLoaded: false,
      browsing: false,
    },
    {
      held: FRAME_B,
      generating: true,
      hasSelection: true,
      finishedLoaded: false,
      browsing: false,
    },
    {
      held: FRAME_B,
      generating: false,
      hasSelection: true,
      finishedLoaded: false,
      browsing: false,
    },
    {
      held: FRAME_B,
      generating: false,
      hasSelection: true,
      finishedLoaded: true,
      browsing: false,
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

// The frame is held past the denoise so the handoff to the finished image never blanks. Holding
// it any longer is its own bug: the gallery blob cache has a byte budget and evicts, and a newly
// loaded page of results streams its blobs in, so "selected but not loaded yet" is an ordinary
// state long after a run. Reaching it with a frame still held painted the previous run's preview
// over whichever image the user had just clicked.

test("the frame is released once the run's own image is up", () => {
  assert.equal(
    releaseHeldPreview({
      held: FRAME_B,
      generating: false,
      finishedLoaded: true,
      selectionMatchesRun: true,
      producedImage: true,
    }),
    true,
  );
});

test("the frame is released when the user picks a different image", () => {
  assert.equal(
    releaseHeldPreview({
      held: FRAME_B,
      generating: false,
      finishedLoaded: false,
      selectionMatchesRun: false,
      producedImage: true,
    }),
    true,
  );
});

test("the handoff itself never releases the frame", () => {
  // Denoise done, record persisting, blob not in yet: exactly what the hold exists for.
  assert.equal(
    releaseHeldPreview({
      held: FRAME_B,
      generating: false,
      finishedLoaded: false,
      selectionMatchesRun: true,
      producedImage: true,
    }),
    false,
  );
  // And nothing is released mid-denoise.
  assert.equal(
    releaseHeldPreview({
      held: FRAME_B,
      generating: true,
      finishedLoaded: false,
      selectionMatchesRun: true,
      producedImage: true,
    }),
    false,
  );
});

test("there is nothing to release when no frame is held", () => {
  assert.equal(
    releaseHeldPreview({
      held: null,
      generating: false,
      finishedLoaded: false,
      selectionMatchesRun: false,
      producedImage: true,
    }),
    false,
  );
});

test("a released frame cannot cover an unrelated image", () => {
  // Replays the reported sequence: a run finishes and hands off, then the user clicks a gallery
  // image whose blob was evicted. Without the release the viewer showed FRAME_B for that image.
  let held: string | null = FRAME_B;
  const step = (state: {
    generating: boolean;
    finishedLoaded: boolean;
    selectionMatchesRun: boolean;
  }) => {
    if (releaseHeldPreview({ held, ...state, producedImage: true })) held = null;
    return previewFrame({
      held,
      generating: state.generating,
      hasSelection: true,
      finishedLoaded: state.finishedLoaded,
      browsing: false,
    });
  };
  // The run's image lands and takes over.
  assert.equal(
    step({ generating: false, finishedLoaded: true, selectionMatchesRun: true }),
    null,
  );
  // A different image is selected before its blob arrives: no stale frame, just the spinner.
  assert.equal(
    step({ generating: false, finishedLoaded: false, selectionMatchesRun: false }),
    null,
  );
  assert.equal(held, null);
});

test("the page releases the held frame rather than leaving it set", () => {
  const page = readFileSync(
    fileURLToPath(
      new URL("../src/features/images/images-page.tsx", import.meta.url),
    ),
    "utf8",
  );
  assert.match(
    page,
    /releaseHeldPreview\(/,
    "the page holds a preview but never releases it",
  );
  assert.match(
    page,
    /previewOwner\.current/,
    "the release needs the run's image to tell a later pick apart",
  );
});

// --- browsing the gallery during a run -------------------------------------------------
// Before previews, the viewer showed whatever thumbnail you clicked, including mid-run.
// Returning the held frame unconditionally while generating took that away: the highlight
// moved but the viewer kept showing the denoise.

test("a mid-run gallery pick takes the viewer back from the preview", () => {
  assert.equal(
    previewFrame({
      held: FRAME_B,
      generating: true,
      hasSelection: true,
      finishedLoaded: true,
      browsing: true,
    }),
    null,
  );
});

test("the preview keeps the viewer while the run is not being browsed away from", () => {
  assert.equal(
    previewFrame({
      held: FRAME_B,
      generating: true,
      hasSelection: true,
      finishedLoaded: true,
      browsing: false,
    }),
    FRAME_B,
  );
});

// --- runs that never produce an image --------------------------------------------------
// A cancelled or failed run satisfies neither release condition: the selection still
// matches the run, and no blob of its own is ever coming.

test("a cancelled run's frame is released rather than held forever", () => {
  assert.equal(
    releaseHeldPreview({
      held: FRAME_B,
      generating: false,
      finishedLoaded: false,
      selectionMatchesRun: true,
      producedImage: false,
    }),
    true,
  );
});

test("a run that did produce an image still gets its handoff", () => {
  assert.equal(
    releaseHeldPreview({
      held: FRAME_B,
      generating: false,
      finishedLoaded: false,
      selectionMatchesRun: true,
      producedImage: true,
    }),
    false,
  );
});

test("a cancelled run cannot leave its frame over a gallery image", () => {
  let held: string | null = FRAME_B;
  if (
    releaseHeldPreview({
      held,
      generating: false,
      finishedLoaded: false,
      selectionMatchesRun: true,
      producedImage: false,
    })
  )
    held = null;
  assert.equal(
    previewFrame({
      held,
      generating: false,
      hasSelection: true,
      finishedLoaded: false,
      browsing: false,
    }),
    null,
    "the cancelled run's last frame must not stand in for the selected image",
  );
});

test("the page tracks whether the run produced a record, and mid-run browsing", () => {
  const page = readFileSync(
    fileURLToPath(
      new URL("../src/features/images/images-page.tsx", import.meta.url),
    ),
    "utf8",
  );
  assert.match(page, /setRunProducedImage\(/, "no record tracking for the release rule");
  assert.match(page, /setBrowsingDuringRun\(/, "no mid-run browsing signal");
  // The owner is adopted late, for the resume path that goes idle before its record lands.
  assert.match(
    page,
    /previewOwner\.current === null && selected/,
    "a run that ends owning nothing must adopt its first selection, not release on it",
  );
});

test("a generation whose POST response was lost still gets its handoff", () => {
  // The secure-mode tunnel caps the response near 100s, so a long run can succeed with its
  // POST lost. settleLostGeneration then proves the record exists off the gallery. That path
  // has an image coming just like a normal success, so it must mark the run as produced --
  // otherwise the release rule treats it as cancelled and drops the frame mid-handoff.
  const page = readFileSync(
    fileURLToPath(
      new URL("../src/features/images/images-page.tsx", import.meta.url),
    ),
    "utf8",
  );
  const recovery = page.indexOf("await settleLostGeneration(");
  assert.ok(recovery > 0, "the lost-response recovery branch moved");
  assert.match(
    page.slice(recovery, recovery + 900),
    /setRunProducedImage\(true\)/,
    "a recovered generation must count as having produced its image",
  );
});

// --- sequential runs (Runs > 1) and resumed runs ---------------------------------------
// Both outcome and progress belong to ONE backend run. Latching either across a batch, or
// carrying one in from a run the page did not start, misreports the run in flight.

test("each attempt in a batch resets its own outcome and progress", () => {
  const page = readFileSync(
    fileURLToPath(
      new URL("../src/features/images/images-page.tsx", import.meta.url),
    ),
    "utf8",
  );
  const loop = page.indexOf("for (let i = 0; i < runs; i++)");
  assert.ok(loop > 0, "the run loop moved");
  const body = page.slice(loop, loop + 700);
  assert.match(
    body,
    /setRunProducedImage\(false\)/,
    "attempt 1's success would otherwise speak for a later cancelled attempt",
  );
  assert.match(
    body,
    /setGenStep\(null\)/,
    "the previous attempt's step count would otherwise suppress the next run's step 0",
  );
});

test("a resumed run records whether it actually produced an image", () => {
  const page = readFileSync(
    fileURLToPath(
      new URL("../src/features/images/images-page.tsx", import.meta.url),
    ),
    "utf8",
  );
  const resume = page.indexOf("const knownBefore = galleryCache.images.length");
  assert.ok(resume > 0, "the resumed-run settle moved");
  const body = page.slice(resume, resume + 400);
  assert.match(body, /await loadGallery\(\)/, "the outcome must be settled before going idle");
  assert.match(
    body,
    /setRunProducedImage\(galleryCache\.images\.length > knownBefore\)/,
    "a resumed run that was stopped must not read as having an image coming",
  );
});
