// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const { INITIAL_AB_STATE, clampPosition, resumeAfterLoad, switchSide } =
  await import("../src/features/audio/components/ab-compare-state.ts");
const compare = readSrc("features/audio/components/ab-compare.tsx");
const page = readSrc("features/audio/pages/edit-page.tsx");

test("the recording takes uploads, recordings and history, not saved voices", () => {
  assert.match(page, /id="edit-recording"[\s\S]*?allowSavedVoice=\{false\}/);
  assert.match(page, /id="edit-recording"[\s\S]*?usesFirstSeconds=\{null\}/);
  assert.match(page, /maxRecordSeconds=\{EDIT_SOURCE_MAX_SECONDS\}/);
  const gallery = readSrc("features/audio/hooks/use-audio-gallery.tsx");
  assert.match(gallery, /clip\.role !== "source"/);
});

test("a history pick fills the transcript only with speech, not a Music description or a file name", async () => {
  const { clipReference } = await import(
    "../src/features/audio/audio-run-request.ts"
  );
  const clip = { id: "c1", duration_s: 4, reference_name: "take.wav" };
  const rows: [string, string, string | null][] = [
    ["Hello there.", "edit", "Hello there."],
    ["calm piano", "music", null],
    ["take.wav", "edit", null],
  ];
  for (const [prompt, workflow, transcript] of rows) {
    const ref = clipReference({ ...clip, prompt, workflow });
    assert.equal(ref.transcript, transcript, prompt);
  }
  assert.equal(clipReference({ ...clip, prompt: "take.wav" }).name, "take.wav");
});

test("a new recording cancels the previous one's transcription", () => {
  const hook = readSrc("features/audio/hooks/use-edit-generation.ts");
  assert.match(
    hook,
    /useEffect\(\(\) => cancelTranscribe, \[sourceId, cancelTranscribe\]\);/,
  );
});

test("a result opens on Edited at 0; switching keeps the moment and the playing flag", () => {
  assert.deepEqual(INITIAL_AB_STATE, {
    side: "edited",
    position: 0,
    playing: false,
  });
  const playing = { side: "edited" as const, position: 2.5, playing: true };
  const toOriginal = switchSide(playing, "original", 4.7);
  assert.deepEqual(toOriginal, { ...playing, side: "original" });
  const paused = { ...playing, playing: false };
  assert.equal(switchSide(paused, "original", 4.7).playing, false);
  assert.equal(switchSide(playing, "edited", 1), playing);
  const late = { side: "original" as const, position: 4.6, playing: true };
  assert.equal(switchSide(late, "edited", 3.2).position, 3.2);
  assert.equal(switchSide(late, "edited", null).position, 4.6);
  assert.equal(clampPosition(-1, 3), 0);
  assert.equal(clampPosition(Number.NaN, 3), 0);
});

test("after loading, the clip resumes at the kept moment, or the top when past its end", () => {
  const state = { side: "edited" as const, position: 2, playing: true };
  const rows: [typeof state, number | null, object][] = [
    [state, 3.2, { time: 2, play: true }],
    [{ ...state, position: 3.2 }, 3.2, { time: 0, play: true }],
    [{ ...state, playing: false }, null, { time: 2, play: false }],
  ];
  for (const [at, duration, resume] of rows) {
    assert.deepEqual(resumeAfterLoad(at, duration), resume);
  }
});

test("the A/B player is one keyboard-operable slider over a single audio element", () => {
  assert.match(compare, /role="slider"/);
  for (const key of ['" "', '"ArrowRight"', '"ArrowLeft"', '"Home"']) {
    assert.ok(compare.includes(`event.key === ${key}`), key);
  }
  assert.equal(compare.match(/<audio\b/g)?.length, 1);
  assert.match(compare, /\}, \[originalKey, editedKey\]\);/);
});

test("the bars come from the server's file, since the CSP blocks fetching object URLs", () => {
  assert.match(compare, /if \(src\.startsWith\("blob:"\)\) return null;/);
  assert.match(page, /fileUrl: galleryFileUrl\(clip\.id\)/);
});
