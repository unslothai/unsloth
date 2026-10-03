// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const page = readSrc("features/audio/pages/edit-page.tsx");

test("the recording takes uploads, recordings and history, not saved voices", () => {
  assert.match(page, /id="edit-recording"[\s\S]*?allowSavedVoice=\{false\}/);
});

test("history hides an edit's original", () => {
  const gallery = readSrc("features/audio/hooks/use-audio-gallery.tsx");
  assert.match(gallery, /clip\.role !== "source"/);
});

test("the compare draws its bars from the server's file, since the CSP blocks fetching object URLs", () => {
  const compare = readSrc("features/audio/components/ab-compare.tsx");
  assert.match(compare, /if \(src\.startsWith\("blob:"\)\) return null;/);
  assert.match(page, /fileUrl: galleryFileUrl\(clip\.id\)/);
});

test("a long recording is refused on Edit, so the card does not promise to use its first 30 s", () => {
  assert.match(page, /id="edit-recording"[\s\S]*?usesFirstSeconds=\{null\}/);
  assert.match(page, /maxRecordSeconds=\{EDIT_SOURCE_MAX_SECONDS\}/);
});

test("a history pick fills the transcript only with speech, not a Music description or a file name", async () => {
  const { clipReference } = await import(
    "../src/features/audio/audio-run-request.ts"
  );
  const clip = { id: "c1", duration_s: 4, reference_name: "take.wav" };
  assert.equal(
    clipReference({ ...clip, prompt: "Hello there.", workflow: "edit" })
      .transcript,
    "Hello there.",
  );
  assert.equal(
    clipReference({ ...clip, prompt: "calm piano", workflow: "music" })
      .transcript,
    null,
  );
  assert.equal(
    clipReference({ ...clip, prompt: "take.wav", workflow: "edit" }).transcript,
    null,
  );
  assert.equal(
    clipReference({ ...clip, prompt: "take.wav", workflow: "edit" }).name,
    "take.wav",
  );
});

test("the compare resets for a new pair, not when the Original's file arrives", () => {
  const compare = readSrc("features/audio/components/ab-compare.tsx");
  assert.match(compare, /\}, \[originalKey, editedKey\]\);/);
  assert.doesNotMatch(compare, /\}, \[original\.src, edited\.src\]\);/);
  assert.match(
    compare,
    /clips\[side\]\.src \? next : \{ \.\.\.next, playing: false \}/,
  );
});

test("a new recording cancels the previous one's transcription", () => {
  const hook = readSrc("features/audio/hooks/use-edit-generation.ts");
  assert.match(
    hook,
    /useEffect\(\(\) => cancelTranscribe, \[sourceId, cancelTranscribe\]\);/,
  );
});
