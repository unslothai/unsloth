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
  assert.match(page, /id="edit-recording"[\s\S]*?trimsLongClips=\{false\}/);
  const card = readSrc("features/audio/components/audio-source-input.tsx");
  assert.match(card, /trimsLongClips &&\s*durationS !== null/);
});

test("a history pick fills ① only with speech, not a Music description or a file name", () => {
  const card = readSrc("features/audio/components/audio-source-input.tsx");
  assert.match(card, /transcript: spokenText\(clip\)/);
  assert.match(card, /clipWorkflow\(clip\) !== "music"/);
  assert.match(card, /clip\.prompt !== clip\.reference_name/);
});

test("the compare resets for a new pair, not when the Original's file arrives", () => {
  const compare = readSrc("features/audio/components/ab-compare.tsx");
  assert.match(compare, /\}, \[originalKey, editedKey\]\);/);
  assert.doesNotMatch(compare, /\}, \[original\.src, edited\.src\]\);/);
});

test("a new recording cancels the previous one's transcription", () => {
  const hook = readSrc("features/audio/hooks/use-edit-generation.ts");
  assert.match(
    hook,
    /useEffect\(\(\) => cancelTranscribe, \[sourceId, cancelTranscribe\]\);/,
  );
});
