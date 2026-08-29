// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  new URL("../src/features/audio/audio-page.tsx", import.meta.url),
  "utf8",
);

test("generation exposes a clickable Stop action wired to the request abort", () => {
  assert.match(
    source,
    /const handleStopGeneration[\s\S]*generateAbort\.current\?\.abort\(\)/,
  );
  assert.match(
    source,
    /onClick=\{\s*busy === "generating"\s*\?\s*handleStopGeneration\s*:\s*handleGenerate\s*\}/,
  );
  assert.match(
    source,
    /busy === "generating"[\s\S]*icon=\{StopIcon\}[\s\S]*Stop/,
  );
});

test("leaving the audio page does NOT abort an in-flight generation", () => {
  // RootLayout keeps this page mounted precisely so synthesis survives a tab
  // switch (see the note in routes/audio.tsx), the clip is persisted server-side
  // so it is waiting in the gallery on return, and neither Images nor Video
  // cancels on deactivation either. Only unmount aborts.
  assert.doesNotMatch(
    source,
    /if \(!active\) generateAbort\.current\?\.abort\(\)/,
  );
  assert.match(
    source,
    /useEffect\(\(\) => \(\) => generateAbort\.current\?\.abort\(\), \[\]\)/,
  );
});

test("leaving the audio page aborts an in-flight transcription", () => {
  assert.match(
    source,
    /const transcriptionAbort = useRef<AbortController \| null>\(null\)/,
  );
  assert.match(
    source,
    /transcribeAudioBlob\(blob, \{[\s\S]*signal: controller\.signal/,
  );
  assert.match(
    source,
    /if \(!active\) \{[\s\S]*transcriptionAbort\.current\?\.abort\(\)/,
  );
  assert.match(
    source,
    /if \(controller\.signal\.aborted \|\| !activeRef\.current\) return/,
  );
});

test("a non-abort generation failure refreshes authoritative residency", () => {
  assert.match(
    source,
    /catch \(error\) \{[\s\S]*if \(!controller\.signal\.aborted\)[\s\S]*await refreshStatus\(\)/,
  );
  assert.match(
    source,
    /const refreshStatus[\s\S]*catch \{[\s\S]*setStatus\(null\)/,
  );
});

test("a saved clip the refresh missed keeps its response audio mounted", () => {
  // selectClip nulls the fallback by default, which undid the setFallbackClip immediately
  // before it and left selectedId pointing at a clip that is not in `clips` yet, so the
  // player rendered the empty state.
  assert.match(
    source,
    /const selectClip = useCallback\(\s*\(id: string, keepFallback = false\) => \{[\s\S]*if \(!keepFallback\) setFallbackClip\(null\);/,
  );
  assert.match(source, /saved: true,\s*\}\);\s*selectClip\(generated\.clip_id, true\);/);
});

test("deleting a clip drops the row without waiting on the refresh", () => {
  // refreshGallery swallows a failed GET and returns the cache without setClips, which left
  // the deleted row on screen against an already-revoked object URL.
  assert.match(
    source,
    /await deleteAudioClip\(id\);[\s\S]*galleryCache\.clips = galleryCache\.clips\.filter\([\s\S]*setClips\(galleryCache\.clips\);/,
  );
});

test("the response fallback is dropped once its gallery record arrives", () => {
  // Kept past that point, deleting the now-visible clip made the fallback reappear from a
  // stale data URL, labelled as saved, as though the delete had not happened.
  assert.match(
    source,
    /fallbackClipRef\.current &&\s*galleryCache\.selectedId &&\s*merged\.some\(\(c\) => c\.id === galleryCache\.selectedId\)\s*\)\s*\{\s*setFallbackClip\(null\);/,
  );
});
