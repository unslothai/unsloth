// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  new URL("../src/features/audio/audio-page.tsx", import.meta.url),
  "utf8",
);

test("generation exposes Stop only while the request controller can abort", () => {
  assert.match(
    source,
    /const handleStopGeneration[\s\S]*const controller = generateAbort\.current;[\s\S]*!controller \|\| controller\.signal\.aborted[\s\S]*updateGenerationPhase\("stopping"\);[\s\S]*controller\.abort\(\)/,
  );
  assert.match(
    source,
    /generationPresentation\?\.canStop\s*\?\s*handleStopGeneration\s*:\s*handleGenerate/,
  );
  assert.match(
    source,
    /disabled=\{[\s\S]*generationPresentation[\s\S]*!generationPresentation\.canStop/,
  );
});

test("generation renders one accessible indeterminate task indicator", () => {
  assert.match(
    source,
    /import \{ Progress \} from "@\/components\/ui\/progress"/,
  );
  assert.match(
    source,
    /busy === "generating" && generationPresentation[\s\S]*<Progress[\s\S]*indeterminate[\s\S]*aria-label="Audio task in progress"/,
  );
  assert.match(
    source,
    /<output[\s\S]*aria-live="polite"[\s\S]*aria-atomic="true"[\s\S]*generationPresentation\.status[\s\S]*<\/output>/,
  );
  assert.doesNotMatch(
    source,
    /<Progress[\s\S]{0,300}aria-valuenow|<Progress[\s\S]{0,300}value=/,
  );
});

test("generation progress follows the request-owned lifecycle", () => {
  const generation = source.slice(
    source.indexOf("const handleGenerate = useCallback"),
    source.indexOf("// --- Transcribe"),
  );
  assert.match(
    generation,
    /busyRef\.current = "generating";\s*setBusy\("generating"\);\s*updateGenerationPhase\("preparing"\);\s*const releaseInFlight/,
  );
  assert.match(
    generation,
    /generateAbort\.current = controller;\s*updateGenerationPhase\("generating"\);\s*try \{\s*const generated = await generateAudio/,
  );
  assert.match(
    generation,
    /const generated = await generateAudio[\s\S]*?\);\s*updateGenerationPhase\("finishing"\);\s*const refreshed = await refreshGallery/,
  );
  assert.match(
    generation,
    /catch \(error\) \{\s*if \(!controller\.signal\.aborted\) \{\s*updateGenerationPhase\("finishing"\);[\s\S]*await refreshStatus\(\)/,
  );
  assert.match(
    generation,
    /finally \{\s*generateAbort\.current = null;\s*updateGenerationPhase\(null\);\s*busyRef\.current = null;\s*setBusy\(null\)/,
  );
});

test("mode transitions read the synchronously authoritative generation phase", () => {
  assert.match(
    source,
    /const generationPhaseRef = useRef<AudioGenerationPhase>\(generationPhase\);\s*const updateGenerationPhase = useCallback\(\s*\(nextPhase: AudioGenerationPhase\) => \{\s*generationPhaseRef\.current = nextPhase;\s*setGenerationPhase\(nextPhase\)/,
  );
  assert.match(
    source,
    /canTransitionAudioMode\(busyRef\.current, generationPhaseRef\.current\)/,
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
    /const dropClip = useCallback\(\(id: string\) => \{[\s\S]*galleryCache\.clips = galleryCache\.clips\.filter\([\s\S]*setClips\(galleryCache\.clips\);/,
  );
  assert.match(
    source,
    /await deleteAudioClip\(id\);\s*dropClip\(id\);\s*await refreshGallery\(id\);/,
  );
});

test("archiving a clip drops the row the same way a delete does", () => {
  // Same revoked-object-URL trap: the clip is gone server-side, so a row left up until a refresh
  // that may fail renders a tile that can no longer play.
  assert.match(
    source,
    /await setAudioClipFlags\(id, \{ archived: true \}\);[\s\S]*?dropClip\(id\);\s*await refreshGallery\(id\);/,
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
