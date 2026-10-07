// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readAudioWorkspaceSource } from "./helpers/audio-workspace.ts";

const source = readAudioWorkspaceSource();

const { audioGenerationPresentation } = await import(
  "../src/features/audio/audio-page-policy.ts"
);

test("a run that reloads the model first can be stopped while it switches", () => {
  assert.deepEqual(audioGenerationPresentation("switching"), {
    status: "Switching model…",
    actionLabel: "Stop",
    canStop: true,
  });
  assert.deepEqual(
    audioGenerationPresentation("switching", "Switching Chatterbox to Convert…"),
    {
      status: "Switching Chatterbox to Convert…",
      actionLabel: "Stop",
      canStop: true,
    },
  );
  assert.equal(
    audioGenerationPresentation("generating", "Switching…")?.status,
    "Generating audio…",
  );
});

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
  // RootLayout keeps this page mounted so synthesis survives tab switches; only unmount aborts.
  assert.doesNotMatch(
    source,
    /if \(!active\) generateAbort\.current\?\.abort\(\)/,
  );
  assert.match(
    source,
    /useEffect\(\(\) => \(\) => generateAbort\.current\?\.abort\(\), \[\]\)/,
  );
});

test("transcription survives page deactivation and aborts on unmount", () => {
  assert.match(
    source,
    /const transcriptionAbort = useRef<AbortController \| null>\(null\)/,
  );
  assert.match(
    source,
    /transcribeSourceWithProgress\(\s*sourceRefOf\(source\),\s*source\.name,\s*\{[\s\S]*signal: controller\.signal/,
  );
  assert.match(
    source,
    /useEffect\(\(\) => \(\) => transcriptionAbort\.current\?\.abort\(\), \[\]\)/,
  );
  assert.doesNotMatch(
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
  assert.match(
    source,
    /const selectClip = useCallback\(\s*\(id: string, keepFallback = false\) => \{[\s\S]*if \(!keepFallback && !otherPage\) setFallbackClip\(null\);/,
  );
  assert.match(source, /saved: true,\s*workflow,\s*\}\);\s*selectClip\(generated\.clip_id, true\);/);
});

test("deleting a clip drops the row without waiting on the refresh", () => {
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
  assert.match(
    source,
    /await setAudioClipFlags\(id, \{ archived: true \}\);[\s\S]*?dropClip\(id\);\s*await refreshGallery\(id\);/,
  );
});

test("the response fallback is dropped once its gallery record arrives", () => {
  assert.match(
    source,
    /fallbackClipRef\.current\?\.saved &&\s*galleryCache\.selectedId &&\s*merged\.some\(\(c\) => c\.id === galleryCache\.selectedId\)\s*\)\s*\{\s*setFallbackClip\(null\);/,
  );
});
