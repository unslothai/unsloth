// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const rail = readSrc("features/audio/pages/transcribe-page.tsx");
const host = readSrc("features/audio/audio-page.tsx");
const api = readSrc("features/audio/transcribe-api.ts");
const hook = readSrc("features/audio/hooks/use-transcription.ts");

test("the rail takes audio through the shared input card, kept as the tour's record target", () => {
  assert.match(
    rail,
    /<div data-tour="audio-record">\s*<AudioSourceInput[\s\S]*?maxRecordSeconds=\{RECORDING_MAX_SECONDS\}/,
  );
  assert.match(rail, /<AudioHistoryProvider value=\{historyClips\}>/);
  // The bare file input and the hook's own recorder are gone.
  assert.doesNotMatch(rail, /<input\b/);
  assert.doesNotMatch(
    hook,
    /createAudioRecorder|getUserMedia|handleRecordToggle/,
  );
});

test("both switches come from the model's capabilities, with their reason in visible text", () => {
  assert.match(rail, /label="Timestamps"\s*value=\{switches\.timestamps\}/);
  assert.match(rail, /label="Speakers"\s*value=\{switches\.speakers\}/);
  assert.match(rail, /disabled=\{disabled \|\| value\.disabled\}/);
  assert.match(rail, /id=\{`\$\{id\}-hint`\}[\s\S]*?\{value\.hint\}/);
  assert.match(host, /transcribeSwitches\(\s*transcribeCaps\.caps,/);
});

test("the footer says why Transcribe is off inline, and Stop replaces it while it runs", () => {
  assert.match(
    rail,
    /id="transcribe-blocker"[\s\S]*?\{blocker\.reason\}[\s\S]*?<GenerateActions actions=\{blocker\.actions\} \/>/,
  );
  assert.match(rail, /onClick=\{running \? onStop : onTranscribe\}/);
  assert.match(host, /reason: "Add audio to transcribe\."/);
  assert.match(host, /reason: "Pick a speech-to-text model to transcribe\."/);
  assert.match(
    host,
    /actions: \[chooseModelAction, \.\.\.recommendedSttActions\]/,
  );
});

test("Mod+Enter transcribes on Transcribe, and a run is announced and focused", () => {
  assert.match(
    host,
    /generateShortcut\.current = \(\) => \{\s*if \(canTranscribe\) handleTranscribe\(\);\s*else if \(canGenerate\)/,
  );
  assert.match(
    host,
    /document\.getElementById\("transcribe-result"\)\?\.focus\(\)/,
  );
  assert.match(
    host,
    /<output aria-live="polite" aria-atomic="true" className="sr-only">\s*\{transcribeAnnouncement\}/,
  );
});

test("runs send the source by id, never a path or the audio itself", () => {
  assert.match(api, /"\/api\/inference\/audio\/transcribe\/source"/);
  assert.match(api, /source: ref,/);
  assert.doesNotMatch(api, /\bpath:|"path"|audio_path|FormData|body: blob/);
  assert.match(hook, /transcribeSourceWithProgress\(\s*sourceRefOf\(source\),/);
  // A 404 on an upload marks the card expired rather than toasting a generic failure.
  assert.match(
    api,
    /throw new AudioApiError\(await readFastApiError\(response\), response\.status\)/,
  );
  assert.match(hook, /error\.status === 404 &&\s*source\.kind === "input"/);
});

test("a run shows its step and elapsed time under the button, model load included", () => {
  assert.match(rail, /const working = running \|\| busy === "loading";/);
  assert.match(rail, /`Loading \$\{modelName \|\| "the model"\}…`/);
  assert.match(rail, /"Downloading the timing aligner…"/);
  assert.match(
    rail,
    /<output\s+aria-live="polite"[\s\S]*?\{status\}[\s\S]*?formatClipDuration\(elapsed\)/,
  );
  // Picking a model only remembers it; the footer says it loads first.
  assert.match(host, /`Loads \$\{transcribeModelName\} first\.`/);
  assert.match(host, /: \(transcribeRepo \?\? undefined\);/);
});

test("an always-on setting reads as a fact and the card does not cap Transcribe's audio", () => {
  assert.match(
    rail,
    /\{value\.always \? \([\s\S]*?Always on[\s\S]*?\) : \(\s*<Switch/,
  );
  assert.match(rail, /usesFirstSeconds=\{null\}/);
  const card = readSrc("features/audio/components/audio-source-input.tsx");
  assert.match(card, /usesFirstSeconds = REFERENCE_MAX_SECONDS,/);
  assert.match(
    card,
    /usesFirstSeconds !== null &&\s*durationS !== null &&\s*durationS > usesFirstSeconds/,
  );
});

test("the run sends the language the rail shows, never a hidden saved one", () => {
  assert.match(rail, /transcribeLanguageFor\(language, languages\) \|\| AUTO/);
  assert.match(
    host,
    /const transcribeLanguage = transcribeLanguageFor\(\s*transcribePrefs\.language,\s*transcribeLanguages,\s*\)/,
  );
  assert.match(host, /language: transcribeLanguage,/);
  assert.doesNotMatch(host, /language: transcribePrefs\.language/);
});

test("the source card speaks for Transcribe, not for a clone reference", () => {
  assert.match(
    rail,
    /recordHint="Record up to 30 minutes, then transcribe it\."/,
  );
  assert.match(rail, /expiredMessage="This upload expired\. Add it again\."/);
  assert.match(rail, /usesFirstSeconds=\{null\}/);
});
