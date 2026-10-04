// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const rail = readSrc("features/audio/pages/transcribe-page.tsx");
const host = readSrc("features/audio/audio-page.tsx");
const api = readSrc("features/audio/transcribe-api.ts");
const hook = readSrc("features/audio/hooks/use-transcription.ts");
const view = readSrc("features/audio/components/transcript-view.tsx");
const gallery = readSrc("features/audio/transcript-gallery.tsx");

test("the rail takes audio through the shared input card, worded for Transcribe and uncapped", () => {
  assert.match(
    rail,
    /<AudioHistoryProvider value=\{historyClips\}>\s*<div data-tour="audio-record">\s*<AudioSourceInput[\s\S]*?maxRecordSeconds=\{RECORDING_MAX_SECONDS\}\s*expiredMessage="This upload expired\. Add it again\."\s*usesFirstSeconds=\{null\}\s*recordHint="Record up to 30 minutes, then transcribe it\."/,
  );
});

test("both switches come from the model's capabilities, an always-on one reads as a fact", () => {
  assert.match(host, /transcribeSwitches\(\s*transcribeCaps\.caps,/);
  assert.match(
    rail,
    /\(\["timestamps", "speakers"\] as const\)\.map[\s\S]*?value=\{switches\[name\]\}/,
  );
  assert.match(
    rail,
    /\{value\.always \? \([\s\S]*?Always on[\s\S]*?\) : \(\s*<Switch/,
  );
});

test("the footer says why Transcribe is off, Stop replaces it, Mod+Enter runs it", () => {
  assert.match(
    rail,
    /id=\{blocker \? "transcribe-blocker" : "transcribe-notice"\}[\s\S]*?\{blocker \? blocker\.reason : notice\}\s*<GenerateActions actions=\{blocker\?\.actions\} \/>/,
  );
  assert.match(rail, /onClick=\{running \? onStop : onTranscribe\}/);
  assert.match(
    host,
    /generateShortcut\.current = \(\) => \{\s*if \(canTranscribe\) handleTranscribe\(\);\s*else if \(canGenerate\)/,
  );
  assert.match(
    host,
    /document\.getElementById\("transcribe-result"\)\?\.focus\(\)/,
  );
});

test("runs send the source by id and the language the rail shows; a 404 upload marks the card expired", () => {
  assert.doesNotMatch(api, /\bpath:|"path"|audio_path|FormData|body: blob/);
  assert.match(hook, /transcribeSourceWithProgress\(\s*sourceRefOf\(source\),/);
  assert.match(
    api,
    /throw new AudioApiError\(await readFastApiError\(response\), response\.status\)/,
  );
  assert.match(
    hook,
    /error\.status === 404 &&\s*source\.kind === "input"\s*\)\s*\{\s*onSourceExpired\?\.\(\);/,
  );
  assert.match(
    host,
    /setExpiredTranscribeSourceId\(source\.id\);\s*transcribeSourceHandle\.current\?\.markExpired\(\);/,
  );
  assert.match(rail, /transcribeLanguageFor\(language, languages\) \|\| AUTO/);
  assert.match(host, /\{ language: transcribeLanguage, \.\.\.request \}/);
});

test("timestamps seek once the audio loaded, nothing autoplays, the playing row is followed", () => {
  assert.match(view, /const player = ready \? control : null;/);
  assert.match(
    view,
    /aria-label=\{`Play from \$\{stamp\}`\}\s*onClick=\{\(\) => player\.current\?\.seek\(segment\.start, true\)\}/,
  );
  assert.match(
    view,
    /controller\.abort\(\);\s*if \(url\) URL\.revokeObjectURL\(url\);/,
  );
  assert.doesNotMatch(view + rail, /autoPlay|\.play\(/);
  assert.match(
    view,
    /if \(!playing \|\| active < 0 \|\| shown !== "segments"\) return;/,
  );
  assert.match(view, /onRename\(id, sanitizeSpeakerName\(draft\)\)/);
});

test("both export menus share the timestamp gate, and a download marks only the transcript it saved", () => {
  assert.equal(rail.match(/<TranscriptExportItems/g)?.length, 1);
  assert.equal(gallery.match(/<TranscriptExportItems/g)?.length, 1);
  assert.match(
    hook,
    /const version = transcriptVersion\.current;[\s\S]*?await downloadTranscript\(format,[\s\S]*?names: speakerNames,[\s\S]*?transcriptVersion\.current === version\s*\)\s*setTranscriptExported\(true\);/,
  );
  assert.match(
    gallery,
    /full = await getTranscript\(record\.id\);[\s\S]*?await downloadTranscript\(format, \{[\s\S]*?details: detailsFrom\(full\),\s*names: full\.speaker_names \?\? \{\},/,
  );
  assert.doesNotMatch(gallery, /downloadFile|record\.text, record\.title/);
});
