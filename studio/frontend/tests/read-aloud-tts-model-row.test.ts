// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(
  new URL("../src/features/settings/tabs/voice-tab.tsx", import.meta.url),
  "utf8",
);
const en = readFileSync(
  new URL("../src/i18n/locales/en.ts", import.meta.url),
  "utf8",
);
const audioSource = readFileSync(
  new URL("../src/features/audio/audio-page.tsx", import.meta.url),
  "utf8",
);

test("the studio TTS row offers the Audio page it tells the user to use", () => {
  // Settings is a modal, so it must close or Audio opens behind it.
  assert.match(
    source,
    /label=\{t\("settings\.voice\.readAloud\.modelLabel"\)\}[\s\S]*?useSettingsDialogStore\.getState\(\)\.closeDialog\(\);[\s\S]{0,200}?void navigate\(\{\s*to: "\/audio",/,
  );
  assert.match(source, /t\("settings\.voice\.readAloud\.openAudioAction"\)/);
  assert.match(en, /openAudioAction: "Open Audio",/);
});

test("the row lands on the TTS selector, not the mode Audio was left in", () => {
  // AudioPage stays mounted and keeps its `mode`, so plain /audio can show the wrong selector.
  assert.match(
    source,
    /to: "\/audio",\s*search: \{ task: "text-to-speech" \},/,
  );
  // Audio ignored a task without a model, so the intent needs handling at the other end.
  assert.match(
    audioSource,
    /const task = routeSearch\.task;\s*if \(!task\) return;\s*const intended =\s*task === "automatic-speech-recognition" \? "transcribe" : "speak";/,
  );
  // A refused switch keeps the parameter, so the retry rides the effect's busy dep.
  assert.match(
    audioSource,
    /if \(intended !== mode && !transitionMode\(intended\)\) return;\s*void navigateSelf\(\{ to: "\/audio", search: \{\}, replace: true \}\);/,
  );
});

test("custom TTS explains the strict voice default", () => {
  assert.match(
    en,
    /customVoiceDescription: "Voice name the endpoint expects; defaults to alloy",/,
  );
  assert.doesNotMatch(en, /customVoiceDescription: .*optional/i);
});

test("a studio preview shows the generate wait instead of an idle button", () => {
  // Stop during the generate wait read as idle, and extra clicks orphaned requests.
  assert.match(
    source,
    /markPreviewing\(true\);\s*setPreparingPreview\(true\);\s*try \{\s*const generate =[\s\S]*?const url = await generate\(/,
  );
  assert.match(
    source,
    /if \(controller\.signal\.aborted\) \{\s*releaseTtsAudioUrl\(url\);\s*return;\s*\}\s*setPreparingPreview\(false\);/,
  );
  // Every other exit goes through markPreviewing, so clearing there covers them all.
  assert.match(
    source,
    /previewingRef\.current = value;\s*setPreviewing\(value\);[\s\S]*?if \(!value\) setPreparingPreview\(false\);/,
  );
  // Preparing outranks previewing: both are true during the wait.
  assert.match(
    source,
    /\{preparingPreview \? \([\s\S]*?<Spinner[\s\S]*?preparingAction"\)\}[\s\S]*?\) : previewing \? \(/,
  );
  assert.match(en, /preparingAction: "Generating…",/);
});
