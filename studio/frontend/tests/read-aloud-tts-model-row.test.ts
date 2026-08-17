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

test("the studio TTS row offers the Audio page it tells the user to use", () => {
  // Settings is a modal over the current route, so it has to close or the Audio page
  // opens behind it.
  assert.match(
    source,
    /label=\{t\("settings\.voice\.readAloud\.modelLabel"\)\}[\s\S]*?useSettingsDialogStore\.getState\(\)\.closeDialog\(\);\s*void navigate\(\{ to: "\/audio", search: \{\} \}\);/,
  );
  assert.match(source, /t\("settings\.voice\.readAloud\.openAudioAction"\)/);
  assert.match(en, /openAudioAction: "Open Audio",/);
});

test("a studio preview shows the generate wait instead of an idle button", () => {
  // Showing Stop for the generate wait read as nothing happening, and each extra click
  // orphaned a request that still counted against the next model load.
  assert.match(
    source,
    /markPreviewing\(true\);\s*setPreparingPreview\(true\);\s*try \{\s*const url = await generateStudioTtsAudio\(/,
  );
  assert.match(
    source,
    /if \(controller\.signal\.aborted\) return;\s*setPreparingPreview\(false\);/,
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
