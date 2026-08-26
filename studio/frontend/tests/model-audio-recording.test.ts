// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const composer = readFileSync(
  new URL("../src/features/chat/shared-composer.tsx", import.meta.url),
  "utf8",
);
const adapter = readFileSync(
  new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  "utf8",
);

test("model-audio recorder is visible only for audio-input models", () => {
  assert.match(composer, /activeModel\?\.hasAudioInput === true && \(/);
  assert.match(composer, /aria-label="Record audio for model"/);
});

test("stopping a recording creates the existing pending-audio attachment", () => {
  assert.match(composer, /createAudioRecorder\(stream\)/);
  assert.match(composer, /new File\(chunks, recordedAudioName\(contentType\), \{/);
  assert.match(composer, /contentType === "audio\/wav"\) \{[\s\S]*return "recording\.wav"/);
  assert.match(composer, /const base64 = await fileToBase64\(file\)/);
  assert.match(composer, /setPendingAudio\(\{[\s\S]*contentType: file\.type/);
  assert.match(composer, /setPendingAudioStore\(base64, file\.name\)/);
});

test("cancelling a recording neither attaches nor sends audio", () => {
  const cancel = composer.slice(
    composer.indexOf("const cancel = useCallback"),
    composer.indexOf("const start = useCallback"),
  );
  assert.match(cancel, /cancelledRef\.current = true/);
  assert.match(cancel, /recorder\.stop\(\)/);
  assert.match(cancel, /cleanup\(\)/);
  assert.match(composer, /if \(wasCancelled\) \{[\s\S]*return/);
});

test("capture failures and oversized clips are reported and release resources", () => {
  assert.match(composer, /toast\.error\("Could not start audio recording"/);
  assert.match(composer, /getAudioSizeError\(MAX_AUDIO_SIZE \+ 1\)/);
  assert.match(composer, /stopMicrophone\(streamRef\.current\)/);
  assert.match(composer, /mountedRef\.current = false;[\s\S]*cancel\(\)/);
});

test("recorded pending audio uses the established audio-base64 request path", () => {
  assert.match(composer, /audio: `data:\$\{submittedAudio\.contentType\};base64,\$\{submittedAudio\.base64\}`/);
  assert.match(adapter, /audio_base64: audioBase64/);
});
