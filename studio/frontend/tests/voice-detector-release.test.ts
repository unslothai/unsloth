// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const chatPage = readSrc("features/chat/chat-page.tsx");
const sessionStore = readSrc("features/chat/voice/voice-session-store.ts");

test("Chat hands voice mode changes and its own unmount to the voice session", () => {
  assert.match(chatPage, /useEffect\(\(\) => \{\s*voiceSession\.setMode\(voiceMode\);\s*\}, \[voiceMode, voiceSession\]\);/);
  assert.match(chatPage, /useEffect\(\(\) => \{\s*voiceSession\.activate\(\);\s*return \(\) => voiceSession\.dispose\(\);\s*\}, \[voiceSession\]\);/);
});

test("the session's listening release is the store one", () => {
  assert.match(sessionStore, /releaseListening: releaseVoiceModels,/);
});

test("releasing voice models includes the transcriber and the speech detector", () => {
  assert.match(
    sessionStore,
    /function releaseVoiceModels\(\): void \{\s*void unloadSttModel\(\)\.catch\(\(\) => \{\}\);\s*void unloadVadModel\(\);/,
  );
});
