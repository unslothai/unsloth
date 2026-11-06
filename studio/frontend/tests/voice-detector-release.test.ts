// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const chatPage = readSrc("features/chat/chat-page.tsx");

test("voice mode going off releases the speech detector, which has no idle unload", () => {
  assert.match(
    chatPage,
    /function releaseVoiceDetector\(\): void \{\s*void unloadVadModel\(\);\s*useChatRuntimeStore\.getState\(\)\.setVadSlotLoaded\(false\);/,
  );
  // Only on the change to off: a page that mounts with voice off sends no unload.
  assert.match(chatPage, /if \(was !== "off"\) releaseVoiceDetector\(\);/);
});

test("leaving Chat with voice mode on releases the speech detector too", () => {
  assert.match(
    chatPage,
    /voiceSlot\.close\(\);\s*if \(prevVoiceModeRef\.current !== "off"\) releaseVoiceDetector\(\);/,
  );
});
