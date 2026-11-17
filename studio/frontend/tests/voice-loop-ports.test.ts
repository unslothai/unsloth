// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const loop = readSrc("features/chat/adapters/studio-voice-loop-adapter.ts");
const engine = readSrc("features/chat/voice/voice-engine.tsx");

test("the voice loop reaches the chat runtime and voice settings only through its ports", () => {
  assert.doesNotMatch(loop, /useChatRuntimeStore|useVoiceSettingsStore|chat-runtime-store|voice-settings-store/);
});

test("a mic request the browser never answers frees the open queue once its session ends", () => {
  assert.match(loop, /const opened = await Promise\.race\(\[opening, endedPromise\.then\(\(\) => null\)\]\);/);
  assert.match(loop, /if \(!opened \|\| ended\) \{\s*void opening\.then\(\s*\(late\) => late\.getTracks\(\)\.forEach\(\(t\) => t\.stop\(\)\),/);
});

test("the engine sets the loop's ports before it registers the listen factory", () => {
  assert.match(engine, /setVoiceLoopPorts\(storeVoiceLoopPorts\);\s*registerVoiceListen\(/);
});
