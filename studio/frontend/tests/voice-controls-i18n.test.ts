// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { en } from "../src/i18n/locales/en.ts";

const read = (relative: string) =>
  readFileSync(new URL(`../src/features/chat/${relative}`, import.meta.url), "utf8");

test("voice buttons and toasts go through i18n", () => {
  const sources = [
    read("voice/voice-engine.tsx"),
    read("hooks/use-tts-player.ts"),
    read("adapters/studio-voice-loop-adapter.ts"),
  ].join("\n");
  for (const text of Object.values(en.voiceControls)) {
    if (text.includes("{")) continue;
    assert.ok(!sources.includes(`"${text}"`), `hardcoded: ${text}`);
  }
  assert.ok(!sources.includes("· hold to float"));
});
