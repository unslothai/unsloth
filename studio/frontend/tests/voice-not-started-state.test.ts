// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const read = (path: string) => readFileSync(new URL(path, import.meta.url), "utf8");
const panel = read("../src/components/assistant-ui/voice-panel.tsx");

test("before voice mode starts the panel says so instead of warning that nothing is loaded", () => {
  // "0 ready" over five amber "Not loaded" rows read as a failure with no way forward.
  assert.match(panel, /const started = useChatRuntimeStore\(\(s\) => s\.voiceMode === "active"\);/);
  assert.match(panel, /: !started\s*\? t\("voicePanel\.notStarted"\)/);
  assert.match(panel, /idle=\{row\.unset \|\| \(!started && row\.state === "not_loaded"\)\}/);
  assert.match(panel, /: !started && row\.state === "not_loaded"\s*\? "voicePanel\.state\.loadsOnStart"/);
});

test("the panel offers the start action until voice mode is running", () => {
  const start = panel.slice(panel.indexOf("const start = () => {"), panel.indexOf("const ttsState"));
  assert.match(start, /store\.setVoiceMode\("active"\);/);
  assert.match(panel, /\{!started && \([\s\S]*?t\("voicePanel\.startHint"\)[\s\S]*?onClick=\{start\}/);
});

test("built-in classifiers with no source read as not set up, not as an error", () => {
  // A default install has none, and voice mode works without them: red "Error" sent users hunting.
  assert.match(
    panel,
    /const decideUnset =\s*decision\.status\?\.current === null &&\s*decision\.status\.builtin_configured === false;/,
  );
  assert.match(panel, /rows\[r\]\.unset \? null : rows\[r\]\.state/);
  assert.match(panel, /row\.unset\s*\? "voicePanel\.state\.notSetUp"/);
});
