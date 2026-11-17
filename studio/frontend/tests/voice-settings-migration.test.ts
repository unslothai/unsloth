// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Voice settings v2: the latency rollout switches, filled in for a v1 save. v3: turn ending and the
// spoken-reply instruction became choices, and a v2 save takes their defaults.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  DEFAULT_STT_MODEL,
  DEFAULT_VOICE_REPLY_INSTRUCTION,
  DEFAULT_VOICE_TURN_ENDING,
  VOICE_ROLLOUT_DEFAULTS,
  migrateVoiceSettings,
} from "../src/features/settings/stores/stt-model-catalog.ts";

const store = readFileSync(
  new URL("../src/features/settings/stores/voice-settings-store.ts", import.meta.url),
  "utf8",
);

test("there is one switch per latency phase", () => {
  assert.deepEqual(Object.keys(VOICE_ROLLOUT_DEFAULTS).sort(), [
    "clauseFirstChunk",
    "heardCut",
    "speechNormalize",
  ]);
});

test("a v1 save gains the switches at their defaults and keeps everything else", () => {
  const saved = { sttModel: "large-v3", ttsEngine: "studio", voiceLogEnabled: true };
  assert.deepEqual(migrateVoiceSettings(saved, 1), { ...VOICE_ROLLOUT_DEFAULTS, ...saved });
});

test("a v1 save that somehow already has a switch keeps its value", () => {
  assert.equal(migrateVoiceSettings({ heardCut: true }, 1)?.heardCut, true);
});

test("a v2 save drops the old turn and reply switches, so the new choices take their defaults", () => {
  const migrated = migrateVoiceSettings({ smartTurn: false, smartTurnJudge: false, replyShaping: false, heardCut: true }, 2);
  assert.deepEqual(migrated, { heardCut: true });
  assert.equal(DEFAULT_VOICE_TURN_ENDING, "smartTurn");
  assert.equal(DEFAULT_VOICE_REPLY_INSTRUCTION, "conversation");
});

test("a v0 save is migrated through both versions", () => {
  const migrated = migrateVoiceSettings({ sttModel: "small" }, 0);
  assert.equal(migrated?.sttModel, DEFAULT_STT_MODEL);
  assert.equal(migrated?.clauseFirstChunk, VOICE_ROLLOUT_DEFAULTS.clauseFirstChunk);
});

test("a v3 save drops its turn ending, so the Smart Turn default applies", () => {
  assert.deepEqual(migrateVoiceSettings({ turnEnding: "smartTurnJudge", heardCut: true }, 3), { heardCut: true });
});

test("a v4 save and an empty store are left alone", () => {
  const saved = { clauseFirstChunk: true };
  assert.equal(migrateVoiceSettings(saved, 4), saved);
  assert.equal(migrateVoiceSettings(undefined, 1), undefined);
});

test("the store persists at v4 and reads each switch as a boolean or its default", () => {
  assert.match(store, /version: 4,\s*migrate: migrateVoiceSettings,/);
  // A saved choice outside the list, or a bar out of range, reads as the default.
  assert.match(store, /turnEnding: VOICE_TURN_ENDINGS\.includes\(/);
  assert.match(store, /replyInstruction: VOICE_REPLY_INSTRUCTIONS\.includes\(/);
  assert.match(store, /turnJudgeBar: clampNumber\(saved\?\.turnJudgeBar, TURN_JUDGE_BAR\.min, TURN_JUDGE_BAR\.max, TURN_JUDGE_BAR\.default\)/);
  assert.match(store, /\.\.\.rolloutSettings\(saved\),/);
  assert.match(store, /if \(typeof value === "boolean"\) out\[key\] = value;/);
  // Settings > Voice > Voice mode sets each switch through this one setter.
  assert.match(store, /setVoiceRollout: \(setting, on\) => set\(\{ \[setting\]: on \}\),/);
});

test("Settings > Voice > Voice mode has a labelled switch for every setting", async () => {
  const ui = readFileSync(new URL("../src/features/settings/components/voice-mode-switches.tsx", import.meta.url), "utf8");
  const section = readFileSync(new URL("../src/features/settings/components/voice-decision-log-section.tsx", import.meta.url), "utf8");
  const { en } = await import("../src/i18n/locales/en.ts");
  assert.match(section, /<VoiceModeSwitches \/>/);
  for (const key of Object.keys(VOICE_ROLLOUT_DEFAULTS)) {
    assert.ok(ui.includes(`setting="${key}"`), key);
    const entry = (en.settings.voice.voiceMode.switches as Record<string, { label: string; description: string }>)[key];
    assert.ok(entry?.label && entry.description, key);
  }
  assert.match(ui, /onCheckedChange=\{\(value\) => setVoiceRollout\(setting, value\)\}/);
  // Every choice has a label and a description.
  const mode = en.settings.voice.voiceMode;
  for (const option of ["smartTurn", "smartTurnJudge", "wait"] as const) {
    assert.ok(mode.turnEnding.options[option].label && mode.turnEnding.options[option].description, option);
  }
  for (const option of ["conversation", "spoken", "voice", "custom", "none"] as const) {
    assert.ok(mode.replyInstruction.options[option].label && mode.replyInstruction.options[option].description, option);
  }
  assert.match(ui, /<TurnEndingRow \/>/);
  assert.match(ui, /<ReplyInstructionRow \/>/);
});

test("the defaults: heard cut off, the rest on", () => {
  assert.deepEqual(VOICE_ROLLOUT_DEFAULTS, {
    clauseFirstChunk: true,
    speechNormalize: true,
    heardCut: false,
  });
});
