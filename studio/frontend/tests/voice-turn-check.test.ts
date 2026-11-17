// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Turn ending has three lanes: the fixed silence wait (turn detection off), Smart Turn alone (the
// default), and Smart Turn with the text judge agreeing (smartTurnJudge, fused by turn-fuser).

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { TURN_THRESHOLDS, fuseTurn } from "../src/features/chat/voice/turn-fuser.ts";
import { DEFAULT_VOICE_TURN_ENDING, VOICE_TURN_ENDINGS } from "../src/features/settings/stores/stt-model-catalog.ts";

const src = (path: string) => readFileSync(new URL(`../src/features/${path}`, import.meta.url), "utf8");
const adapter = src("chat/adapters/studio-voice-loop-adapter.ts");
const engine = src("chat/voice/voice-engine.tsx");

test("Smart Turn is the default; the judge lane and the fixed pause stay choices", () => {
  assert.equal(DEFAULT_VOICE_TURN_ENDING, "smartTurn");
  assert.deepEqual([...VOICE_TURN_ENDINGS], ["smartTurn", "smartTurnJudge", "wait"]);
});

test("Settings lets the owner pick the decision model, and None is a choice", () => {
  const ui = src("settings/components/voice-mode-switches.tsx");
  assert.match(ui, /<DecisionModelRow \/>/);
  assert.match(ui, /setVoiceDecision\(value === NO_DECISION_MODEL \? null : value\)/);
  // Only a pick of your own selects a model; without one the "None" row names what decides instead.
  assert.match(ui, /const current = !status \|\| status\.source !== "setting" \? NO_DECISION_MODEL : \(status\.current \?\? NO_DECISION_MODEL\);/);
  assert.match(ui, /t\("voicePanel\.builtinClassifiers"\)/);
  assert.match(ui, /current === NO_DECISION_MODEL && !decidesWithoutPick/);
  // A non-owner cannot read or change the installation-wide pick.
  assert.match(ui, /disabled=\{failed \|\| !status \|\| saving\}/);
});

test("turn detection off never loads the end-of-turn model, so the silence wait alone ends a turn", () => {
  assert.match(
    adapter,
    /if \(useVoiceSettingsStore\.getState\(\)\.turnEnding !== "wait"\) \{\s*void loadTurnModel\(\)\.then\(\(ready\) => \{\s*turnReady = ready;/,
  );
  // askTurn needs turnReady: no model, no early end.
  assert.match(adapter, /const askTurn = turnReady && !bargedIn;/);
});

test("Smart Turn's last answer is kept for the judge lane", () => {
  assert.match(adapter, /turnP = answer\.p;/);
});

test("a turn that sounded finished asks the judge only in the judge lane", () => {
  assert.match(
    adapter,
    /\(soundedFinished\s*\? turnEnding === "smartTurnJudge"\s*\? await turnUnfinished\(transcript, modelId, turnP, turnJudgeBar\)\s*: readsCutOff\(transcript, modelId\)\s*: await turnUnfinished\(transcript, modelId\)\)/,
  );
  // The judge lane fuses both scores; a missing Smart Turn score asks the judge alone.
  assert.match(
    adapter,
    /soundedP === null\s*\? !r\.finished\s*: fuseTurn\(\{ pText: r\.p, pAudio: soundedP \}, \{ text: judgeBar, audio: TURN_THRESHOLDS\.audio \}\) !== "confirmed"/,
  );
});

test("the fuser needs both scores over their bars", () => {
  assert.deepEqual(TURN_THRESHOLDS, { text: 0.65, audio: 0.5 });
  assert.equal(fuseTurn({ pText: 0.9, pAudio: 0.9 }), "confirmed");
  assert.equal(fuseTurn({ pText: 0.5, pAudio: 0.9 }), "today");
  assert.equal(fuseTurn({ pText: 0.9, pAudio: 0.3 }), "today");
  assert.equal(fuseTurn({ pText: null, pAudio: 0.9 }), "confirmed");
  assert.equal(fuseTurn({ pText: 0.9, pAudio: null }), "today");
});

test("a turn that sounded finished skips only the plain coalesce window", () => {
  assert.match(adapter, /submitLater\(transcript, \{ turnId: turnId \?\? undefined, confirmedEarly: soundedFinished \}\)/);
  assert.match(
    engine,
    /holdOn\s*\? VOICE_HOLD_COALESCE_MS\s*: cutRun\s*\? VOICE_RESTART_COALESCE_MS\s*: confirmedEarly\s*\? 0\s*: VOICE_COALESCE_MS/,
  );
  assert.match(engine, /commitUtterance\(text, meta\?\.confirmedEarly\);/);
});

test("the settings page offers each lane, and the judge bar only in the judge lane", () => {
  const ui = src("settings/components/voice-mode-switches.tsx");
  assert.match(ui, /VOICE_TURN_ENDINGS\.map\(/);
  assert.match(ui, /\{ending === "smartTurnJudge" \? \(/);
  assert.match(ui, /setTurnJudgeBar\(values\[0\]\)/);
});
