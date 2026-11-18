// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Turn ending has three lanes: the fixed silence wait (turn detection off), Smart Turn alone (the
// default), and Smart Turn with the text judge agreeing (smartTurnJudge, fused by turn-fuser).

import assert from "node:assert/strict";
import test from "node:test";

import { DEFAULT_VOICE_TURN_ENDING, VOICE_TURN_ENDINGS } from "../src/features/settings/stores/stt-model-catalog.ts";

test("Smart Turn is the default; the judge lane and the fixed pause stay choices", () => {
  assert.equal(DEFAULT_VOICE_TURN_ENDING, "smartTurn");
  assert.deepEqual([...VOICE_TURN_ENDINGS], ["smartTurn", "smartTurnJudge", "wait"]);
});
