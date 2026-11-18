// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { fuseTurn, } from "../src/features/chat/voice/turn-fuser.ts";

test("a turn is confirmed only when the text and the audio both say it ended", () => {
  assert.equal(fuseTurn({ pText: 0.9, pAudio: 0.9 }), "confirmed");
  assert.equal(fuseTurn({ pText: 0.65, pAudio: 0.5 }), "confirmed");
  assert.equal(fuseTurn({ pText: 0.64, pAudio: 0.99 }), "today");
  assert.equal(fuseTurn({ pText: 0.99, pAudio: 0.49 }), "today");
});

test("with no judge the audio decides alone; with no audio score today's path stays", () => {
  assert.equal(fuseTurn({ pText: null, pAudio: 0.99 }), "confirmed");
  assert.equal(fuseTurn({ pText: null, pAudio: 0.49 }), "today");
  assert.equal(fuseTurn({ pText: 0.99, pAudio: null }), "today");
  assert.equal(fuseTurn({ pText: null, pAudio: null }), "today");
});

test("thresholds can be passed in", () => {
  assert.equal(fuseTurn({ pText: 0.5, pAudio: 0.3 }, { text: 0.4, audio: 0.2 }), "confirmed");
});
