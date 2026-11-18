// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { isRevealPrefix, isTypedVoiceSubmit } from "../src/features/chat/voice/typed-send.ts";

test("Enter mid-reveal is not a typed send of the cut transcript", () => {
  const revealing = "What is the weather like today?";
  assert.equal(isRevealPrefix("What is the weath", revealing), true);
  assert.equal(isRevealPrefix("What is ", revealing), true);
  assert.equal(
    isTypedVoiceSubmit({ box: "What is the weath", shown: "", paused: false, revealing }),
    false,
  );
});

test("typing something else during a reveal is still a typed send", () => {
  const revealing = "What is the weather like today?";
  assert.equal(isRevealPrefix("Actually never mind", revealing), false);
  assert.equal(
    isTypedVoiceSubmit({ box: "Actually never mind", shown: "", paused: false, revealing }),
    true,
  );
  assert.equal(isRevealPrefix("What is the weath", null), false);
  assert.equal(isTypedVoiceSubmit({ box: "What is the weath", shown: "", paused: false }), true);
});
