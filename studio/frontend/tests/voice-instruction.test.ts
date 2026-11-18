// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Phase 3: a cacheable voice prompt. The instruction is fixed text appended last, voice sends no
// seed and the lowest reasoning effort, and none of it applies outside voice mode.

import assert from "node:assert/strict";
import test from "node:test";

import {
  CONVERSATION_INSTRUCTION,
  SPOKEN_REPLY_INSTRUCTIONS,
  VOICE_INSTRUCTION,
  withReplyInstruction,
  } from "../src/features/chat/voice/voice-instruction.ts";

test("each spoken-reply choice adds its text last, and none or a blank custom adds nothing", () => {
  const base = "Be kind.";
  assert.equal(withReplyInstruction(base, "conversation", ""), `Be kind.\n\n${CONVERSATION_INSTRUCTION}`);
  assert.equal(withReplyInstruction(base, "spoken", ""), `Be kind.\n\n${SPOKEN_REPLY_INSTRUCTIONS}`);
  assert.equal(withReplyInstruction(base, "voice", ""), `Be kind.\n\n${VOICE_INSTRUCTION}`);
  assert.equal(withReplyInstruction(base, "custom", "  One sentence only.  "), "Be kind.\n\nOne sentence only.");
  assert.equal(withReplyInstruction(base, "custom", "   "), base);
  assert.equal(withReplyInstruction(base, "none", "ignored"), base);
  assert.equal(withReplyInstruction("", "voice", ""), VOICE_INSTRUCTION);
});

test("the conversation prompt covers misheard words, tools and thinking, and stays fixed text", () => {
  assert.match(CONVERSATION_INSTRUCTION, /misheard/);
  assert.match(CONVERSATION_INSTRUCTION, /never say tool names/);
  assert.match(CONVERSATION_INSTRUCTION, /not to any thinking before it\.$/);
  assert.doesNotMatch(CONVERSATION_INSTRUCTION, /\{|\}/);
});
