// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Phase 3: a cacheable voice prompt. The instruction is fixed text appended last, voice sends no
// seed and the lowest reasoning effort, and none of it applies outside voice mode.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  CONVERSATION_INSTRUCTION,
  SPOKEN_REPLY_INSTRUCTIONS,
  VOICE_INSTRUCTION,
  withReplyInstruction,
  withVoiceInstruction,
} from "../src/features/chat/voice/voice-instruction.ts";

const adapter = readFileSync(new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url), "utf8");

test("the voice instruction comes last, after everything the chat already had", () => {
  const prompt = "You are a helpful assistant.\n\nProject notes.";
  const out = withVoiceInstruction(prompt);
  assert.ok(out.startsWith(prompt));
  assert.ok(out.endsWith(VOICE_INSTRUCTION));
  assert.equal(withVoiceInstruction(""), VOICE_INSTRUCTION);
  assert.equal(withVoiceInstruction("Be brief.  \n"), `Be brief.\n\n${VOICE_INSTRUCTION}`);
  // Fixed text: the same prompt gives the same bytes every turn, so the prefix cache holds.
  assert.equal(withVoiceInstruction(prompt), out);
  assert.doesNotMatch(VOICE_INSTRUCTION, /\d{4}|\{|\}/);
});

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

test("only voice mode changes the request, and no seed or low effort with no instruction", () => {
  assert.match(adapter, /if \(getVoiceMode\(\) !== "active"\) return instructions;/);
  assert.match(
    adapter,
    /function voiceReplyShaping\(\): boolean \{\s*return getVoiceMode\(\) === "active" && useVoiceSettingsStore\.getState\(\)\.replyInstruction !== "none";\s*\}/,
  );
});

test("the instruction goes only to the model, never into copies or exports", () => {
  assert.match(adapter, /return opts\?\.agentsMd \? withVoiceReplyShaping\(instructions\) : instructions;/);
  assert.match(adapter, /return withReplyInstruction\(instructions, replyInstruction, replyInstructionCustom\);/);
  // The token count builds the same prompt, so it counts the instruction too.
  assert.match(adapter, /const combinedSystemPrompt = withVoiceReplyShaping\(\s*composeChatInstructions\(\{/);
  // Both model-bound callers ask for it; the prompt-storage copy does not.
  assert.equal(adapter.match(/readThreadRecord,\s*\{ agentsMd: true \},/g)?.length, 2);
});

test("no seed is sent in voice mode", () => {
  assert.match(
    adapter,
    /\.\.\.\(params\.seed == null \|\|\s*voiceReplyShaping\(\) \|\|\s*!modelReadsSamplingSeed\(activeModel, runtime\.loadedEngine\)\s*\? \{\}\s*: \{ seed: params\.seed \}\)/,
  );
});

test("voice asks for the lowest reasoning effort wherever the effort is clamped", () => {
  assert.match(
    adapter,
    /function requestReasoningEffort<T extends string>\(effort: T, levels: readonly T\[\]\): T \{\s*return voiceReplyShaping\(\) && levels\.length > 0 \? levels\[0\] : effort;/,
  );
  const clamps = adapter.match(/clampReasoningEffortToLevels\(\s*([^,]+),/g) ?? [];
  assert.equal(clamps.length, 3);
  for (const clamp of clamps) assert.match(clamp, /\(\s*requestReasoningEffort\(reasoningEffort,$/);
});

test("the conversation prompt covers misheard words, tools and thinking, and stays fixed text", () => {
  assert.match(CONVERSATION_INSTRUCTION, /misheard/);
  assert.match(CONVERSATION_INSTRUCTION, /never say tool names/);
  assert.match(CONVERSATION_INSTRUCTION, /not to any thinking before it\.$/);
  assert.doesNotMatch(CONVERSATION_INSTRUCTION, /\{|\}/);
});
