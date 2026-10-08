// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  geminiThoughtReplayPart,
  pinGeminiPartThoughtSignature,
  withGeminiThoughtReplayParts,
} from "../src/features/chat/gemini-thought-replay.ts";
import { readSrc } from "./helpers/kit.ts";

const READ_REPLAY_PART = /geminiThoughtReplayPart\(part\)/;
const WRITE_REPLAY_PARTS = /withGeminiThoughtReplayParts\(/;
const PIN_THOUGHT_SIGNATURE =
  /pinGeminiPartThoughtSignature\([\s\S]*latestGeminiThoughtSignature,[\s\S]*true,[\s\S]*\)/;
const PIN_TEXT_SIGNATURE =
  /pinGeminiPartThoughtSignature\([\s\S]*latestGeminiTextSignature,[\s\S]*false,[\s\S]*\)/;

test("a signed thought stays on the reasoning part", () => {
  const parts: Array<{
    type: string;
    text: string;
    _google_thought_signature?: string;
  }> = [
    { type: "reasoning", text: "consider the clues" },
    { type: "text", text: "the answer" },
  ];

  pinGeminiPartThoughtSignature(parts, "SIG-THOUGHT", true);
  pinGeminiPartThoughtSignature(parts, "SIG-ANSWER", false);

  assert.equal(parts[0]._google_thought_signature, "SIG-THOUGHT");
  assert.equal(parts[1]._google_thought_signature, "SIG-ANSWER");
  assert.deepEqual(geminiThoughtReplayPart(parts[0]), {
    text: "consider the clues",
    thoughtSignature: "SIG-THOUGHT",
  });
});

test("signed thoughts use a separate replay envelope from signed answer text", () => {
  const extra = withGeminiThoughtReplayParts(
    { google: { thought_signature: "SIG-ANSWER" } },
    [{ text: "consider the clues", thoughtSignature: "SIG-THOUGHT" }],
  );

  assert.deepEqual(extra, {
    google: {
      thought_signature: "SIG-ANSWER",
      thought_parts: [
        { text: "consider the clues", thought_signature: "SIG-THOUGHT" },
      ],
    },
  });
});

test("the chat adapter retains thought and answer signatures independently", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(adapter, READ_REPLAY_PART);
  assert.match(adapter, WRITE_REPLAY_PARTS);
  assert.match(adapter, PIN_THOUGHT_SIGNATURE);
  assert.match(adapter, PIN_TEXT_SIGNATURE);
});
