// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type GeminiThoughtReplayPart,
  type PositionedGeminiThoughtReplayPart,
  appendGeminiThoughtReplayPart,
  collectGeminiThoughtReplayParts,
  continuationGeminiReplayTurns,
  geminiContinuationReplayEntries,
  geminiThoughtReplayParts,
  pinGeminiTextThoughtSignature,
  pinGeminiThoughtReplayParts,
  readGeminiContinuationReplay,
  withGeminiThoughtReplayParts,
} from "../src/features/chat/gemini-thought-replay.ts";
import { readSrc } from "./helpers/kit.ts";

const READ_REPLAY_PARTS = /geminiThoughtReplayParts\(part\)/;
const WRITE_REPLAY_PARTS = /withGeminiThoughtReplayParts\(/;
const PIN_THOUGHT_PARTS = /pinGeminiThoughtReplayParts\(/;
const CAPTURE_THOUGHT_PARTS =
  /appendGeminiThoughtReplayPart\([\s\S]{0,220}toolCallParts\.length/;
const FLUSH_BEFORE_REPLAY_CAPTURE =
  /if \(part\.type === "reasoning"\) \{[\s\S]{0,180}flushAssistantAndToolResults\(\);[\s\S]{0,180}pendingGeminiThoughtParts\.push/;
const PIN_TEXT_SIGNATURE =
  /pinGeminiTextThoughtSignature\([\s\S]*latestGeminiTextSignature,[\s\S]*\)/;

test("every signed thought stays separate from signed answer text", () => {
  const parts: Array<{
    type: string;
    text?: string;
    _google_thought_signature?: string;
    _google_thought_parts?: GeminiThoughtReplayPart[];
  }> = [
    { type: "reasoning", text: "first thought" },
    { type: "tool-call" },
    { type: "reasoning", text: "second thought" },
    { type: "text", text: "the answer" },
  ];
  const thoughtParts: PositionedGeminiThoughtReplayPart[] = [];
  appendGeminiThoughtReplayPart(thoughtParts, "first ", "SIG-THOUGHT-1", 0);
  appendGeminiThoughtReplayPart(thoughtParts, "thought", "SIG-THOUGHT-1", 0);
  appendGeminiThoughtReplayPart(
    thoughtParts,
    "second thought",
    "SIG-THOUGHT-2",
    1,
  );

  pinGeminiThoughtReplayParts(parts, thoughtParts);
  pinGeminiTextThoughtSignature(parts, "SIG-ANSWER");

  assert.equal(parts[3]._google_thought_signature, "SIG-ANSWER");
  assert.deepEqual(geminiThoughtReplayParts(parts[0]), [
    { text: "first thought", thoughtSignature: "SIG-THOUGHT-1" },
  ]);
  assert.deepEqual(geminiThoughtReplayParts(parts[2]), [
    { text: "second thought", thoughtSignature: "SIG-THOUGHT-2" },
  ]);
  assert.deepEqual(collectGeminiThoughtReplayParts(parts), [
    { text: "first thought", thoughtSignature: "SIG-THOUGHT-1" },
    { text: "second thought", thoughtSignature: "SIG-THOUGHT-2" },
  ]);
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

test("continued Gemini turns keep the hidden user boundary", () => {
  const metadata = {
    custom: {
      geminiContinuationReplay: {
        turns: [
          {
            text: "first answer",
            thoughtSignature: "SIG-ANSWER-1",
            thoughtParts: [
              { text: "first thought", thoughtSignature: "SIG-THOUGHT-1" },
            ],
          },
        ],
        visiblePrefix: "first answer",
      },
    },
  };

  assert.deepEqual(readGeminiContinuationReplay(metadata), {
    turns: [
      {
        text: "first answer",
        thoughtSignature: "SIG-ANSWER-1",
        thoughtParts: [
          { text: "first thought", thoughtSignature: "SIG-THOUGHT-1" },
        ],
      },
    ],
    visiblePrefix: "first answer",
  });
  assert.deepEqual(
    continuationGeminiReplayTurns(metadata, {
      text: "first answer and the rest",
      thoughtSignature: "SIG-ANSWER-2",
      thoughtParts: [
        { text: "second thought", thoughtSignature: "SIG-THOUGHT-2" },
      ],
    }),
    [
      {
        text: "first answer",
        thoughtSignature: "SIG-ANSWER-1",
        thoughtParts: [
          { text: "first thought", thoughtSignature: "SIG-THOUGHT-1" },
        ],
      },
      {
        text: " and the rest",
        thoughtSignature: "SIG-ANSWER-2",
        thoughtParts: [
          { text: "second thought", thoughtSignature: "SIG-THOUGHT-2" },
        ],
      },
    ],
  );
  assert.deepEqual(
    geminiContinuationReplayEntries(
      [
        { text: "first", thoughtSignature: "SIG-1" },
        { text: "second", thoughtSignature: "SIG-2" },
      ],
      true,
    ),
    [
      {
        role: "assistant",
        turn: { text: "first", thoughtSignature: "SIG-1" },
      },
      { role: "user" },
      {
        role: "assistant",
        turn: { text: "second", thoughtSignature: "SIG-2" },
      },
      { role: "user" },
    ],
  );
});

test("the chat adapter retains thought and answer signatures independently", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(adapter, READ_REPLAY_PARTS);
  assert.match(adapter, WRITE_REPLAY_PARTS);
  assert.match(adapter, PIN_THOUGHT_PARTS);
  assert.match(adapter, CAPTURE_THOUGHT_PARTS);
  assert.match(adapter, FLUSH_BEFORE_REPLAY_CAPTURE);
  assert.match(adapter, PIN_TEXT_SIGNATURE);
});
