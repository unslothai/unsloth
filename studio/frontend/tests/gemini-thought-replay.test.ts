// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type GeminiAnswerReplayPart,
  type GeminiThoughtReplayPart,
  type PositionedGeminiAnswerReplayPart,
  type PositionedGeminiThoughtReplayPart,
  appendGeminiAnswerReplayPart,
  appendGeminiThoughtReplayPart,
  collectGeminiAnswerReplayParts,
  collectGeminiThoughtReplayParts,
  continuationGeminiReplayTurns,
  geminiContinuationReplayEntries,
  geminiThoughtReplayParts,
  pinGeminiAnswerReplayParts,
  pinGeminiTextThoughtSignature,
  pinGeminiThoughtReplayParts,
  readGeminiContinuationReplay,
  withGeminiAnswerReplayParts,
  withGeminiThoughtReplayParts,
} from "../src/features/chat/gemini-thought-replay.ts";
import { readSrc } from "./helpers/kit.ts";

const READ_REPLAY_PARTS = /geminiThoughtReplayParts\(part\)/;
const WRITE_REPLAY_PARTS = /withGeminiThoughtReplayParts\(/;
const PIN_THOUGHT_PARTS = /pinGeminiThoughtReplayParts\(/;
const CAPTURE_THOUGHT_PARTS =
  /appendGeminiThoughtReplayPart\([\s\S]{0,220}toolCallParts\.length/;
const CAPTURE_NATIVE_THOUGHT_PART =
  /const thoughtPart = googleRecord\.thought_part/;
const MERGE_UNSIGNED_THOUGHT_INTO_LATER_SIGNATURE = /pendingGeminiThoughtText/;
const FLUSH_BEFORE_REPLAY_CAPTURE =
  /if \(part\.type === "reasoning"\) \{[\s\S]{0,180}flushAssistantAndToolResults\(\);[\s\S]{0,180}pendingGeminiThoughtParts\.push/;
const PIN_TEXT_SIGNATURE =
  /pinGeminiTextThoughtSignature\([\s\S]*legacyGeminiTextSignature,[\s\S]*\)/;

test("every native thought boundary stays separate from signed answer text", () => {
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
  appendGeminiThoughtReplayPart(thoughtParts, "unsigned thought", undefined, 0);
  appendGeminiThoughtReplayPart(thoughtParts, "signed thought", "SIG-THOUGHT-1", 0);
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
    { text: "unsigned thought" },
    { text: "signed thought", thoughtSignature: "SIG-THOUGHT-1" },
  ]);
  assert.deepEqual(geminiThoughtReplayParts(parts[2]), [
    { text: "second thought", thoughtSignature: "SIG-THOUGHT-2" },
  ]);
  assert.deepEqual(collectGeminiThoughtReplayParts(parts), [
    { text: "unsigned thought" },
    { text: "signed thought", thoughtSignature: "SIG-THOUGHT-1" },
    { text: "second thought", thoughtSignature: "SIG-THOUGHT-2" },
  ]);
});

test("thought parts use a separate replay envelope from signed answer text", () => {
  const extra = withGeminiThoughtReplayParts(
    { google: { thought_signature: "SIG-ANSWER" } },
    [
      { text: "unsigned preface" },
      { text: "consider the clues", thoughtSignature: "SIG-THOUGHT" },
    ],
  );

  assert.deepEqual(extra, {
    google: {
      thought_signature: "SIG-ANSWER",
      thought_parts: [
        { text: "unsigned preface" },
        { text: "consider the clues", thought_signature: "SIG-THOUGHT" },
      ],
    },
  });
});

test("signed and unsigned Gemini answer-part boundaries replay exactly", () => {
  const parts: Array<{
    type: string;
    text?: string;
    _google_answer_parts?: GeminiAnswerReplayPart[];
  }> = [{ type: "text", text: "firstsecond" }];
  const answerParts: PositionedGeminiAnswerReplayPart[] = [];
  appendGeminiAnswerReplayPart(answerParts, { text: "first" }, 0);
  appendGeminiAnswerReplayPart(
    answerParts,
    { text: "second", thoughtSignature: "SIG-SECOND" },
    0,
  );
  appendGeminiAnswerReplayPart(
    answerParts,
    { text: "", thoughtSignature: "SIG-EMPTY" },
    0,
  );
  pinGeminiAnswerReplayParts(parts, answerParts);
  assert.deepEqual(collectGeminiAnswerReplayParts(parts), [
    { text: "first" },
    { text: "second", thoughtSignature: "SIG-SECOND" },
    { text: "", thoughtSignature: "SIG-EMPTY" },
  ]);
  assert.deepEqual(
    withGeminiAnswerReplayParts(undefined, collectGeminiAnswerReplayParts(parts)),
    {
      google: {
        answer_parts: [
          { text: "first" },
          { text: "second", thought_signature: "SIG-SECOND" },
          { text: "", thought_signature: "SIG-EMPTY" },
        ],
      },
    },
  );
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
            answerParts: [
              { text: "first answer" },
              { text: "", thoughtSignature: "SIG-ANSWER-1" },
            ],
          },
        ],
        visiblePrefix: "first answer",
        stripVisiblePrefix: true,
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
        answerParts: [
          { text: "first answer" },
          { text: "", thoughtSignature: "SIG-ANSWER-1" },
        ],
      },
    ],
    visiblePrefix: "first answer",
    stripVisiblePrefix: true,
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
        answerParts: [
          { text: "first answer" },
          { text: "", thoughtSignature: "SIG-ANSWER-1" },
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
    continuationGeminiReplayTurns(
      {
        custom: {
          geminiContinuationReplay: {
            turns: [{ text: "first answer", thoughtSignature: "SIG-ANSWER-1" }],
            visiblePrefix: "first answer",
            stripVisiblePrefix: false,
          },
        },
      },
      {
        text: "first answer restarted in full",
        thoughtSignature: "SIG-ANSWER-2",
      },
    ),
    [
      { text: "first answer", thoughtSignature: "SIG-ANSWER-1" },
      {
        text: "first answer restarted in full",
        thoughtSignature: "SIG-ANSWER-2",
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
  assert.match(adapter, CAPTURE_NATIVE_THOUGHT_PART);
  assert.doesNotMatch(adapter, MERGE_UNSIGNED_THOUGHT_INTO_LATER_SIGNATURE);
  assert.match(adapter, FLUSH_BEFORE_REPLAY_CAPTURE);
  assert.match(adapter, PIN_TEXT_SIGNATURE);
});
