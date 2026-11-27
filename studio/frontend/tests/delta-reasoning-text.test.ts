// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readText, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { deltaReasoningText, parseAssistantContent } = await import(
  "../src/features/chat/utils/parse-assistant-content.ts"
);
const { appendRecoveredDelta, generationChunkHasSubstantiveDelta } = await import(
  "../src/features/chat/utils/chat-generation-recovery.ts"
);

const openRouterChunks = [
  {
    choices: [
      {
        delta: {
          content: "",
          reasoning: "Let me think. ",
          reasoning_details: [
            { type: "reasoning.text", text: "Let me think. ", format: "unknown", index: 0 },
          ],
        },
      },
    ],
  },
  {
    choices: [
      {
        delta: {
          content: "",
          reasoning_details: [
            { type: "reasoning.encrypted", data: "gAAAA==", format: "openai-responses-v1", index: 1 },
          ],
        },
      },
    ],
  },
  { choices: [{ delta: { content: "The answer is 4." } }] },
];

test("an OpenRouter thought is read from reasoning_details once", () => {
  assert.deepEqual(
    openRouterChunks.map((chunk) => deltaReasoningText(chunk.choices[0].delta)),
    ["Let me think. ", "", ""],
  );
  assert.equal(deltaReasoningText({ reasoning_content: "Kimi thought" }), "Kimi thought");
  assert.equal(deltaReasoningText(undefined), "");
  assert.equal(generationChunkHasSubstantiveDelta(openRouterChunks[0]), true);
  assert.equal(generationChunkHasSubstantiveDelta(openRouterChunks[1]), false);
});

test("a recovered OpenRouter reply keeps its Reasoning block", () => {
  let reply = { raw: "", reasoningOpen: false, parseThink: false, splitReasoningSeen: false };
  for (const chunk of openRouterChunks) {
    reply = appendRecoveredDelta(reply, chunk.choices[0].delta);
  }
  assert.equal(reply.raw, "<think>Let me think. </think>The answer is 4.");
  assert.deepEqual(
    parseAssistantContent(reply.raw, { parseThink: reply.parseThink }).map((part) => part.type),
    ["reasoning", "text"],
  );
});

test("the recovery follow replays chunks through appendRecoveredDelta", () => {
  const source = readText("../src/features/chat/runtime-provider.tsx");
  const start = source.indexOf("function scheduleGenerationRecovery");
  assert.ok(start > 0, "scheduleGenerationRecovery is gone");
  const recovery = source.slice(start, source.indexOf("\nfunction ", start + 1));
  assert.match(recovery, /appendRecoveredDelta\(/);
  assert.doesNotMatch(recovery, /reasoning_content/);
});
