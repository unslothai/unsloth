// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { extractDeltaText, parseAssistantContent } = await import(
  "../src/features/chat/utils/parse-assistant-content.ts"
);
const { createSegmentedAssistantText } = await import(
  "../src/features/chat/utils/incremental-assistant-content.ts"
);
const {
  generationRawContent,
  requestParsesThinkTags,
  restoreCarriedPartsFromRaw,
} = await import(
  "../src/features/chat/utils/chat-generation-recovery.ts"
);

const LITERAL = "Use <think>x</think> tags, or a lone </think> and <think>";

test("thinking off keeps literal think tags as reply text", () => {
  assert.deepEqual(
    parseAssistantContent("Use <think>x</think> tags", { parseThink: false }),
    [{ type: "text", text: "Use <think>x</think> tags" }],
  );
  assert.deepEqual(parseAssistantContent("", { parseThink: false }), []);
});

test("thinking on still splits think tags into reasoning", () => {
  assert.deepEqual(parseAssistantContent("Use <think>x</think> tags"), [
    { type: "text", text: "Use " },
    { type: "reasoning", text: "x" },
    { type: "text", text: " tags" },
  ]);
});

test("the streaming parse matches the full parse with think parsing off", () => {
  const boundaries = [4, 20];
  const segmented = createSegmentedAssistantText({ parseThink: false });
  let raw = "";
  for (const char of LITERAL) {
    raw += char;
    segmented.appendText(char);
    const cuts = boundaries.filter((at) => at <= raw.length);
    const expected = [0, ...cuts].map((from, index) =>
      parseAssistantContent(raw.slice(from, cuts[index] ?? raw.length), {
        parseThink: false,
      }),
    );
    assert.deepEqual(segmented.runs(raw, cuts), expected);
  }
  assert.deepEqual(segmented.runs(LITERAL, boundaries), [
    [{ type: "text", text: LITERAL.slice(0, 4) }],
    [{ type: "text", text: LITERAL.slice(4, 20) }],
    [{ type: "text", text: LITERAL.slice(20) }],
  ]);
});

test("reasoning that arrives on a thinking-off turn still renders as reasoning", () => {
  const raw = "Answer<think>real reasoning</think>Use <think>x</think>";
  const reparsed = createSegmentedAssistantText({ parseThink: true });
  reparsed.appendText("<think>real reasoning");
  assert.deepEqual(reparsed.runs(raw, []), [parseAssistantContent(raw)]);
  assert.equal(
    reparsed.runs(raw, [])[0].some((part) => part.type === "reasoning"),
    true,
  );
  assert.equal(
    extractDeltaText([{ type: "thinking", thinking: "plan" }])
      .hasStructuredReasoning,
    true,
  );
  assert.equal(
    extractDeltaText([{ type: "text", text: "<think>x</think>" }])
      .hasStructuredReasoning,
    false,
  );
});

test("recovery replay keeps literal think text when the turn stamped it", () => {
  const stored = [
    { type: "text", text: "Use <think>x</think> tags " },
    { type: "tool-call", toolCallId: "call_0", toolName: "web_search" },
    { type: "text", text: "then <think>" },
  ];
  const { raw, carried } = generationRawContent(stored);
  assert.deepEqual(
    restoreCarriedPartsFromRaw(raw, carried, { parseThink: false }),
    stored,
  );
  assert.deepEqual(restoreCarriedPartsFromRaw(raw, [], { parseThink: false }), [
    { type: "text", text: raw },
  ]);
  assert.equal(
    restoreCarriedPartsFromRaw(raw, carried).some(
      (part) => part.type === "reasoning",
    ),
    true,
  );
});

test("a recovery before the first client save reads the choice from the request", () => {
  assert.equal(
    requestParsesThinkTags({ thinking: { type: "disabled" } }),
    false,
  );
  assert.equal(requestParsesThinkTags({ enable_thinking: false }), false);
  assert.equal(requestParsesThinkTags({ reasoning_effort: "none" }), false);
  assert.equal(requestParsesThinkTags({ thinking: { type: "enabled" } }), true);
  assert.equal(
    requestParsesThinkTags({ enable_thinking: true, reasoning_effort: "high" }),
    true,
  );
  assert.equal(requestParsesThinkTags({ reasoning_effort: "low" }), true);
  assert.equal(requestParsesThinkTags({}), true);

  const { raw, carried } = generationRawContent([]);
  const replayed = `${raw}Use <think>hi</think> in your prompt.`;
  assert.deepEqual(
    restoreCarriedPartsFromRaw(replayed, carried, {
      parseThink: requestParsesThinkTags({ thinking: { type: "disabled" } }),
    }),
    [{ type: "text", text: "Use <think>hi</think> in your prompt." }],
  );
});

test("the adapter and recovery follow the turn's think parse state", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(adapter, /parseThinkTags: parseThink,/);
  assert.match(
    adapter,
    /createSegmentedAssistantText\(\{\s*trustAppends,\s*parseThink,\s*\}\)/,
  );
  assert.match(
    adapter,
    /if \(reasoning \|\| hasStructuredReasoning\) \{\s*setParseThink\(true\);\s*\}\s*if \(reasoning\) \{\s*if \(!reasoningContentOpen\) \{/,
  );
  assert.match(adapter, /parseThink && thinkTags\.endsInsideThink\(\)/);
  assert.match(
    adapter,
    /: !supportsReasoning \|\|\s*reasoningEnabled \|\|\s*reasoningAlwaysOn \|\|\s*reasoningStyle === "reasoning_effort",/,
    "a local model that cannot stop thinking must keep parsing its tags",
  );

  const recovery = readSrc("features/chat/runtime-provider.tsx");
  assert.match(
    recovery,
    /let parseThink = metadata\.parseThinkTags !== false;/,
  );
  assert.match(recovery, /carried,\s*\{ parseThink \},/);
  assert.match(
    recovery,
    /if \(typeof metadata\.parseThinkTags !== "boolean"\) \{\s*parseThink = requestParsesThinkTags\(update\.run\.requestPayload\);\s*currentMetadata = \{\s*\.\.\.currentMetadata,\s*parseThinkTags: parseThink,\s*\};/,
    "the server placeholder has no flag until the first client save",
  );
  assert.match(
    recovery,
    /if \(!parseThink && \(reasoning \|\| hasStructuredReasoning\)\) \{\s*parseThink = true;\s*currentMetadata = \{ \.\.\.currentMetadata, parseThinkTags: true \};/,
  );
});
