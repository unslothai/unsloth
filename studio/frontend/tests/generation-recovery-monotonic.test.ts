// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A durable generation run has two readers while the tab that started it is
// still streaming: the adapter, which publishes the whole reply about once per
// animation frame, and the recovery follower, which replays the run's stored
// events and pays a write to storage before it reads the next one. The follower
// is therefore hundreds of characters behind, and its publish used to be
// imported into the thread unconditionally, so the reasoning pane rewound to
// its opening lines for the frame or two before the next adapter yield.
//
// The trace below is the one from the report: "Create a detailed SVG image of a
// cute kitten", reasoning only, no tool calls, no continuation.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { generationRawContent, recoveredContentToImport } = await import(
  "../src/features/chat/utils/chat-generation-recovery.ts"
);
const { parseAssistantContent } = await import(
  "../src/features/chat/utils/parse-assistant-content.ts"
);

const PARAGRAPHS = [
  "The user wants a detailed SVG image of a cute kitten. I should create a rich, crafted SVG, not a trivial circle-cat. Let me design a sitting kitten with fur texture, big eyes, whiskers, tabby markings, gradients, shading, maybe a soft background scene.",
  "Let me think about composition: a seated kitten facing viewer, three-quarter view. Big glossy eyes, pink nose, whiskers, inner ear pink, small front paws, tail curled around the body, and a soft ground shadow beneath it.",
  "Canvas: viewBox 0 0 800 800. Body: an egg shape centred around (400, 520), width about 300, height about 300. Head: a circle at (400, 300) with radius 150, slightly flattened at the top.",
  "Ears: triangles with rounded tips at roughly (280, 190) and (520, 190). Inner ear in a warmer pink, inset about 18 units and shorter, so the fur edge reads.",
  "Eyes: two ellipses at (340, 300) and (460, 300), rx 42 ry 48. Iris in green with a radial gradient, pupil a vertical slit, two white highlights.",
];
const TRACE = PARAGRAPHS.join("\n\n");

/** What the adapter yields once `chars` of the trace have arrived. */
const liveContent = (chars: number) =>
  parseAssistantContent(`<think>${TRACE.slice(0, chars)}`);

/**
 * What a recovery publish carries once its replay has reached `chars`. It
 * closes the block it is inside, exactly as `publish` does before it parses.
 */
const recoveredContent = (chars: number) =>
  parseAssistantContent(`<think>${TRACE.slice(0, chars)}</think>`);

const reasoningLength = (content: unknown): number => {
  if (!Array.isArray(content)) {
    return 0;
  }
  let total = 0;
  for (const part of content) {
    if (part && typeof part === "object" && part.type === "reasoning") {
      total += String(part.text ?? "").length;
    }
  }
  return total;
};

test("a recovery publish behind the live reply leaves the view alone", () => {
  // The frame the report caught: the reader is eleven paragraphs in, the
  // follower has replayed 323 characters.
  const view = liveContent(2400);
  const recovered = recoveredContent(323);
  assert.equal(recoveredContentToImport(view, recovered), view);
});

test("reasoning never goes backwards across an interleaved run", () => {
  // One recovery publish per twenty-eight adapter yields, which is the ratio a
  // ~500 ms storage round trip has against a per-frame publish. The follower
  // advances one replayed chunk each time, as the recording's episodes do.
  let shown: unknown = [];
  const lengths: number[] = [];
  let replayed = 0;
  let yields = 0;
  for (let live = 40; live <= TRACE.length; live += 40) {
    shown = liveContent(live);
    lengths.push(reasoningLength(shown));
    yields += 1;
    if (yields % 4 === 0) {
      replayed += 6;
      shown = recoveredContentToImport(shown, recoveredContent(replayed));
      lengths.push(reasoningLength(shown));
    }
  }
  assert.ok(replayed > 0, "the run must actually publish a recovery body");
  assert.ok(
    replayed < TRACE.length,
    "the follower must stay behind, or there is nothing to test",
  );
  for (let i = 1; i < lengths.length; i += 1) {
    assert.ok(
      lengths[i] >= lengths[i - 1],
      `reasoning shrank from ${lengths[i - 1]} to ${lengths[i]} at step ${i}`,
    );
  }
});

test("a recovery ahead of the view still wins", () => {
  // The case the follower exists for: a reload left the thread holding the last
  // saved prefix and the run kept going on the server.
  const view = liveContent(200);
  const recovered = recoveredContent(2400);
  assert.equal(recoveredContentToImport(view, recovered), recovered);
});

test("an empty view takes the recovered body", () => {
  const recovered = recoveredContent(900);
  assert.equal(recoveredContentToImport([], recovered), recovered);
});

test("a recovered body that disagrees with the view still wins", () => {
  // Not a prefix, so storage is repairing something rather than lagging. Shorter
  // than the view and still imported.
  const view = liveContent(2400);
  const recovered = parseAssistantContent(
    "<think>The server recorded a different reply for this run.</think>",
  );
  assert.equal(recoveredContentToImport(view, recovered), recovered);
});

test("an equal body is still imported, so metadata-only publishes land", () => {
  const view = liveContent(900);
  const recovered = liveContent(900);
  assert.equal(recoveredContentToImport(view, recovered), recovered);
});

test("the comparison reads tool calls as neither text nor reasoning", () => {
  // A view carrying a tool-call part must not read as longer than the same
  // reply without one, or a live view would be preferred for the wrong reason.
  const withTool = [
    { type: "reasoning", text: TRACE.slice(0, 100) },
    { type: "tool-call", toolCallId: "call_0", toolName: "search", args: {} },
  ];
  assert.equal(
    generationRawContent(withTool).raw,
    `<think>${TRACE.slice(0, 100)}`,
  );
});
