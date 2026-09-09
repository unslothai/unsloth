// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A recovery rebuilds the reply from the raw text a run's chunk events carried, and the
// projection it rebuilds through only understands text and reasoning. Every other part was
// dropped on the way in, so the rebuilt body lost the turn's tool calls and the follower then
// wrote that body to storage, making the loss permanent.
//
// On screen: the thinking blocks and the prose survive, because both round-trip through
// <think> tags, and the "Used tool: edit_file" cards between them are gone, leaving the
// model's colon preamble introducing nothing. A recovery is armed by `online`, `pageshow`,
// `visibilitychange` and history load, so a long local run that the reader tabs away from
// hits it.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { generationRawContent, recoveredContentToImport, restoreCarriedParts } =
  await import("../src/features/chat/utils/chat-generation-recovery.ts");
const { parseAssistantContent } = await import(
  "../src/features/chat/utils/parse-assistant-content.ts"
);

type Part = {
  type: string;
  text?: string;
  toolName?: string;
  toolCallId?: string;
};

/** What the follower does to a stored body on every publish. */
function recoverBody(content: Part[]): Part[] {
  const { raw, reasoningOpen, carried } = generationRawContent(content);
  return restoreCarriedParts(
    parseAssistantContent(reasoningOpen ? `${raw}</think>` : raw) as Part[],
    carried,
  );
}

const TURN: Part[] = [
  { type: "reasoning", text: "planning the GLSL repair" },
  {
    type: "text",
    text: "That patch mangled the GLSL line, repairing properly:",
  },
  { type: "tool-call", toolCallId: "call_0", toolName: "edit_file" },
  { type: "reasoning", text: "checking the result" },
  { type: "text", text: "Run 7, the Drowned Nave:" },
  { type: "tool-call", toolCallId: "call_1", toolName: "render_html" },
];

test("a recovered turn keeps the tool calls it was streamed with", () => {
  const recovered = recoverBody(TURN);

  assert.deepEqual(
    recovered.map((part) => part.type),
    TURN.map((part) => part.type),
  );
  assert.deepEqual(
    recovered
      .filter((part) => part.type === "tool-call")
      .map((p) => p.toolCallId),
    ["call_0", "call_1"],
  );
});

test("the text a recovery rebuilds is unchanged by carrying the parts", () => {
  // The projection is what the follower compares and stores against, so it must not move.
  assert.equal(
    generationRawContent(recoverBody(TURN)).raw,
    generationRawContent(TURN).raw,
  );
});

test("a call between two runs of text splits the text it sits in", () => {
  // The rebuild coalesces "A" and "B" into one part, so the call has to land by offset.
  const body: Part[] = [
    { type: "text", text: "A" },
    { type: "tool-call", toolCallId: "call_0", toolName: "python" },
    { type: "text", text: "B" },
  ];
  assert.deepEqual(recoverBody(body), body);
});

test("a turn with no tool calls rebuilds exactly as before", () => {
  const body: Part[] = [
    { type: "reasoning", text: "thinking" },
    { type: "text", text: "answer" },
  ];
  assert.deepEqual(recoverBody(body), body);
});

test("an unfinished reply keeps its calls and its open reasoning", () => {
  const body: Part[] = [
    { type: "text", text: "one moment:" },
    { type: "tool-call", toolCallId: "call_0", toolName: "web_search" },
    { type: "reasoning", text: "still going" },
  ];
  const recovered = recoverBody(body);
  assert.deepEqual(
    recovered.map((part) => part.type),
    ["text", "tool-call", "reasoning"],
  );
});

test("a publish cannot swap a body that has cards for one that lost them", () => {
  // The prefix guard reads both sides through the projection, which sees no tool call, so a
  // stripped body of equal or greater length used to win outright.
  const stripped = TURN.filter((part) => part.type !== "tool-call");
  const kept = recoveredContentToImport(TURN, stripped) as Part[];

  assert.deepEqual(
    kept.filter((part) => part.type === "tool-call").map((p) => p.toolCallId),
    ["call_0", "call_1"],
  );
  assert.equal(
    generationRawContent(kept).raw,
    generationRawContent(stripped).raw,
  );
});

test("a publish that is genuinely behind is still refused", () => {
  const behind: Part[] = [{ type: "text", text: "That patch" }];
  const full: Part[] = [{ type: "text", text: "That patch mangled the line" }];
  assert.deepEqual(recoveredContentToImport(full, behind), full);
});

test("a publish carrying new text still wins", () => {
  const view: Part[] = [{ type: "text", text: "That patch" }];
  const ahead: Part[] = [{ type: "text", text: "That patch mangled the line" }];
  assert.deepEqual(recoveredContentToImport(view, ahead), ahead);
});
