// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
  // The prefix guard compares projections, which hide tool calls: a stripped body used to win.
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

test("calls split reasoning into the original groups", () => {
  const body: Part[] = [
    { type: "reasoning", text: "before" },
    { type: "tool-call", toolCallId: "one", toolName: "edit_file" },
    { type: "reasoning", text: "between" },
    { type: "tool-call", toolCallId: "two", toolName: "web_search" },
    { type: "reasoning", text: "after" },
    { type: "text", text: "done" },
  ];
  assert.deepEqual(recoverBody(body), body);
  assert.deepEqual(recoverBody(recoverBody(body)), body);
});

test("cards at shared offsets retain their order", () => {
  const body: Part[] = [
    { type: "text", text: "A\u{1D11E}" },
    { type: "tool-call", toolCallId: "one" },
    { type: "tool-call", toolCallId: "two" },
    { type: "text", text: "B" },
  ];
  assert.deepEqual(recoverBody(body), body);
  assert.deepEqual(
    recoveredContentToImport(
      body,
      body.filter((p) => p.toolCallId !== "one"),
    ),
    body,
  );
});

test("a divergent recovered reply does not inherit old cards or attachments", () => {
  for (const type of ["tool-call", "source", "file", "image"]) {
    const view: Part[] = [{ type: "text", text: "old response" }, { type }];
    const recovered: Part[] = [
      { type: "text", text: "server repaired response" },
    ];
    assert.equal(recoveredContentToImport(view, recovered), recovered);
  }
});

test("compatible replies retain recovered cards and their authoritative results", () => {
  const old = { type: "tool-call", toolCallId: "shared", result: "old" };
  const current = { ...old, result: "new" };
  const missing = { type: "tool-call", toolCallId: "missing" };
  const added = { type: "tool-call", toolCallId: "added" };
  const view = [{ type: "text", text: "A" }, old, missing];
  const recovered = [
    { type: "text", text: "A" },
    current,
    added,
    { type: "text", text: "B" },
  ];
  const merged = recoveredContentToImport(view, recovered);
  assert.deepEqual(merged, [
    { type: "text", text: "A" },
    current,
    missing,
    added,
    { type: "text", text: "B" },
  ]);
});

test("missing cards stay beside shared anchors without reordering recovered cards", () => {
  const card = (toolCallId: string) => ({ type: "tool-call", toolCallId });
  for (const [liveIds, recoveredIds, expected] of [
    [
      ["b", "c"],
      ["a", "b"],
      ["a", "b", "c"],
    ],
    [["a", "b"], ["b"], ["a", "b"]],
    [
      ["b", "c", "d"],
      ["a", "b", "e"],
      ["a", "b", "c", "d", "e"],
    ],
    [
      ["b", "c", "a"],
      ["a", "b"],
      ["a", "b", "c"],
    ],
    [
      ["a", "b", "c"],
      ["c", "a"],
      ["c", "a", "b"],
    ],
  ]) {
    const view = [{ type: "text", text: "Working:" }, ...liveIds.map(card)];
    const recovered = [
      { type: "text", text: "Working:" },
      ...recoveredIds.map(card),
      { type: "text", text: "Done" },
    ];
    const merged = recoveredContentToImport(view, recovered);
    assert.equal(generationRawContent(merged).raw, "Working:Done");
    assert.deepEqual(
      merged
        .filter((part) => part.type === "tool-call")
        .map((part) => (part as Part).toolCallId),
      expected,
    );
  }
});

test("replay identities match provider cards without merging different rounds", () => {
  const card = {
    type: "tool-call",
    toolName: "edit_file",
    backendToolCallId: "call_0",
    generationToolCallId: "run:1",
    toolCallId: "provider-id",
  };
  const recovered = [{ ...card, toolCallId: "call_0:run:1", result: "ok" }];
  assert.equal(recoveredContentToImport([card], recovered), recovered);
  const next = {
    ...card,
    toolCallId: "next-id",
    generationToolCallId: "run:3",
  };
  assert.equal(recoveredContentToImport([next], recovered).length, 2);
});

test("legacy calls at different positions remain separate", () => {
  const card = {
    type: "tool-call",
    toolName: "edit_file",
    toolCallId: "call_0:live",
  };
  const recovered = [
    {
      ...card,
      toolCallId: "call_0:run:1",
      backendToolCallId: "call_0",
      generationToolCallId: "run:1",
    },
    { type: "text", text: "Done" },
  ];
  const merged = recoveredContentToImport(
    [{ type: "text", text: "Done" }, card],
    recovered,
  );
  assert.equal(merged.filter((part) => part.type === "tool-call").length, 2);
});

test("legacy matching cannot cross a shared card into another round", () => {
  const card = (toolCallId: string) => ({
    type: "tool-call",
    toolName: "edit_file",
    toolCallId,
  });
  const first = card("call_0:live-0");
  const shared = card("call_0:live-1");
  const last = {
    ...card("call_0:run:5"),
    backendToolCallId: "call_0",
    generationToolCallId: "run:5",
  };
  assert.deepEqual(recoveredContentToImport([first, shared], [shared, last]), [
    first,
    shared,
    last,
  ]);
  const replayedFirst = {
    ...first,
    toolCallId: "call_0:run:1",
    backendToolCallId: "call_0",
    generationToolCallId: "run:1",
  };
  const liveLast = card("call_0:live-2");
  assert.deepEqual(
    recoveredContentToImport([shared, liveLast], [replayedFirst, shared]),
    [replayedFirst, shared, liveLast],
  );
});
