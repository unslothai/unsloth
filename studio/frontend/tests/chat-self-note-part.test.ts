// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";
import ts from "typescript";

import { readSrc } from "./helpers/kit.ts";

const adapter = readSrc("features/chat/api/chat-adapter.ts");

// --------------------------------------------------------------------------
// Part 1: the streaming tool-event branch that latches a `self_note` chunk
// (`{"_toolEvent": {"type": "self_note", "content": "<note>"}}`, emitted from
// routes/inference.py's `_self_note_sse_chunk`) into `selfNotePart`, and that
// it does so BEFORE the unconditional `closeReasoningContent()` call that
// follows -- so a self_note event does not force-close an open reasoning
// block, unlike tool_start/tool_end which legitimately do.
// --------------------------------------------------------------------------

/**
 * Executes the adapter's real tool-event dispatch, from the container_ready
 * branch through the self_note branch and into the unconditional
 * closeReasoningContent() call that follows -- the exact region whose
 * ordering this test is about. Mirrors the slice-and-vm-execute technique
 * `generation-tool-recovery.test.ts` uses for the same switch.
 */
function runToolEventDispatch(toolEvent: Record<string, unknown>): {
  selfNotePart: { type: string; content: string } | undefined;
  reasoningClosed: boolean;
} {
  const start = adapter.indexOf('if (toolEvent.type === "container_ready") {');
  // tool_start is the first branch reachable only after the unconditional
  // closeReasoningContent() call, so stopping just before its `if` captures
  // that call without pulling in the (heavily-dependent) tool_start body.
  const end = adapter.indexOf('if (toolEvent.type === "tool_start") {', start);
  assert.ok(start > 0 && end > start, "adapter tool-event region not found");
  const body = adapter.slice(start, end);

  const source = `
    // A generator, not a plain function: some sibling branches (e.g.
    // context_window_exceeded, tool_args) yield an interim publish before
    // their "continue", exactly as they do inside the adapter's real
    // for-await/yield loop. Draining it below runs them to completion.
    function* receive(toolEvent) {
      let selfNotePart;
      let reasoningClosed = false;
      const closeReasoningContent = () => { reasoningClosed = true; };
      // The sliced body's "continue" statements belong to the adapter's real
      // for-await loop; this dummy single-pass loop gives them somewhere
      // legal to target without changing what they do (skip the rest).
      for (const _once of [1]) {
        ${body}
      }
      return { selfNotePart, reasoningClosed };
    }
  `;
  const compiled = ts.transpileModule(source, {
    compilerOptions: { target: ts.ScriptTarget.ES2022 },
  }).outputText;
  // resolvedThreadId is a free variable the container_ready/container_invalidated
  // branches read; irrelevant to self_note, but they sit in the same slice and
  // must still evaluate. Left undefined/falsy so their `if` bodies are skipped.
  const context = vm.createContext({ resolvedThreadId: undefined });
  vm.runInContext(compiled, context);
  const gen = context.receive(toolEvent);
  let step = gen.next();
  while (!step.done) step = gen.next();
  // Round-tripped in the HOST realm: the vm realm's JSON.parse/stringify still
  // produces vm-realm objects, whose prototype differs from this file's
  // Object even when every field matches deepStrictEqual field-by-field.
  return JSON.parse(JSON.stringify(step.value));
}

test("a self_note tool event latches a self_note content part with the right content", () => {
  const { selfNotePart } = runToolEventDispatch({
    type: "self_note",
    content: "Remember: the user wants dark mode by default.",
  });
  assert.deepEqual(selfNotePart, {
    type: "self_note",
    content: "Remember: the user wants dark mode by default.",
  });
});

test("an empty or non-string note is not latched", () => {
  assert.equal(runToolEventDispatch({ type: "self_note", content: "" }).selfNotePart, undefined);
  assert.equal(
    runToolEventDispatch({ type: "self_note", content: "   " }).selfNotePart,
    undefined,
  );
  assert.equal(
    runToolEventDispatch({ type: "self_note", content: undefined }).selfNotePart,
    undefined,
  );
});

test("a self_note event does not force-close an open reasoning block", () => {
  // Deliberate choice (see the brief's gap #2): self_note is state-only, like
  // container_ready/context_window_exceeded, so it is handled and `continue`s
  // BEFORE the unconditional closeReasoningContent() call that tool_start/
  // tool_end rely on. This asserts that choice directly: reasoning must stay
  // open across a self_note event.
  const { reasoningClosed } = runToolEventDispatch({
    type: "self_note",
    content: "note",
  });
  assert.equal(
    reasoningClosed,
    false,
    "self_note must not force-close reasoning; it has no card to open, unlike tool_start",
  );
});

test("a sibling event (context_window_exceeded) also skips the unconditional close, for comparison", () => {
  // Not asserting new behaviour -- confirms the harness captures the real
  // control flow, since context_window_exceeded's own early return means it
  // never reaches the loop's yield in this sliced region either.
  const { reasoningClosed } = runToolEventDispatch({
    type: "container_invalidated",
  });
  assert.equal(reasoningClosed, false);
});

test("the self_note branch sits before the unconditional closeReasoningContent(), like its state-only siblings", () => {
  const dispatchStart = adapter.indexOf(
    'if (toolEvent.type === "container_ready") {',
  );
  const selfNoteAt = adapter.indexOf(
    'if (toolEvent.type === "self_note") {',
    dispatchStart,
  );
  const closeAt = adapter.indexOf("closeReasoningContent();", selfNoteAt);
  const toolStartAt = adapter.indexOf(
    'if (toolEvent.type === "tool_start") {',
    dispatchStart,
  );
  assert.ok(dispatchStart > 0 && selfNoteAt > dispatchStart);
  assert.ok(
    selfNoteAt < closeAt && closeAt < toolStartAt,
    "self_note must be matched before the unconditional closeReasoningContent() call",
  );
});

// --------------------------------------------------------------------------
// Part 2: the final assistant content the adapter yields at end-of-stream
// carries the latched note as a `self_note` part, spliced in next to
// `documentCitationParts` (both are tool-event-derived parts that only
// materialize in the FINAL content, never in an interim streamed yield).
// --------------------------------------------------------------------------

test("the final yielded content splices in the self_note part, mirroring documentCitationParts", () => {
  const finalContentAt = adapter.indexOf(
    "...buildAssistantContent(mergeContinuation(cumulativeText, { final: true })),",
  );
  assert.ok(finalContentAt > 0);
  // Widened from 400: the splice carries an explanatory comment about the cast that
  // `self_note` needs (it is not an assistant-ui part variant), which pushed the spread
  // itself past a 400-char window.
  const region = adapter.slice(finalContentAt, finalContentAt + 900);
  assert.match(region, /\.\.\.documentCitationParts,/);
  // Matched loosely around the cast, so a change to HOW the part is typed does not fail a
  // test whose subject is WHETHER the part reaches the final content.
  assert.match(region, /\.\.\.\(selfNotePart\s*\?\s*\[selfNotePart[\s\S]*?\]\s*:\s*\[\]\),/);
});

test("an assistant message with no note produces no self_note part in the final content splice", () => {
  // selfNotePart starts undefined and only the self_note branch (Part 1)
  // assigns it, so a stream carrying none never enters the spread -- verified
  // structurally alongside the declaration and initial value.
  const declAt = adapter.indexOf("let selfNotePart:");
  assert.ok(declAt > 0);
  assert.match(
    adapter.slice(declAt, declAt + 200),
    /let selfNotePart: \{ type: "self_note"; content: string \} \| undefined;/,
  );
});

// --------------------------------------------------------------------------
// Part 3: the re-send path. `collectAssistantSelfNote` reads the part back
// off a stored assistant message; `attachAssistantSelfNotePart` writes it
// onto the OUTBOUND SerializedMessage so the backend's `latest_self_note`
// (checkpoint.py) can read it on the next request. Both are small, dependency
// -free functions, executed directly rather than only pattern-matched.
// --------------------------------------------------------------------------

function loadSelfNoteHelpers(): {
  collectAssistantSelfNote: (message: { content?: unknown }) => string | undefined;
  attachAssistantSelfNotePart: (
    messages: Array<{ role: string; content: unknown }>,
    note: string | undefined,
  ) => void;
} {
  const start = adapter.indexOf("function collectAssistantSelfNote(");
  const end = adapter.indexOf("function serializeAssistantReplayMessages(");
  assert.ok(start > 0 && end > start, "self-note helpers not found");
  const body = adapter.slice(start, end);
  const compiled = ts.transpileModule(
    `${body}\nmodule.exports = { collectAssistantSelfNote, attachAssistantSelfNotePart };`,
    { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } },
  ).outputText;
  const moduleObj = { exports: {} as ReturnType<typeof loadSelfNoteHelpers> };
  const context = vm.createContext({ module: moduleObj, exports: moduleObj.exports });
  vm.runInContext(compiled, context);
  return moduleObj.exports;
}

test("collectAssistantSelfNote reads the part back off the assistant message", () => {
  const { collectAssistantSelfNote } = loadSelfNoteHelpers();
  const message = {
    content: [
      { type: "text", text: "Here is my answer." },
      { type: "self_note", content: "Task: finish the migration." },
    ],
  };
  assert.equal(collectAssistantSelfNote(message), "Task: finish the migration.");
});

test("collectAssistantSelfNote returns undefined when there is no note", () => {
  const { collectAssistantSelfNote } = loadSelfNoteHelpers();
  assert.equal(
    collectAssistantSelfNote({ content: [{ type: "text", text: "hi" }] }),
    undefined,
  );
  assert.equal(collectAssistantSelfNote({ content: "plain string" }), undefined);
  assert.equal(collectAssistantSelfNote({}), undefined);
});

test("collectAssistantSelfNote takes the newest note, matching the backend's newest-wins contract", () => {
  const { collectAssistantSelfNote } = loadSelfNoteHelpers();
  const message = {
    content: [
      { type: "self_note", content: "old note" },
      { type: "text", text: "..." },
      { type: "self_note", content: "new note" },
    ],
  };
  assert.equal(collectAssistantSelfNote(message), "new note");
});

test("attachAssistantSelfNotePart re-sends the note in the outbound request body's last assistant message", () => {
  const { attachAssistantSelfNotePart } = loadSelfNoteHelpers();
  const messages = [
    { role: "user", content: "hello" },
    { role: "assistant", content: "hi there" },
  ];
  attachAssistantSelfNotePart(messages, "Task: finish the migration.");
  const assistant = messages[1] as { role: string; content: unknown };
  assert.ok(Array.isArray(assistant.content), "content must become a list -- latest_self_note only reads list-shaped content");
  // Round-tripped: attachAssistantSelfNotePart builds the part inside the vm
  // realm, so its object identity/prototype differs from this test's `Object`
  // even when every field matches -- exactly the cross-realm distinction that
  // wire serialisation (this function's whole purpose) also erases.
  const parts = JSON.parse(JSON.stringify(assistant.content)) as Array<
    Record<string, unknown>
  >;
  assert.deepEqual(
    parts.find((p) => p.type === "self_note"),
    { type: "self_note", content: "Task: finish the migration." },
  );
  // The original text must survive alongside the note, not be replaced by it.
  assert.deepEqual(
    parts.find((p) => p.type === "text"),
    { type: "text", text: "hi there" },
  );
});

test("attachAssistantSelfNotePart is a no-op when there is nothing to attach", () => {
  const { attachAssistantSelfNotePart } = loadSelfNoteHelpers();
  const messages = [{ role: "assistant", content: "hi there" }];
  attachAssistantSelfNotePart(messages, undefined);
  assert.equal(messages[0].content, "hi there", "content must stay a plain string when there is no note");
});

test("attachAssistantSelfNotePart promotes null content to a list so the part is not dropped", () => {
  const { attachAssistantSelfNotePart } = loadSelfNoteHelpers();
  const messages = [{ role: "assistant", content: null }];
  attachAssistantSelfNotePart(messages, "note");
  assert.deepEqual(
    JSON.parse(JSON.stringify(messages[0].content)),
    [{ type: "self_note", content: "note" }],
  );
});

test("serializeAssistantReplayMessages wires the note through on every call, not just the thought-signature path", () => {
  // Structural: confirms attachAssistantSelfNotePart is actually invoked from
  // the function that builds every outbound assistant replay, immediately
  // alongside the pre-existing attachAssistantThoughtSignature call it
  // mirrors -- so the wiring in Part 3 above is not dead code.
  const fnStart = adapter.indexOf("function serializeAssistantReplayMessages(");
  const fnEnd = adapter.indexOf("function toOpenAIMessages(", fnStart);
  assert.ok(fnStart > 0 && fnEnd > fnStart);
  const body = adapter.slice(fnStart, fnEnd);
  assert.match(
    body,
    /attachAssistantSelfNotePart\(messages, collectAssistantSelfNote\(message\)\);/,
  );
  const thoughtAt = body.indexOf("attachAssistantThoughtSignature(");
  const noteAt = body.indexOf("attachAssistantSelfNotePart(messages,");
  assert.ok(thoughtAt > 0 && noteAt > thoughtAt, "self-note attach must run after the thought-signature attach, mirroring it");
});
