// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";
import ts from "typescript";

import { readSrc } from "./helpers/kit.ts";

const adapter = readSrc("features/chat/api/chat-adapter.ts");


function runToolEventDispatch(toolEvent: Record<string, unknown>): {
  selfNotePart: { type: string; content: string } | undefined;
  reasoningClosed: boolean;
} {
  const start = adapter.indexOf('if (toolEvent.type === "container_ready") {');
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
  const context = vm.createContext({ resolvedThreadId: undefined });
  vm.runInContext(compiled, context);
  const gen = context.receive(toolEvent);
  let step = gen.next();
  while (!step.done) step = gen.next();
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


test("the final yielded content splices in the self_note part", () => {
  const finalContentAt = adapter.indexOf("            ...finalContent,\n");
  assert.ok(finalContentAt > 0);
  const region = adapter.slice(finalContentAt, finalContentAt + 400);
  assert.match(region, /\.\.\.\(selfNotePart\s*\?\s*\[selfNotePart[\s\S]*?\]\s*:\s*\[\]\),/);
});

test("an assistant message with no note produces no self_note part in the final content splice", () => {
  const declAt = adapter.indexOf("let selfNotePart:");
  assert.ok(declAt > 0);
  assert.match(
    adapter.slice(declAt, declAt + 200),
    /let selfNotePart: \{ type: "self_note"; content: string \} \| undefined;/,
  );
});


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
  const parts = JSON.parse(JSON.stringify(assistant.content)) as Array<
    Record<string, unknown>
  >;
  assert.deepEqual(
    parts.find((p) => p.type === "self_note"),
    { type: "self_note", content: "Task: finish the migration." },
  );
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
