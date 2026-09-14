// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const adapter = readSrc("features/chat/api/chat-adapter.ts");

test("a Stop with no output is recognised by what it puts on the wire", () => {
  assert.match(adapter, /function isAbandonedAssistantTurn\(/);
  assert.match(
    adapter,
    /!hasReplayContent\(only\.content\) &&\s*!only\.tool_calls &&\s*!only\.reasoning_content/,
  );
});

test("whitespace is not content, the way the backend already reads it", () => {
  assert.match(adapter, /function hasReplayContent\(/);
  assert.match(
    adapter,
    /if \(typeof content === "string"\) return content\.trim\(\)\.length > 0;/,
  );
  assert.match(
    adapter,
    /part\.type === "text" && part\.text\.trim\(\)\.length > 0/,
  );
});

test("a turn that carries payload of its own is never abandoned", () => {
  assert.match(adapter, /function assistantTurnCarriesPayload\(/);
  assert.match(
    adapter,
    /if \(assistantTurnCarriesPayload\(message\)\) return false;/,
  );
});

test("a tool call is left to the wire shape, not counted as payload up front", () => {
  const start = adapter.indexOf("function assistantTurnCarriesPayload(");
  const end = adapter.indexOf("function hasReplayContent(");
  assert.ok(start > 0 && end > start);
  // A call the replay can carry already leaves tool_calls on the wire; one it cannot carry
  // reaches the provider as nothing; short-circuiting here would keep the empty turn the
  // backend drops, stranding the pair this prune exists to repair.
  assert.doesNotMatch(adapter.slice(start, end), /part\.type === "tool-call"/);
});

test("a turn that finished on reasoning alone keeps its prompt", () => {
  assert.match(adapter, /function assistantTurnEndedEarly\(/);
  assert.match(adapter, /!assistantTurnEndedEarly\(message\) &&/);
});

test("an abandoned turn is pruned with the user prompt that triggered it", () => {
  assert.match(
    adapter,
    /if \(last && last\.role === "user"\) surviving\.pop\(\)/,
  );
});

test("a stopped empty assistant is filled before the prune sees it", () => {
  assert.match(adapter, /function fillStoppedAssistantReplay\(/);
  assert.match(
    adapter,
    /return fillStoppedAssistantReplay\(\s*message,\s*serializeAssistantReplayMessages\(/,
  );
  assert.match(adapter, /function stoppedAssistantReplayText\(/);
  assert.match(adapter, /incompleteLabel\(info\?\.reason \?\? fromStatus\)/);
});

test("the fill leaves a refusal suppressed", () => {
  assert.match(
    adapter,
    /if \(serialized\.length === 0\) \{[\s\S]{0,200}?return serialized;/,
  );
  assert.doesNotMatch(
    adapter,
    /serialized\.length === 0[\s\S]{0,200}content: stoppedAssistantReplayText/,
  );
});

test("a Stop while a model loads arrives as a Stop", () => {
  assert.match(
    adapter,
    /reject\(abortSignal\.reason \?\? new DOMException\("Aborted", "AbortError"\)\)/,
  );
});

test("a Stop that beats the durable admission arrives as a Stop", () => {
  assert.match(adapter, /if \(runSignal\.aborted\) \{/);
  assert.match(
    adapter,
    /throw runSignal\.reason \?\?\s*new DOMException\("Aborted", "AbortError"\)/,
  );
});

test("an admission that answered without a run is not replayed as a Stop", () => {
  // Ungated, a transport failure files as a cancellation: no Retry, and a stop label on the wire.
  assert.match(
    adapter,
    /throw new Error\(\s*"The server accepted the request without starting a generation run",/,
  );
});

test("assistant-ui still stops a run with an AbortError, not a bare detach marker", () => {
  // The forwarded reason only reads as a Stop while cancelRun throws this class, not the plain
  // `{ detach }` our tests stand in with; otherwise it must be normalised before it is thrown.
  const core = new URL(
    "../node_modules/@assistant-ui/core/dist/runtimes/local/local-thread-runtime-core.js",
    import.meta.url,
  );
  if (!existsSync(core)) return;
  const source = readFileSync(core, "utf8");
  assert.match(
    source,
    /class AbortError extends Error \{\s*name = "AbortError";/,
  );
  assert.match(
    source,
    /cancelRun\(\) \{\s*const error = new AbortError\(false\);/,
  );
  assert.match(source, /detach\(\) \{\s*const error = new AbortError\(true\);/);
});

test("only a deliberate Stop is replayed as one", () => {
  // Collapsing this back to a constant tells the model a failed turn was stopped.
  assert.match(
    adapter,
    /status\?\.type !== "incomplete" \|\| status\.reason === "cancelled"/,
  );
});

test("a trailing abandoned turn keeps the prompt it followed", () => {
  assert.match(adapter, /if \(refused \|\| index < lastSurviving\) \{/);
});

test("the send and token-count paths prune through the same helper", () => {
  assert.match(
    adapter,
    /const survivingMessages = pruneOutboundHistory\(messages, true\)/,
  );
  assert.match(
    adapter,
    /const survivingMessages = pruneOutboundHistory\(\s*messages,\s*!isExternalRequest,\s*\);/,
  );
});
