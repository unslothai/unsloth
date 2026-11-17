// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A spoken reply cut before any of it was heard is dropped from the history, but the question it
// answered is not: it joins the next prompt, so the model still sees it and roles keep alternating.

import assert from "node:assert/strict";
import { stripTypeScriptTypes } from "node:module";
import test from "node:test";

import { withHeardCut } from "../src/features/chat/api/mcp-image.ts";
import { readSrc } from "./helpers/kit.ts";

const adapter = readSrc("features/chat/api/chat-adapter.ts");
const body = adapter.slice(adapter.indexOf("function pruneOutboundHistory("), adapter.indexOf("function extractImageBase64("));

type Msg = { id: string; role: string; content: { type: string; text?: string }[]; metadata?: unknown; attachments?: unknown[] };
// Abandoned: an assistant turn with no text left, which is what the real check reads off the wire.
const isAbandonedAssistantTurn = (m: Msg) => m.role === "assistant" && !m.content.some((p) => p.text?.trim());
const prune = new Function(
  "withHeardCut",
  "isAbandonedAssistantTurn",
  "isAnthropicRefusalMessage",
  `${stripTypeScriptTypes(body)}\nreturn pruneOutboundHistory;`,
)(withHeardCut, isAbandonedAssistantTurn, () => false) as (m: Msg[], r: boolean) => Msg[];

const text = (t: string) => [{ type: "text", text: t }];
const cutAt = (offset: number) => ({ custom: { voiceHeard: { offset, at: 1 } } });

test("a reply cut before any of it was heard keeps its question, joined to the next prompt", () => {
  const out = prune([
    { id: "u1", role: "user", content: text("What's the weather in Paris?") },
    { id: "a1", role: "assistant", content: text("It is sunny in Paris today."), metadata: cutAt(0) },
    { id: "u2", role: "user", content: text("and tomorrow?"), attachments: [{ name: "x" }] },
  ], false);
  assert.deepEqual(out.map((m) => m.role), ["user"]);
  assert.equal(out[0].id, "u2");
  assert.deepEqual(out[0].content.map((p) => p.text), ["What's the weather in Paris?", "and tomorrow?"]);
  assert.deepEqual(out[0].attachments, [{ name: "x" }]);
});

test("a reply abandoned on its own (a Stop before any output) still takes its prompt with it", () => {
  const out = prune([
    { id: "u1", role: "user", content: text("Hello?") },
    { id: "a1", role: "assistant", content: text("") },
    { id: "u2", role: "user", content: text("Hi.") },
  ], false);
  assert.deepEqual(out.map((m) => m.id), ["u2"]);
  assert.deepEqual(out[0].content.map((p) => p.text), ["Hi."]);
});

test("a reply heard in part stays, cut, and nothing is joined", () => {
  const out = prune([
    { id: "u1", role: "user", content: text("Tell me two things.") },
    { id: "a1", role: "assistant", content: text("First one. Second one."), metadata: cutAt(10) },
    { id: "u2", role: "user", content: text("Stop.") },
  ], false);
  assert.deepEqual(out.map((m) => m.id), ["u1", "a1", "u2"]);
});
