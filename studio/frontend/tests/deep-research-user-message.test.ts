// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "@typescript/typescript6";

import {
  appendQuoteContextToText,
  buildQuoteContextEnvelope,
  quoteContextTransport,
} from "../src/components/assistant-ui/message-quote-context.ts";
import type { QuoteAnchor } from "../src/components/assistant-ui/message-quote-model.ts";
import { deepResearchUserMessageRecord } from "../src/features/chat/utils/deep-research-user-message.ts";
import { readSrc } from "./helpers/kit.ts";

const anchor: QuoteAnchor = {
  messageId: "assistant-1",
  turn: 1,
  text: "quoted passage",
  markdown: "quoted passage",
  start: 0,
  end: 14,
  prefix: "",
  suffix: "",
  occurrence: 1,
};

function quoteTransport(comment = "Keep this comment.") {
  const envelope = buildQuoteContextEnvelope(
    [
      {
        id: "quote-1",
        anchor,
        comment,
        createdAt: 1,
      },
    ],
    "ctx-deep-research",
  );
  assert.ok(envelope);
  return quoteContextTransport(envelope);
}

const quote = quoteTransport();
const COMMENT_RE = /Keep this comment\./;
const PASSAGE_RE = /quoted passage/;

function record(
  overrides: {
    storedMetadata?: Record<string, unknown>;
    liveMetadata?: Record<string, unknown>;
  } = {},
) {
  return deepResearchUserMessageRecord({
    id: "user-1",
    threadId: "thread-1",
    parentId: null,
    content: [{ type: "text", text: "Research this" }],
    attachments: [],
    createdAt: 123,
    ...overrides,
  });
}

test("stored metadata is authoritative and explicit null clears survive", () => {
  const built = record({
    storedMetadata: {
      quote,
      revision: 7,
      scheduledTask: { id: "task-1" },
      cleared: null,
    },
    liveMetadata: {
      quote: { messageId: "stale", text: "stale" },
      revision: 99,
      cleared: "should not return",
      liveOnly: "fallback",
    },
  });

  assert.deepEqual(built.metadata, {
    quote,
    revision: 7,
    scheduledTask: { id: "task-1" },
    cleared: null,
    liveOnly: "fallback",
  });
});

test("live metadata is the fallback when the stored row has no value", () => {
  const built = record({
    liveMetadata: { quote, liveOnly: true },
  });

  assert.deepEqual(built.metadata, { quote, liveOnly: true });
});

test("a missing stored row keeps the live quote and foreign fields", () => {
  const built = record({
    liveMetadata: { quote, friendly: "field" },
  });

  assert.equal(built.metadata?.quote, quote);
  assert.equal(built.metadata?.friendly, "field");
});

test("no metadata produces no empty metadata field", () => {
  assert.equal(Object.hasOwn(record(), "metadata"), false);
});

test("quote comments remain available to the next model request", () => {
  const text = appendQuoteContextToText("", quote);
  assert.match(text, COMMENT_RE);
  assert.match(text, PASSAGE_RE);
});

function liftedSaveCall() {
  const text = readSrc("features/chat/api/chat-adapter.ts");
  const source = ts.createSourceFile(
    "chat-adapter.ts",
    text,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TS,
  );
  let start: ts.Node | undefined;
  const findStart = (node: ts.Node) => {
    if (
      ts.isVariableDeclaration(node) &&
      node.name.getText(source) === "startDeepResearch"
    ) {
      start = node;
    }
    ts.forEachChild(node, findStart);
  };
  findStart(source);
  assert.ok(start, "startDeepResearch declaration missing");

  const calls: ts.CallExpression[] = [];
  const findCall = (node: ts.Node) => {
    if (
      ts.isCallExpression(node) &&
      ts.isIdentifier(node.expression) &&
      node.expression.text === "saveStoredChatMessage"
    ) {
      calls.push(node);
    }
    ts.forEachChild(node, findCall);
  };
  findCall(start);
  assert.equal(
    calls.length,
    1,
    "startDeepResearch must save one complete user-message snapshot",
  );
  const call = calls[0];
  assert.ok(call);
  return ts.transpileModule(
    `return (async () => (${call.getText(source)}))();`,
    {
      compilerOptions: {
        target: ts.ScriptTarget.ES2022,
        module: ts.ModuleKind.None,
      },
    },
  ).outputText;
}

test("the actual startDeepResearch save payload includes quote metadata", async () => {
  const saved: Record<string, unknown>[] = [];
  const saveCall = new Function(
    "deepResearchUserMessageRecord",
    "saveStoredChatMessage",
    "userMessage",
    "storedUserMessage",
    "userMessageParentId",
    "resolvedThreadId",
    liftedSaveCall(),
  );
  const userMessage = {
    id: "user-1",
    content: [{ type: "text", text: "Research this" }],
    // The helper must read the nested live shape, never metadata.quote.
    metadata: { custom: { quote, liveOnly: true } },
    createdAt: new Date(456),
  };
  const storedUserMessage = {
    id: "user-1",
    parentId: null,
    metadata: {
      quote,
      foreign: { preserved: true },
      cleared: null,
    },
  };

  const built = await saveCall(
    deepResearchUserMessageRecord,
    (payload: Record<string, unknown>) => {
      saved.push(payload);
      return payload;
    },
    userMessage,
    storedUserMessage,
    "assistant-parent",
    "thread-1",
  );

  assert.equal(saved.length, 1, "the actual save call ran");
  const [record] = saved;
  assert.ok(record);
  const metadata = record.metadata as Record<string, unknown>;
  assert.equal(metadata.quote, quote);
  assert.deepEqual(metadata.foreign, { preserved: true });
  assert.equal(metadata.cleared, null);
  assert.equal(metadata.liveOnly, true);
  assert.equal(record.parentId, null);
  assert.equal(record.createdAt, 456);
  assert.deepEqual(record, built);
});

test("the actual save call handles a source row that has not persisted yet", async () => {
  const saved: Record<string, unknown>[] = [];
  const saveCall = new Function(
    "deepResearchUserMessageRecord",
    "saveStoredChatMessage",
    "userMessage",
    "storedUserMessage",
    "userMessageParentId",
    "resolvedThreadId",
    liftedSaveCall(),
  );
  await saveCall(
    deepResearchUserMessageRecord,
    (payload: Record<string, unknown>) => {
      saved.push(payload);
      return payload;
    },
    {
      id: "user-1",
      content: [{ type: "text", text: "Research this" }],
      metadata: { custom: { quote } },
      createdAt: new Date(456),
    },
    undefined,
    "parent-1",
    "thread-1",
  );

  const [savedRecord] = saved;
  assert.ok(savedRecord);
  const metadata = savedRecord.metadata as Record<string, unknown>;
  assert.equal(metadata.quote, quote);
  assert.equal(savedRecord.parentId, "parent-1");
});
