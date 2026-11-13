// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// #10288: a leaf still mounted from the outgoing thread re-renders before React unmounts it and
// reads the incoming thread through its old index. assistant-ui 0.12 threw there and took down the app.

import assert from "node:assert/strict";
import test from "node:test";

import {
  EMPTY_TEXT_PART,
  createStaleSafeMessageResolver,
  textPartOrEmpty,
} from "../src/components/assistant-ui/stale-safe-scope-logic.ts";
import { readSrc } from "./helpers/kit.ts";

function lookup<T>(items: readonly T[], index: number): T {
  if (index < 0 || index >= items.length) {
    throw new Error(
      `tapClientLookup: Index ${index} out of bounds (length: ${items.length})`,
    );
  }
  return items[index];
}

/** A message client whose `part({ index })` throws exactly as assistant-ui's tapClientLookup does. */
function message(name: string, parts: string[] = ["text"]) {
  return {
    name,
    getState: () => ({ parts }),
    part: (selector: { index: number } | { toolCallId: string }) =>
      "index" in selector
        ? `${name}.${lookup(parts, selector.index)}`
        : `${name}.tool`,
  };
}
type Msg = ReturnType<typeof message>;

function thread(messages: Msg[]) {
  return {
    getState: () => ({ messages }),
    message: ({ index }: { index: number }) => lookup(messages, index),
  };
}

const resolver = (index: number) =>
  createStaleSafeMessageResolver<string, Msg>(index);

test("New chat: the outgoing reply row keeps its message instead of throwing on the empty thread", () => {
  const resolve = resolver(1);
  assert.equal(
    resolve(thread([message("user"), message("assistant")])).name,
    "assistant",
  );
  // switchToNewThread() installs an empty thread while the row at index 1 is still mounted.
  const empty = thread([]);
  assert.throws(
    () => empty.message({ index: 1 }),
    /Index 1 out of bounds \(length: 0\)/,
  );
  assert.equal(resolve(empty).name, "assistant");
  // Opening a saved chat that is shorter, or still loading, is the same window.
  assert.equal(resolve(thread([message("other user")])).name, "assistant");
});

test("an in-range index always reads the current thread, so a same-length switch shows the new chat", () => {
  const resolve = resolver(1);
  assert.equal(resolve(thread([message("u"), message("first")])).name, "first");
  assert.equal(
    resolve(thread([message("u"), message("second")])).name,
    "second",
  );
  assert.equal(
    resolve(thread([message("u"), message("a"), message("u"), message("b")]))
      .name,
    "a",
  );
});

test("a part leaf past the incoming message's parts keeps its last part instead of throwing", () => {
  const resolve = resolver(1);
  const streaming = message("streaming", ["reasoning", "text"]);
  assert.equal(
    resolve(thread([message("u"), streaming])).part({ index: 1 }),
    "streaming.text",
  );
  // The chat opened mid-reply has one part at that message index.
  const saved = message("saved", ["text"]);
  assert.throws(
    () => saved.part({ index: 1 }),
    /Index 1 out of bounds \(length: 1\)/,
  );
  const row = resolve(thread([message("u"), saved]));
  assert.equal(row.name, "saved");
  assert.equal(row.part({ index: 1 }), "streaming.text");
  assert.equal(row.part({ index: 0 }), "saved.text");
  assert.equal(row.part({ toolCallId: "call_0" }), "saved.tool");
  // One wrapper per message client, so the scope's identity is stable across reads.
  assert.equal(resolve(thread([message("u"), saved])), row);
});

test("wrapping a real assistant-ui client proxy reads through it without tripping Proxy invariants", () => {
  // Shaped like @assistant-ui/store's ClientProxyHandler: an empty target, and a
  // getOwnPropertyDescriptor (BaseProxyHandler) that reports every key as non-configurable.
  const OUTPUT = Symbol("assistant-ui.store.getValue");
  const impl: Record<PropertyKey, unknown> = {
    ...message("assistant", ["text"]),
    [OUTPUT]: { getState: () => ({ parts: ["text"] }) },
  };
  const client = new Proxy({} as Msg, {
    get: (_, prop) => impl[prop],
    has: (_, prop) => prop in impl,
    getOwnPropertyDescriptor: (_, prop) =>
      impl[prop] === undefined
        ? undefined
        : {
            value: impl[prop],
            writable: false,
            enumerable: true,
            configurable: false,
          },
  });
  const row = resolver(1)(thread([message("u"), client]));
  assert.equal(row.name, "assistant");
  assert.equal(
    (row as unknown as Record<symbol, unknown>)[OUTPUT],
    impl[OUTPUT],
  );
  assert.ok(OUTPUT in row);
  assert.equal(row.part({ index: 0 }), "assistant.text");
});

test("a row or part that never resolved still reports the lookup error", () => {
  assert.throws(
    () => resolver(3)(thread([message("u")])),
    /Index 3 out of bounds \(length: 1\)/,
  );
  const row = resolver(0)(thread([message("u", ["text"])]));
  assert.throws(
    () => row.part({ index: 2 }),
    /Index 2 out of bounds \(length: 1\)/,
  );
});

test("a text leaf pointed at a tool call reads one frozen empty text part", () => {
  const tool = {
    type: "tool-call",
    toolName: "python",
    status: { type: "running" },
  };
  const first = textPartOrEmpty(tool as never);
  assert.equal(first, EMPTY_TEXT_PART);
  // Same object every time: useSyncExternalStore loops on a snapshot that changes per read.
  assert.equal(textPartOrEmpty(tool as never), first);
  assert.ok(Object.isFrozen(first));
  assert.deepEqual(first, {
    type: "text",
    text: "",
    status: { type: "complete" },
  });
  for (const part of [
    { type: "text", text: "hi", status: { type: "running" } },
    { type: "reasoning", text: "hmm", status: { type: "complete" } },
  ]) {
    assert.equal(textPartOrEmpty(part as never), part);
  }
});

test("Studio reads rows and text parts only through the stale-safe scopes", () => {
  const rows = readSrc("components/assistant-ui/progressive-messages.tsx");
  assert.match(
    rows,
    /import \{ MessageByIndexProvider \} from "@\/components\/assistant-ui\/stale-safe-scopes";/,
  );
  const markdown = readSrc("components/assistant-ui/markdown-text.tsx");
  assert.match(markdown, /useMessagePartTextSafe\(\)/);
  assert.doesNotMatch(markdown, /\buseMessagePartText\b/);
  const scopes = readSrc("components/assistant-ui/stale-safe-scopes.tsx");
  assert.match(scopes, /get: \(aui\) => resolve\(aui\.thread\(\)\),/);
  assert.match(
    scopes,
    /get: \(aui\) => resolve\(aui\.thread\(\)\)\.composer\(\),/,
  );
});
