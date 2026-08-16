// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// #8977: useOwnsResearchMessage exported the whole thread inside a per-message render
// body, so one render pass over N messages exported N times and inspected N*N items.
// The answer is a property of the thread revision, so it is derived once and shared.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  getResearchRunId,
  researchOwnerIds,
} from "../src/features/chat/utils/research-ownership.ts";

type Item = { parentId: string | null; message: { metadata?: unknown } };

/** A thread of `n` messages whose reply to `m1` is a research report. */
function thread(n: number): { revision: object[]; items: Item[] } {
  const items: Item[] = [];
  for (let i = 0; i < n; i++) {
    items.push({
      parentId: i === 0 ? null : `m${i - 1}`,
      message: {
        metadata:
          i === 2 ? { custom: { researchRunId: "run-1" } } : { custom: {} },
      },
    });
  }
  return { revision: items.map(() => ({})), items };
}

test("a render pass over 200 messages exports the thread once", () => {
  const { revision, items } = thread(200);
  let exports = 0;
  const exportItems = () => {
    exports += 1;
    return items;
  };

  const owners = revision.map((_, i) =>
    researchOwnerIds(revision, exportItems).has(`m${i}`),
  );

  assert.equal(exports, 1);
  assert.equal(owners.filter(Boolean).length, 1);
  assert.equal(owners[1], true, "m1 owns the research reply");
  assert.equal(owners[0], false);
  assert.equal(owners[2], false);
});

test("a new revision is derived again", () => {
  const first = thread(4);
  let exports = 0;
  const count = (items: Item[]) => () => {
    exports += 1;
    return items;
  };
  researchOwnerIds(first.revision, count(first.items));
  researchOwnerIds(first.revision, count(first.items));
  assert.equal(exports, 1);

  // What a delete, a fork or an edit produces: a new message list.
  const second = thread(3);
  assert.equal(researchOwnerIds(second.revision, count(second.items)).has("m1"), true);
  assert.equal(exports, 2);

  // A revision whose research reply is gone owns nothing.
  const third = { revision: [{}], items: [{ parentId: null, message: {} }] };
  assert.equal(researchOwnerIds(third.revision, count(third.items)).size, 0);
  assert.equal(exports, 3);
});

test("ownership reads both metadata shapes, and a root message owns nothing", () => {
  const revision = [{}];
  const items: Item[] = [
    { parentId: null, message: { metadata: { custom: { researchRunId: "r" } } } },
    { parentId: "a", message: { metadata: { custom: { researchRun: { id: "r" } } } } },
    { parentId: "b", message: { metadata: { custom: { researchRunId: 7 } } } },
    { parentId: "c", message: {} },
  ];
  const owners = researchOwnerIds(revision, () => items);
  assert.equal(owners.has("a"), true);
  assert.equal(owners.has("b"), false, "a non-string run id is not a research reply");
  assert.equal(owners.has("c"), false);
  assert.equal(owners.size, 1);
});

test("getResearchRunId keeps its contract", () => {
  assert.equal(getResearchRunId({ custom: { researchRunId: "r" } }), "r");
  assert.equal(getResearchRunId({ custom: { researchRun: { id: "r" } } }), "r");
  assert.equal(getResearchRunId({ custom: {} }), null);
  assert.equal(getResearchRunId(undefined), null);
});

test("the message render body no longer exports the thread", () => {
  const thread = readFileSync(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(thread, /\.thread\(\)\s*\.export\(\)\s*\.messages\.some\(/s);
  assert.match(thread, /researchOwnerIds\(\s*messages,/);
});
