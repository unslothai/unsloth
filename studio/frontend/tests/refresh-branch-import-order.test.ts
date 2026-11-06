// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { ThreadMessage } from "@assistant-ui/core";
import { MessageRepository } from "@assistant-ui/core/internal"; // intentional internal API
import {
  compareStoredMessages,
  createParentResolver,
  orderParentsFirst,
  resolveSavedBranchHead,
} from "../src/features/chat/utils/message-order.ts";

type Row = {
  id: string;
  parentId: string | null;
  createdAt: number;
  role: string;
};

// edited user messages can be stored after replies, so createdAt alone orders replies first.
const ANOMALOUS = [
  { id: "R", parentId: null, createdAt: 1, role: "user" },
  { id: "A1", parentId: "R", createdAt: 2, role: "assistant" },
  { id: "E1", parentId: "A1", createdAt: 10, role: "user" },
  { id: "R1", parentId: "E1", createdAt: 9, role: "assistant" },
  { id: "E2", parentId: "A1", createdAt: 20, role: "user" },
  { id: "R2", parentId: "E2", createdAt: 19, role: "assistant" },
];

// use the production comparator to exercise the load path's ordering.
function byCreatedAt(rows: Row[]): Row[] {
  return [...rows].sort(compareStoredMessages);
}

// assistant-ui import fails when a message precedes its parent.
function countMissingParents(ordered: Row[]): number {
  const seen = new Set<string>();
  let missing = 0;
  for (const row of ordered) {
    if (row.parentId !== null && !seen.has(row.parentId)) missing += 1;
    seen.add(row.id);
  }
  return missing;
}

// user and assistant messages require distinct properties, so assert the literal as the union type.
function threadMessage(id: string, role: string): ThreadMessage {
  return {
    id,
    role: role as "user" | "assistant",
    content: [{ type: "text" as const, text: `text-${id}` }],
    createdAt: new Date(0),
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {},
    },
    ...(role === "user"
      ? { attachments: [] }
      : { status: { type: "complete" as const, reason: "stop" as const } }),
  } as ThreadMessage;
}

test("createdAt order alone misorders a reply before its edited user parent", () => {
  assert.ok(countMissingParents(byCreatedAt(ANOMALOUS)) > 0);
});

test("orderParentsFirst emits every parent before its child", () => {
  const ordered = orderParentsFirst(ANOMALOUS);
  assert.equal(countMissingParents(ordered), 0);
});

test("orderParentsFirst handles a long reversed chain without recursion", () => {
  const rows = Array.from({ length: 10_000 }, (_, index) => ({
    id: String(index),
    parentId: index === 0 ? null : String(index - 1),
  })).reverse();
  const ordered = orderParentsFirst(rows);
  assert.equal(ordered.length, rows.length);
  assert.equal(ordered[0]?.id, "0");
  assert.equal(ordered.at(-1)?.id, "9999");
});

test("the fallback head stays on the newest row's branch after reordering", () => {
  const rows = byCreatedAt([
    { id: "root", parentId: null, createdAt: 0, role: "user" },
    { id: "reply", parentId: "edited", createdAt: 1, role: "assistant" },
    { id: "other", parentId: "root", createdAt: 2, role: "assistant" },
    { id: "edited", parentId: "root", createdAt: 3, role: "user" },
  ]);
  const resolveParent = createParentResolver();
  const ordered = orderParentsFirst(
    rows.map((record) => ({
      record,
      id: record.id,
      parentId: resolveParent(record),
    })),
  );
  assert.equal(ordered.at(-1)?.id, "other");
  assert.equal(resolveSavedBranchHead(rows, rows.at(-1)?.id), "reply");
});

test("the saved-branch head resolves to a leaf, so import keeps every reply", () => {
  // mirror the load path through a real repository import.
  const resolveParent = createParentResolver();
  const ordered = orderParentsFirst(
    byCreatedAt(ANOMALOUS).map((m) => ({
      record: m,
      id: m.id,
      parentId: resolveParent(m),
    })),
  );
  // choose leaf R2 because resetHead drops a non-leaf head's descendants.
  const headId = resolveSavedBranchHead(byCreatedAt(ANOMALOUS), "A1");
  assert.equal(headId, "R2");
  const repo = new MessageRepository();
  repo.import({
    headId,
    messages: ordered.map(({ record, parentId }) => ({
      parentId,
      message: threadMessage(record.id, record.role),
    })),
  });
  const present = new Set(repo.export().messages.map((m) => m.message.id));
  for (const row of ANOMALOUS)
    assert.ok(present.has(row.id), `${row.id} was dropped`);
});

test("a missing parent is treated as a root rather than crashing or looping", () => {
  const rows = [
    { id: "orphan", parentId: "gone", createdAt: 1, role: "user" },
    { id: "child", parentId: "orphan", createdAt: 2, role: "assistant" },
  ];
  // ignore missing-parent edges so present parents stay ordered and the walk terminates.
  assert.deepEqual(
    orderParentsFirst(rows).map((r) => r.id),
    ["orphan", "child"],
  );
});
