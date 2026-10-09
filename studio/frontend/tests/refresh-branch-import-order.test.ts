// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { ThreadMessage } from "@assistant-ui/core";
import { MessageRepository } from "@assistant-ui/core/internal"; // internal API: intentional
import {
  compareStoredMessages,
  createParentResolver,
  orderParentsFirst,
  resolveSavedBranchHead,
} from "../src/features/chat/utils/message-order.ts";

type Row = { id: string; parentId: string | null; createdAt: number; role: string };

// Mirrors the real failure: an edited user message is written after its reply, so the
// reply's createdAt predates its user parent. createdAt order alone puts the reply first.
const ANOMALOUS = [
  { id: "R", parentId: null, createdAt: 1, role: "user" },
  { id: "A1", parentId: "R", createdAt: 2, role: "assistant" },
  { id: "E1", parentId: "A1", createdAt: 10, role: "user" },
  { id: "R1", parentId: "E1", createdAt: 9, role: "assistant" }, // before E1
  { id: "E2", parentId: "A1", createdAt: 20, role: "user" },
  { id: "R2", parentId: "E2", createdAt: 19, role: "assistant" }, // before E2
];

// Same comparator the production load path uses, so the test exercises the real ordering.
function byCreatedAt(rows: Row[]): Row[] {
  return [...rows].sort(compareStoredMessages);
}

// assistant-ui's import() throws on the first item whose parentId isn't inserted yet.
function countMissingParents(ordered: Row[]): number {
  const seen = new Set<string>();
  let missing = 0;
  for (const row of ordered) {
    if (row.parentId !== null && !seen.has(row.parentId)) missing += 1;
    seen.add(row.id);
  }
  return missing;
}

// Minimal ThreadMessage for a real MessageRepository.import(). The user/assistant branches
// carry different required props, so the literal is asserted to the union type.
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
  // Documents the bug: in raw storage order the import would throw.
  assert.ok(countMissingParents(byCreatedAt(ANOMALOUS)) > 0);
});

test("orderParentsFirst emits every parent before its child", () => {
  const ordered = orderParentsFirst(ANOMALOUS);
  assert.equal(countMissingParents(ordered), 0);
});

test("the saved-branch head resolves to a leaf, so import keeps every reply", () => {
  // Mirrors the load() path: reorder parents-first, resolve the head via the saved-branch
  // logic (newest leaf below the saved head), and hand it to a real import().
  const resolveParent = createParentResolver();
  const ordered = orderParentsFirst(
    byCreatedAt(ANOMALOUS).map((m) => ({
      record: m,
      id: m.id,
      parentId: resolveParent(m),
    })),
  );
  // The user was last viewing A1 (a non-leaf); the head must advance to a leaf below it.
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
  // import drops a non-leaf head's descendants, so the leaf head must keep every row.
  const present = new Set(repo.export().messages.map((m) => m.message.id));
  for (const row of ANOMALOUS) assert.ok(present.has(row.id), `${row.id} was dropped`);
});

test("a missing parent is treated as a root rather than crashing or looping", () => {
  const rows = [
    { id: "orphan", parentId: "gone", createdAt: 1, role: "user" },
    { id: "child", parentId: "orphan", createdAt: 2, role: "assistant" },
  ];
  // The dangling reference is not an ordering constraint; the child must still follow
  // its (present) parent, and the walk must terminate.
  assert.deepEqual(orderParentsFirst(rows).map((r) => r.id), ["orphan", "child"]);
});
