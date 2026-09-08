// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  createParentResolver,
  orderBySelectedBranch,
} from "../src/features/chat/utils/message-order.ts";

type Shape = { id: string; parentId?: string | null };

function resolveAll(rows: Shape[]): (string | null)[] {
  const resolveParent = createParentResolver();
  return rows.map((row) => resolveParent(row));
}

const EDITED_ROOT = [
  { id: "a", parentId: null, createdAt: 1, role: "user" },
  { id: "reply-a", parentId: "a", createdAt: 2, role: "assistant" },
  { id: "b", parentId: null, createdAt: 3, role: "user" },
  { id: "reply-b", parentId: "b", createdAt: 4, role: "assistant" },
];

test("a stored null is a root once the thread has recorded a real parent", () => {
  assert.deepEqual(
    resolveAll([
      { id: "a", parentId: null },
      { id: "reply-a", parentId: "a" },
      { id: "b", parentId: null },
    ]),
    [null, "a", null],
  );
});

test("a record from before the field existed still chains to its predecessor", () => {
  assert.deepEqual(resolveAll([{ id: "a" }, { id: "b" }]), [null, "a"]);
});

test("an undefined parentId is treated as absent, not as a root", () => {
  assert.deepEqual(
    resolveAll([{ id: "a" }, { id: "b", parentId: undefined }]),
    [null, "a"],
  );
});

test("a leading run of stored nulls is legacy, so storage order stands in", () => {
  // Server rows always carry the key; with no recorded parent yet, these nulls chain.
  assert.deepEqual(
    resolveAll([
      { id: "old-1", parentId: null },
      { id: "old-2", parentId: null },
      { id: "new", parentId: "old-2" },
      { id: "root", parentId: null },
    ]),
    [null, "old-1", "old-2", null],
  );
});

test("editing the first message leaves two branches, not one conversation", () => {
  assert.deepEqual(
    orderBySelectedBranch(EDITED_ROOT).map(({ id }) => id),
    ["b", "reply-b"],
  );
});

test("a legacy thread with no parentIds at all still loads in order", () => {
  const legacy = [
    { id: "u1", createdAt: 1, role: "user" },
    { id: "a1", createdAt: 2, role: "assistant" },
    { id: "u2", createdAt: 3, role: "user" },
    { id: "a2", createdAt: 4, role: "assistant" },
  ];

  assert.deepEqual(
    orderBySelectedBranch(legacy).map(({ id }) => id),
    ["u1", "a1", "u2", "a2"],
  );
});

test("a mixed thread keeps its legacy chain instead of dropping it", () => {
  const mixed = [
    { id: "old-1", createdAt: 1, role: "user" },
    { id: "old-2", createdAt: 2, role: "assistant" },
    { id: "new-1", parentId: "old-2", createdAt: 3, role: "user" },
    { id: "new-2", parentId: "new-1", createdAt: 4, role: "assistant" },
  ];

  assert.deepEqual(
    orderBySelectedBranch(mixed).map(({ id }) => id),
    ["old-1", "old-2", "new-1", "new-2"],
  );
});

test("a server-backed legacy thread survives its first parent-linked turn", () => {
  // Rooting these explicit legacy nulls dropped every turn before the first linked one.
  const serverBacked = [
    { id: "old-u1", parentId: null, createdAt: 1, role: "user" },
    { id: "old-a1", parentId: null, createdAt: 2, role: "assistant" },
    { id: "old-u2", parentId: null, createdAt: 3, role: "user" },
    { id: "old-a2", parentId: null, createdAt: 4, role: "assistant" },
    { id: "new-u3", parentId: "old-a2", createdAt: 5, role: "user" },
    { id: "new-a3", parentId: "new-u3", createdAt: 6, role: "assistant" },
  ];

  assert.deepEqual(
    orderBySelectedBranch(serverBacked).map(({ id }) => id),
    ["old-u1", "old-a1", "old-u2", "old-a2", "new-u3", "new-a3"],
  );
});

test("editing the first message of a server-backed thread keeps both branches", () => {
  const edited = [
    { id: "u1", parentId: null, createdAt: 1, role: "user" },
    { id: "a1", parentId: "u1", createdAt: 2, role: "assistant" },
    { id: "u1-edited", parentId: null, createdAt: 3, role: "user" },
    { id: "a1-edited", parentId: "u1-edited", createdAt: 4, role: "assistant" },
  ];

  assert.deepEqual(
    orderBySelectedBranch(edited).map(({ id }) => id),
    ["u1-edited", "a1-edited"],
  );
});

test("retrying a reply mid-thread still selects the newest branch", () => {
  const retried = [
    { id: "u1", parentId: null, createdAt: 1, role: "user" },
    { id: "a1-old", parentId: "u1", createdAt: 2, role: "assistant" },
    { id: "a1-new", parentId: "u1", createdAt: 3, role: "assistant" },
    { id: "u2", parentId: "a1-new", createdAt: 4, role: "user" },
    { id: "a2", parentId: "u2", createdAt: 5, role: "assistant" },
  ];

  assert.deepEqual(
    orderBySelectedBranch(retried).map(({ id }) => id),
    ["u1", "a1-new", "u2", "a2"],
  );
});

type Row = {
  id: string;
  parentId?: string | null;
  createdAt?: number;
  role?: string;
};

function rng(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1103515245 + 12345) % 2147483648;
    return s / 2147483648;
  };
}

function generate(r: () => number): Row[] {
  const n = 2 + Math.floor(r() * 10);
  const rows: Row[] = [];
  for (let i = 0; i < n; i++) {
    const roll = r();
    let parentId: string | null | undefined;
    if (i === 0) {
      parentId = null;
    } else if (roll < 0.15) {
      parentId = null; // a second root
    } else if (roll < 0.3) {
      parentId = undefined; // pre-parentId record
    } else if (roll < 0.4) {
      parentId = `ghost-${i}`; // dangling reference
    } else {
      parentId = rows[Math.floor(r() * rows.length)].id;
    }
    const row: Row = {
      id: `m${i}`,
      createdAt: i,
      role: i % 2 ? "assistant" : "user",
    };
    if (parentId !== undefined) {
      row.parentId = parentId;
    }
    rows.push(row);
  }
  return rows;
}

// The previous rule, `parentId ?? previous`, kept verbatim as the reference.
function orderByPreviousRule(rows: Row[]): string[] {
  const sorted = [...rows].sort(
    (a, b) => (a.createdAt ?? 0) - (b.createdAt ?? 0),
  );
  const parentOf = new Map<string, string | null>();
  let prev: string | null = null;
  for (const m of sorted) {
    parentOf.set(m.id, m.parentId ?? prev);
    prev = m.id;
  }
  const chain: string[] = [];
  const seen = new Set<string>();
  let cur: string | null = sorted.at(-1)?.id ?? null;
  while (cur != null && !seen.has(cur)) {
    seen.add(cur);
    if (!sorted.some((x) => x.id === cur)) {
      break;
    }
    chain.push(cur);
    cur = parentOf.get(cur) ?? null;
  }
  return chain.reverse();
}

test("threads with no explicit null parent are completely unaffected", () => {
  // Only a stored null changes meaning; everything else must order as before.
  const r = rng(7);
  for (let c = 0; c < 300; c++) {
    const rows = generate(r).filter((m) => m.parentId !== null);
    assert.deepEqual(
      orderBySelectedBranch(rows).map((m) => m.id),
      orderByPreviousRule(rows),
    );
  }
});

test("the resolver is total over the three shapes", () => {
  assert.deepEqual(
    resolveAll([
      { id: "p", parentId: null },
      { id: "link", parentId: "p" },
      { id: "root", parentId: null },
      { id: "absent" },
      { id: "dangling", parentId: "gone" },
    ]),
    [null, "p", null, "root", "gone"],
  );
});
