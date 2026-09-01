// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  orderByParentChain,
  orderBySelectedBranch,
  resolveParentId,
} from "../src/features/chat/utils/message-order.ts";

const EDITED_ROOT = [
  { id: "a", parentId: null, createdAt: 1, role: "user" },
  { id: "reply-a", parentId: "a", createdAt: 2, role: "assistant" },
  { id: "b", parentId: null, createdAt: 3, role: "user" },
  { id: "reply-b", parentId: "b", createdAt: 4, role: "assistant" },
];

test("a stored null parentId is a root, not a link to the message before it", () => {
  assert.equal(resolveParentId({ id: "b", parentId: null }, "reply-a"), null);
});

test("a record from before the field existed still chains to its predecessor", () => {
  assert.equal(resolveParentId({ id: "b" }, "reply-a"), "reply-a");
  assert.equal(resolveParentId({ id: "b" }, null), null);
});

test("an undefined parentId is treated as absent, not as a root", () => {
  assert.equal(
    resolveParentId({ id: "b", parentId: undefined }, "reply-a"),
    "reply-a",
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

test("a mixed thread keeps its legacy chain and its recorded roots apart", () => {
  const mixed = [
    { id: "old-1", createdAt: 1, role: "user" },
    { id: "old-2", createdAt: 2, role: "assistant" },
    { id: "new-root", parentId: null, createdAt: 3, role: "user" },
    { id: "new-reply", parentId: "new-root", createdAt: 4, role: "assistant" },
  ];

  assert.deepEqual(
    orderBySelectedBranch(mixed).map(({ id }) => id),
    ["new-root", "new-reply"],
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

type Row = { id: string; parentId?: string | null; createdAt?: number; role?: string };

function rng(seed: number) { let s = seed; return () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648; }

function generate(r: () => number): Row[] {
  const n = 2 + Math.floor(r() * 10);
  const rows: Row[] = [];
  for (let i = 0; i < n; i++) {
    const roll = r();
    let parentId: string | null | undefined;
    if (i === 0) parentId = null;
    else if (roll < 0.15) parentId = null;                       // a second root
    else if (roll < 0.30) parentId = undefined;                  // pre-parentId record
    else if (roll < 0.40) parentId = `ghost-${i}`;               // dangling reference
    else parentId = rows[Math.floor(r() * rows.length)].id;
    const row: Row = { id: `m${i}`, createdAt: i, role: i % 2 ? "assistant" : "user" };
    if (parentId !== undefined) row.parentId = parentId;
    rows.push(row);
  }
  return rows;
}

test("the walk always terminates and never repeats a message", () => {
  const r = rng(4242);
  for (let c = 0; c < 500; c++) {
    const rows = generate(r);
    const out = orderBySelectedBranch(rows);
    const ids = out.map((m) => m.id);
    assert.equal(new Set(ids).size, ids.length, `repeat in ${JSON.stringify(ids)}`);
    assert.ok(out.length <= rows.length);
    for (const m of out) assert.ok(rows.some((x) => x.id === m.id));
  }
});

test("a cycle cannot hang the walk", () => {
  const cyclic: Row[] = [
    { id: "a", parentId: "b", createdAt: 1 },
    { id: "b", parentId: "a", createdAt: 2 },
  ];
  const out = orderBySelectedBranch(cyclic);
  assert.ok(out.length <= 2);
});

test("a self-parent cannot hang the walk", () => {
  const out = orderBySelectedBranch([{ id: "a", parentId: "a", createdAt: 1 }]);
  assert.deepEqual(out.map((m) => m.id), ["a"]);
});

test("the branch always ends at the newest message", () => {
  const r = rng(99);
  for (let c = 0; c < 300; c++) {
    const rows = generate(r);
    const out = orderBySelectedBranch(rows);
    const newest = [...rows].sort((a, b) => (a.createdAt ?? 0) - (b.createdAt ?? 0)).at(-1);
    assert.equal(out.at(-1)?.id, newest?.id);
  }
});

test("threads with no explicit null parent are completely unaffected", () => {
  // The blast radius: only a STORED null changes meaning. Records that all carry a real
  // parent, or none at all, must order exactly as they did before.
  const r = rng(7);
  for (let c = 0; c < 300; c++) {
    const rows = generate(r).filter((m) => m.parentId !== null);
    const viaOld = (() => {                       // main's rule: `parentId ?? previous`
      const sorted = [...rows].sort((a, b) => (a.createdAt ?? 0) - (b.createdAt ?? 0));
      const parentOf = new Map<string, string | null>();
      let prev: string | null = null;
      for (const m of sorted) { parentOf.set(m.id, m.parentId ?? prev); prev = m.id; }
      const chain: string[] = []; const seen = new Set<string>();
      let cur: string | null = sorted.at(-1)?.id ?? null;
      while (cur != null && !seen.has(cur)) {
        seen.add(cur);
        if (!sorted.some((x) => x.id === cur)) break;
        chain.push(cur); cur = parentOf.get(cur) ?? null;
      }
      return chain.reverse();
    })();
    assert.deepEqual(orderBySelectedBranch(rows).map((m) => m.id), viaOld);
  }
});

test("orderByParentChain with siblings still returns every message", () => {
  const r = rng(31);
  for (let c = 0; c < 300; c++) {
    const rows = generate(r);
    const out = orderByParentChain(rows);
    assert.equal(new Set(out.map((m) => m.id)).size, rows.length);
  }
});

test("resolveParentId is total over the three shapes", () => {
  assert.equal(resolveParentId({ id: "x", parentId: null }, "p"), null);
  assert.equal(resolveParentId({ id: "x", parentId: "q" }, "p"), "q");
  assert.equal(resolveParentId({ id: "x" }, "p"), "p");
  assert.equal(resolveParentId({ id: "x" }, null), null);
});
