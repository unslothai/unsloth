// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The sidebar's chat rows are derived from the whole thread list by groupThreads, which sorts and
// allocates a fresh array of fresh objects. Unmemoized that ran on every render of every consumer,
// and because the identity changed each time it also broke every downstream useMemo in
// app-sidebar.tsx that is keyed on the result. These tests pin the two halves of the fix:
//
//   1. the grouping itself is unchanged (it moved file, so every branch of it is re-pinned here),
//   2. the derived lists keep a STABLE IDENTITY while the threads have not changed, and a FRESH
//      identity the moment they have. Only asserting the first half would bless a stale rail.

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import React from "react";

import {
  SIDEBAR_THREAD_FIELDS,
  groupThreads,
  sameSidebarThreads,
  useSidebarThreadGroups,
} from "../src/features/chat/hooks/sidebar-thread-groups.ts";
import type { ThreadRecord } from "../src/features/chat/types.ts";

// ── a minimal hook runner ────────────────────────────────────────────────────
//
// There is no DOM and no renderer in this test suite, so hooks are driven through React's own
// dispatcher slot: React.useMemo resolves the dispatcher on every call, so installing one here
// runs the REAL React entry point against hook slots this file owns. Deps are compared the way
// React compares them (Object.is, elementwise), so a hook that memoizes on a value it rebuilds
// every render fails here exactly as it would in the browser.

interface MemoSlot {
  deps: unknown[] | undefined;
  value: unknown;
}

const reactInternals = (
  React as unknown as {
    __CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE?: {
      H: unknown;
    };
  }
).__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;

function depsEqual(
  previous: unknown[] | undefined,
  next: unknown[] | undefined,
): boolean {
  if (previous === undefined || next === undefined) return false;
  if (previous.length !== next.length) return false;
  for (let i = 0; i < previous.length; i += 1) {
    if (!Object.is(previous[i], next[i])) return false;
  }
  return true;
}

function createHookRunner<A, R>(hook: (arg: A) => R) {
  const slots: MemoSlot[] = [];
  let cursor = 0;
  let memoCalls = 0;
  const dispatcher = {
    useMemo<T>(create: () => T, deps: unknown[] | undefined): T {
      memoCalls += 1;
      const slot = slots[cursor];
      if (slot && depsEqual(slot.deps, deps)) {
        cursor += 1;
        return slot.value as T;
      }
      const value = create();
      slots[cursor] = { deps, value };
      cursor += 1;
      return value;
    },
  };
  return {
    render(arg: A): R {
      // Missing internals is a hard failure, never a skip: a runner that silently stopped driving
      // React would let every identity assertion below pass without testing anything.
      assert.ok(
        reactInternals,
        "React client internals are missing; the hook runner cannot drive useMemo",
      );
      cursor = 0;
      const previous = reactInternals.H;
      reactInternals.H = dispatcher;
      try {
        return hook(arg);
      } finally {
        reactInternals.H = previous;
      }
    },
    memoCalls: () => memoCalls,
  };
}

// ── fixtures ─────────────────────────────────────────────────────────────────

const EPOCH = 1_700_000_000_000;

function thread(overrides: Partial<ThreadRecord> & { id: string }): ThreadRecord {
  return {
    title: `chat ${overrides.id}`,
    modelType: "base",
    archived: false,
    createdAt: EPOCH,
    updatedAt: EPOCH,
    ...overrides,
  } as ThreadRecord;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── 1. the grouping is unchanged ─────────────────────────────────────────────

test("groupThreads on an empty history returns empty lists for both flags", () => {
  assert.deepEqual(groupThreads([]), []);
  assert.deepEqual(groupThreads([], true), []);
});

test("groupThreads treats a legacy record with no archived field as not archived", () => {
  const legacy = [
    thread({ id: "no-field" }),
    thread({ id: "explicit-false", archived: false }),
    thread({ id: "explicit-true", archived: true }),
  ];
  // Legacy rows predating the archived column: undefined and null, not false.
  delete (legacy[0] as { archived?: boolean }).archived;
  (legacy[1] as { archived: unknown }).archived = null;

  assert.deepEqual(
    groupThreads(legacy).map((item) => item.id).sort(),
    ["explicit-false", "no-field"],
  );
  assert.deepEqual(
    groupThreads(legacy, true).map((item) => item.id),
    ["explicit-true"],
  );
});

test("groupThreads collapses a compare pair into one row keyed by pairId", () => {
  const items = groupThreads([
    thread({ id: "left", pairId: "pair-1", createdAt: EPOCH, updatedAt: EPOCH }),
    thread({
      id: "right",
      pairId: "pair-1",
      createdAt: EPOCH + 10,
      updatedAt: EPOCH + 20,
    }),
  ]);
  assert.equal(items.length, 1);
  assert.equal(items[0]?.type, "compare");
  assert.equal(items[0]?.id, "pair-1");
  assert.deepEqual(items[0]?.threadIds, ["left", "right"]);
  assert.equal(items[0]?.createdAt, EPOCH + 10);
  assert.equal(items[0]?.updatedAt, EPOCH + 20);
});

test("groupThreads sorts newest activity first and falls back to createdAt", () => {
  const withoutUpdatedAt = thread({ id: "middle", createdAt: EPOCH + 50 });
  delete (withoutUpdatedAt as { updatedAt?: number }).updatedAt;
  const items = groupThreads([
    thread({ id: "oldest", createdAt: EPOCH, updatedAt: EPOCH }),
    withoutUpdatedAt,
    thread({ id: "newest", createdAt: EPOCH, updatedAt: EPOCH + 100 }),
  ]);
  assert.deepEqual(
    items.map((item) => item.id),
    ["newest", "middle", "oldest"],
  );
  assert.equal(items[1]?.updatedAt, EPOCH + 50);
});

test("groupThreads carries the fork flag and the project id onto single rows", () => {
  const items = groupThreads([
    thread({ id: "forked", forkedFromThreadId: "parent", projectId: "proj-1" }),
    thread({ id: "plain" }),
  ]);
  const forked = items.find((item) => item.id === "forked");
  const plain = items.find((item) => item.id === "plain");
  assert.equal(forked?.isFork, true);
  assert.equal(forked?.projectId, "proj-1");
  assert.equal(plain?.isFork, false);
  assert.equal(plain?.projectId, null);
  assert.deepEqual(plain?.threadIds, ["plain"]);
});

// ── 2. the memo key ──────────────────────────────────────────────────────────

test("sameSidebarThreads accepts two distinct arrays holding equal records", () => {
  const a = [thread({ id: "a" }), thread({ id: "b", pairId: "p" })];
  const b = a.map((record) => ({ ...record }));
  assert.notEqual(a, b);
  assert.notEqual(a[0], b[0]);
  assert.equal(sameSidebarThreads(a, b), true);
});

test("sameSidebarThreads rejects a list that gained or lost a thread", () => {
  const a = [thread({ id: "a" })];
  assert.equal(sameSidebarThreads(a, [...a, thread({ id: "b" })]), false);
  assert.equal(sameSidebarThreads([...a, thread({ id: "b" })], a), false);
});

test("sameSidebarThreads rejects a reordered list", () => {
  const a = [thread({ id: "a" }), thread({ id: "b" })];
  assert.equal(sameSidebarThreads(a, [a[1] as ThreadRecord, a[0] as ThreadRecord]), false);
});

test("sameSidebarThreads rejects a change to every field the grouping reads", () => {
  const changes: Record<(typeof SIDEBAR_THREAD_FIELDS)[number], unknown> = {
    id: "changed-id",
    title: "changed title",
    archived: true,
    pairId: "changed-pair",
    projectId: "changed-project",
    createdAt: EPOCH + 1,
    updatedAt: EPOCH + 2,
    forkedFromThreadId: "changed-parent",
  };
  // Every field in the key gets its own turn. A field dropped from
  // SIDEBAR_THREAD_FIELDS leaves its entry here unvisited and the count fails.
  let visited = 0;
  for (const field of SIDEBAR_THREAD_FIELDS) {
    const base = [thread({ id: "a", pairId: "p", projectId: null })];
    const changed = [{ ...(base[0] as ThreadRecord), [field]: changes[field] }];
    assert.equal(
      sameSidebarThreads(base, changed as ThreadRecord[]),
      false,
      `a change to ${field} must invalidate the memo`,
    );
    visited += 1;
  }
  assert.equal(visited, Object.keys(changes).length);
});

test("sameSidebarThreads only says yes when the grouping really is identical", () => {
  // The direction that matters. If groupThreads ever starts reading a field the key does not
  // compare, this finds it: the pairs below differ in EVERY other ThreadRecord field.
  const rand = mulberry32(0x5f_1d_eb_a1);
  const pick = <T,>(values: T[]): T =>
    values[Math.floor(rand() * values.length)] as T;
  const ignored = (): Partial<ThreadRecord> => ({
    modelType: pick(["base", "lora", "model1", "model2"] as const),
    modelId: pick(["m-1", "m-2", undefined]),
    openaiCodeExecContainerId: pick(["c-1", null, undefined]),
    anthropicCodeExecContainerId: pick(["c-2", null, undefined]),
    forkedFromMessageId: pick(["msg-1", null, undefined]),
  });
  const keyed = (index: number): Partial<ThreadRecord> => ({
    id: `t-${pick([index, index + 1, index + 2])}`,
    title: pick(["alpha", "beta", "gamma"]),
    archived: pick([true, false]),
    pairId: pick(["pair-a", "pair-b", undefined]),
    projectId: pick(["proj-a", null, undefined]),
    createdAt: EPOCH + pick([0, 10, 20]),
    updatedAt: EPOCH + pick([0, 10, 20]),
    forkedFromThreadId: pick(["parent", null, undefined]),
  });

  let agreed = 0;
  for (let trial = 0; trial < 400; trial += 1) {
    const size = 1 + Math.floor(rand() * 5);
    const left: ThreadRecord[] = [];
    const right: ThreadRecord[] = [];
    for (let i = 0; i < size; i += 1) {
      const key = keyed(i);
      const clone = rand() < 0.5;
      left.push(thread({ ...ignored(), ...key } as ThreadRecord & { id: string }));
      right.push(
        thread({
          ...ignored(),
          ...(clone ? key : keyed(i)),
        } as ThreadRecord & { id: string }),
      );
    }
    if (!sameSidebarThreads(left, right)) continue;
    agreed += 1;
    assert.deepEqual(groupThreads(left), groupThreads(right));
    assert.deepEqual(groupThreads(left, true), groupThreads(right, true));
  }
  // Guards the guard: a key that never agrees would make every assertion above unreachable.
  assert.ok(agreed > 20, `expected the key to agree sometimes, agreed ${agreed} times`);
});

// ── 3. identity across renders ───────────────────────────────────────────────

test("useSidebarThreadGroups keeps both lists identical across a re-render", () => {
  const runner = createHookRunner(useSidebarThreadGroups);
  const threads = [
    thread({ id: "a" }),
    thread({ id: "b", archived: true }),
    thread({ id: "c", updatedAt: EPOCH + 5 }),
  ];
  const first = runner.render(threads);
  const second = runner.render(threads);
  const third = runner.render(threads);

  assert.ok(runner.memoCalls() >= 6, "the hook must derive both lists through useMemo");
  assert.equal(
    first.items,
    second.items,
    "items got a new identity from a render that changed nothing",
  );
  assert.equal(second.items, third.items);
  assert.equal(
    first.archivedItems,
    second.archivedItems,
    "archivedItems got a new identity from a render that changed nothing",
  );
  assert.equal(second.archivedItems, third.archivedItems);
});

test("useSidebarThreadGroups keeps the empty case stable too", () => {
  const runner = createHookRunner(useSidebarThreadGroups);
  const first = runner.render(undefined);
  const second = runner.render(undefined);
  assert.deepEqual(first.items, []);
  assert.deepEqual(first.archivedItems, []);
  assert.equal(first.items, second.items);
  assert.equal(first.archivedItems, second.archivedItems);
});

test("useSidebarThreadGroups regroups when the threads actually change", () => {
  const runner = createHookRunner(useSidebarThreadGroups);
  const before = [thread({ id: "a" })];
  const after = [thread({ id: "a" }), thread({ id: "b", updatedAt: EPOCH + 9 })];
  const first = runner.render(before);
  const second = runner.render(after);

  assert.notEqual(
    first.items,
    second.items,
    "a changed thread list must produce a new items identity, or the rail shows stale rows",
  );
  assert.notEqual(first.archivedItems, second.archivedItems);
  assert.deepEqual(
    second.items.map((item) => item.id),
    ["b", "a"],
  );
});

test("useSidebarThreadGroups does not conflate the archived list with the live one", () => {
  const runner = createHookRunner(useSidebarThreadGroups);
  const threads = [
    thread({ id: "live" }),
    thread({ id: "gone", archived: true }),
  ];
  const rendered = runner.render(threads);
  assert.deepEqual(
    rendered.items.map((item) => item.id),
    ["live"],
  );
  assert.deepEqual(
    rendered.archivedItems.map((item) => item.id),
    ["gone"],
  );
  assert.notEqual(rendered.items, rendered.archivedItems);
});

// ── 4. the wiring in useChatSidebarItems ─────────────────────────────────────

test("useChatSidebarItems derives its lists through the memo, not per render", async () => {
  const source = await readFile(
    new URL(
      "../src/features/chat/hooks/use-chat-sidebar-items.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const body = source.slice(source.indexOf("export function useChatSidebarItems"));
  assert.match(body, /useSidebarThreadGroups\(allThreads\)/);
  assert.doesNotMatch(
    body,
    /const\s+(items|archivedItems)\s*=\s*groupThreads\(/,
    "the render body regrouped the whole history again",
  );
  // The refetch bail-out: without it every debounced refresh hands React a new array and the
  // memo above cannot hold, however well it is keyed.
  assert.match(body, /setAllThreads\(\s*\(previous\)\s*=>[\s\S]{0,120}sameSidebarThreads\(/);
  // The other groupThreads caller in the file is archiveAllChatItems, which counts rows once per
  // click and is not on a render path. It is expected to stay a direct call.
  assert.match(source, /return groupThreads\(toArchive\)\.length;/);
});

test("groupThreads is still exported from its original module path", async () => {
  const source = await readFile(
    new URL(
      "../src/features/chat/hooks/use-chat-sidebar-items.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(source, /export \{ groupThreads \} from "\.\/sidebar-thread-groups";/);
  assert.match(
    source,
    /export type \{ SidebarItem \} from "\.\/sidebar-thread-groups";/,
  );
});
