// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  countByStatus,
  filterEntries,
  mergeMonitorResponse,
} from "../src/features/api-monitor/monitor-snapshot.ts";
import type {
  ApiMonitorEntry,
  ApiMonitorResponse,
} from "../src/features/chat/types/api.ts";

const PROGRESS = {
  total: 100,
  processed: null,
  cached: null,
  time_ms: null,
  percent: null,
};

function entry(overrides: Partial<ApiMonitorEntry>): ApiMonitorEntry {
  return {
    id: "a",
    status: "completed",
    model: "m",
    endpoint: "/v1/chat/completions",
    method: "POST",
    started_at: 1,
    updated_at: 1,
    prompt_preview: "hi",
    reply_preview: "hello",
    ...overrides,
  } as ApiMonitorEntry;
}

function snapshot(
  entries: ApiMonitorEntry[],
  serverTime = 100,
): ApiMonitorResponse {
  return {
    status: "ready",
    server_time: serverTime,
    active_model: "m",
    active_requests: 0,
    queue: { active: 0, capacity: 4, queued: 0, free: 4 },
    entries,
  } as ApiMonitorResponse;
}

test("a poll that only moved server_time keeps the previous snapshot", () => {
  const first = snapshot([entry({ id: "a" }), entry({ id: "b" })]);
  const second = snapshot([entry({ id: "a" }), entry({ id: "b" })], 101.5);
  assert.equal(mergeMonitorResponse(first, second), first);
});

test("a changed entry is replaced while unchanged ones keep their identity", () => {
  const first = snapshot([
    entry({ id: "a", status: "running", updated_at: 1 }),
    entry({ id: "b" }),
  ]);
  const second = snapshot([
    entry({ id: "a", status: "running", updated_at: 2, reply_preview: "hel" }),
    entry({ id: "b" }),
  ]);
  const merged = mergeMonitorResponse(first, second);
  assert.notEqual(merged, first);
  assert.notEqual(merged.entries[0], first.entries[0]);
  assert.equal(merged.entries[0].reply_preview, "hel");
  assert.equal(merged.entries[1], first.entries[1]);
});

test("a new entry, a reorder or a queue change produces a new snapshot", () => {
  const first = snapshot([entry({ id: "a" })]);
  const added = mergeMonitorResponse(
    first,
    snapshot([entry({ id: "b" }), entry({ id: "a" })]),
  );
  assert.equal(added.entries.length, 2);
  assert.equal(added.entries[1], first.entries[0]);

  const queue = snapshot([entry({ id: "a" })]);
  queue.queue = { active: 1, capacity: 4, queued: 0, free: 3 };
  const merged = mergeMonitorResponse(first, queue);
  assert.notEqual(merged, first);
  assert.equal(merged.entries, first.entries);
  assert.equal(merged.queue?.active, 1);
});

test("nested fields such as prompt progress count as changes", () => {
  const first = snapshot([
    entry({
      id: "a",
      status: "running",
      prompt_progress: { ...PROGRESS, percent: 10 },
    }),
  ]);
  const second = snapshot([
    entry({
      id: "a",
      status: "running",
      prompt_progress: { ...PROGRESS, percent: 40 },
    }),
  ]);
  const merged = mergeMonitorResponse(first, second);
  assert.equal(merged.entries[0].prompt_progress?.percent, 40);
});

test("status tabs count requests only, and a failed load lists under Errors", () => {
  const entries = [
    entry({ id: "r1", status: "completed" }),
    entry({ id: "r2", status: "error", error: "boom" }),
    entry({ id: "r3", status: "running" }),
    entry({ id: "l1", kind: "lifecycle", status: "completed" }),
    entry({ id: "l2", kind: "lifecycle", status: "running" }),
    entry({
      id: "l3",
      kind: "lifecycle",
      status: "error",
      error: "out of memory",
    }),
  ];
  const counts = countByStatus(entries);
  assert.deepEqual(counts, {
    all: 6,
    running: 1,
    completed: 1,
    error: 2,
    cancelled: 0,
  });
  for (const status of [
    "running",
    "completed",
    "error",
    "cancelled",
  ] as const) {
    assert.equal(filterEntries(entries, status, "").length, counts[status]);
  }
  assert.equal(filterEntries(entries, "all", "").length, 6);
  assert.deepEqual(
    filterEntries(entries, "error", "memory").map((e) => e.id),
    ["l3"],
  );
});

test("the floating monitor folds a fresh first snapshot and pauses while hidden", async () => {
  const { readSrc } = await import("./helpers/kit.ts");
  const overlay = readSrc("features/api-monitor/api-monitor-overlay.tsx");
  assert.match(
    overlay,
    /fresh \? next : mergeMonitorResponse\(previous, next\)/,
  );
  assert.match(overlay, /if \(!document\.hidden\) poll\(\);/);
  assert.match(
    overlay,
    /addEventListener\("visibilitychange", onVisibilityChange\)/,
  );
});
