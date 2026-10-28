// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  createRemeasureQueue,
  createSettleQueue,
} from "../src/components/assistant-ui/reasoning-settle-queue.ts";

const frames = () => {
  let queue: (() => void)[] = [];
  return {
    request: (callback: () => void) => {
      queue.push(callback);
    },
    run: () => {
      const due = queue;
      queue = [];
      for (const callback of due) callback();
      return due.length;
    },
    pending: () => queue.length,
  };
};

test("a row settles only after one whole frame has laid it out unsettled", () => {
  const clock = frames();
  const settled: string[] = [];
  const queue = createSettleQueue<string>(clock.request, (row) => settled.push(row));
  queue.add("a");
  queue.add("b");
  assert.equal(clock.pending(), 1, "one frame request covers every row queued before it");
  clock.run();
  assert.deepEqual(settled, [], "the first frame lays the rows out and records their size");
  clock.run();
  assert.deepEqual(settled, ["a", "b"]);
  assert.equal(clock.pending(), 0, "no frame is requested once nothing waits");
});

test("a row queued while a settle is one frame along waits its own whole frame", () => {
  const clock = frames();
  const settled: string[] = [];
  const queue = createSettleQueue<string>(clock.request, (row) => settled.push(row));
  queue.add("mounted");
  clock.run();
  queue.add("unsettled by a width change");
  clock.run();
  assert.deepEqual(settled, ["mounted"], "the late row has not been through a layout yet, so it must not be skipped on a stale size");
  clock.run();
  assert.deepEqual(settled, ["mounted", "unsettled by a width change"]);
});

test("re-queuing a ripening row restarts its wait, and a forgotten row never settles", () => {
  const clock = frames();
  const settled: string[] = [];
  const queue = createSettleQueue<string>(clock.request, (row) => settled.push(row));
  queue.add("a");
  queue.add("b");
  clock.run();
  queue.add("a");
  queue.forget("b");
  clock.run();
  assert.deepEqual(settled, []);
  clock.run();
  assert.deepEqual(settled, ["a"]);
  assert.equal(clock.pending(), 0);
});

test("a row forgotten before its first frame never settles and keeps no frame requested", () => {
  const clock = frames();
  const settled: string[] = [];
  const queue = createSettleQueue<string>(clock.request, (row) => settled.push(row));
  queue.add("a");
  queue.forget("a");
  clock.run();
  assert.equal(clock.pending(), 0);
  clock.run();
  assert.deepEqual(settled, []);
});

test("forgetting a row at either stage leaves the rows beside it", () => {
  const clock = frames();
  const settled: string[] = [];
  const queue = createSettleQueue<string>(clock.request, (row) => settled.push(row));
  queue.add("ripe kept");
  queue.add("ripe forgotten");
  clock.run();
  queue.add("queued kept");
  queue.add("queued forgotten");
  queue.forget("ripe forgotten");
  queue.forget("queued forgotten");
  clock.run();
  assert.deepEqual(settled, ["ripe kept"]);
  clock.run();
  assert.deepEqual(settled, ["ripe kept", "queued kept"]);
  assert.equal(clock.pending(), 0);
});

test("remeasuring long transcripts lays out one row per frame and preserves the settle delay", () => {
  const clock = frames();
  const visible = new Set<number>();
  const settled: number[] = [];
  const settle = createSettleQueue<number>(clock.request, (row) => {
    visible.delete(row);
    settled.push(row);
  });
  const remeasure = createRemeasureQueue<number>(clock.request, (row) => {
    visible.add(row);
    settle.add(row);
  }, () => false);
  for (let i = 0; i < 100; i++) remeasure.add(i);
  clock.run();
  assert.deepEqual([...visible], [0]);
  assert.deepEqual(settled, []);
  clock.run();
  assert.deepEqual([...visible], [0, 1]);
  assert.deepEqual(settled, []);
  for (let i = 0; clock.pending() && i < 200; i++) {
    clock.run();
    assert.ok(visible.size <= 3);
  }
  assert.deepEqual(settled, Array.from({ length: 100 }, (_, i) => i));
  assert.equal(clock.pending(), 0);
});

test("a drag pauses remeasurement, repeated resizes coalesce, and removed rows stay removed", () => {
  const clock = frames();
  let dragging = true;
  const measured: string[] = [];
  const queue = createRemeasureQueue<string>(clock.request, (row) => measured.push(row), () => dragging);
  queue.add("first");
  queue.add("removed");
  queue.add("last");
  clock.run();
  assert.deepEqual(measured, []);
  queue.add("first");
  queue.forget("removed");
  dragging = false;
  clock.run();
  assert.deepEqual(measured, ["first"]);
  clock.run();
  assert.deepEqual(measured, ["first", "last"]);
  assert.equal(clock.pending(), 0);
});
