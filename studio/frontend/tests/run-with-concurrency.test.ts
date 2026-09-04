import assert from "node:assert/strict";
import test from "node:test";

import { runWithConcurrency } from "../src/features/chat/utils/run-with-concurrency.ts";

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

test("never runs more than the limit at once", async () => {
  const gates = Array.from({ length: 10 }, deferred);
  let active = 0;
  let peak = 0;
  const started: number[] = [];

  const all = runWithConcurrency([...gates.keys()], 3, async (index) => {
    started.push(index);
    active += 1;
    peak = Math.max(peak, active);
    await gates[index].promise;
    active -= 1;
  });

  await Promise.resolve();
  assert.deepEqual(started, [0, 1, 2]);
  for (const gate of gates) gate.resolve();
  await all;
  assert.equal(peak, 3);
  assert.equal(started.length, 10);
});

test("a limit above the item count still runs every item once", async () => {
  const seen: number[] = [];
  await runWithConcurrency([1, 2, 3], 99, async (n) => {
    seen.push(n);
  });
  assert.deepEqual(seen.sort(), [1, 2, 3]);
});

test("an empty list resolves without running anything", async () => {
  let ran = false;
  await runWithConcurrency([], 4, async () => {
    ran = true;
  });
  assert.equal(ran, false);
});
