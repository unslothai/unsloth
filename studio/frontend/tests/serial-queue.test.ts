import assert from "node:assert/strict";
import test from "node:test";

import { createSerialQueue } from "../src/features/chat/utils/serial-queue.ts";

function deferred() {
  let resolve!: () => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<void>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

test("a second caller waits rather than running alongside the first", async () => {
  // Two sidebars can be mounted at once. Without this their passes overlap and
  // the write concurrency cap is only per pass.
  const run = createSerialQueue();
  const first = deferred();
  let active = 0;
  let peak = 0;

  const a = run(async () => {
    active += 1;
    peak = Math.max(peak, active);
    await first.promise;
    active -= 1;
  });
  const b = run(async () => {
    active += 1;
    peak = Math.max(peak, active);
    active -= 1;
  });

  await Promise.resolve();
  assert.equal(active, 1);
  first.resolve();
  await Promise.all([a, b]);
  assert.equal(peak, 1);
});

test("a rejected task does not block the next one", async () => {
  const run = createSerialQueue();
  let ran = false;
  const failed = run(async () => {
    throw new Error("read failed");
  });
  const after = run(async () => {
    ran = true;
    return 7;
  });
  await assert.rejects(failed, /read failed/);
  assert.equal(await after, 7);
  assert.equal(ran, true);
});

test("tasks run in the order they were queued", async () => {
  const run = createSerialQueue();
  const order: number[] = [];
  await Promise.all(
    [1, 2, 3].map((n) =>
      run(async () => {
        order.push(n);
      }),
    ),
  );
  assert.deepEqual(order, [1, 2, 3]);
});
