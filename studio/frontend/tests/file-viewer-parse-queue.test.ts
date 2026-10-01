// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { MAX_QUEUED_PARSES, queueParse } from "../src/components/file-viewer/parse-queue.ts";

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

test("no more than the limit run at once, and the rest start as slots free", { timeout: 2000 }, async () => {
  let running = 0;
  let peak = 0;
  const gates = Array.from({ length: 6 }, deferred);
  const results = gates.map((gate, index) =>
    queueParse(async () => {
      running += 1;
      peak = Math.max(peak, running);
      await gate.promise;
      running -= 1;
      return index;
    }, () => false),
  );
  await Promise.resolve();
  assert.equal(running, MAX_QUEUED_PARSES);
  for (const gate of gates) gate.resolve();
  assert.deepEqual(await Promise.all(results), [0, 1, 2, 3, 4, 5]);
  assert.equal(peak, MAX_QUEUED_PARSES);
});

test("a task cancelled while waiting never runs and frees no slot it never took", { timeout: 2000 }, async () => {
  const gates = Array.from({ length: MAX_QUEUED_PARSES }, deferred);
  const busy = gates.map((gate) => queueParse(() => gate.promise, () => false));
  let ran = false;
  let cancelled = false;
  const skipped = queueParse(async () => {
    ran = true;
  }, () => cancelled);
  const after = queueParse(async () => "after", () => false);
  cancelled = true;
  for (const gate of gates) gate.resolve();
  await Promise.all(busy);
  assert.equal(await skipped, null);
  assert.equal(await after, "after");
  assert.equal(ran, false);
});

test("a failed task still frees its slot", { timeout: 2000 }, async () => {
  await assert.rejects(queueParse(() => Promise.reject(new Error("bad file")), () => false), /bad file/);
  await new Promise((settle) => setImmediate(settle));
  let running = 0;
  const gates = Array.from({ length: MAX_QUEUED_PARSES }, deferred);
  const busy = gates.map((gate) =>
    queueParse(async () => {
      running += 1;
      await gate.promise;
    }, () => false),
  );
  assert.equal(running, MAX_QUEUED_PARSES);
  for (const gate of gates) gate.resolve();
  await Promise.all(busy);
});
