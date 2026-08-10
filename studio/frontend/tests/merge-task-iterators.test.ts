// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { mergeTaskIterators } from "../src/features/hub/lib/merge-task-iterators.ts";

const FIRST_TASK_FAILURE_RE = /failed:one/;

function deferred(): { promise: Promise<void>; resolve: () => void } {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

async function collect<T>(iter: AsyncGenerator<T>): Promise<T[]> {
  const values: T[] = [];
  for await (const value of iter) {
    values.push(value);
  }
  return values;
}

test("starts task searches concurrently and yields deterministic priority rounds", async () => {
  const primaryGate = deferred();
  const started: string[] = [];
  const iter = mergeTaskIterators(
    ["primary", "secondary"],
    async function* (task) {
      started.push(task ?? "default");
      if (task === "primary") {
        await primaryGate.promise;
      }
      yield { name: `${task}-1` };
      yield { name: `${task}-2` };
    },
  );

  const first = iter.next();
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.deepEqual(started, ["primary", "secondary"]);
  primaryGate.resolve();

  assert.deepEqual((await first).value, { name: "primary-1" });
  assert.deepEqual(await collect(iter), [
    { name: "secondary-1" },
    { name: "primary-2" },
    { name: "secondary-2" },
  ]);
});

test("keeps healthy task results when a sibling task fails", async () => {
  const values = await collect(
    mergeTaskIterators(["failed", "healthy"], async function* (task) {
      await Promise.resolve();
      if (task === "failed") {
        throw new Error("rate limited");
      }
      yield { name: "healthy-1" };
      yield { name: "healthy-2" };
    }),
  );

  assert.deepEqual(values, [{ name: "healthy-1" }, { name: "healthy-2" }]);
});

test("surfaces an error when every task fails before yielding", async () => {
  await assert.rejects(
    collect(
      mergeTaskIterators(["one", "two"], async function* (task) {
        await Promise.resolve();
        if (task) {
          throw new Error(`failed:${task}`);
        }
        yield { name: "unreachable" };
      }),
    ),
    FIRST_TASK_FAILURE_RE,
  );
});

test("aborts and closes pending task iterators when the consumer stops", async () => {
  const closed: string[] = [];
  const iter = mergeTaskIterators(
    ["primary", "secondary"],
    async function* (task, signal) {
      try {
        if (task === "primary") {
          yield { name: task };
        }
        await new Promise<void>((_, reject) => {
          signal.addEventListener(
            "abort",
            () => reject(new DOMException("Aborted", "AbortError")),
            { once: true },
          );
        });
      } finally {
        closed.push(task ?? "default");
      }
    },
  );

  assert.deepEqual((await iter.next()).value, { name: "primary" });
  await iter.return(undefined);
  assert.deepEqual(closed.sort(), ["primary", "secondary"]);
});
