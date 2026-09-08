// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { ThreadRecordWriteCoordinator } from "../src/features/chat/utils/thread-record-write-coordinator.ts";

class DeletedError extends Error {}

function coordinator(): ThreadRecordWriteCoordinator {
  return new ThreadRecordWriteCoordinator(
    (threadId) => new Error(`clear:${threadId}`),
    (error) => error instanceof DeletedError,
  );
}

function deferred(): {
  promise: Promise<void>;
  resolve: () => void;
  reject: (error: Error) => void;
} {
  let resolve!: () => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<void>((done, fail) => {
    resolve = done;
    reject = fail;
  });
  return { promise, resolve, reject };
}

test("exposes a write id before its request starts", async () => {
  const writes = coordinator();
  const pending = deferred();
  let started = false;
  const work = writes.write("thread-1", () => {
    started = true;
    return pending.promise;
  });

  assert.equal(started, false);
  assert.deepEqual(writes.idsRequiringFence(), ["thread-1"]);
  await Promise.resolve();
  assert.equal(started, true);

  pending.resolve();
  await work;
  assert.deepEqual(writes.idsRequiringFence(), []);
});

test("retains an unconfirmed id after a transport failure", async () => {
  const writes = coordinator();
  const failure = new Error("connection lost");

  await assert.rejects(
    writes.write("thread-1", () => Promise.reject(failure)),
    failure,
  );
  assert.deepEqual(writes.idsRequiringFence(), ["thread-1"]);

  writes.confirmFinalState(["thread-1"]);
  assert.deepEqual(writes.idsRequiringFence(), []);
});

test("one successful write does not hide a concurrent ambiguous write", async () => {
  const writes = coordinator();
  const ambiguous = deferred();
  const first = writes.write("thread-1", () => ambiguous.promise);
  const second = writes.write("thread-1", () => Promise.resolve());

  await second;
  ambiguous.reject(new Error("connection lost"));
  await assert.rejects(first);

  assert.deepEqual(writes.idsRequiringFence(), ["thread-1"]);
});

test("a backend tombstone confirms that a failed save is absent", async () => {
  const writes = coordinator();
  await assert.rejects(
    writes.write("thread-1", () => Promise.reject(new DeletedError())),
    DeletedError,
  );
  assert.deepEqual(writes.idsRequiringFence(), []);
});

test("clear closes admission before taking its fence snapshot", async () => {
  const writes = coordinator();
  const reopen = writes.closeAdmission();
  let started = false;

  await assert.rejects(
    writes.write("thread-1", () => {
      started = true;
      return Promise.resolve();
    }),
    /clear:thread-1/,
  );
  assert.equal(started, false);
  assert.deepEqual(writes.idsRequiringFence(), []);

  reopen();
  await writes.write("thread-1", () => {
    started = true;
    return Promise.resolve();
  });
  assert.equal(started, true);
});

test("settles all work already observed for a thread", async () => {
  const writes = coordinator();
  const first = deferred();
  const second = deferred();
  writes.observe("thread-1", first.promise);
  writes.observe("thread-1", second.promise);

  let settled = false;
  const waiting = writes.settleCurrent("thread-1").then(() => {
    settled = true;
  });
  first.reject(new Error("failed"));
  await Promise.resolve();
  assert.equal(settled, false);
  second.resolve();
  await waiting;
  assert.equal(settled, true);
});
