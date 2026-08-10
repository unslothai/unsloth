// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  createScopedSingleFlightRequest,
  createSingleFlightRequest,
} from "../src/features/training/lib/single-flight-request.ts";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });
  return { promise, resolve, reject };
}

test("concurrent callers share one request until it settles", async () => {
  const first = deferred<number>();
  let calls = 0;
  const request = createSingleFlightRequest(() => {
    calls += 1;
    return first.promise;
  });

  const left = request();
  const right = request();
  assert.strictEqual(left, right);
  assert.equal(calls, 0);

  await Promise.resolve();
  assert.equal(calls, 1);
  first.resolve(7);
  assert.deepEqual(await Promise.all([left, right]), [7, 7]);
});

test("settled and rejected requests release the next attempt", async () => {
  const attempts = [deferred<number>(), deferred<number>(), deferred<number>()];
  let calls = 0;
  const request = createSingleFlightRequest(() => attempts[calls++].promise);

  const first = request();
  attempts[0].resolve(1);
  assert.equal(await first, 1);
  await Promise.resolve();

  const second = request();
  attempts[1].reject(new Error("offline"));
  await assert.rejects(second, /offline/);
  await Promise.resolve();

  const third = request();
  attempts[2].resolve(3);
  assert.equal(await third, 3);
  assert.equal(calls, 3);
});

test("a new runtime scope aborts the previous scoped request", async () => {
  const pending = new Map<string, ReturnType<typeof deferred<string>>>();
  const request = createScopedSingleFlightRequest((scope, _input, signal) => {
    const operation = deferred<string>();
    pending.set(scope, operation);
    signal.addEventListener(
      "abort",
      () => operation.reject(new DOMException("Superseded", "AbortError")),
      { once: true },
    );
    return operation.promise;
  });

  const oldLeft = request.run("scope-a", undefined);
  const oldRight = request.run("scope-a", undefined);
  await Promise.resolve();
  const current = request.run("scope-b", undefined);
  await Promise.resolve();

  assert.strictEqual(oldLeft, oldRight);
  assert.notStrictEqual(oldLeft, current);
  await assert.rejects(oldLeft, { name: "AbortError" });
  pending.get("scope-b")?.resolve("current");
  assert.equal(await current, "current");
});

test("a scoped refresh replaces an in-flight request in the same scope", async () => {
  const attempts: Array<ReturnType<typeof deferred<number>>> = [];
  const request = createScopedSingleFlightRequest((_scope, _input, signal) => {
    const operation = deferred<number>();
    attempts.push(operation);
    signal.addEventListener(
      "abort",
      () => operation.reject(new DOMException("Refreshed", "AbortError")),
      { once: true },
    );
    return operation.promise;
  });

  const first = request.run("scope-a", undefined);
  await Promise.resolve();
  const fresh = request.refresh("scope-a", undefined);
  await Promise.resolve();

  await assert.rejects(first, { name: "AbortError" });
  assert.equal(attempts.length, 2);
  attempts[1]?.resolve(2);
  assert.equal(await fresh, 2);
});
