// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { beginInferenceStatusRefresh, resetInferenceStatusRefreshSeqForTests } =
  await import("../src/features/chat/lib/inference-status-refresh-seq.ts");

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

const flush = () => new Promise((r) => setTimeout(r, 0));

async function settledEarly(promise: Promise<void>): Promise<boolean> {
  let done = false;
  void promise.then(() => {
    done = true;
  });
  await flush();
  return done;
}

test("a superseded chat status refresh writes nothing and waits for the newer read", async () => {
  resetInferenceStatusRefreshSeqForTests();
  let store = "stale";
  const first = beginInferenceStatusRefresh();
  const firstRead = deferred<void>();
  const firstSettled = first.register(
    firstRead.promise.then(() => {
      if (!first.isCurrent()) return first.superseded();
      store = "first";
    }),
  );

  const second = beginInferenceStatusRefresh();
  const secondRead = deferred<void>();
  const secondSettled = second.register(
    secondRead.promise.then(() => {
      if (!second.isCurrent()) return second.superseded();
      store = "second";
    }),
  );

  firstRead.resolve();
  assert.equal(await settledEarly(firstSettled), false);
  assert.equal(store, "stale");

  secondRead.resolve();
  await firstSettled;
  assert.equal(store, "second");
  await secondSettled;
});

test("out-of-order chat status responses still leave the newest read in charge", async () => {
  resetInferenceStatusRefreshSeqForTests();
  let store = "stale";
  const first = beginInferenceStatusRefresh();
  const firstRead = deferred<void>();
  const firstSettled = first.register(
    firstRead.promise.then(() => {
      if (!first.isCurrent()) return first.superseded();
      store = "first";
    }),
  );

  const second = beginInferenceStatusRefresh();
  const secondRead = deferred<void>();
  const secondSettled = second.register(
    secondRead.promise.then(() => {
      if (!second.isCurrent()) return second.superseded();
      store = "second";
    }),
  );

  secondRead.resolve();
  firstRead.resolve();
  await firstSettled;
  assert.equal(store, "second");
  await secondSettled;
});
