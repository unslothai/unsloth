// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  type ManagedModelDownloadDependencies,
  coordinateManagedModelDownload,
} from "../src/features/chat/api/managed-model-download.ts";

const REQUEST = {
  repoId: "unsloth/Qwen3.5-4B-MTP-GGUF",
  variant: "UD-Q4_K_XL",
  expectedBytes: 0,
};

function createHarness(
  outcome: "started" | "existing" | "conflict" | "busy" | "error" = "started",
) {
  const order: string[] = [];
  const cancelledKeys: string[] = [];
  let listeners: Parameters<ManagedModelDownloadDependencies["subscribe"]>[2] =
    {};
  let unsubscribed = false;
  const dependencies: ManagedModelDownloadDependencies = {
    requestStart: async (request) => {
      order.push("start");
      assert.deepEqual(request, { kind: "model", ...REQUEST });
      return outcome;
    },
    cancel: async (key) => {
      cancelledKeys.push(key);
    },
    subscribe: (kind, repoId, nextListeners) => {
      order.push("subscribe");
      assert.equal(kind, "model");
      assert.equal(repoId, REQUEST.repoId);
      listeners = nextListeners;
      return () => {
        unsubscribed = true;
      };
    },
    jobKey: (kind, repoId, variant) => `${kind}:${repoId}#${variant}`,
  };
  return {
    dependencies,
    order,
    cancelledKeys,
    listeners: () => listeners,
    unsubscribed: () => unsubscribed,
  };
}

test("subscribes before start and completes only for the requested variant", async () => {
  const harness = createHarness();
  const controller = new AbortController();
  const pending = coordinateManagedModelDownload(
    REQUEST,
    controller.signal,
    harness.dependencies,
  );

  assert.deepEqual(harness.order, ["subscribe", "start"]);
  let settled = false;
  void pending.then(() => {
    settled = true;
  });
  harness.listeners().onComplete?.("Q8_0", 123);
  await Promise.resolve();
  assert.equal(settled, false);

  harness.listeners().onComplete?.("ud-q4_k_xl", 456);
  assert.equal(await pending, "complete");
  assert.equal(harness.unsubscribed(), true);
});

test("chat abort cancels the exact manager job and preserves the abort reason", async () => {
  const harness = createHarness();
  const controller = new AbortController();
  const pending = coordinateManagedModelDownload(
    REQUEST,
    controller.signal,
    harness.dependencies,
  );
  const reason = new Error("Stopped by user");

  controller.abort(reason);

  await assert.rejects(pending, (error) => error === reason);
  const expectedKey = `model:${REQUEST.repoId}#${REQUEST.variant}`;
  assert.ok(harness.cancelledKeys.length >= 1);
  assert.ok(
    harness.cancelledKeys.every((cancelledKey) => cancelledKey === expectedKey),
  );
  assert.equal(harness.unsubscribed(), true);
});

test("chat abort does not cancel a pre-existing managed download", async () => {
  const harness = createHarness("existing");
  const controller = new AbortController();
  const pending = coordinateManagedModelDownload(
    REQUEST,
    controller.signal,
    harness.dependencies,
  );
  const reason = new Error("Stopped by user");

  controller.abort(reason);

  await assert.rejects(pending, (error) => error === reason);
  assert.deepEqual(harness.cancelledKeys, []);
  assert.equal(harness.unsubscribed(), true);
});

test("chat abort re-cancels a job that starts after async preflight", async () => {
  const harness = createHarness();
  const controller = new AbortController();
  let finishStart: ((outcome: "started") => void) | undefined;
  harness.dependencies.requestStart = () =>
    new Promise((resolve) => {
      finishStart = resolve;
    });
  const pending = coordinateManagedModelDownload(
    REQUEST,
    controller.signal,
    harness.dependencies,
  );
  const reason = new Error("Stopped during preflight");

  controller.abort(reason);
  await assert.rejects(pending, (error) => error === reason);
  assert.deepEqual(harness.cancelledKeys, []);

  finishStart?.("started");
  await Promise.resolve();
  assert.deepEqual(harness.cancelledKeys, [
    `model:${REQUEST.repoId}#${REQUEST.variant}`,
  ]);
});

test("manager cancellation settles the waiting first chat", async () => {
  const harness = createHarness();
  const pending = coordinateManagedModelDownload(
    REQUEST,
    new AbortController().signal,
    harness.dependencies,
  );

  harness.listeners().onCancelled?.(REQUEST.variant);

  assert.equal(await pending, "cancelled");
  assert.equal(harness.unsubscribed(), true);
});

for (const outcome of ["conflict", "busy", "error"] as const) {
  test(`returns ${outcome} without waiting for a terminal event`, async () => {
    const harness = createHarness(outcome);
    const result = await coordinateManagedModelDownload(
      REQUEST,
      new AbortController().signal,
      harness.dependencies,
    );

    assert.equal(result, outcome);
    assert.equal(harness.unsubscribed(), true);
  });
}

test("turns an unexpected start rejection into a managed error", async () => {
  const harness = createHarness();
  harness.dependencies.requestStart = async () => {
    throw new Error("start failed");
  };

  assert.equal(
    await coordinateManagedModelDownload(
      REQUEST,
      new AbortController().signal,
      harness.dependencies,
    ),
    "error",
  );
  assert.equal(harness.unsubscribed(), true);
});
