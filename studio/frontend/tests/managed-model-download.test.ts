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

test("one consumer abort cannot cancel a shared owned download", async () => {
  const listeners: Array<
    Parameters<ManagedModelDownloadDependencies["subscribe"]>[2]
  > = [];
  const cancelledKeys: string[] = [];
  let starts = 0;
  const dependencies: ManagedModelDownloadDependencies = {
    requestStart: async () => {
      starts += 1;
      return "started";
    },
    cancel: async (key) => {
      cancelledKeys.push(key);
    },
    subscribe: (_kind, _repoId, nextListeners) => {
      listeners.push(nextListeners);
      return () => undefined;
    },
    jobKey: (kind, repoId, variant) => `${kind}:${repoId}#${variant}`,
  };
  const firstController = new AbortController();
  const secondController = new AbortController();
  const first = coordinateManagedModelDownload(
    REQUEST,
    firstController.signal,
    dependencies,
  );
  const second = coordinateManagedModelDownload(
    REQUEST,
    secondController.signal,
    dependencies,
  );
  await Promise.resolve();

  firstController.abort(new Error("first consumer left"));
  await assert.rejects(first);
  assert.equal(starts, 1);
  assert.deepEqual(cancelledKeys, []);

  for (const listener of listeners) {
    listener.onComplete?.(REQUEST.variant, 123);
  }
  assert.equal(await second, "complete");
  assert.deepEqual(cancelledKeys, []);
});

test("the last shared consumer cancels an owned download exactly once", async () => {
  const cancelledKeys: string[] = [];
  const dependencies: ManagedModelDownloadDependencies = {
    requestStart: async () => "started",
    cancel: async (key) => {
      cancelledKeys.push(key);
    },
    subscribe: () => () => undefined,
    jobKey: (kind, repoId, variant) => `${kind}:${repoId}#${variant}`,
  };
  const firstController = new AbortController();
  const secondController = new AbortController();
  const first = coordinateManagedModelDownload(
    REQUEST,
    firstController.signal,
    dependencies,
  );
  const second = coordinateManagedModelDownload(
    REQUEST,
    secondController.signal,
    dependencies,
  );
  await Promise.resolve();

  firstController.abort();
  await assert.rejects(first);
  assert.deepEqual(cancelledKeys, []);

  secondController.abort();
  await assert.rejects(second);
  assert.deepEqual(cancelledKeys, [
    `model:${REQUEST.repoId}#${REQUEST.variant}`,
  ]);
});

test("a successor waits for owned cancellation before starting", async () => {
  const listeners: Array<
    Parameters<ManagedModelDownloadDependencies["subscribe"]>[2]
  > = [];
  let starts = 0;
  let finishCancellation: (() => void) | undefined;
  const cancellation = new Promise<void>((resolve) => {
    finishCancellation = resolve;
  });
  const dependencies: ManagedModelDownloadDependencies = {
    requestStart: async () => {
      starts += 1;
      return "started";
    },
    cancel: () => cancellation,
    subscribe: (_kind, _repoId, nextListeners) => {
      listeners.push(nextListeners);
      return () => undefined;
    },
    jobKey: (kind, repoId, variant) => `${kind}:${repoId}#${variant}`,
  };
  const firstController = new AbortController();
  const first = coordinateManagedModelDownload(
    REQUEST,
    firstController.signal,
    dependencies,
  );
  await Promise.resolve();
  firstController.abort(new Error("restart"));
  let firstSettled = false;
  void first.catch(() => {
    firstSettled = true;
  });

  const successor = coordinateManagedModelDownload(
    REQUEST,
    new AbortController().signal,
    dependencies,
  );
  await Promise.resolve();
  assert.equal(starts, 1);
  assert.equal(firstSettled, false);

  finishCancellation?.();
  await assert.rejects(first);
  await Promise.resolve();
  assert.equal(starts, 2);

  listeners.at(-1)?.onComplete?.(REQUEST.variant, 123);
  assert.equal(await successor, "complete");
});


test("a terminal event during start preflight does not request cancellation", async () => {
  const harness = createHarness();
  let finishStart: ((outcome: "started") => void) | undefined;
  harness.dependencies.requestStart = () =>
    new Promise((resolve) => {
      finishStart = resolve;
    });
  const pending = coordinateManagedModelDownload(
    REQUEST,
    new AbortController().signal,
    harness.dependencies,
  );

  harness.listeners().onComplete?.(REQUEST.variant, 123);
  assert.equal(await pending, "complete");
  finishStart?.("started");
  await Promise.resolve();
  assert.deepEqual(harness.cancelledKeys, []);
});
