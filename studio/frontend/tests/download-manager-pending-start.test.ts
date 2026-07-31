import assert from "node:assert/strict";
import test from "node:test";

import {
  type DownloadStartOutcome,
  type PendingStartMap,
  hasPendingStartForRepo,
  resolveJoinedPendingStart,
  runOrJoinPendingStart,
} from "../src/features/hub/download-manager/pending-start.ts";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((ok, fail) => {
    resolve = ok;
    reject = fail;
  });
  return { promise, resolve, reject };
}

test("exact starts share one action and clean up", async () => {
  const pending: PendingStartMap = new Map();
  const gate = deferred<DownloadStartOutcome>();
  let actions = 0;
  const start = () =>
    runOrJoinPendingStart(
      pending,
      "model:org/repo#q4",
      () => {
        actions += 1;
        return gate.promise;
      },
      () => assert.fail("unexpected start error"),
    );
  const first = start();
  const second = start();
  assert.equal(first, second);
  assert.equal(actions, 0);
  await Promise.resolve();
  assert.equal(actions, 1);
  gate.resolve("started");
  assert.deepEqual(await Promise.all([first, second]), ["started", "started"]);
  assert.equal(pending.size, 0);
});

test("an exact join observes cancellation independently of the owner", async () => {
  const gate = deferred<DownloadStartOutcome>();
  let cancelling = false;
  let stateListener = () => {};
  const owner = gate.promise;
  const joined = resolveJoinedPendingStart(
    owner,
    () => cancelling,
    (listener) => {
      stateListener = listener;
      return () => {};
    },
  );

  cancelling = true;
  stateListener();
  cancelling = false;
  gate.resolve("started");

  assert.equal(await owner, "started");
  assert.equal(await joined, "cancelling");
});

test("an exact join preserves cancellation visible when it joined", async () => {
  const gate = deferred<DownloadStartOutcome>();
  let cancelling = true;
  const joined = resolveJoinedPendingStart(
    gate.promise,
    () => cancelling,
    () => () => {},
  );
  cancelling = false;
  gate.resolve("started");
  assert.equal(await joined, "cancelling");
});

test("repo conflicts are boundary-safe across variants and snapshots", () => {
  const classify = (pending: PendingStartMap, key: string, repo: string) =>
    pending.has(key)
      ? "join"
      : hasPendingStartForRepo(pending, repo)
        ? "busy"
        : "free";
  const pending: PendingStartMap = new Map([
    ["model:org/repo#q4", Promise.resolve("started")],
  ]);
  assert.equal(
    classify(pending, "model:org/repo#q4", "model:org/repo"),
    "join",
  );
  assert.equal(
    classify(pending, "model:org/repo#q8", "model:org/repo"),
    "busy",
  );
  assert.equal(classify(pending, "model:org/repo", "model:org/repo"), "busy");
  assert.equal(
    classify(pending, "model:org/repository#q4", "model:org/repository"),
    "free",
  );
  assert.equal(
    classify(pending, "model:other/repo#q4", "model:other/repo"),
    "free",
  );
  pending.clear();
  pending.set("model:org/repo", Promise.resolve("started"));
  assert.equal(
    classify(pending, "model:org/repo#q4", "model:org/repo"),
    "busy",
  );
});

test("rejections report once, retry, and identity-safe cleanup", async () => {
  const pending: PendingStartMap = new Map();
  const errors: unknown[] = [];
  const failed = deferred<DownloadStartOutcome>();
  const start = (action: () => Promise<DownloadStartOutcome>) =>
    runOrJoinPendingStart(pending, "model:org/repo#q4", action, (error) =>
      errors.push(error),
    );
  const first = start(() => failed.promise);
  const joined = start(() => Promise.resolve("conflict"));
  failed.reject(new Error("boom"));
  assert.deepEqual(await Promise.all([first, joined]), ["error", "error"]);
  assert.equal(errors.length, 1);
  assert.equal(pending.size, 0);
  assert.equal(await start(() => Promise.resolve("started")), "started");

  const oldGate = deferred<DownloadStartOutcome>();
  const old = start(() => oldGate.promise);
  await Promise.resolve();
  pending.clear();
  const replacementGate = deferred<DownloadStartOutcome>();
  const replacement = start(() => replacementGate.promise);
  await Promise.resolve();
  oldGate.resolve("started");
  await old;
  assert.equal(pending.get("model:org/repo#q4"), replacement);
  replacementGate.resolve("started");
  await replacement;
  assert.equal(pending.size, 0);
});
