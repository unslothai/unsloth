// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import ts from "typescript";
import { readSrc } from "./helpers/kit.ts";

// Execute the production adapter wrapper around a controlled pending history write.
const source = ts.createSourceFile(
  "runtime-provider.tsx",
  readSrc("features/chat/runtime-provider.tsx"),
  ts.ScriptTarget.Latest,
  true,
  ts.ScriptKind.TSX,
);
const declaration = source.statements.find(
  (node) =>
    ts.isFunctionDeclaration(node) &&
    node.name?.text === "createPersistedRunAdapter",
);
assert.ok(declaration);
const js = ts.transpileModule(declaration.getText(source), {
  compilerOptions: {
    target: ts.ScriptTarget.ES2022,
    module: ts.ModuleKind.None,
  },
}).outputText;

function pendingRun(hasReservation = true) {
  const controller = new AbortController();
  const stops: string[][] = [];
  const failures: string[] = [];
  let cancelled = false;
  let released = 0;
  let calls = 0;
  let resolve!: (id: string) => void;
  let reject!: (error: Error) => void;
  const preflight = new Promise<string>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  const deps = {
    runStartThreadIdsForMessages: () => ["chat"],
    preStreamRunThreadIdsForRuntime: (ids: string[]) => [
      ...new Set(ids.filter(Boolean)),
    ],
    useChatRuntimeStore: { getState: () => ({ activeThreadId: "chat" }) },
    findPreStreamRunReservation: () => (hasReservation ? "reservation" : null),
    claimPreStreamRunReservation: () => {},
    isPreStreamRunReservationCancelled: () => cancelled,
    releasePreStreamRunReservation: () => {
      released += 1;
    },
    waitForRunStartHistoryAppend: () => preflight,
    requestPromptQueueStop: (ids: string[]) => stops.push(ids),
    notifyPromptQueueRunFailed: (id: string) => failures.push(id),
    adoptPreStreamRunReservation: () => {},
  };
  const factory = new Function(
    ...Object.keys(deps),
    `${js}; return createPersistedRunAdapter;`,
  )(...Object.values(deps));
  const adapter = factory({
    async *run() {
      calls += 1;
      yield { content: "answer" };
    },
  });
  const run = adapter.run({
    messages: [],
    unstable_threadId: "chat",
    abortSignal: controller.signal,
  }) as AsyncGenerator<{ content: string }>;
  return {
    run,
    resolve,
    reject,
    stops,
    failures,
    controller,
    cancelReservation: () => {
      cancelled = true;
    },
    released: () => released,
    calls: () => calls,
  };
}

test("steering during history persistence cancels only the old run", async () => {
  const w = pendingRun();
  const next = w.run.next();
  w.cancelReservation();
  w.controller.abort();
  w.resolve("chat");
  await assert.rejects(next, { name: "AbortError" });
  assert.deepEqual(
    w.stops,
    [],
    "an intentional cancellation must not delete accepted follow-ups",
  );
  assert.deepEqual(w.failures, []);
  assert.equal(w.calls(), 0);
  assert.equal(w.released(), 1);
});

test("reservation cancellation before signal propagation does not stop the follow-up queue", async () => {
  const w = pendingRun();
  const next = w.run.next();
  w.cancelReservation();
  w.resolve("chat");
  await assert.rejects(next, { name: "AbortError" });
  assert.deepEqual(w.stops, []);
  assert.deepEqual(w.failures, []);
  assert.equal(w.calls(), 0);
});

test("an aborted run's late persistence rejection does not stop replacement work", async () => {
  const w = pendingRun(false);
  const next = w.run.next();
  w.controller.abort();
  w.reject(new DOMException("cancelled", "AbortError"));
  await assert.rejects(next, { name: "AbortError" });
  assert.deepEqual(w.stops, []);
  assert.deepEqual(w.failures, []);
});

test("a genuine history persistence failure still stops and reports the affected queue", async () => {
  const w = pendingRun();
  const next = w.run.next();
  w.reject(new Error("history write failed"));
  await assert.rejects(next, /history write failed/);
  assert.deepEqual(w.stops, [["chat"]]);
  assert.deepEqual(w.failures, ["chat"]);
  assert.equal(w.calls(), 0);
  assert.equal(w.released(), 1);
});

test("successful history persistence starts the adapter once", async () => {
  const w = pendingRun();
  const next = w.run.next();
  w.resolve("chat");
  assert.deepEqual(await next, { value: { content: "answer" }, done: false });
  assert.equal((await w.run.next()).done, true);
  assert.equal(w.calls(), 1);
  assert.deepEqual(w.stops, []);
  assert.deepEqual(w.failures, []);
});
