// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./helpers/export-store-resolver.mjs", import.meta.url);

const stub = await import("./helpers/export-api-stub.mjs");
const { useExportRuntimeStore } = await import(
  "../src/features/export/stores/export-runtime-store.ts"
);

type Status = Parameters<
  ReturnType<typeof useExportRuntimeStore.getState>["applyBackendStatus"]
>[0];

const idle = { is_export_active: false } as Status;

function params(overrides: Record<string, unknown> = {}) {
  return {
    sourceMode: "model",
    checkpointPath: null,
    source: "unsloth/Qwen3-0.6B",
    modelSource: "hf",
    trustRemoteCode: false,
    exportMethod: "gguf",
    isAdapter: false,
    quantLevels: ["q4_k_m"],
    saveDirectory: "out",
    destination: "local",
    privateRepo: false,
    summary: {},
    ...overrides,
  } as unknown as Parameters<
    ReturnType<typeof useExportRuntimeStore.getState>["runExport"]
  >[0];
}

function deferred(): {
  promise: Promise<unknown>;
  resolve: (value: unknown) => void;
} {
  let resolve!: (value: unknown) => void;
  const promise = new Promise<unknown>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

const tick = () => new Promise<void>((resolve) => setImmediate(resolve));

const operationPosts = () =>
  stub.calls
    .map((entry) => entry.name)
    .filter((name) =>
      ["loadCheckpoint", "exportGGUF", "exportMerged", "exportLoRA"].includes(
        name,
      ),
    );

test("an unchanged idle status poll does not notify store listeners", () => {
  useExportRuntimeStore.getState().reset();
  useExportRuntimeStore.getState().applyBackendStatus(idle);
  const hydrated = useExportRuntimeStore.getState();
  assert.equal(hydrated.hasHydrated, true);

  let notified = 0;
  const unsubscribe = useExportRuntimeStore.subscribe(() => {
    notified += 1;
  });
  try {
    useExportRuntimeStore.getState().applyBackendStatus(idle);
    useExportRuntimeStore.getState().applyBackendStatus({ ...idle });
    assert.equal(notified, 0);
    assert.equal(useExportRuntimeStore.getState(), hydrated);

    useExportRuntimeStore
      .getState()
      .applyBackendStatus({ is_export_active: true } as Status);
    assert.equal(notified, 1, "a real change still publishes");
    assert.equal(useExportRuntimeStore.getState().backendActive, true);
  } finally {
    unsubscribe();
    useExportRuntimeStore.getState().reset();
  }
});

test("a cancel during the status preflight blocks every operation POST", async () => {
  const store = useExportRuntimeStore.getState();
  store.reset();
  stub.resetStub();
  const preflight = deferred();
  stub.responses.set("getExportStatus", () => preflight.promise);
  try {
    const run = store.runExport(params());
    assert.ok(
      stub.calls.some((entry) => entry.name === "getExportStatus"),
      "the run waits on the status preflight before its first POST",
    );

    // The backend has no active op yet, so its cancel is a no-op; the intent
    // lives only in the store until the next operation would dispatch.
    await store.requestCancel();
    preflight.resolve({ last_op_seq: 0 });
    await run;

    const state = useExportRuntimeStore.getState();
    assert.equal(state.phase, "canceled");
    assert.equal(state.isExporting, false);
    assert.equal(state.ownsRun, false);
    assert.deepEqual(operationPosts(), []);
    assert.deepEqual(
      stub.calls.map((entry) => entry.name),
      ["getExportStatus", "cancelExport", "cleanupExport"],
    );

    // The next run starts clean: runExport resets the recorded cancel intent.
    stub.resetStub();
    await store.runExport(params());
    const rerun = useExportRuntimeStore.getState();
    assert.equal(rerun.phase, "success");
    assert.deepEqual(operationPosts(), ["loadCheckpoint", "exportGGUF"]);
  } finally {
    stub.resetStub();
    useExportRuntimeStore.getState().reset();
  }
});

test("a cancel between two export operations blocks the pending POST", async () => {
  const store = useExportRuntimeStore.getState();
  store.reset();
  stub.resetStub();
  const late = deferred();
  let statusCalls = 0;
  stub.responses.set("getExportStatus", () => {
    statusCalls += 1;
    return statusCalls < 3
      ? Promise.resolve({ last_op_seq: statusCalls })
      : late.promise;
  });
  try {
    const run = store.runExport(
      params({
        exportMethod: "merged",
        mergedSelections: [
          { formatType: "16-bit (FP16)", compressedMethod: null, label: "FP16" },
          { formatType: "4-bit (NF4)", compressedMethod: "nf4", label: "NF4" },
        ],
      }),
    );
    while (statusCalls < 3) {
      await tick();
    }
    assert.deepEqual(operationPosts(), ["loadCheckpoint", "exportMerged"]);

    await store.requestCancel();
    late.resolve({ last_op_seq: 9 });
    await run;

    const state = useExportRuntimeStore.getState();
    assert.equal(state.phase, "canceled");
    assert.equal(state.isExporting, false);
    assert.deepEqual(operationPosts(), ["loadCheckpoint", "exportMerged"]);
    assert.equal(
      stub.calls.filter((entry) => entry.name === "exportMerged").length,
      1,
      "the second selection must never dispatch",
    );
  } finally {
    late.resolve({ last_op_seq: 9 });
    stub.resetStub();
    useExportRuntimeStore.getState().reset();
  }
});

test("an uncanceled run dispatches every operation", async () => {
  const store = useExportRuntimeStore.getState();
  store.reset();
  stub.resetStub();
  const preflight = deferred();
  stub.responses.set("getExportStatus", () => preflight.promise);
  try {
    const run = store.runExport(params());
    preflight.resolve({ last_op_seq: 0 });
    await run;

    assert.equal(useExportRuntimeStore.getState().phase, "success");
    assert.deepEqual(operationPosts(), ["loadCheckpoint", "exportGGUF"]);
  } finally {
    stub.resetStub();
    useExportRuntimeStore.getState().reset();
  }
});
