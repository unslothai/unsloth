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

function refusingBackend() {
  const cancelledIds = new Set<unknown>();
  const begun: string[] = [];
  stub.responses.set("cancelExport", (id: unknown) => {
    if (id) cancelledIds.add(id);
    return Promise.resolve({ success: true, message: "ok" });
  });
  const op =
    (name: string, beforeBegin?: () => Promise<void>) =>
    async (body: { op_id?: string }) => {
      await beforeBegin?.();
      if (cancelledIds.has(body.op_id)) throw new Error("Export cancelled");
      begun.push(name);
      return { success: true, message: "ok", details: null };
    };
  return { begun, op };
}

test("a cancel landing after the dispatch check but before the op begins is refused", async () => {
  const store = useExportRuntimeStore.getState();
  store.reset();
  stub.resetStub();
  const backend = refusingBackend();
  stub.responses.set(
    "loadCheckpoint",
    backend.op("loadCheckpoint", () => store.requestCancel()),
  );
  stub.responses.set("exportGGUF", backend.op("exportGGUF"));
  try {
    await store.runExport(params());
    assert.equal(useExportRuntimeStore.getState().phase, "canceled");
    assert.deepEqual(backend.begun, []);
    assert.deepEqual(operationPosts(), ["loadCheckpoint"]);
  } finally {
    stub.resetStub();
    useExportRuntimeStore.getState().reset();
  }
});

test("a cancel during an op targets that op's id", async () => {
  const store = useExportRuntimeStore.getState();
  store.reset();
  stub.resetStub();
  const load = deferred();
  stub.responses.set("loadCheckpoint", () => load.promise);
  try {
    const run = store.runExport(params());
    while (!operationPosts().length) await tick();
    await store.requestCancel();
    load.resolve({ success: true, message: "ok", details: null });
    await run;

    const sent = stub.calls.find((c) => c.name === "loadCheckpoint")!
      .args[0] as { op_id?: string };
    const cancel = stub.calls.find((c) => c.name === "cancelExport")!;
    assert.ok(sent.op_id);
    assert.equal(cancel.args[0], sent.op_id);
    assert.equal(useExportRuntimeStore.getState().phase, "canceled");
    assert.deepEqual(operationPosts(), ["loadCheckpoint"]);
  } finally {
    stub.resetStub();
    useExportRuntimeStore.getState().reset();
  }
});

test("each op of a run carries its own id", async () => {
  const store = useExportRuntimeStore.getState();
  store.reset();
  stub.resetStub();
  try {
    await store.runExport(params());
    const ids = stub.calls
      .filter((c) => ["loadCheckpoint", "exportGGUF"].includes(c.name))
      .map((c) => (c.args[0] as { op_id?: string }).op_id);
    assert.equal(ids.length, 2);
    assert.ok(ids[0] && ids[1] && ids[0] !== ids[1]);
  } finally {
    stub.resetStub();
    useExportRuntimeStore.getState().reset();
  }
});

test("cancelling a run attached from another tab stops whatever runs", async () => {
  const store = useExportRuntimeStore.getState();
  store.reset();
  stub.resetStub();
  try {
    await store.runExport(params());
    store.applyBackendStatus({ is_export_active: true } as Status);
    assert.equal(useExportRuntimeStore.getState().ownsRun, false);
    stub.resetStub();
    await store.requestCancel();
    const cancel = stub.calls.find((c) => c.name === "cancelExport")!;
    assert.equal(cancel.args[0], null);
  } finally {
    stub.resetStub();
    useExportRuntimeStore.getState().reset();
  }
});

test("timeout recovery waits for the export's own operation result", async (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  const store = useExportRuntimeStore.getState();
  store.reset();
  stub.resetStub();
  let exportId: string | undefined;
  let statusRequests = 0;
  stub.responses.set("exportGGUF", ({ op_id }: { op_id: string }) => {
    exportId = op_id;
    throw new Error("Cloudflare 524");
  });
  stub.responses.set("getExportStatus", () => {
    statusRequests += 1;
    const status = {
      is_export_active: false,
      last_op_seq: statusRequests,
      last_op_kind: "export_gguf",
      last_op_status: "success",
    };
    return {
      ...status,
      last_op_id: statusRequests === 4 ? exportId : "another-operation",
      last_op_output_path: statusRequests === 4 ? "my-export" : "another-export",
    };
  });

  try {
    const run = store.runExport(params());
    await tick();
    assert.equal(useExportRuntimeStore.getState().reconnecting, true);

    // The first poll sees a different export finish. The next sees our result.
    t.mock.timers.tick(1500);
    await tick();
    t.mock.timers.tick(1500);
    await run;

    const result = useExportRuntimeStore.getState();
    assert.equal(result.phase, "success");
    assert.equal(result.result?.outputPath, "my-export");
  } finally {
    stub.resetStub();
    useExportRuntimeStore.getState().reset();
  }
});
