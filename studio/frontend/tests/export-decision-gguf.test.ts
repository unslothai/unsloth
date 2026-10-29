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

function params(overrides: Record<string, unknown>) {
  return {
    sourceMode: "checkpoint",
    checkpointPath: "/runs/clef",
    source: "clef",
    modelSource: "local",
    trustRemoteCode: true,
    exportMethod: "gguf",
    isAdapter: false,
    quantLevels: ["q8_0", "q4_k_m"],
    saveDirectory: "clef-gguf",
    destination: "local",
    privateRepo: false,
    summary: {},
    ...overrides,
  } as unknown as Parameters<
    ReturnType<typeof useExportRuntimeStore.getState>["runExport"]
  >[0];
}

async function run(overrides: Record<string, unknown>, details: unknown) {
  stub.resetStub();
  stub.responses.set("exportGGUF", {
    success: true,
    message: "ok",
    details,
  });
  await useExportRuntimeStore.getState().runExport(params(overrides));
  return useExportRuntimeStore.getState().result;
}

const label = (quants: string[]) => `Decision GGUF (${quants.join(", ")})`;

test("a decision export lists the quantizations the run folder now holds", async () => {
  const result = await run(
    { decisionOutputLabel: label },
    { output_path: "/runs/clef/gguf", quantizations: ["Q8_0", "Q4_K_M", "F16"] },
  );
  assert.deepEqual(result?.outputPaths, [
    { label: "Decision GGUF (Q8_0, Q4_K_M, F16)", path: "/runs/clef/gguf" },
  ]);
  const body = stub.calls.find((c) => c.name === "exportGGUF")?.args[0] as
    | Record<string, unknown>
    | undefined;
  assert.deepEqual(body?.quantization_method, ["q8_0", "q4_k_m"]);
});

test("a decision export without export.json details falls back to the requested quants", async () => {
  const result = await run(
    { decisionOutputLabel: label },
    { output_path: "/runs/clef/gguf" },
  );
  assert.equal(result?.outputPaths[0]?.label, "Decision GGUF (Q8_0, Q4_K_M)");
});

test("other GGUF exports keep the plain label", async () => {
  const result = await run({}, { output_path: "exports/model-gguf" });
  assert.deepEqual(result?.outputPaths, [
    { label: "GGUF", path: "exports/model-gguf" },
  ]);
});
