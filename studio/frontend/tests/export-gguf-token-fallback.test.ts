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

// A local imatrix export resolves the matrix from a Hub repo, but export-page.tsx only sets
// `token` for a hub push, so the GGUF request falls back to the load token as the LoRA request
// already does. Asserting the emitted body, not the source, so a refactor still passes.

function params(overrides: Record<string, unknown>) {
  return {
    sourceMode: "model",
    checkpointPath: null,
    source: "unsloth/Qwen3-0.6B",
    modelSource: "hf",
    trustRemoteCode: false,
    exportMethod: "gguf",
    isAdapter: false,
    quantLevels: ["iq2_xxs"],
    saveDirectory: "out",
    destination: "local",
    privateRepo: false,
    summary: {},
    ...overrides,
  } as unknown as Parameters<
    ReturnType<typeof useExportRuntimeStore.getState>["runExport"]
  >[0];
}

async function ggufRequest(overrides: Record<string, unknown>) {
  stub.resetStub();
  await useExportRuntimeStore.getState().runExport(params(overrides));
  const call = stub.calls.find((entry) => entry.name === "exportGGUF");
  assert.ok(call, "no GGUF export request was made");
  return call.args[0] as Record<string, unknown>;
}

test("a local imatrix export falls back to the load token", async () => {
  const body = await ggufRequest({
    useImatrix: true,
    loadToken: "hf_load",
    token: undefined,
    destination: "local",
  });

  assert.equal(body.hf_token, "hf_load");
  assert.equal(body.imatrix, true);
  assert.equal(body.push_to_hub, false);
});

test("a hub push prefers its own upload token over the load token", async () => {
  const body = await ggufRequest({
    useImatrix: true,
    loadToken: "hf_load",
    token: "hf_upload",
    destination: "hub",
    repoId: "org/model",
  });

  assert.equal(body.hf_token, "hf_upload");
  assert.equal(body.repo_id, "org/model");
  assert.equal(body.push_to_hub, true);
});

test("no token anywhere sends null rather than undefined", async () => {
  const body = await ggufRequest({ useImatrix: true, loadToken: null, token: undefined });

  // undefined would be dropped by JSON.stringify and the field would go missing entirely.
  assert.equal(body.hf_token, null);
  assert.ok("hf_token" in body);
});

test("the fallback matches the LoRA request, which already had it", async () => {
  const gguf = await ggufRequest({ useImatrix: true, loadToken: "hf_load", token: undefined });

  stub.resetStub();
  await useExportRuntimeStore.getState().runExport(
    params({ exportMethod: "lora", loadToken: "hf_load", token: undefined }),
  );
  const lora = stub.calls.find((entry) => entry.name === "exportLoRA");
  assert.ok(lora, "no LoRA export request was made");

  assert.equal(gguf.hf_token, (lora.args[0] as Record<string, unknown>).hf_token);
});

test("the load phase keeps using the load token on its own", async () => {
  stub.resetStub();
  await useExportRuntimeStore.getState().runExport(
    params({ useImatrix: true, loadToken: "hf_load", token: "hf_upload", destination: "hub",
             repoId: "org/model" }),
  );
  const load = stub.calls.find((entry) => entry.name === "loadCheckpoint");
  assert.ok(load, "no load request was made");
  assert.equal((load.args[0] as Record<string, unknown>).hf_token, "hf_load");
});
