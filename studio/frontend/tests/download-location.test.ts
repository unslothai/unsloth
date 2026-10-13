// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Download to (#10872): a model download carries the chosen folder, and nothing when none is chosen.

import assert from "node:assert/strict";
import test, { after } from "node:test";
import { register } from "node:module";
import { installLocalStorageFake } from "./helpers/kit.ts";

register("./helpers/download-lifecycle-resolver.mjs", import.meta.url);

const { storage } = installLocalStorageFake();
const originalFetch = globalThis.fetch;
after(() => {
  globalThis.fetch = originalFetch;
});

const bodies: Record<string, unknown>[] = [];
globalThis.fetch = (async (_input: string, init?: RequestInit) => {
  bodies.push(JSON.parse(String(init?.body)));
  return new Response(JSON.stringify({ job_key: "k", state: "running" }), {
    status: 202,
    headers: { "Content-Type": "application/json" },
  });
}) as typeof fetch;

const { apiStart } = await import(
  "../src/features/hub/download-manager/download-api-adapter.ts"
);
const { DOWNLOAD_DIR_STORAGE_KEY } = await import(
  "../src/features/hub/download-manager/download-location.ts"
);

const model = {
  kind: "model" as const,
  repoId: "Org/Model-GGUF",
  variant: "Q4_K_M",
  expectedBytes: 0,
};

test("no folder chosen sends the same body as before", async () => {
  storage.removeItem(DOWNLOAD_DIR_STORAGE_KEY);
  bodies.length = 0;
  await apiStart(model, false, null);
  assert.equal(bodies.length, 1);
  assert.equal("download_dir" in bodies[0], false);
});

test("a chosen folder rides on the model download", async () => {
  storage.setItem(DOWNLOAD_DIR_STORAGE_KEY, "/mnt/e/AI/Models");
  bodies.length = 0;
  await apiStart(model, false, null);
  assert.equal(bodies[0].download_dir, "/mnt/e/AI/Models");
});

test("datasets never carry the folder", async () => {
  storage.setItem(DOWNLOAD_DIR_STORAGE_KEY, "/mnt/e/AI/Models");
  bodies.length = 0;
  await apiStart({ ...model, kind: "dataset" as const, variant: null }, false, null);
  assert.equal("download_dir" in bodies[0], false);
});
