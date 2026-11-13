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

const { apiStart } =
  await import("../src/features/hub/download-manager/download-api-adapter.ts");
const { DOWNLOAD_DIR_STORAGE_KEY } =
  await import("../src/features/hub/download-manager/download-location.ts");

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
  await apiStart(
    { ...model, kind: "dataset" as const, variant: null },
    false,
    null,
  );
  assert.equal("download_dir" in bodies[0], false);
});

test("a folder pinned on the request wins over a later choice", async () => {
  storage.setItem(DOWNLOAD_DIR_STORAGE_KEY, "/mnt/b/Models");
  bodies.length = 0;
  await apiStart({ ...model, downloadDir: "/mnt/a/Models" }, false, null);
  await apiStart({ ...model, downloadDir: null }, false, null);
  assert.equal(bodies[0].download_dir, "/mnt/a/Models");
  assert.equal("download_dir" in bodies[1], false);
});

test("Resume after a cancel keeps the folder; a failed start takes the new choice", async () => {
  const { withDownloadDir } =
    await import("../src/features/hub/download-manager/transport-conflict.ts");
  const { putJob, jobKeyOf } =
    await import("../src/features/hub/download-manager/download-manager-state.ts");
  storage.setItem(DOWNLOAD_DIR_STORAGE_KEY, "/mnt/b/Models");
  const base = {
    key: jobKeyOf("model", model.repoId, model.variant),
    kind: "model" as const,
    repoId: model.repoId,
    variant: model.variant,
    downloadedBytes: 0,
    completedBytes: 0,
    completeOnDisk: false,
    expectedBytes: 0,
    fraction: 0,
    bytesPerSec: 0,
    etaSeconds: 0,
    error: null,
    startedAt: 1,
    downloadDir: "/mnt/a/Models",
  };
  putJob({ ...base, state: "cancelled" });
  assert.equal(withDownloadDir(model).downloadDir, "/mnt/a/Models");
  putJob({
    ...base,
    state: "error",
    error: "The parent folder does not exist.",
  });
  assert.equal(withDownloadDir(model).downloadDir, "/mnt/b/Models");
});
