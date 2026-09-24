// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A repo whose download is still running holds an `.incomplete` blob, so the cache scan reports it
// partial, and the live-download row the inventory injects is partial by construction. Every
// surface then drew the stopped-download treatment: an amber "Partial download, open it to
// finish" dot in the On Device list and a "Partial" tag with a "Resume" button on the card. A
// scoped job (an image model's "Required assets", a staged "Model file") was worse, since the
// repo's own card keys on the snapshot job and saw no download at all. Users read all of this as
// the download being paused while the downloads panel and toast said it was running.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const { activeDownloadRepoKeys, markDownloadingRows } = await import(
  "../src/features/hub/inventory/inventory-dedupe.ts"
);
const { createLiveInventoryJobsSelector } = await import(
  "../src/features/hub/inventory/use-hub-inventory.ts"
);
const { buildDiscoverRows } = await import(
  "../src/features/hub/lib/view-models.ts"
);
const { findActiveScopedJobForRepo, selectActiveJob } = await import(
  "../src/features/hub/download-manager/download-manager-state.ts"
);
const { isRepoDownloadProgress } = await import(
  "../src/features/hub/catalog/use-download-card-state.ts"
);
const { createScopedLiveGgufFilesSelector, isScopedLiveVariant } =
  await import("../src/features/hub/catalog/gguf-live-variant-states.ts");

const REPO = "unsloth/Qwen-Image-2.1-FP8";

function job(over: Record<string, unknown> = {}) {
  const variant = (over.variant as string | null | undefined) ?? null;
  return {
    key: `model:${REPO.toLowerCase()}${variant ? `#${variant.toLowerCase()}` : ""}`,
    kind: "model",
    repoId: REPO,
    variant,
    state: "running",
    startedAt: 1,
    downloadedBytes: 1_200_000_000,
    completedBytes: 0,
    completeOnDisk: false,
    expectedBytes: 9_400_000_000,
    fraction: 0.13,
    bytesPerSec: 0,
    etaSeconds: 0,
    error: null,
    ...over,
  } as never;
}

function scannedRow(over: Record<string, unknown> = {}) {
  return {
    kind: "cache" as const,
    id: `safetensors:${REPO}`,
    loadId: REPO,
    repoId: REPO,
    owner: "unsloth",
    repo: "Qwen-Image-2.1-FP8",
    isGguf: false,
    modelFormat: "safetensors" as const,
    capabilities: {} as never,
    bytes: 9_400_000_000,
    partial: true,
    downloading: undefined as boolean | undefined,
    ...over,
  };
}

function jobsRecord(...list: ReturnType<typeof job>[]) {
  return Object.fromEntries(
    list.map((entry) => [(entry as { key: string }).key, entry]),
  );
}

test("a partial row with a running scoped job reads as downloading, not paused", () => {
  const scoped = job({ variant: "@hub-required-assets", scopedFiles: ["vae/a.safetensors"] });
  const live = createLiveInventoryJobsSelector(false)({ jobs: jsonJobs(scoped) });
  const keys = activeDownloadRepoKeys(live);
  assert.deepEqual([...keys], [REPO.toLowerCase()]);

  const [row] = markDownloadingRows([scannedRow()], (r) => r.repoId, keys);
  assert.equal(row.partial, true);
  assert.equal(row.downloading, true);
});

test("a stopped partial stays a partial", () => {
  for (const state of ["cancelled", "error"]) {
    const stopped = job({ state });
    const live = createLiveInventoryJobsSelector(false)({ jobs: jsonJobs(stopped) });
    const keys = activeDownloadRepoKeys(live);
    assert.equal(keys.size, 0, state);
    const rows = [scannedRow()];
    const marked = markDownloadingRows(rows, (r) => r.repoId, keys);
    assert.equal(marked, rows, "unchanged rows keep their identity");
    assert.equal(marked[0].downloading, undefined, state);
  }
});

test("a complete row is never marked downloading", () => {
  const keys = activeDownloadRepoKeys([{ repoId: REPO, state: "running" }]);
  const rows = [scannedRow({ partial: false })];
  assert.equal(markDownloadingRows(rows, (r) => r.repoId, keys), rows);
});

test("the flag clears once the job leaves the running states", () => {
  const rows = [scannedRow({ downloading: true })];
  const [row] = markDownloadingRows(rows, (r) => r.repoId, new Set());
  assert.equal(row.downloading, false);
});

test("repo matching ignores case, as the cache dir does", () => {
  const keys = activeDownloadRepoKeys([
    { repoId: "UNSLOTH/qwen-image-2.1-fp8", state: "cancelling" },
  ]);
  const [row] = markDownloadingRows([scannedRow()], (r) => r.repoId, keys);
  assert.equal(row.downloading, true);
});

test("discover rows carry the downloading state of their inventory row", () => {
  const result = { id: REPO, isGguf: false, tags: [] } as never;
  const [downloading] = buildDiscoverRows(
    [result],
    [scannedRow({ downloading: true })] as never,
    [],
  );
  assert.equal(downloading.isPartialOnDevice, true);
  assert.equal(downloading.isDownloadingOnDevice, true);

  const [stopped] = buildDiscoverRows([result], [scannedRow()] as never, []);
  assert.equal(stopped.isPartialOnDevice, true);
  assert.equal(stopped.isDownloadingOnDevice, false);
});

test("the repo card finds a running scoped job the snapshot key does not", () => {
  const scoped = job({ variant: "@hub-required-assets" });
  const state = {
    jobs: jsonJobs(scoped),
    conflicts: {},
    completedHintSignature: "",
    completedInventoryHints: [],
  } as never;
  // The snapshot key the Safetensors card asks for has no job.
  assert.equal(selectActiveJob(state, "model", REPO, null), null);
  const found = findActiveScopedJobForRepo(jsonJobs(scoped), "model", REPO);
  assert.equal(found?.variant, "@hub-required-assets");
  assert.equal(isRepoDownloadProgress({ variant: found?.variant ?? null }), true);
});

test("a GGUF quant job is not a repo-level download, and stopped scoped jobs are ignored", () => {
  const quant = job({ variant: "Q8_0" });
  const cancelled = job({ variant: "@other", state: "cancelled" });
  assert.equal(
    findActiveScopedJobForRepo(jsonJobs(quant, cancelled), "model", REPO),
    null,
  );
  assert.equal(isRepoDownloadProgress({ variant: "Q8_0" }), false);
  assert.equal(isRepoDownloadProgress({ variant: null }), true);
  assert.equal(isRepoDownloadProgress(null), false);
});

test("a GGUF quant written by a scoped Model file job reads as downloading", () => {
  const gguf = "unsloth/Qwen-Image-2.1-GGUF";
  const scoped = job({
    repoId: gguf,
    key: "model:unsloth/qwen-image-2.1-gguf#@images",
    variant: "@images",
    scopedFiles: ["qwen-image-2.1-Q8_0.gguf"],
  });
  const select = createScopedLiveGgufFilesSelector(gguf);
  const files = select({ jobs: jsonJobs(scoped) });
  assert.equal(isScopedLiveVariant({ filename: "qwen-image-2.1-Q8_0.gguf" }, files), true);
  assert.equal(isScopedLiveVariant({ filename: "qwen-image-2.1-Q4_K_M.gguf" }, files), false);
  assert.equal(select({ jobs: jsonJobs(scoped) }), files, "stable identity while unchanged");

  const done = select({ jobs: jsonJobs(job({ ...(scoped as object), state: "complete" })) });
  assert.equal(isScopedLiveVariant({ filename: "qwen-image-2.1-Q8_0.gguf" }, done), false);
});

function jsonJobs(...list: ReturnType<typeof job>[]) {
  return jobsRecord(...list) as never;
}

test("every partial marker on the hub has a downloading branch", () => {
  const card = readSrc("features/hub/catalog/safetensors-download-card.tsx");
  assert.match(card, /includeScopedJobs: true/);
  assert.match(card, /isRepoDownloadProgress\(progress\)/);
  assert.match(card, /tone="downloading" label="Downloading"/);

  const rows = readSrc("features/hub/catalog/models-catalog-rows.tsx");
  // The amber dot is only drawn through PartialStatusDot, which picks Downloading when live.
  assert.equal(rows.match(/<StatusDot tone="warning" label="Partial download" \/>/g)?.length, 1);
  assert.match(rows, /<PartialStatusDot downloading=\{downloading\} \/>/);

  for (const file of ["models-table.tsx", "model-card.tsx"]) {
    const source = readSrc(`features/hub/catalog/${file}`);
    assert.match(source, /partial && !downloading && \(/, file);
    assert.match(source, /aria-label="Downloading"/, file);
  }
});
