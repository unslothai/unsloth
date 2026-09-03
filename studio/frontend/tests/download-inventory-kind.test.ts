// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerStoreStubResolver } from "./helpers/kit.ts";

import {
  downloadInventoryHintKind,
  downloadRequestInventoryKind,
  scopedDownloadInventoryKind,
} from "../src/features/hub/download-manager/download-manager-types.ts";

registerStoreStubResolver();

const {
  createLiveInventoryJobsSelector,
  liveInventorySelectionKey,
  observedKeyProtections,
} = await import("../src/features/hub/inventory/use-hub-inventory.ts");

function liveJob(over: Record<string, unknown>) {
  return {
    kind: "model",
    repoId: "Org/Repo",
    variant: null,
    state: "running",
    startedAt: 1,
    downloadedBytes: 0,
    completedBytes: 0,
    expectedBytes: 0,
    fraction: 0,
    bytesPerSec: 0,
    etaSeconds: 0,
    error: null,
    completeOnDisk: false,
    ...over,
  } as never;
}

test("classifies quant variants as GGUF", () => {
  assert.equal(downloadInventoryHintKind("model", "Q4_K_M"), "gguf");
});

test("classifies legacy scoped variants as model downloads", () => {
  assert.equal(downloadInventoryHintKind("model", "@diffusion"), "model");
});

test("uses the caller's format for scoped downloads", () => {
  assert.equal(
    downloadInventoryHintKind("model", "@diffusion", "gguf"),
    "gguf",
  );
  assert.equal(
    downloadInventoryHintKind("model", "@diffusion", "model"),
    "model",
  );
});

test("keeps dataset jobs out of model inventory", () => {
  assert.equal(downloadInventoryHintKind("dataset", null), "dataset");
});

test("recovers scoped inventory format from backend files", () => {
  assert.equal(
    scopedDownloadInventoryKind(["transformer/model-Q4_K_M.gguf"]),
    "gguf",
  );
  assert.equal(
    scopedDownloadInventoryKind(["transformer/model.safetensors"]),
    "model",
  );
  assert.equal(scopedDownloadInventoryKind(undefined), "model");
});

test("classifies staged single-file checkpoints from their file set", () => {
  // A `.safetensors` checkpoint can sit in the field named gguf_filename, so that field is not evidence of GGUF.
  assert.equal(
    downloadRequestInventoryKind({
      kind: "model",
      variant: "@diffusion",
      files: ["flux1-dev.safetensors"],
    }),
    "model",
  );
  assert.equal(
    downloadRequestInventoryKind({
      kind: "model",
      variant: "@diffusion",
      files: ["flux1-dev-Q8_0.gguf"],
    }),
    "gguf",
  );
  assert.equal(scopedDownloadInventoryKind(["flux1-dev.safetensors"]), "model");
});

test("infers missing scoped request formats during adoption", () => {
  assert.equal(
    downloadRequestInventoryKind({
      kind: "model",
      variant: "@rag-embedding",
      files: ["model-Q4_K_M.gguf"],
    }),
    "gguf",
  );
  assert.equal(
    downloadRequestInventoryKind({
      kind: "model",
      variant: "@diffusion",
      files: ["transformer/model.safetensors"],
    }),
    "model",
  );
  assert.equal(
    downloadRequestInventoryKind({
      kind: "model",
      variant: "@rag-embedding",
      inventoryKind: "model",
      files: ["model-Q4_K_M.gguf"],
    }),
    "model",
  );
  assert.equal(
    downloadRequestInventoryKind({
      kind: "model",
      variant: "@rag-embedding",
    }),
    undefined,
  );

});

test("separates live inventory rows for hybrid repository formats", () => {
  // Two scoped jobs on one repo share the variant SHAPE, so only the resolved inventory kind separates them; keying on the variant collapses both rows onto one.
  const selector = createLiveInventoryJobsSelector(false);
  const rows = selector({
    jobs: {
      a: liveJob({
        key: "a",
        variant: "@rag-embedding",
        inventoryKind: "gguf",
      }),
      b: liveJob({ key: "b", variant: "@diffusion", inventoryKind: "model" }),
    },
  });

  assert.equal(rows.length, 2);
  assert.deepEqual(
    rows.map((row) => row.inventoryKind).sort(),
    ["gguf", "model"],
  );
  assert.equal(
    liveInventorySelectionKey(rows[0]!) === liveInventorySelectionKey(rows[1]!),
    false,
    "two formats of one repo collapsed onto a single live row",
  );
});

test("same-format live jobs for one repository still collapse", () => {
  const selector = createLiveInventoryJobsSelector(false);
  const rows = selector({
    jobs: {
      a: liveJob({ key: "a", variant: "Q4_K_M", inventoryKind: "gguf" }),
      b: liveJob({ key: "b", variant: "Q8_0", inventoryKind: "gguf" }),
    },
  });
  assert.equal(rows.length, 1);
});

test("protects observed keys with the job's explicit inventory kind", () => {
  // A scoped job classified as gguf must protect the gguf key, not the key its `@variant` shape alone implies.
  const keys = observedKeyProtections({
    scoped: liveJob({
      key: "scoped",
      variant: "@rag-embedding",
      inventoryKind: "gguf",
    }),
  });
  assert.equal(keys.gguf.has("org/repo"), true);
  assert.equal(keys.model.has("org/repo"), false);

  const unclassified = observedKeyProtections({
    scoped: liveJob({ key: "scoped", variant: "@diffusion" }),
  });
  assert.equal(unclassified.model.has("org/repo"), true);
  assert.equal(unclassified.gguf.has("org/repo"), false);
});
