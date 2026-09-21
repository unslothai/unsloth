// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Jobs persisted before `inventoryKind` still carry `scopedFiles`, so hydration recovers the kind rather than giving a scoped GGUF a safetensors row after an update.

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();

const PERSIST_KEY = "unsloth.studio.downloads";

function persistedJob(
  repoId: string,
  variant: string | null,
  extra: Record<string, unknown> = {},
) {
  return {
    key: `model:${repoId}:${variant ?? ""}`,
    kind: "model",
    repoId,
    variant,
    state: "running",
    downloadedBytes: 25,
    completedBytes: 0,
    expectedBytes: 100,
    fraction: 0.25,
    error: null,
    startedAt: 1,
    ...extra,
  };
}

store.set(
  PERSIST_KEY,
  JSON.stringify({
    state: {
      jobs: {
        // Written before `inventoryKind` existed: a scoped GGUF embedding download.
        legacyGguf: persistedJob("org/embedder", "@rag-embedding", {
          scopedFiles: ["nomic-embed-text-v1.5.Q8_0.gguf"],
        }),
        legacyModel: persistedJob("org/diffusion", "@diffusion", {
          scopedFiles: ["transformer/diffusion_pytorch_model.safetensors"],
        }),
        explicit: persistedJob("org/explicit", "@diffusion", {
          scopedFiles: ["model.safetensors"],
          inventoryKind: "gguf",
        }),
        noEvidence: persistedJob("org/unknown-scope", "@diffusion"),
        quant: persistedJob("org/quant", "Q4_K_M"),
      },
      conflicts: {},
    },
    version: 2,
  }),
);

const { getState, jobKeyOf } = await import(
  "../src/features/hub/download-manager/download-manager-state.ts"
);
const { downloadInventoryHintKind } = await import(
  "../src/features/hub/download-manager/download-manager-types.ts"
);

function job(repoId: string, variant: string | null) {
  return getState().jobs[jobKeyOf("model", repoId, variant)];
}

test("a scoped GGUF job persisted before inventoryKind is recovered", () => {
  const recovered = job("org/embedder", "@rag-embedding");
  assert.ok(recovered, "the legacy job did not survive hydration");
  assert.equal(recovered.inventoryKind, "gguf");
  assert.equal(
    downloadInventoryHintKind(
      recovered.kind,
      recovered.variant,
      recovered.inventoryKind,
    ),
    "gguf",
    "a scoped GGUF download came back as a model download after a reload",
  );
});

test("a scoped safetensors job keeps the model classification", () => {
  const recovered = job("org/diffusion", "@diffusion");
  assert.ok(recovered);
  assert.equal(recovered.inventoryKind, "model");
});

test("an explicitly stored kind still wins over the file list", () => {
  assert.equal(job("org/explicit", "@diffusion")?.inventoryKind, "gguf");
});

test("a scoped job with no files stays unclassified", () => {
  // Guessing here would publish a wrong row; backend adoption resolves it.
  assert.equal(
    job("org/unknown-scope", "@diffusion")?.inventoryKind,
    undefined,
  );
});

test("an unscoped quant download is untouched", () => {
  const quant = job("org/quant", "Q4_K_M");
  assert.ok(quant);
  assert.equal(quant.inventoryKind, undefined);
  assert.equal(
    downloadInventoryHintKind(quant.kind, quant.variant, quant.inventoryKind),
    "gguf",
  );
});
