// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  downloadInventoryHintKind,
  downloadRequestInventoryKind,
  scopedDownloadInventoryKind,
} from "../src/features/hub/download-manager/download-manager-types.ts";

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
  const source = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/hub/download-manager/use-staged-download.ts",
        import.meta.url,
      ),
    ),
    "utf-8",
  );
  assert.match(
    source,
    /inventoryKind: scopedDownloadInventoryKind\(current\.files\)/,
  );
  assert.doesNotMatch(source, /inventoryKind: current\.ggufFilename/);
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

  const source = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/hub/download-manager/poll-loop.ts",
        import.meta.url,
      ),
    ),
    "utf-8",
  );
  assert.match(
    source,
    /if \(probeDescribesCurrentRun\(known, generation\)\)[\s\S]*?const inventoryKind = downloadRequestInventoryKind\(req\)[\s\S]*?scopedFiles: \[\.\.\.req\.files\][\s\S]*?\{ inventoryKind \}/,
  );
});

test("separates live inventory rows for hybrid repository formats", () => {
  const source = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/hub/inventory/use-hub-inventory.ts",
        import.meta.url,
      ),
    ),
    "utf-8",
  );
  assert.match(
    source,
    /function liveInventorySelectionKey[\s\S]*?inventoryHintKey\(\s*downloadInventoryHintKind\(job\.kind, job\.variant, job\.inventoryKind\),\s*job\.repoId,\s*\)/,
  );
  assert.match(
    source,
    /const selectionKey = liveInventorySelectionKey\(job\)[\s\S]*?selectedByRepoAndKind\.set\(selectionKey, job\)/,
  );
});

test("protects observed keys with the job's explicit inventory kind", () => {
  const source = readFileSync(
    fileURLToPath(
      new URL(
        "../src/features/hub/inventory/use-hub-inventory.ts",
        import.meta.url,
      ),
    ),
    "utf-8",
  );
  assert.match(
    source,
    /keys\[\s*downloadInventoryHintKind\(job\.kind, job\.variant, job\.inventoryKind\)\s*\]/,
  );
});
