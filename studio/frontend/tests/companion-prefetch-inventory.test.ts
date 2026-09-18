// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import {
  companionPrefetchDownloadHint,
  formatCachedComponentsSummary,
  formatPipelineComponentLabel,
} from "../src/features/hub/lib/pipeline-components.ts";
import { modelDownloadState } from "../src/features/hub/catalog/model-download-state.ts";

const ROOT = join(import.meta.dirname, "..");

function read(relPath: string): string {
  return readFileSync(join(ROOT, relPath), "utf8");
}

test("companion prefetch rows are not treated as partial downloads", () => {
  const state = modelDownloadState({
    id: "Org/Base",
    loadId: "Org/Base",
    kind: "cache",
    displayId: "Org/Base",
    hubRepoId: "Org/Base",
    owner: "Org",
    title: "Base",
    summary: "",
    sourceLabel: "Hub cache",
    path: "/cache",
    isLocal: false,
    isGguf: false,
    modelFormat: "safetensors",
    isDownloaded: false,
    runtimeCanChat: false,
    isPartial: false,
    companionPrefetch: true,
    cachedComponents: ["text_encoder", "vae"],
    capabilities: [],
    license: null,
  });

  assert.equal(state.isPartial, false);
  assert.equal(state.companionPrefetch, true);
  assert.equal(state.isDownloaded, false);
});

test("partial download rows stay partial", () => {
  const state = modelDownloadState({
    id: "Org/Base",
    loadId: "Org/Base",
    kind: "cache",
    displayId: "Org/Base",
    hubRepoId: "Org/Base",
    owner: "Org",
    title: "Base",
    summary: "",
    sourceLabel: "Hub cache",
    path: "/cache",
    isLocal: false,
    isGguf: false,
    modelFormat: "safetensors",
    isDownloaded: false,
    runtimeCanChat: false,
    isPartial: true,
    partialTransport: "http",
    partialResumable: true,
    capabilities: [],
    license: null,
  });

  assert.equal(state.isPartial, true);
  assert.equal(state.companionPrefetch, false);
  assert.equal(state.isDownloaded, false);
});

test("cached component labels are human readable", () => {
  assert.equal(formatPipelineComponentLabel("text_encoder"), "Text encoder");
  assert.equal(formatPipelineComponentLabel("vae"), "VAE");
  assert.equal(
    formatCachedComponentsSummary(["text_encoder", "vae"]),
    "Text encoder, VAE",
  );
});

test("companion prefetch hint names cached assets and full download", () => {
  const hint = companionPrefetchDownloadHint(["text_encoder", "vae"]);
  assert.match(hint, /Text encoder, VAE cached/);
  assert.match(hint, /Full pipeline weights are not installed/);
  assert.match(hint, /Click Download/);
});

test("safetensors download card shows cached assets instead of partial", () => {
  const card = read("src/features/hub/catalog/safetensors-download-card.tsx");
  assert.match(card, /companionPrefetch && !downloading/);
  assert.match(card, /label="Cached assets"/);
  assert.doesNotMatch(
    card.split('label="Cached assets"')[0],
    /companionPrefetch[\s\S]*label="Partial"/,
  );
});

test("inventory rows expose cached assets status", () => {
  const rows = read("src/features/hub/catalog/models-catalog-rows.tsx");
  assert.match(rows, /companionPrefetchRepoId/);
  assert.match(rows, /label="Cached assets"/);
});
