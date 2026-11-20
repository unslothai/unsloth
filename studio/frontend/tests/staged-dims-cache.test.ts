// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  cachedStagedDims,
  rememberStagedDims,
  stagedDimsLastInconclusive,
} from "../src/features/model-picker/model-config/staged-dims-cache.ts";

const read = { contextLength: 32768, layerCount: 28, moeLayerCount: null, isDiffusion: false };

test("a header that was read is reused on reopen", () => {
  rememberStagedDims({ key: "org/m-GGUF\nQ4_K_M", ...read });
  assert.equal(cachedStagedDims("org/m-GGUF\nQ4_K_M")?.layerCount, 28);
  assert.equal(cachedStagedDims(null), null);
});

test("an undownloaded GGUF's empty answer is not cached, so it re-probes after download", () => {
  const none = { contextLength: null, layerCount: null, moeLayerCount: null };
  rememberStagedDims({ key: "unknown", ...none, isDiffusion: false, diffusionUnknown: true });
  rememberStagedDims({ key: "by-name", ...none, isDiffusion: true, diffusionUnknown: false });
  assert.equal(cachedStagedDims("unknown"), null);
  assert.equal(cachedStagedDims("by-name"), null);
  assert.equal(stagedDimsLastInconclusive("unknown"), true);
  assert.equal(stagedDimsLastInconclusive("never-probed"), false);
  rememberStagedDims({ key: "unknown", ...read });
  assert.equal(cachedStagedDims("unknown")?.layerCount, 28, "the post-download header replaces it");
  assert.equal(stagedDimsLastInconclusive("unknown"), false);
});

test("the cache keeps the most recent 32 headers", () => {
  for (let i = 0; i < 40; i++) rememberStagedDims({ key: `k${i}`, ...read });
  assert.equal(cachedStagedDims("k7"), null);
  assert.equal(cachedStagedDims("k8")?.key, "k8");
});
