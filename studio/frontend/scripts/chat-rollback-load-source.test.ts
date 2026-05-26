// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  buildActiveModelLoadSource,
  resolveRollbackLoadOptions,
} from "../src/features/chat/model-runtime/rollback-load-source.ts";

test("cached Hub rollback preserves local-only load options", () => {
  const previousLoadSource = buildActiveModelLoadSource({
    source: "hub",
    isDownloaded: true,
    isPartial: false,
    preferLocalCache: true,
    localPath: "/cache/models--Org--Model",
    modelFormat: "safetensors",
  });

  const options = resolveRollbackLoadOptions({
    previousCheckpoint: "Org/Model",
    previousVariant: null,
    previousIsLora: false,
    previousLoadSource,
  });

  assert.equal(options.localFilesOnly, true);
  assert.equal(options.localPath, "/cache/models--Org--Model");
  assert.equal(options.modelFormat, "safetensors");
});

test("native local rollback remains local-only by path token", () => {
  const previousLoadSource = buildActiveModelLoadSource({
    source: "local",
    isDownloaded: true,
    isPartial: false,
    localPath: "/models/model.gguf",
    modelFormat: "gguf",
    nativePathToken: "native-token",
  });

  const options = resolveRollbackLoadOptions({
    previousCheckpoint: "Local GGUF model",
    previousVariant: null,
    previousIsLora: false,
    previousActiveNativePathToken: "native-token",
    previousLoadSource,
  });

  assert.equal(options.localFilesOnly, true);
  assert.equal(options.localPath, "/models/model.gguf");
  assert.equal(options.modelFormat, "gguf");
});

test("remote rollback does not force local-only loading", () => {
  const previousLoadSource = buildActiveModelLoadSource({
    source: "hub",
    isDownloaded: false,
    isPartial: false,
    preferLocalCache: false,
    localPath: null,
    modelFormat: "safetensors",
  });

  const options = resolveRollbackLoadOptions({
    previousCheckpoint: "Org/Remote",
    previousVariant: null,
    previousIsLora: false,
    previousLoadSource,
  });

  assert.equal(options.localFilesOnly, false);
  assert.equal(options.localPath, null);
  assert.equal(options.modelFormat, "safetensors");
});
