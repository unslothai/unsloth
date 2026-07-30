// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { toTrainingDownloadState } from "../src/features/studio/training-download-progress.ts";

const overlaySource = readFileSync(
  new URL(
    "../src/features/studio/training-start-overlay.tsx",
    import.meta.url,
  ),
  "utf8",
);
const chatApiSource = readFileSync(
  new URL("../src/features/chat/api/chat-api.ts", import.meta.url),
  "utf8",
);
const runtimeStoreSource = readFileSync(
  new URL(
    "../src/features/training/stores/training-runtime-store.ts",
    import.meta.url,
  ),
  "utf8",
);

const TEN_GIB = Math.round(9.57 * 1024 ** 3);

test("a cached path without disk verification stays in progress", () => {
  const state = toTrainingDownloadState({
    downloaded_bytes: 0,
    expected_bytes: TEN_GIB,
    progress: 0,
    complete_on_disk: false,
    cache_path: "~/.cache/huggingface/hub/models--unsloth--gemma-4-E2B-it",
  });

  assert.equal(state.downloadedBytes, 0);
  assert.equal(state.totalBytes, TEN_GIB);
  assert.equal(state.percent, 0);
  assert.equal(state.completeOnDisk, false);
});

test("byte totals cannot round an unverified download up to ready", () => {
  const state = toTrainingDownloadState({
    downloaded_bytes: TEN_GIB,
    expected_bytes: TEN_GIB,
    progress: 1,
    complete_on_disk: false,
    cache_path: "~/.cache/huggingface/hub/models--unsloth--gemma-4-E2B-it",
  });

  assert.equal(state.percent, 99);
  assert.equal(state.completeOnDisk, false);
});

test("manifest-backed disk completion is the only 100% state", () => {
  const state = toTrainingDownloadState({
    downloaded_bytes: TEN_GIB,
    expected_bytes: TEN_GIB,
    progress: 1,
    complete_on_disk: true,
    cache_path: "~/.cache/huggingface/hub/models--unsloth--gemma-4-E2B-it",
  });

  assert.equal(state.percent, 100);
  assert.equal(state.completeOnDisk, true);
});

test("training progress never promotes a cache path to ready", () => {
  assert.doesNotMatch(
    overlaySource,
    /coerceCachedStateReady/,
    "a cache directory can exist while Hugging Face is still fetching the resolved model",
  );
});

test("training readiness follows manifest-backed disk completion", () => {
  assert.match(
    chatApiSource,
    /complete_on_disk:\s*boolean/,
    "the training progress API type must expose backend disk verification",
  );
  assert.match(
    overlaySource,
    /prog\.complete_on_disk\s*===\s*true/,
    "the overlay must consume manifest-backed completion",
  );
  assert.match(
    overlaySource,
    /completeOnDisk/,
    "the rendered state must retain disk completion separately from rounded percentage",
  );
});

test("training polls the model repo resolved by the worker", () => {
  assert.match(
    runtimeStoreSource,
    /payload\.details\?\.model_download_repo_id/,
    "the runtime store must retain the worker's resolved download repo",
  );
  assert.match(
    overlaySource,
    /modelDownloadRepoId\s*\?\?/,
    "the overlay must prefer the resolved repo over the configured repo",
  );
});
