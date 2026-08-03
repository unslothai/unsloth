// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  coerceCachedStateReady,
  downloadStateFromProgress,
  isDownloadComplete,
  parsePreparationProgress,
  resolvePreparationMessage,
  shouldShowPreparationStatus,
} from "../src/features/studio/training-start-progress.ts";

test("verified snapshots complete the row even while transport progress is capped", () => {
  const state = downloadStateFromProgress({
    downloaded_bytes: 1_510_000_000,
    expected_bytes: 1_510_000_000,
    progress: 0.99,
    complete_on_disk: true,
    cache_path: "/tmp/dataset",
  });

  assert.equal(state.percent, 100);
  assert.equal(state.completeOnDisk, true);
  assert.equal(isDownloadComplete(state), true);
});

test("incomplete snapshots keep the determinate download state", () => {
  const state = downloadStateFromProgress({
    downloaded_bytes: 990,
    expected_bytes: 1000,
    progress: 0.99,
    complete_on_disk: false,
  });

  assert.equal(state.percent, 99);
  assert.equal(state.completeOnDisk, false);
  assert.equal(isDownloadComplete(state), false);
});

test("rounded progress cannot mark an unverified snapshot complete", () => {
  const state = downloadStateFromProgress({
    downloaded_bytes: 995,
    expected_bytes: 1000,
    progress: 0.995,
    complete_on_disk: false,
  });

  assert.equal(state.percent, 100);
  assert.equal(isDownloadComplete(state), false);
});

test("completed cache state stays complete after the download phase", () => {
  const state = coerceCachedStateReady({
    downloadedBytes: 1000,
    totalBytes: 1000,
    percent: 99,
    cachePath: "/tmp/model",
    completeOnDisk: true,
  });

  assert.deepEqual(state, {
    downloadedBytes: 1000,
    totalBytes: 1000,
    percent: 100,
    cachePath: "/tmp/model",
    completeOnDisk: true,
  });
});

test("equal byte counts settle a capped row once preparation starts", () => {
  const state = coerceCachedStateReady({
    downloadedBytes: 1000,
    totalBytes: 1000,
    percent: 99,
    cachePath: "/tmp/dataset",
    completeOnDisk: false,
  });

  assert.equal(state.completeOnDisk, true);
  assert.equal(state.percent, 100);
});

test("preparation settles a cached row even when byte totals are not exact", () => {
  const state = coerceCachedStateReady(
    {
      downloadedBytes: 999,
      totalBytes: 1000,
      percent: 99,
      cachePath: "/tmp/dataset",
      completeOnDisk: false,
    },
    true,
  );

  assert.equal(state.completeOnDisk, true);
  assert.equal(state.percent, 100);
});

test("partial cache data does not get promoted to a completed row", () => {
  const state = coerceCachedStateReady({
    downloadedBytes: 900,
    totalBytes: 1000,
    percent: 90,
    cachePath: "/tmp/dataset",
    completeOnDisk: false,
  });

  assert.equal(state.completeOnDisk, false);
  assert.equal(state.percent, 90);
});

test("preparation status replaces stale download messaging", () => {
  assert.equal(
    shouldShowPreparationStatus("downloading_model", 0, true),
    false,
  );
  assert.equal(shouldShowPreparationStatus("configuring", 0, false), true);
  assert.equal(shouldShowPreparationStatus("training", 0, false), true);
  assert.equal(shouldShowPreparationStatus("training", 1, false), false);
  assert.equal(
    resolvePreparationMessage("Downloading model...", "Preparing"),
    "Preparing",
  );
  assert.equal(
    resolvePreparationMessage('Tokenizing ["text"] 15%', "Preparing"),
    'Tokenizing ["text"] 15%',
  );
});

test("quantitative preparation messages produce determinate progress", () => {
  assert.deepEqual(
    parsePreparationProgress(
      'Tokenizing ["text"] (num_proc=4) 15% (32,000/207,865)',
      "Preparing",
    ),
    {
      title: 'Tokenizing ["text"]',
      detail: "15% (32,000/207,865)",
      percent: (32000 / 207865) * 100,
    },
  );
  assert.deepEqual(
    parsePreparationProgress(
      "Filter (num_proc=4) 7% (16,000/207,865)",
      "Preparing",
    ),
    {
      title: "Filter",
      detail: "8% (16,000/207,865)",
      percent: (16000 / 207865) * 100,
    },
  );
});

test("non-quantitative preparation messages stay indeterminate", () => {
  assert.deepEqual(parsePreparationProgress("Loading model...", "Preparing"), {
    title: "Loading model",
    detail: null,
    percent: null,
  });
  assert.deepEqual(
    parsePreparationProgress("Formatting dataset (207,865 rows)...", "Preparing"),
    {
      title: "Formatting dataset (207,865 rows)",
      detail: null,
      percent: null,
    },
  );
});

test("invalid quantitative messages do not render a false determinate bar", () => {
  assert.deepEqual(
    parsePreparationProgress("Filter 100% (10/0)", "Preparing"),
    { title: "Filter", detail: null, percent: null },
  );
  assert.deepEqual(
    parsePreparationProgress("Filter 100% (11/10)", "Preparing"),
    { title: "Filter", detail: null, percent: null },
  );
});
