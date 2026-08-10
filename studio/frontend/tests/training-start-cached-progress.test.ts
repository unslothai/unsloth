// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The training-start overlay showed cached resources as "Downloading -- 99%" with no
// download running (#7858): the backend caps progress at 99% until a Studio manifest
// verifies the snapshot, and cache entries made outside Studio never have one. These
// pin the distinction the overlay now draws -- finalized bytes decide whether anything
// is still arriving -- including the resumed-download case the 99% cap exists for.

import assert from "node:assert/strict";
import test from "node:test";

import {
  type DownloadState,
  coerceCachedStateReady,
} from "../src/features/studio/download-state.ts";

const MB = 1e6;
const GB = 1e9;

function state(over: Partial<DownloadState> = {}): DownloadState {
  return {
    downloadedBytes: 0,
    completedBytes: 0,
    totalBytes: 0,
    percent: 0,
    cachePath: "/home/u/.cache/huggingface/hub/datasets--unsloth--alpaca-cleaned",
    ...over,
  };
}

test("a cache entry with no manifest reads as ready, not as a transfer", () => {
  // Observed with unsloth/alpaca-cleaned: 42.3 MB of 42.3 MB, no worker running.
  const coerced = coerceCachedStateReady(
    state({
      downloadedBytes: 42.3 * MB,
      completedBytes: 42.3 * MB,
      totalBytes: 42.3 * MB,
      percent: 99,
    }),
  );
  assert.equal(coerced.percent, 100);
});

test("a resumed download still reports progress while blobs are incomplete", () => {
  // The case the backend's 99% cap protects: finalized bytes sit just under the
  // total while the remaining `.incomplete` blob is still growing.
  const resuming = state({
    downloadedBytes: 20 * GB,
    completedBytes: 19.9 * GB,
    totalBytes: 20 * GB,
    percent: 99,
  });
  assert.deepEqual(coerceCachedStateReady(resuming), resuming);
});

test("an ordinary in-flight download is left alone", () => {
  const downloading = state({
    downloadedBytes: 5 * GB,
    completedBytes: 4.8 * GB,
    totalBytes: 20 * GB,
    percent: 25,
  });
  assert.deepEqual(coerceCachedStateReady(downloading), downloading);
});

test("a resource with no cache path is never coerced", () => {
  const uncached = state({
    downloadedBytes: 42.3 * MB,
    completedBytes: 42.3 * MB,
    totalBytes: 42.3 * MB,
    percent: 99,
    cachePath: null,
  });
  assert.deepEqual(coerceCachedStateReady(uncached), uncached);
});

test("a cached entry of unknown size still settles instead of hanging", () => {
  const unsized = coerceCachedStateReady(
    state({ downloadedBytes: 0, completedBytes: 0, totalBytes: 0, percent: 0 }),
  );
  assert.equal(unsized.percent, 100);
});
