// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The training-start overlay showed resources as "Downloading -- 99%" with no download
// running (#7858). The backend caps progress at 0.99 until it verifies the snapshot, and
// verification compares against `expected_bytes`, which counts every file in the repo while a
// training run fetches a subset -- so the cap never lifts. These pin the settling rule that
// replaces it, and the readings it must refuse to settle.

import assert from "node:assert/strict";
import test from "node:test";

import {
  type DownloadProgressReading,
  type DownloadState,
  EMPTY_DOWNLOAD_STATE,
  coerceCachedStateReady,
  downloadStateFromProgress,
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
    completeOnDisk: false,
    settled: false,
    ...over,
  };
}

/** Feed the same reading twice, as the 1.5s poll would when nothing is moving. */
function pollTwice(reading: DownloadProgressReading): DownloadState {
  const first = downloadStateFromProgress(reading, EMPTY_DOWNLOAD_STATE);
  return downloadStateFromProgress(reading, first);
}

test("a verified snapshot settles on the first reading", () => {
  const verified = downloadStateFromProgress({
    downloaded_bytes: 1.51 * GB,
    completed_bytes: 1.51 * GB,
    expected_bytes: 1.51 * GB,
    progress: 1,
    complete_on_disk: true,
    cache_path: "/home/u/.cache/huggingface/hub/datasets--unsloth--LaTeX_OCR",
  });
  assert.equal(verified.percent, 100);
  assert.equal(verified.settled, true);
});

test("a subset fetch settles once its bytes stop moving", () => {
  // Qwen3.5-0.8B-Base: expected counts README.md, LICENSE and .gitattributes, which the
  // trainer never fetches, so completed sits 16,640 bytes short and progress is pinned at
  // the 0.99 cap for a model that is entirely present.
  const reading: DownloadProgressReading = {
    downloaded_bytes: 1_769_897_109,
    completed_bytes: 1_769_897_109,
    expected_bytes: 1_769_913_749,
    progress: 0.99,
    complete_on_disk: false,
    cache_path: "/home/u/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base",
  };
  assert.equal(downloadStateFromProgress(reading).settled, false);
  assert.equal(pollTwice(reading).settled, true);
});

test("a settled subset fetch reports the bytes it actually holds", () => {
  // Not `expected_bytes`: OpenThoughts-1k-sample ships a second config load_dataset never
  // wants, so settling at the expected total would claim 28.1 MB for a 14.0 MB fetch.
  const settled = coerceCachedStateReady(
    pollTwice({
      downloaded_bytes: 14_002_749,
      completed_bytes: 14_002_749,
      expected_bytes: 28_059_770,
      progress: 0.499,
      complete_on_disk: false,
      cache_path: "/home/u/.cache/huggingface/hub/datasets--ryanmarten--OpenThoughts-1k-sample",
    }),
  );
  assert.equal(settled.percent, 100);
  assert.equal(settled.totalBytes, 14_002_749);
});

test("one quiet reading is not enough, because blobs finalize between files", () => {
  // Mid-download, huggingface_hub has just linked a blob and not yet opened the next, so
  // downloaded == completed for this single tick.
  const betweenFiles = downloadStateFromProgress({
    downloaded_bytes: 5 * GB,
    completed_bytes: 5 * GB,
    expected_bytes: 20 * GB,
    progress: 0.25,
    complete_on_disk: false,
    cache_path: "/home/u/.cache/huggingface/hub/models--unsloth--gpt-oss-120b",
  });
  assert.equal(betweenFiles.settled, false);
  assert.equal(coerceCachedStateReady(betweenFiles).percent, 25);
});

test("bytes in flight never settle, however long they sit", () => {
  // A resumed download: finalized bytes near the total with an `.incomplete` blob growing.
  const resuming: DownloadProgressReading = {
    downloaded_bytes: 20 * GB,
    completed_bytes: 19.9 * GB,
    expected_bytes: 20 * GB,
    progress: 0.99,
    complete_on_disk: false,
    cache_path: "/home/u/.cache/huggingface/hub/models--unsloth--gpt-oss-120b",
  };
  assert.equal(pollTwice(resuming).settled, false);
});

test("a growing download never settles", () => {
  const first = downloadStateFromProgress({
    downloaded_bytes: 5 * GB,
    completed_bytes: 5 * GB,
    expected_bytes: 20 * GB,
    progress: 0.25,
    complete_on_disk: false,
    cache_path: "/home/u/.cache/huggingface/hub/models--unsloth--gpt-oss-120b",
  });
  const second = downloadStateFromProgress(
    {
      downloaded_bytes: 6 * GB,
      completed_bytes: 6 * GB,
      expected_bytes: 20 * GB,
      progress: 0.3,
      complete_on_disk: false,
      cache_path: "/home/u/.cache/huggingface/hub/models--unsloth--gpt-oss-120b",
    },
    first,
  );
  assert.equal(second.settled, false);
});

test("a transfer that stalled and resumed stops reading as settled", () => {
  // A slow multi-file download can go quiet for two polls between files. Latching settlement
  // would then show Ready, with no rate or progress, for the rest of the transfer.
  const reading: DownloadProgressReading = {
    downloaded_bytes: 5 * GB,
    completed_bytes: 5 * GB,
    expected_bytes: 20 * GB,
    progress: 0.25,
    complete_on_disk: false,
    cache_path: "/home/u/.cache/huggingface/hub/models--unsloth--gpt-oss-120b",
  };
  const stalled = pollTwice(reading);
  assert.equal(stalled.settled, true);

  const resumed = downloadStateFromProgress(
    { ...reading, downloaded_bytes: 5.2 * GB, completed_bytes: 5 * GB, progress: 0.26 },
    stalled,
  );
  assert.equal(resumed.settled, false);
  assert.equal(coerceCachedStateReady(resumed).percent, 26);
});

test("a cache dir with bytes still expected is not ready", () => {
  // The repo dir exists from an earlier attempt, but nothing has arrived for this one.
  const empty = state({ downloadedBytes: 0, completedBytes: 0, totalBytes: 20 * GB });
  assert.deepEqual(coerceCachedStateReady(empty), empty);
});

test("a resource with no cache path is never coerced", () => {
  const uncached = state({
    downloadedBytes: 42.3 * MB,
    completedBytes: 42.3 * MB,
    totalBytes: 42.3 * MB,
    percent: 99,
    cachePath: null,
    settled: true,
  });
  assert.deepEqual(coerceCachedStateReady(uncached), uncached);
});

test("a cached entry of unknown size still settles instead of hanging", () => {
  const unsized = coerceCachedStateReady(state());
  assert.equal(unsized.percent, 100);
  assert.equal(unsized.settled, true);
});
