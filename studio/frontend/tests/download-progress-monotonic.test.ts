// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The byte counters on a download card are high-water marks, the way the
// fraction already was. Only the GGUF *total* is backend-owned: the backend
// recomputes the counters from the shared per-repo blobs/ dir, so one poll that
// cannot resolve a variant's expected files reports zero, and letting that
// through rewrote a finished card to "0 B of 33 GB" for good. Completion needs
// completedBytes to reach expectedBytes, so the job then never finalized and
// kept its Retry/Resume controls on a download that had succeeded.

import assert from "node:assert/strict";
import test from "node:test";

import type {
  ManagedDownload,
  ProgressLike,
} from "../src/features/hub/download-manager/download-manager-types.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { hasObservedExpectedBytes, resolveProgressUpdate } = await import(
  "../src/features/hub/download-manager/progress-reconcile.ts"
);

const GB = 1024 ** 3;
const EXPECTED = 33 * GB;

function job(overrides: Partial<ManagedDownload> = {}): ManagedDownload {
  return {
    key: "model:org/model-gguf::Q4_K_M",
    kind: "model",
    repoId: "org/model-GGUF",
    variant: "Q4_K_M",
    state: "running",
    downloadedBytes: EXPECTED,
    completedBytes: EXPECTED,
    expectedBytes: EXPECTED,
    completeOnDisk: false,
    fraction: 0.99,
    bytesPerSec: 0,
    error: null,
    startedAt: 1,
    ...overrides,
  } as ManagedDownload;
}

function progress(overrides: Partial<ProgressLike> = {}): ProgressLike {
  return {
    downloaded_bytes: 0,
    completed_bytes: 0,
    expected_bytes: EXPECTED,
    progress: 0,
    complete_on_disk: false,
    ...overrides,
  } as ProgressLike;
}

test("a zero-byte poll cannot rewrite a variant card to 0 B", () => {
  const resolved = resolveProgressUpdate(job(), progress());

  assert.equal(resolved.downloadedBytes, EXPECTED);
  assert.equal(resolved.completedBytes, EXPECTED);
  // The total stays backend-owned, so a variant that was re-resolved smaller
  // still reports the new total.
  assert.equal(resolved.expected, EXPECTED);
  assert.equal(resolved.madeProgress, false);
});

test("a variant total may shrink while its byte counters hold", () => {
  const resolved = resolveProgressUpdate(
    job(),
    progress({ expected_bytes: 30 * GB, downloaded_bytes: 0 }),
  );

  assert.equal(resolved.expected, 30 * GB);
  assert.equal(resolved.downloadedBytes, EXPECTED);
});

test("the high-water mark still lets real progress through", () => {
  const resolved = resolveProgressUpdate(
    job({ downloadedBytes: 2 * GB, completedBytes: GB, fraction: 0.06 }),
    progress({
      downloaded_bytes: 5 * GB,
      completed_bytes: 4 * GB,
      progress: 0.15,
    }),
  );

  assert.equal(resolved.downloadedBytes, 5 * GB);
  assert.equal(resolved.completedBytes, 4 * GB);
  assert.equal(resolved.madeProgress, true);
});

test("resetMonotonic still drops the mark for a new generation", () => {
  // An XET redownload, a restart or a re-adoption starts the bytes over, and
  // the escape hatch has to keep working or the bar would sit at 100% through
  // the whole second run.
  const resolved = resolveProgressUpdate(job(), progress(), {
    resetMonotonic: true,
  });

  assert.equal(resolved.downloadedBytes, 0);
  assert.equal(resolved.completedBytes, 0);
  assert.equal(resolved.madeProgress, true);
});

test("a snapshot job keeps the monotonic counters it always had", () => {
  const resolved = resolveProgressUpdate(
    job({ variant: null, key: "model:org/model" }),
    progress({ downloaded_bytes: 0, completed_bytes: 0 }),
  );

  assert.equal(resolved.downloadedBytes, EXPECTED);
  assert.equal(resolved.completedBytes, EXPECTED);
});

test("a finished variant finalizes instead of holding Retry/Resume", () => {
  // The whole point of the floor: the backend confirms the snapshot on disk on
  // a poll whose byte counters came back empty. hasObservedExpectedBytes is
  // what moves the job out of ACTIVE_STATES, and it reads completedBytes.
  const resolved = resolveProgressUpdate(
    job(),
    progress({ complete_on_disk: true }),
  );
  const settled = {
    ...job(),
    downloadedBytes: resolved.downloadedBytes,
    completedBytes: resolved.completedBytes,
    expectedBytes: resolved.expected,
    completeOnDisk: resolved.completeOnDisk,
  } as ManagedDownload;

  assert.equal(hasObservedExpectedBytes(settled), true);
});

test("negative byte counts from a malformed reading are floored at zero", () => {
  const resolved = resolveProgressUpdate(
    job({ downloadedBytes: 0, completedBytes: 0 }),
    progress({ downloaded_bytes: -1, completed_bytes: -1 }),
  );

  assert.equal(resolved.downloadedBytes, 0);
  assert.equal(resolved.completedBytes, 0);
});
