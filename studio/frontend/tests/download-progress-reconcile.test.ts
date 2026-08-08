// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A download card holds its last reading through a poll that measured nothing,
// instead of publishing "I could not measure this" as a measurement. The
// backend recomputes both byte counters from the shared per-repo blobs/ dir
// every poll and returns an all-zero reading when it cannot resolve a variant's
// expected files; that failure is negatively cached, so every poll for the
// whole TTL says the same thing and a finished card read "0 B of 33 GB".
//
// The rule is NOT the high-water mark the fraction uses. Bytes have legitimate
// reasons to fall inside one generation -- an XET run falling back to HTTP
// re-claims with the same generation and a recomputed completed_baseline_bytes,
// and an XET resume purges the partial -- so only a zero is ignored.

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
  // Keyed the way jobKeyOf spells it, so the fixture stays recognisable.
  const base: ManagedDownload = {
    key: "model:org/model-gguf#q4_k_m",
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
  };
  return { ...base, ...overrides };
}

/** The backend's `_empty_progress` reply: it found nothing to measure. */
function emptyReading(overrides: Partial<ProgressLike> = {}): ProgressLike {
  return {
    downloaded_bytes: 0,
    completed_bytes: 0,
    expected_bytes: EXPECTED,
    progress: 0,
    complete_on_disk: false,
    ...overrides,
  };
}

test("an all-zero reading cannot rewrite a variant card to 0 B", () => {
  const resolved = resolveProgressUpdate(job(), emptyReading());

  assert.equal(resolved.downloadedBytes, EXPECTED);
  assert.equal(resolved.completedBytes, EXPECTED);
  // The total stays backend-owned, so a re-resolved variant size still lands.
  assert.equal(resolved.expected, EXPECTED);
  assert.equal(resolved.madeProgress, false);
});

test("a real drop still lands, so a fallback retry tracks its own bytes", () => {
  // An XET run falling back to HTTP re-claims with the SAME generation and a
  // freshly computed completed_baseline_bytes, so the reported figures
  // legitimately restart low. A high-water mark would pin this card near full
  // for the whole retry and leave the rate estimator with a flat series.
  const resolved = resolveProgressUpdate(
    job(),
    emptyReading({ downloaded_bytes: GB, completed_bytes: 0, progress: 0.03 }),
  );

  assert.equal(resolved.downloadedBytes, GB);
  assert.equal(resolved.madeProgress, false); // lower is not progress
});

test("an XET resume that purged the partial reports what is left", () => {
  const resolved = resolveProgressUpdate(
    job({ downloadedBytes: 20 * GB, completedBytes: 12 * GB }),
    emptyReading({ downloaded_bytes: 12 * GB, completed_bytes: 12 * GB }),
  );

  assert.equal(resolved.downloadedBytes, 12 * GB);
  assert.equal(resolved.completedBytes, 12 * GB);
});

test("the held reading still lets real progress through", () => {
  const resolved = resolveProgressUpdate(
    job({ downloadedBytes: 2 * GB, completedBytes: GB, fraction: 0.06 }),
    emptyReading({
      downloaded_bytes: 5 * GB,
      completed_bytes: 4 * GB,
      progress: 0.15,
    }),
  );

  assert.equal(resolved.downloadedBytes, 5 * GB);
  assert.equal(resolved.completedBytes, 4 * GB);
  assert.equal(resolved.madeProgress, true);
});

test("resetMonotonic publishes the zero for a new generation", () => {
  // The escape hatch has to keep working, or a restart would show the previous
  // run's bytes until the new one passed them.
  const resolved = resolveProgressUpdate(job(), emptyReading(), {
    resetMonotonic: true,
  });

  assert.equal(resolved.downloadedBytes, 0);
  assert.equal(resolved.completedBytes, 0);
  assert.equal(resolved.madeProgress, true);
});

test("a snapshot job holds its reading the same way a variant does", () => {
  const resolved = resolveProgressUpdate(
    job({ variant: null, key: "model:org/model" }),
    emptyReading(),
  );

  assert.equal(resolved.downloadedBytes, EXPECTED);
  assert.equal(resolved.completedBytes, EXPECTED);
});

test("a held reading cannot manufacture a completion on its own", () => {
  // completeOnDisk is never held over, and the backend only sets it on a
  // reading whose own completed_bytes already cleared the bar, so the two
  // halves of hasObservedExpectedBytes can never come from different polls.
  const held = resolveProgressUpdate(job(), emptyReading());
  const afterEmptyPoll = job({
    downloadedBytes: held.downloadedBytes,
    completedBytes: held.completedBytes,
    completeOnDisk: held.completeOnDisk,
  });
  assert.equal(hasObservedExpectedBytes(afterEmptyPoll), false);

  const confirmed = resolveProgressUpdate(
    afterEmptyPoll,
    emptyReading({
      downloaded_bytes: EXPECTED,
      completed_bytes: EXPECTED,
      progress: 1,
      complete_on_disk: true,
    }),
  );

  assert.equal(
    hasObservedExpectedBytes(
      job({
        downloadedBytes: confirmed.downloadedBytes,
        completedBytes: confirmed.completedBytes,
        expectedBytes: confirmed.expected,
        completeOnDisk: confirmed.completeOnDisk,
      }),
    ),
    true,
  );
});

test("a negative byte count reads as no measurement, not as a drop", () => {
  const resolved = resolveProgressUpdate(
    job({ downloadedBytes: 4 * GB, completedBytes: 4 * GB }),
    emptyReading({ downloaded_bytes: -1, completed_bytes: -1 }),
  );

  assert.equal(resolved.downloadedBytes, 4 * GB);
  assert.equal(resolved.completedBytes, 4 * GB);
});
