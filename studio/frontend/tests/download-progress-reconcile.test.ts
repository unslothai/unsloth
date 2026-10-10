// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// An all-zero reading means unmeasured, not empty; only zeros are ignored because bytes can
// legitimately fall within one generation (XET to HTTP reclaim, partial purge).

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
    etaSeconds: 0,
    error: null,
    startedAt: 1,
  };
  return { ...base, ...overrides };
}

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
  assert.equal(resolved.expected, EXPECTED);
  assert.equal(resolved.madeProgress, false);
});

test("a real drop still lands, so a fallback retry tracks its own bytes", () => {
  const resolved = resolveProgressUpdate(
    job(),
    emptyReading({ downloaded_bytes: GB, completed_bytes: 0, progress: 0.03 }),
  );

  assert.equal(resolved.downloadedBytes, GB);
  assert.equal(resolved.madeProgress, false);
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

test("an unmeasured scan keeps the idle grace from finalizing the job", () => {
  const resolved = resolveProgressUpdate(
    job(),
    emptyReading({ cache_measured: false }),
  );

  assert.equal(resolved.madeProgress, true);
  assert.equal(resolved.downloadedBytes, EXPECTED);
});

test("resetMonotonic publishes the zero for a new generation", () => {
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

test("a fresh complete_on_disk cannot promote a HELD byte count to a completion", () => {
  // Holding last poll's completed_bytes beside this poll's complete_on_disk must not retire the card.
  const resolved = resolveProgressUpdate(
    job(),
    emptyReading({ complete_on_disk: true }),
  );

  assert.equal(resolved.completedBytes, EXPECTED);
  assert.equal(resolved.completeOnDisk, false);
  assert.equal(
    hasObservedExpectedBytes(
      job({
        completedBytes: resolved.completedBytes,
        expectedBytes: resolved.expected,
        completeOnDisk: resolved.completeOnDisk,
      }),
    ),
    false,
  );
});

test("a measured completion in the same reading still settles", () => {
  const resolved = resolveProgressUpdate(
    job({ downloadedBytes: 32 * GB, completedBytes: 32 * GB, fraction: 0.97 }),
    emptyReading({
      downloaded_bytes: EXPECTED,
      completed_bytes: EXPECTED,
      progress: 1,
      complete_on_disk: true,
    }),
  );

  assert.equal(resolved.completeOnDisk, true);
  assert.equal(
    hasObservedExpectedBytes(
      job({
        completedBytes: resolved.completedBytes,
        expectedBytes: resolved.expected,
        completeOnDisk: resolved.completeOnDisk,
      }),
    ),
    true,
  );
});

test("a reading with no total keeps the card's own total, not zero", () => {
  for (const resetMonotonic of [false, true]) {
    const resolved = resolveProgressUpdate(
      job({ downloadedBytes: GB, completedBytes: GB }),
      emptyReading({ downloaded_bytes: 2 * GB, expected_bytes: 0 }),
      { resetMonotonic },
    );
    assert.equal(resolved.expected, EXPECTED, `resetMonotonic=${resetMonotonic}`);
  }
});

test("a snapshot total is a high-water mark, a variant total is backend-owned", () => {
  const lower = 20 * GB;
  const variant = resolveProgressUpdate(job(), emptyReading({ expected_bytes: lower }));
  assert.equal(variant.expected, lower);

  const snapshot = resolveProgressUpdate(
    job({ variant: null, key: "model:org/model" }),
    emptyReading({ expected_bytes: lower }),
  );
  assert.equal(snapshot.expected, EXPECTED);
});

test("the bar is capped below full until the backend verifies completion", () => {
  const resolved = resolveProgressUpdate(
    job({ fraction: 0 }),
    emptyReading({ downloaded_bytes: EXPECTED, progress: 1 }),
  );

  assert.equal(resolved.fraction, 0.99);
});

test("a non-finite reading is not a measurement", () => {
  // ProgressLike is unvalidated JSON; NaN would reach the bar as "NaN%".
  for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, null, undefined]) {
    const resolved = resolveProgressUpdate(
      job({ downloadedBytes: 4 * GB, completedBytes: 4 * GB, fraction: 0.12 }),
      emptyReading({
        downloaded_bytes: bad as unknown as number,
        completed_bytes: bad as unknown as number,
        expected_bytes: bad as unknown as number,
        progress: bad as unknown as number,
      }),
      { resetMonotonic: true },
    );

    assert.ok(Number.isFinite(resolved.downloadedBytes), `downloaded ${String(bad)}`);
    assert.ok(Number.isFinite(resolved.completedBytes), `completed ${String(bad)}`);
    assert.ok(Number.isFinite(resolved.expected), `expected ${String(bad)}`);
    assert.ok(Number.isFinite(resolved.fraction), `fraction ${String(bad)}`);
    assert.equal(resolved.expected, EXPECTED);
  }
});

test("a new generation resets the fraction, not only the bytes", () => {
  const resolved = resolveProgressUpdate(job({ fraction: 0.99 }), emptyReading(), {
    resetMonotonic: true,
  });

  assert.equal(resolved.downloadedBytes, 0);
  assert.equal(resolved.fraction, 0);

  const held = resolveProgressUpdate(job({ fraction: 0.99 }), emptyReading());
  assert.equal(held.fraction, 0.99);
});

test("a held transfer reading is reported as held, not as a measurement", () => {
  const held = resolveProgressUpdate(
    job({ downloadedBytes: 3 * GB, completedBytes: 3 * GB }),
    emptyReading({ expected_bytes: GB / 2 }),
  );

  assert.equal(held.measuredTransfer, false);
  assert.equal(held.downloadedBytes, 3 * GB);
  assert.equal(held.expected, GB / 2);

  const measured = resolveProgressUpdate(
    job({ downloadedBytes: 3 * GB, completedBytes: 3 * GB }),
    emptyReading({ downloaded_bytes: GB / 10, expected_bytes: GB / 2 }),
  );

  assert.equal(measured.measuredTransfer, true);
  assert.equal(measured.downloadedBytes, GB / 10);

  const reset = resolveProgressUpdate(job({ downloadedBytes: 3 * GB }), emptyReading(), {
    resetMonotonic: true,
  });

  assert.equal(reset.measuredTransfer, true);
  assert.equal(reset.downloadedBytes, 0);
});
