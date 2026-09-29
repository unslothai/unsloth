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
    etaSeconds: 0,
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

test("an unmeasured scan keeps the idle grace from finalizing the job", () => {
  // cache_measured false is the backend saying it could not read the cache at all, and the
  // reply is then all zeroes. The adopt probe already refuses to retire on that shape; without
  // the same rule here the protection lasted only until the adopted poll loop's grace ran out,
  // and a download whose cache was merely unreadable was finalized as gone.
  const resolved = resolveProgressUpdate(
    job(),
    emptyReading({ cache_measured: false }),
  );

  assert.equal(resolved.madeProgress, true);
  // Still not a measurement: the card holds its figures.
  assert.equal(resolved.downloadedBytes, EXPECTED);
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

test("a fresh complete_on_disk cannot promote a HELD byte count to a completion", () => {
  // The dangerous direction. Holding the last reading over an unmeasured poll
  // made a pairing reachable that publishing the zero never was: LAST poll's
  // completed_bytes beside THIS poll's complete_on_disk. Neither reading showed
  // a completion; together they retire the card and drop the download. The
  // backend does not currently emit this shape, but nothing on this side checks
  // that, and the cost of being wrong is a download silently declared finished.
  const resolved = resolveProgressUpdate(
    job(),
    emptyReading({ complete_on_disk: true }),
  );

  assert.equal(resolved.completedBytes, EXPECTED); // still held, for the card
  assert.equal(resolved.completeOnDisk, false); // but not asserted as finished
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
  // The guard above must not cost the real case: when the poll measured the
  // counter it is making a claim about, the completion lands as before.
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
  // The "of 33 GB" half of the reported card. expected_bytes is the caller's
  // catalog hint echoed back, and a backend that could not size the repo
  // returns 0; publishing that would blank the denominator on a card that
  // already knew it. Untested before: replacing the whole expected branch with
  // `reported` left every other assertion in this file green.
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
  // The two halves of the expected branch differ, and only one of them is a
  // floor. A variant's total is re-derived every poll and legitimately shrinks
  // (a re-resolved file set); a snapshot's must not dip on a jittery reading.
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
  // Untested before: removing the cap left all eight original tests green. The
  // cap is what stops a card sitting at 100% while files are still landing.
  const resolved = resolveProgressUpdate(
    job({ fraction: 0 }),
    emptyReading({ downloaded_bytes: EXPECTED, progress: 1 }),
  );

  assert.equal(resolved.fraction, 0.99);
});

test("a non-finite reading is not a measurement", () => {
  // ProgressLike is cast straight off raw JSON with no runtime validation, and
  // a NaN survives every comparison here to reach the progress bar as a width
  // of "NaN%". Treat it as "not measured" and hold the card's last figures.
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
  // Another client restarting the same GGUF job is exactly what resetMonotonic signals, and it
  // already clears the byte counters. The GGUF high-water mark carried the OLD generation's
  // fraction over, so a retry starting at 0 B sat pinned at the previous run's 99% for its
  // whole life -- the stale card this path exists to remove.
  const resolved = resolveProgressUpdate(job({ fraction: 0.99 }), emptyReading(), {
    resetMonotonic: true,
  });

  assert.equal(resolved.downloadedBytes, 0);
  assert.equal(resolved.fraction, 0);

  // Without the reset the mark still holds, which is what keeps a sibling-quant dip off the bar.
  const held = resolveProgressUpdate(job({ fraction: 0.99 }), emptyReading());
  assert.equal(held.fraction, 0.99);
});

test("a held transfer reading is reported as held, not as a measurement", () => {
  // The card may keep drawing the last figure, but a consumer that subtracts it
  // from the CURRENT expectedBytes is mixing two plans: after an XET-to-HTTP
  // reclaim the retry's first reading is a legitimate zero against a shrunken
  // total, and the bytes held behind it belong to the previous, larger one.
  const held = resolveProgressUpdate(
    job({ downloadedBytes: 3 * GB, completedBytes: 3 * GB }),
    emptyReading({ expected_bytes: GB / 2 }),
  );

  assert.equal(held.measuredTransfer, false);
  assert.equal(held.downloadedBytes, 3 * GB);
  assert.equal(held.expected, GB / 2);

  // The next poll that carries bytes is a measurement again.
  const measured = resolveProgressUpdate(
    job({ downloadedBytes: 3 * GB, completedBytes: 3 * GB }),
    emptyReading({ downloaded_bytes: GB / 10, expected_bytes: GB / 2 }),
  );

  assert.equal(measured.measuredTransfer, true);
  assert.equal(measured.downloadedBytes, GB / 10);

  // A generation bump measures the new run's counter outright, zero included.
  const reset = resolveProgressUpdate(job({ downloadedBytes: 3 * GB }), emptyReading(), {
    resetMonotonic: true,
  });

  assert.equal(reset.measuredTransfer, true);
  assert.equal(reset.downloadedBytes, 0);
});
