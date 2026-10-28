// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { registerBundlerResolver } from "./helpers/kit.ts";
registerBundlerResolver();
const { stagedEntryTransferredBytes } = await import(
  "../src/features/hub/download-manager/staged-download-progress.ts"
);

const GB = 1e9;
// The FLUX.3 Balanced plan: 11.1 GB of missing transformer shards in a 16.7 GB file scope whose encoder and VAE
// (5.6 GB) are already cached. A job resuming a cancelled attempt measures the whole scope.
const missing = 11.1 * GB;
const scope = 16.7 * GB;

test("cached files in the job's scope do not count as staged progress", () => {
  assert.equal(
    stagedEntryTransferredBytes(missing, {
      expectedBytes: scope,
      downloadedBytes: 5.6 * GB,
    }),
    0,
  );
  const halfway = stagedEntryTransferredBytes(missing, {
    expectedBytes: scope,
    downloadedBytes: 5.6 * GB + missing / 2,
  });
  assert.ok(Math.abs(halfway - missing / 2) < 1, String(halfway));
  // The reading that showed "11 GB of 11 GB, 100%" while the panel had 15 GB of 17 GB.
  const late = stagedEntryTransferredBytes(missing, {
    expectedBytes: scope,
    downloadedBytes: 15 * GB,
  });
  assert.ok(late < missing && Math.abs(late - 9.4 * GB) < 1, String(late));
  assert.equal(
    stagedEntryTransferredBytes(missing, {
      expectedBytes: scope,
      downloadedBytes: scope,
    }),
    missing,
  );
});

test("an entry whose scope is all missing keeps the plain reading", () => {
  assert.equal(
    stagedEntryTransferredBytes(100, {
      expectedBytes: 100,
      downloadedBytes: 40,
    }),
    40,
  );
  // A job still seeded with the plan's own total.
  assert.equal(
    stagedEntryTransferredBytes(100, { expectedBytes: 0, downloadedBytes: 40 }),
    40,
  );
  assert.equal(
    stagedEntryTransferredBytes(100, {
      expectedBytes: 80,
      downloadedBytes: 90,
    }),
    90,
  );
  assert.equal(
    stagedEntryTransferredBytes(100, {
      expectedBytes: 100,
      downloadedBytes: 140,
    }),
    100,
  );
  assert.equal(stagedEntryTransferredBytes(100, null), 0);
});

test("a held sample keeps the last measured reading instead of pricing it against the new total", () => {
  // A retry's first poll reports 0 bytes against a smaller total; the job holds the previous 3 GB beside it.
  const held = { expectedBytes: 0.5 * GB, downloadedBytes: 3 * GB, measuredTransfer: false };
  assert.equal(stagedEntryTransferredBytes(0.5 * GB, held), 0);
  assert.equal(stagedEntryTransferredBytes(0.5 * GB, held, 0.2 * GB), 0.2 * GB);
  assert.equal(stagedEntryTransferredBytes(0.5 * GB, held, 0.9 * GB), 0.5 * GB);
  // A measured sample ignores `previous`.
  assert.equal(
    stagedEntryTransferredBytes(100, { expectedBytes: 100, downloadedBytes: 40, measuredTransfer: true }, 90),
    40,
  );
});
