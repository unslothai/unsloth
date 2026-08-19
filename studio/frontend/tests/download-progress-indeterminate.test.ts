// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { isIndeterminateProgress } = await import(
  "../src/features/hub/download-manager/progress-reconcile.ts"
);

test("zero measured bytes and fraction reads as indeterminate", () => {
  // A Xet transfer commits bytes in batches: a small file reads 0 B for its
  // whole life. The card must show activity rather than a stuck 0% bar.
  assert.equal(
    isIndeterminateProgress({ downloadedBytes: 0, fraction: 0 }),
    true,
  );
});

test("the first measured byte is determinate", () => {
  assert.equal(
    isIndeterminateProgress({ downloadedBytes: 1, fraction: 0 }),
    false,
  );
});

test("a nonzero fraction alone is determinate too", () => {
  // resolveProgressUpdate keeps GGUF fractions monotonic, so a fraction can
  // lead the byte counter; that is a measurement, not a stall.
  assert.equal(
    isIndeterminateProgress({ downloadedBytes: 0, fraction: 0.4 }),
    false,
  );
});

test("a pending cancellation is not transfer activity", () => {
  // The row keeps rendering the bar while state is "cancelling", directly above
  // a "Cancelling..." status. An animated "Transferring..." there would say the
  // cancel did not take, so the measured reading stands instead.
  assert.equal(
    isIndeterminateProgress({ downloadedBytes: 0, fraction: 0 }, true),
    false,
  );
});
