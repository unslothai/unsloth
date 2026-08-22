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
  // The row still renders the bar while cancelling, above a "Cancelling..."
  // status; an animated "Transferring..." there would contradict it.
  assert.equal(
    isIndeterminateProgress({ downloadedBytes: 0, fraction: 0 }, true),
    false,
  );
});
