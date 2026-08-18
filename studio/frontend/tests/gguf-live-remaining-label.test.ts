// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A row that is downloading says "N left", so N has to follow the transfer. The live overlay
// only carried the expected size, leaving the label on whatever the one-time variant fetch had
// measured -- or on the full total for a download that started after it, which reads as no
// progress at all.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { applyLiveGgufVariantStates } = await import(
  "../src/features/hub/catalog/gguf-live-variant-states.ts"
);
const { ggufVariantTransferLabel } = await import(
  "../src/features/hub/lib/gguf-variant-sort.ts"
);

const GB = 1000 ** 3;

const variant = (over: Record<string, unknown> = {}) => ({
  quant: "Q4_K_M",
  filename: "model-Q4_K_M.gguf",
  size_bytes: 4 * GB,
  download_size_bytes: 4 * GB,
  download_remaining_bytes: null,
  downloaded: false,
  partial: false,
  ...over,
});

const live = (over: Record<string, unknown> = {}) =>
  new Map([
    [
      "q4_k_m",
      {
        state: "running",
        expectedBytes: 4 * GB,
        transferredBytes: 3 * GB,
        startedAt: 0,
        ...over,
      },
    ],
  ]);

test("a running download prices the remainder from its own progress", () => {
  const [row] = applyLiveGgufVariantStates([variant()], live() as never);

  assert.equal(row.partial, true);
  assert.equal(row.download_remaining_bytes, 1 * GB);
  assert.equal(ggufVariantTransferLabel(row), "1.0 GB left");
});

test("progress replaces the remainder the one-time fetch measured", () => {
  // The fetch saw 3.5 GB outstanding; 3 GB have since arrived.
  const [row] = applyLiveGgufVariantStates(
    [variant({ partial: true, download_remaining_bytes: 3.5 * GB })],
    live() as never,
  );

  assert.equal(row.download_remaining_bytes, 1 * GB);
});

test("a job that has moved no bytes keeps the measured remainder", () => {
  const [row] = applyLiveGgufVariantStates(
    [variant({ partial: true, download_remaining_bytes: 3.5 * GB })],
    live({ transferredBytes: 0 }) as never,
  );

  assert.equal(row.download_remaining_bytes, 3.5 * GB);
});

test("the remainder never goes negative when progress overruns the estimate", () => {
  const [row] = applyLiveGgufVariantStates(
    [variant()],
    live({ transferredBytes: 5 * GB }) as never,
  );

  assert.equal(row.download_remaining_bytes, 0);
});

test("a completed download is left alone, so no row reads as partial", () => {
  const [row] = applyLiveGgufVariantStates(
    [variant({ partial: true, download_remaining_bytes: 3.5 * GB })],
    live({ state: "complete", transferredBytes: 4 * GB }) as never,
  );

  assert.equal(row.downloaded, true);
  assert.equal(row.partial, false);
  assert.equal(ggufVariantTransferLabel(row), "4.0 GB");
});

test("a variant with no live job is returned untouched", () => {
  const original = variant({ partial: true, download_remaining_bytes: 2 * GB });
  const [row] = applyLiveGgufVariantStates([original], new Map() as never);

  assert.equal(row, original);
});
