// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { applyLiveGgufVariantStates, createLiveGgufVariantStatesSelector } =
  await import("../src/features/hub/catalog/gguf-live-variant-states.ts");
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
        measuredTransfer: true,
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

test("a cancelled job keeps the remainder the backend measured", () => {
  // From huggingface_hub 1.18 the partial is process-unique and unlinked, so it is refetched whole.
  for (const state of ["cancelled", "error"]) {
    const [row] = applyLiveGgufVariantStates(
      [
        variant({
          size_bytes: 18 * GB,
          download_size_bytes: 18 * GB,
          partial: true,
          download_remaining_bytes: 18 * GB,
        }),
      ],
      live({
        state,
        expectedBytes: 18 * GB,
        transferredBytes: 17 * GB,
      }) as never,
    );

    assert.equal(row.partial, true);
    assert.equal(row.download_remaining_bytes, 18 * GB);
    assert.equal(ggufVariantTransferLabel(row), "18 GB left");
  }
});

test("an XET fallback does not price the retry against the dead run's bytes", () => {
  // After an XET-to-HTTP fallback the reclaim recomputes the baseline from disk.
  const select = createLiveGgufVariantStatesSelector("unsloth/model-GGUF");
  const states = select({
    jobs: {
      "model:unsloth/model-GGUF:Q4_K_M": {
        kind: "model",
        repoId: "unsloth/model-GGUF",
        variant: "Q4_K_M",
        state: "running",
        expectedBytes: 0.5 * GB,
        downloadedBytes: 0.1 * GB,
        completedBytes: 3 * GB,
        startedAt: 0,
      },
    },
  } as never);

  const [row] = applyLiveGgufVariantStates(
    [
      variant({
        size_bytes: 3.5 * GB,
        download_size_bytes: 3.5 * GB,
        partial: true,
        download_remaining_bytes: 3.5 * GB,
      }),
    ],
    states as never,
  );

  assert.equal(row.download_remaining_bytes, 0.4 * GB);
});

test("a retry that has not measured a byte yet keeps the backend remainder", () => {
  const select = createLiveGgufVariantStatesSelector("unsloth/model-GGUF");
  const states = select({
    jobs: {
      "model:unsloth/model-GGUF:Q4_K_M": {
        kind: "model",
        repoId: "unsloth/model-GGUF",
        variant: "Q4_K_M",
        state: "running",
        expectedBytes: 0.5 * GB,
        downloadedBytes: 3 * GB,
        measuredTransfer: false,
        completedBytes: 3 * GB,
        startedAt: 0,
      },
    },
  } as never);

  const [row] = applyLiveGgufVariantStates(
    [
      variant({
        size_bytes: 3.5 * GB,
        download_size_bytes: 3.5 * GB,
        partial: true,
        download_remaining_bytes: 0.5 * GB,
      }),
    ],
    states as never,
  );

  assert.equal(row.download_remaining_bytes, 0.5 * GB);
  assert.equal(ggufVariantTransferLabel(row), "500 MB left");
});

test("the retry prices itself off the first reading that does measure", () => {
  const select = createLiveGgufVariantStatesSelector("unsloth/model-GGUF");
  const states = select({
    jobs: {
      "model:unsloth/model-GGUF:Q4_K_M": {
        kind: "model",
        repoId: "unsloth/model-GGUF",
        variant: "Q4_K_M",
        state: "running",
        expectedBytes: 0.5 * GB,
        downloadedBytes: 0.1 * GB,
        measuredTransfer: true,
        completedBytes: 0,
        startedAt: 0,
      },
    },
  } as never);

  const [row] = applyLiveGgufVariantStates(
    [
      variant({
        size_bytes: 3.5 * GB,
        download_size_bytes: 3.5 * GB,
        partial: true,
        download_remaining_bytes: 0.5 * GB,
      }),
    ],
    states as never,
  );

  assert.equal(row.download_remaining_bytes, 0.4 * GB);
});

test("a variant with no live job is returned untouched", () => {
  const original = variant({ partial: true, download_remaining_bytes: 2 * GB });
  const [row] = applyLiveGgufVariantStates([original], new Map() as never);

  assert.equal(row, original);
});

test("a reused mmproj does not come back as bytes still to fetch", () => {
  const [row] = applyLiveGgufVariantStates(
    [variant({ size_bytes: 5 * GB, download_size_bytes: 5 * GB })],
    live({ expectedBytes: 4 * GB, transferredBytes: 1 * GB }) as never,
  );

  assert.equal(row.download_remaining_bytes, 3 * GB);
  assert.equal(row.download_size_bytes, 5 * GB);
});
