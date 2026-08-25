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

test("a cancelled job keeps the remainder the backend measured", () => {
  // Progress is not reusable bytes: from huggingface_hub 1.18 the partial is
  // process-unique and unlinked in a finally, so an interrupted in-file transfer
  // is refetched whole. Subtracting the dead job's 17 GB reported "1.0 GB left"
  // for a resume that still has all 18 GB to fetch.
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
  // The XET attempt finalized 3 GB of a 3.5 GB quant, then fell back to HTTP.
  // The reclaim recomputes completed_baseline_bytes from disk, so the retry
  // reports a 0.5 GB total with 0.1 GB moved and completed_bytes 0. Taking the
  // max against the held 3 GB read "0 B left" with 0.4 GB still to fetch.
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
  // The reading right after the reclaim, before the HTTP run moves anything: a
  // real downloaded_bytes 0 against a 0.5 GB total, behind which
  // resolveProgressUpdate holds the dead run's 3 GB. Pricing the remainder off
  // that held figure read "0 B left" with all 0.5 GB still to fetch.
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
  // snapshot_progress nets the baseline out of both of the job's counters, so
  // 5 GB plan - 1 GB already on disk = a 4 GB job, 1 GB of which has arrived.
  const [row] = applyLiveGgufVariantStates(
    [variant({ size_bytes: 5 * GB, download_size_bytes: 5 * GB })],
    live({ expectedBytes: 4 * GB, transferredBytes: 1 * GB }) as never,
  );

  assert.equal(row.download_remaining_bytes, 3 * GB);
  // The catalog total still drives the size the row reports, untouched.
  assert.equal(row.download_size_bytes, 5 * GB);
});
