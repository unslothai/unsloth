// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The download card defaults to the FIRST variant of the sorted list, so the
 * comparator's tie-break direction is a product decision: fitting tiers largest
 * first (best quality that fits), everything past memory smallest first
 * (closest to viable). The regression this file pins: when the refusal tier
 * moved from rank 3 to rank 4, the `=== 3` ascending special-case silently
 * flipped all-refused hosts to largest-first, and a machine with no measured
 * budget defaulted its card to the biggest quant.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { compareGgufVariantFitAndSize } = await import(
  "../src/features/hub/lib/gguf-variant-sort.ts"
);
type GgufVariantDetail = import("../src/features/hub/inventory/index.ts").GgufVariantDetail;

const GB = 1024 ** 3;

function variant(sizeGb: number): GgufVariantDetail {
  return { quant: `Q${sizeGb}`, size_bytes: sizeGb * GB } as GgufVariantDetail;
}

test("an unmeasured host sorts refused variants smallest first", () => {
  // No GPU and no RAM figures: every variant classifies as a refusal.
  const none = { gpuGb: 0, systemRamGb: 0 };
  const sorted = [variant(354), variant(73), variant(188)].sort((a, b) =>
    compareGgufVariantFitAndSize(a, b, none),
  );
  assert.deepEqual(
    sorted.map((v) => v.size_bytes / GB),
    [73, 188, 354],
  );
});

test("fitting tiers still sort largest first", () => {
  const roomy = { gpuGb: 80, systemRamGb: 160 };
  const sorted = [variant(10), variant(40), variant(20)].sort((a, b) =>
    compareGgufVariantFitAndSize(a, b, roomy),
  );
  assert.deepEqual(
    sorted.map((v) => v.size_bytes / GB),
    [40, 20, 10],
  );
});
