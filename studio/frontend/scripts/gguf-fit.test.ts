// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { classifyGgufFit, ggufFitTier } from "../src/lib/gguf-fit.ts";

const oneGiB = 1024 ** 3;

test("GGUF CPU-only hosts use RAM fallback instead of GPU fit", () => {
  const fit = classifyGgufFit(4 * oneGiB, {
    gpuGb: 0,
    systemRamGb: 16,
  });

  assert.equal(fit, "ram");
  assert.equal(ggufFitTier(fit), "tight");
});

test("GGUF CPU-only hosts exceed budget when RAM is insufficient", () => {
  assert.equal(
    classifyGgufFit(12 * oneGiB, {
      gpuGb: 0,
      systemRamGb: 8,
    }),
    "oom",
  );
});

test("GGUF GPU hosts distinguish partial RAM offload from full offload", () => {
  assert.equal(
    classifyGgufFit(8 * oneGiB, {
      gpuGb: 8,
      systemRamGb: 16,
    }),
    "partial",
  );
});
