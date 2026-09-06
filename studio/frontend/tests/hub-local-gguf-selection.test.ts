// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { resolveLocalGgufVariant } = await import(
  "../src/features/hub/lib/gguf-variant-sort.ts"
);

const variants = [{ quant: "Q4_K_M" }, { quant: "Q8_0" }, { quant: "Q6_K" }];

test("an explicit Hub quant selection takes priority", () => {
  assert.equal(
    resolveLocalGgufVariant(variants, {
      selectedVariant: "q6_k",
      activeVariant: "Q8_0",
      defaultVariant: "Q4_K_M",
    })?.quant,
    "Q6_K",
  );
});

test("the resident quant takes priority over the repository default", () => {
  assert.equal(
    resolveLocalGgufVariant(variants, {
      activeVariant: "q8_0",
      defaultVariant: "Q4_K_M",
    })?.quant,
    "Q8_0",
  );
});

test("selection falls back through default, first variant, and empty state", () => {
  assert.equal(
    resolveLocalGgufVariant(variants, {
      activeVariant: "missing",
      defaultVariant: "q4_k_m",
    })?.quant,
    "Q4_K_M",
  );
  assert.equal(resolveLocalGgufVariant(variants, {})?.quant, "Q4_K_M");
  assert.equal(resolveLocalGgufVariant([], {}), null);
  assert.equal(resolveLocalGgufVariant(null, {}), null);
});
