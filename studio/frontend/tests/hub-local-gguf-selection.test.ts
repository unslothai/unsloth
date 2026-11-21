// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { resolveLocalGgufVariant, sortLocalGgufVariants } = await import(
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

test("blank preferences preserve the first-variant fallback", () => {
  const withBlankQuant = [{ quant: "Q4_K_M" }, { quant: "" }];
  assert.equal(resolveLocalGgufVariant(withBlankQuant, {})?.quant, "Q4_K_M");
  assert.equal(
    resolveLocalGgufVariant(withBlankQuant, {
      selectedVariant: null,
      activeVariant: null,
      defaultVariant: null,
    })?.quant,
    "Q4_K_M",
  );
  assert.equal(
    resolveLocalGgufVariant(withBlankQuant, {
      selectedVariant: "   ",
      activeVariant: "",
    })?.quant,
    "Q4_K_M",
  );
  assert.equal(
    resolveLocalGgufVariant([{ quant: "" }, { quant: "Q8_0" }], {
      selectedVariant: "Q8_0",
    })?.quant,
    "Q8_0",
  );
});

const localVariants = [
  { quant: "", filename: "blank-key.gguf", size_bytes: 100 * 1024 ** 3 },
  { quant: "Q4_K_M", filename: "model-Q4_K_M.gguf", size_bytes: 4 * 1024 ** 3 },
  { quant: "Q8_0", filename: "model-Q8_0.gguf", size_bytes: 8 * 1024 ** 3 },
];

test("sorting and selection ignore blank defaults", () => {
  for (const defaultVariant of [undefined, null, "", " \t\n", "missing"]) {
    const sorted = sortLocalGgufVariants(localVariants, {
      defaultVariant,
      gpuGb: 24,
    });
    assert.deepEqual(sorted, [
      localVariants[2],
      localVariants[1],
      localVariants[0],
    ]);
    assert.equal(
      resolveLocalGgufVariant(sorted, { defaultVariant }),
      localVariants[2],
    );
  }
  assert.deepEqual(
    localVariants.map((variant) => variant.quant),
    ["", "Q4_K_M", "Q8_0"],
  );
});

test("blank selections preserve resident and default preferences after sorting", () => {
  const defaultVariant = " q4_k_m ";
  const sorted = sortLocalGgufVariants(localVariants, {
    defaultVariant,
    gpuGb: 24,
  });
  assert.equal(sorted[0], localVariants[1]);
  for (const selectedVariant of [undefined, null, "", " \t\n", "missing"]) {
    assert.equal(
      resolveLocalGgufVariant(sorted, {
        selectedVariant,
        activeVariant: " q8_0 ",
        defaultVariant,
      }),
      localVariants[2],
    );
    assert.equal(
      resolveLocalGgufVariant(sorted, {
        selectedVariant,
        activeVariant: " ",
        defaultVariant,
      }),
      localVariants[1],
    );
  }
});

test("a blank-quant variant remains eligible for the ranked fallback", () => {
  const blank = {
    quant: "",
    filename: "blank-key.gguf",
    size_bytes: 4 * 1024 ** 3,
  };
  const oversized = {
    quant: "Q8_0",
    filename: "model-Q8_0.gguf",
    size_bytes: 100 * 1024 ** 3,
  };
  const sorted = sortLocalGgufVariants([oversized, blank], { gpuGb: 24 });
  assert.equal(resolveLocalGgufVariant(sorted, {}), blank);
});

test("the memory bar reads the cached revision while an update is published", async () => {
  const { localGgufMemoryBarVariant } = await import(
    "../src/features/hub/lib/gguf-variant-sort.ts"
  );
  const { ggufVariantLookupTableBytes } = await import(
    "../src/lib/gguf-fit.ts"
  );
  const GiB = 1024 ** 3;
  const cached = {
    quant: "Q4_K_M",
    size_bytes: 70 * GiB,
    lookup_table_bytes: 20 * GiB,
  };
  // The merged row keeps the cached size but carries the published table.
  const merged = {
    ...cached,
    lookup_table_bytes: 30 * GiB,
    download_size_bytes: 80 * GiB,
    update_available: true,
  };
  const barVariant = localGgufMemoryBarVariant(merged, [
    { quant: "Q8_0", size_bytes: 1, lookup_table_bytes: 0 },
    cached,
  ]);
  assert.equal(barVariant?.size_bytes, 70 * GiB);
  assert.equal(ggufVariantLookupTableBytes(barVariant!), 20 * GiB);

  // No update, or no cached row, keeps the selected variant.
  const current = { ...cached, update_available: false };
  assert.equal(localGgufMemoryBarVariant(current, [cached]), current);
  assert.equal(localGgufMemoryBarVariant(merged, null), merged);
  assert.equal(localGgufMemoryBarVariant(null, [cached]), null);
});
