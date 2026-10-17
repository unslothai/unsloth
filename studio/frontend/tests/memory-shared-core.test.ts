// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  formatBytesGiB,
  formatGiB,
  formatKvRate,
} from "../src/lib/memory/format.ts";
import {
  DEFAULT_VRAM_BUDGET_FRACTION,
  MEMORY_FIT_TIGHT_RATIO,
  PRESSURE_CRITICAL_PCT,
  PRESSURE_HIGH_PCT,
} from "../src/lib/memory/thresholds.ts";
import {
  classifyMemoryFit,
  fromModelMemoryStatus,
  toModelMemoryStatus,
  worseMemoryFit,
} from "../src/lib/memory/verdict.ts";

import { readSrc } from "./helpers/kit.ts";

const GIB = 1024 ** 3;

test("every formatter names a BINARY unit, because every divide is binary", () => {
  // Each divides by 1024, so each must name a binary unit.
  assert.equal(formatGiB(7.24), "7.2 GiB");
  assert.equal(formatGiB(24), "24 GiB");
  assert.equal(formatBytesGiB(24 * GIB), "24.00 GiB");
  assert.equal(formatKvRate(6234), "6.1 KiB");
  assert.equal(formatKvRate(1024 * 1024 * 3), "3.0 MiB");
});

test("the two size formatters take DIFFERENT units, and say so in their names", () => {
  // The old pair shared a name and signature but took bytes vs GiB, off by 1024^3.
  const oneGibAsBytes = GIB;
  const oneGib = 1;
  assert.equal(formatBytesGiB(oneGibAsBytes), "1.00 GiB");
  assert.equal(formatGiB(oneGib), "1.0 GiB");
  assert.notEqual(formatGiB(oneGibAsBytes), "1.0 GiB");
});

test("no formatter renders a number that does not exist", () => {
  // Figures come off the wire; NaN or negative must not read as measurements.
  for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, -5, undefined]) {
    assert.equal(formatGiB(bad as number), "0 GiB");
    assert.equal(formatBytesGiB(bad as number), "0.00 GiB");
    assert.equal(formatKvRate(bad as number), "0 KiB");
  }
});

test("the budget fraction matches what the loader admits at", () => {
  // 0.97 is _CTX_FIT_VRAM_FRACTION in core/inference/llama_cpp.py; verdicts must match admission.
  assert.equal(DEFAULT_VRAM_BUDGET_FRACTION, 0.97);
  // Not 0.90: llama_cpp.py records it as reverted (dropped fits to CPU offload).
  assert.notEqual(DEFAULT_VRAM_BUDGET_FRACTION, 0.9);
});

test("the fit threshold and the pressure ramp stay distinct", () => {
  assert.equal(MEMORY_FIT_TIGHT_RATIO, 0.85);
  assert.equal(PRESSURE_HIGH_PCT, 80);
  assert.equal(PRESSURE_CRITICAL_PCT, 90);
  assert.ok(PRESSURE_HIGH_PCT < PRESSURE_CRITICAL_PCT);
});

test("a figure that does not exist is never a confident fit", () => {
  // NaN fails every comparison, so `<= 0` alone lets it fall through to "fits".
  assert.equal(classifyMemoryFit(Number.NaN, 24), "unknown");
  assert.equal(classifyMemoryFit(8 * GIB, Number.NaN), "unknown");
  assert.equal(classifyMemoryFit(Number.POSITIVE_INFINITY, 24), "unknown");
  assert.equal(classifyMemoryFit(8 * GIB, 0), "unknown");
});

test("the bands land where the thresholds say", () => {
  assert.equal(classifyMemoryFit(8 * GIB, 24), "fits");
  // Just over 85% of 24 GiB.
  assert.equal(classifyMemoryFit(21 * GIB, 24), "tight");
  assert.equal(classifyMemoryFit(25 * GIB, 24), "exceeds");
});

test("unknown never erases a real verdict", () => {
  assert.equal(worseMemoryFit("unknown", "exceeds"), "exceeds");
  assert.equal(worseMemoryFit("fits", "tight"), "tight");
  assert.equal(worseMemoryFit("tight", "exceeds"), "exceeds");
});

test("neither surface loses a distinction it used to make", () => {
  assert.equal(classifyMemoryFit(21 * GIB, 24), "tight");
  assert.equal(
    toModelMemoryStatus({ verdict: "exceeds", cause: "context" }),
    "context-exceeds",
  );
  assert.equal(
    toModelMemoryStatus({ verdict: "exceeds", cause: "irreducible" }),
    "model-exceeds",
  );
});

test("tight folds into fits for the bar, which colours that band instead", () => {
  // The bar has no "tight" status; its pressure ramp expresses that band.
  assert.equal(toModelMemoryStatus({ verdict: "tight", cause: null }), "fits");
  assert.equal(toModelMemoryStatus({ verdict: "fits", cause: null }), "fits");
  assert.equal(toModelMemoryStatus({ verdict: "unknown", cause: null }), "unknown");
});

test("the bar's vocabulary round-trips, and is honest about what it loses", () => {
  for (const status of ["unknown", "fits", "context-exceeds", "model-exceeds"] as const) {
    assert.equal(toModelMemoryStatus(fromModelMemoryStatus(status)), status);
  }
  assert.equal(fromModelMemoryStatus("fits").verdict, "fits");
});

test("the Hub fit badge and the memory bar share one budget constant", async () => {
  // These render on the same row, so they must share one budget.
  const { VRAM_HEADROOM_RATIO } = await import("../src/lib/gguf-fit.ts");
  assert.equal(
    VRAM_HEADROOM_RATIO,
    DEFAULT_VRAM_BUDGET_FRACTION,
    "the badge and the bar are judging against different budgets again",
  );
});

test("aligning the constant narrows the badge/bar gap without closing it", async () => {
  // Residual disagreement is the estimator difference, which a shared constant cannot remove.
  const { classifyGgufFit, requiredGgufMemoryGb } = await import(
    "../src/lib/gguf-fit.ts"
  );
  const bytes = 20 * 1024 ** 3;
  assert.ok(
    requiredGgufMemoryGb(bytes) > 20,
    "the badge's heuristic no longer inflates, so this note is stale",
  );
  assert.notEqual(classifyGgufFit(bytes, { gpuGb: 24, systemRamGb: 64 }), "fits");
});

test("a saved VRAM Budget moves the badge, not only the bar", async () => {
  // The badge must use the user's saved fraction, as the bar beside it does.
  const { classifyGgufFit, requiredGgufMemoryGb } = await import(
    "../src/lib/gguf-fit.ts"
  );
  const bytes = 18 * 1024 ** 3;
  const required = requiredGgufMemoryGb(bytes);
  assert.ok(
    required > 24 * 0.9 && required <= 24 * DEFAULT_VRAM_BUDGET_FRACTION,
    `fixture must sit between the two budgets; required=${required}`,
  );
  assert.equal(
    classifyGgufFit(bytes, { gpuGb: 24, systemRamGb: 64, budgetFraction: 0.9 }),
    "marginal",
    "a saved 0.90 must push this over the line the loader draws",
  );
  assert.equal(
    classifyGgufFit(bytes, { gpuGb: 24, systemRamGb: 64 }),
    "fits",
    "and without a saved fraction the default still applies, unchanged",
  );
});

test("an absent or unusable budget falls back rather than refusing everything", async () => {
  // Absent budgetFraction (first paint, old backend) must mean the default, not 1 or 0.
  const { classifyGgufFit } = await import("../src/lib/gguf-fit.ts");
  const bytes = 18 * 1024 ** 3;
  const withDefault = classifyGgufFit(bytes, { gpuGb: 24, systemRamGb: 64 });
  for (const bad of [undefined, 0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.equal(
      classifyGgufFit(bytes, {
        gpuGb: 24,
        systemRamGb: 64,
        budgetFraction: bad as number,
      }),
      withDefault,
      `budgetFraction=${String(bad)} must fall back to the shared default`,
    );
  }
  assert.equal(
    classifyGgufFit(bytes, { gpuGb: 24, systemRamGb: 64, budgetFraction: 1 }),
    "fits",
  );
});

test("the badge's call sites read the live fraction", async () => {
  // Asserted on source because the .tsx call sites cannot render here.
  const card = readSrc("features/hub/catalog/gguf-download-card.tsx");
  assert.match(
    card,
    /useVramBudgetFraction\(\)/,
    "the Hub card must read the saved budget, not rely on the default",
  );
  // The sort uses the same classifier, so it must get the fraction too.
  const passes = card.match(/budgetFraction,/g) ?? [];
  assert.ok(
    passes.length >= 3,
    `expected the fraction at the badge, the sort and the menu; found ${passes.length}`,
  );
});

test("the offload band credits the budget, not the whole card", async () => {
  // Once layers spill, only the budgeted GPU share counts; driven at the legal minimum 0.80.
  const { classifyGgufFit, requiredGgufMemoryGb } = await import(
    "../src/lib/gguf-fit.ts"
  );
  const input = { gpuGb: 24, systemRamGb: 16, budgetFraction: 0.8 };
  for (const sizeGb of [23, 24, 25, 26]) {
    const bytes = sizeGb * 1024 ** 3;
    const required = requiredGgufMemoryGb(bytes);
    // Beyond what budget plus offloadable RAM can hold: 19.2 + 8 = 27.2.
    assert.ok(required > 27.2, `fixture ${sizeGb} must exceed the real ceiling`);
    assert.equal(
      classifyGgufFit(bytes, input),
      "oom",
      `a ${sizeGb} GiB quant needs ${required.toFixed(2)} GiB, and 0.80 of a ` +
        "24 GiB card plus half of 16 GiB of RAM cannot hold it",
    );
  }
});

test("marginal stays on the raw card, or the band cannot be reached", async () => {
  // Not scored against the budget, or this band would be dead code behind `fits`.
  const { classifyGgufFit } = await import("../src/lib/gguf-fit.ts");
  // 20 GiB needs 24.00 GiB: over 0.80 of the card (19.2) and exactly at the card.
  assert.equal(
    classifyGgufFit(20 * 1024 ** 3, {
      gpuGb: 24,
      systemRamGb: 16,
      budgetFraction: 0.8,
    }),
    "marginal",
    "a load between the budget and the card must still be reachable",
  );
});

test("every fit-scoring surface reads the saved budget, not just the Hub card", async () => {
  // The On Device card shows a bar and a classifyGgufFit-sorted menu; both need the fraction.
  for (const rel of [
    "features/hub/catalog/gguf-download-card.tsx",
    "features/hub/catalog/local-on-device-card.tsx",
  ]) {
    const source = readSrc(rel);
    assert.match(
      source,
      /useVramBudgetFraction\(\)/,
      `${rel} scores GGUF fit but does not read the saved VRAM Budget`,
    );
    assert.match(
      source,
      /budgetFraction,/,
      `${rel} reads the fraction but never passes it to the classifier`,
    );
  }
});

test("the budget read is shared, not one request per mounted card", async () => {
  // loadVramBudgetSettings has no cache, so a per-card call would GET once per card.
  const source = readSrc("hooks/use-vram-budget-fraction.ts");
  assert.match(
    source,
    /let cachedFraction/,
    "the fraction must be cached module-wide; it is one host-wide value",
  );
  assert.match(
    source,
    /let inFlight/,
    "cards mounting in the same tick must share one request",
  );
  assert.match(
    source,
    /routeAbsent/,
    "a 404 must be remembered, or every card retries a route that does not exist",
  );
  assert.match(
    source,
    /subscribeVramBudgetSettings\(\([\s\S]{0,200}?cachedFraction = settings\.fraction/,
    "the change event must refresh the cache, or a save never reaches the cards",
  );
});

test("a variant's fit counts the companions fetched with it", async () => {
  // Bartowski Muse Glimmer Q3_K_S: weights, projector, DFlash drafter; only the weights fit 16 GiB.
  const { classifyGgufVariantFit, ggufVariantFitSizeBytes } = await import(
    "../src/lib/gguf-fit.ts"
  );
  const weights = 12_789_199_648;
  const variant = {
    size_bytes: weights,
    download_size_bytes: weights + 3_849_173_920 + 1_451_094_176,
  };
  const budget = { gpuGb: 16, systemRamGb: 32 };

  assert.equal(ggufVariantFitSizeBytes(variant), variant.download_size_bytes);
  assert.equal(
    classifyGgufVariantFit({ size_bytes: weights }, budget),
    "fits",
    "the fixture only proves anything while the weights alone still fit",
  );
  assert.equal(classifyGgufVariantFit(variant, budget), "partial");
});

test("a total below the weights never lowers the estimate", async () => {
  // A positive-but-smaller total is the case a `??` or `||` fallback lets through.
  const { ggufVariantFitSizeBytes } = await import("../src/lib/gguf-fit.ts");
  const bytes = 12 * GIB;
  for (const download_size_bytes of [undefined, 0, bytes - 1, 1]) {
    assert.equal(
      ggufVariantFitSizeBytes({ size_bytes: bytes, download_size_bytes }),
      bytes,
      `a total of ${download_size_bytes} must not score below the weights`,
    );
  }
});
