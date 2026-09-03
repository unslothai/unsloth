// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The shared presentation core, and the property that justifies it: the Load
// Model panel and the Hub memory bar cannot describe one load differently.
//
// Before this module the two surfaces disagreed in every category that decides
// what a user reads -- unit labels, rounding, fit thresholds, the fraction of a
// card a load may claim, and the verdict vocabulary itself -- while being fed
// byte-identical figures from one backend planner. Each difference on its own
// was defensible; together they meant the same model could read as fitting on
// one screen and not on the other.
//
// These tests are written against the SHARED module rather than through either
// surface, because a test that goes through one surface only proves that
// surface is self-consistent, which was never the problem.

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

const GIB = 1024 ** 3;

// ---------------------------------------------------------------------------
// Units

test("every formatter names a BINARY unit, because every divide is binary", () => {
  // The whole point of the rename. Each of these divides by 1024, so each must
  // say so; the panel's old formatter divided by 1024**3 and said "GB".
  assert.equal(formatGiB(7.24), "7.2 GiB");
  assert.equal(formatGiB(24), "24 GiB");
  assert.equal(formatBytesGiB(24 * GIB), "24.00 GiB");
  assert.equal(formatKvRate(6234), "6.1 KiB");
  assert.equal(formatKvRate(1024 * 1024 * 3), "3.0 MiB");
});

test("the two size formatters take DIFFERENT units, and say so in their names", () => {
  // This is the bug the rename exists to prevent: the old pair were both called
  // formatMemoryGb, both `(number) => string`, and one took bytes while the
  // other took gibibytes. Passing the wrong one was off by 1024^3 and
  // typechecked cleanly.
  const oneGibAsBytes = GIB;
  const oneGib = 1;
  assert.equal(formatBytesGiB(oneGibAsBytes), "1.00 GiB");
  assert.equal(formatGiB(oneGib), "1.0 GiB");
  // Feeding bytes to the gibibyte formatter is the mistake, and it produces an
  // absurd number rather than a plausible one, which is the best available
  // outcome now that the names differ.
  assert.notEqual(formatGiB(oneGibAsBytes), "1.0 GiB");
});

test("no formatter renders a number that does not exist", () => {
  // Every figure here comes off the wire. "NaN GiB" and "-3.0 GiB" both read as
  // measurements rather than as the missing readings they are.
  for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, -5, undefined]) {
    assert.equal(formatGiB(bad as number), "0 GiB");
    assert.equal(formatBytesGiB(bad as number), "0.00 GiB");
    assert.equal(formatKvRate(bad as number), "0 KiB");
  }
});

// ---------------------------------------------------------------------------
// Thresholds

test("the budget fraction matches what the loader admits at", () => {
  // 0.97 is _CTX_FIT_VRAM_FRACTION in core/inference/llama_cpp.py. A verdict
  // drawn against a different fraction than admission uses is wrong by
  // construction, in whichever direction it differs.
  assert.equal(DEFAULT_VRAM_BUDGET_FRACTION, 0.97);
  // Specifically NOT 0.90, which the bar used. llama_cpp.py records that value
  // as measured and reverted: "0.90 dropped 91-94% fits to CPU offload, #5106".
  assert.notEqual(DEFAULT_VRAM_BUDGET_FRACTION, 0.9);
});

test("the fit threshold and the pressure ramp stay distinct", () => {
  // These measure different things -- "will it fit" against "how full does it
  // look" -- so they are allowed to differ. What they are NOT allowed to do is
  // differ per surface, which is what collecting them here prevents.
  assert.equal(MEMORY_FIT_TIGHT_RATIO, 0.85);
  assert.equal(PRESSURE_HIGH_PCT, 80);
  assert.equal(PRESSURE_CRITICAL_PCT, 90);
  assert.ok(PRESSURE_HIGH_PCT < PRESSURE_CRITICAL_PCT);
});

// ---------------------------------------------------------------------------
// Verdicts

test("a figure that does not exist is never a confident fit", () => {
  // `<= 0` alone does not cover it: NaN fails every comparison, so `NaN <= 0` is
  // false and the ratio test falls through to "fits" -- a green verdict printed
  // from a number that is not there. JSON.parse turns 1e999 into Infinity, so a
  // malformed response reaches this without trying.
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
  // One half being unmeasurable must not silently improve the other half's
  // answer.
  assert.equal(worseMemoryFit("unknown", "exceeds"), "exceeds");
  assert.equal(worseMemoryFit("fits", "tight"), "tight");
  assert.equal(worseMemoryFit("tight", "exceeds"), "exceeds");
});

test("neither surface loses a distinction it used to make", () => {
  // The panel's contribution: TIGHT, the warning before anything is wrong.
  assert.equal(classifyMemoryFit(21 * GIB, 24), "tight");
  // The bar's contribution: WHY it exceeds, which decides the remedy.
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
  // The bar has no "tight" status; it expresses the same band through its
  // pressure ramp. Folding it into "fits" here loses nothing that renders.
  assert.equal(toModelMemoryStatus({ verdict: "tight", cause: null }), "fits");
  assert.equal(toModelMemoryStatus({ verdict: "fits", cause: null }), "fits");
  assert.equal(toModelMemoryStatus({ verdict: "unknown", cause: null }), "unknown");
});

test("the bar's vocabulary round-trips, and is honest about what it loses", () => {
  for (const status of ["unknown", "fits", "context-exceeds", "model-exceeds"] as const) {
    assert.equal(toModelMemoryStatus(fromModelMemoryStatus(status)), status);
  }
  // The lossy direction, asserted so it is a known property rather than a
  // surprise: the bar never recorded "tight", so it cannot be recovered.
  assert.equal(fromModelMemoryStatus("fits").verdict, "fits");
});

// ---------------------------------------------------------------------------
// The fit badge draws against the same budget as the bar (Codex P2 on 9830)

test("the Hub fit badge and the memory bar share one budget constant", async () => {
  // These render on the SAME ROW, so two budgets meant two verdicts for one
  // model. gguf-fit.ts held 0.90 while the bar used the loader's 0.97, which is
  // what admission actually applies.
  const { VRAM_HEADROOM_RATIO } = await import("../src/lib/gguf-fit.ts");
  assert.equal(
    VRAM_HEADROOM_RATIO,
    DEFAULT_VRAM_BUDGET_FRACTION,
    "the badge and the bar are judging against different budgets again",
  );
});

test("aligning the constant narrows the badge/bar gap without closing it", async () => {
  // Honest about what the fix buys. Measured over 15-24 GiB on a 24 GiB card the
  // disagreement count goes 11/19 -> 8/19. The residual 8 are the ESTIMATOR
  // difference -- gguf-fit scores `size * 1.15 + 1 GB` while the bar uses the
  // planner's real figures -- so sharing a constant cannot remove them, and
  // claiming otherwise would be the wrong lesson to leave here.
  const { classifyGgufFit, requiredGgufMemoryGb } = await import(
    "../src/lib/gguf-fit.ts"
  );
  const bytes = 20 * 1024 ** 3;
  // The badge still scores a 20 GiB file at 24.0 GiB of "required" memory.
  assert.ok(
    requiredGgufMemoryGb(bytes) > 20,
    "the badge's heuristic no longer inflates, so this note is stale",
  );
  // So on a 24 GiB card it still refuses a file the bar happily fits.
  assert.notEqual(classifyGgufFit(bytes, { gpuGb: 24, systemRamGb: 64 }), "fits");
});

// ---------------------------------------------------------------------------
// The badge must score against the SAVED budget, not just the default
// (Codex P2 on 9830)

test("a saved VRAM Budget moves the badge, not only the bar", async () => {
  // Sharing the default constant fixed the loading and old-backend case. It did
  // not fix the case where the user has actually saved a fraction: the badge
  // still applied 0.97 while the bar beside it consumed the live value from
  // `use-model-memory.ts`. Measured on a 24 GiB card with a saved 0.90, where the
  // loader admits at 21.6 GiB and the default would score against 23.28 GiB.
  const { classifyGgufFit, requiredGgufMemoryGb } = await import(
    "../src/lib/gguf-fit.ts"
  );
  const bytes = 18 * 1024 ** 3;
  const required = requiredGgufMemoryGb(bytes);
  // Between the two budgets, which is the whole window in which they can differ.
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
  // `budgetFraction` arrives from a settings route. Absent is the normal state
  // during the first paint and the permanent state on a backend predating the
  // route, so it must mean "use the default" and not "the whole card" (which
  // would admit loads the loader refuses) or "0" (which would refuse all of
  // them).
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
  // And 1.0 is a legitimate saved value, not a rejected one: it means the user
  // allowed the whole card.
  assert.equal(
    classifyGgufFit(bytes, { gpuGb: 24, systemRamGb: 64, budgetFraction: 1 }),
    "fits",
  );
});

test("the badge's call sites read the live fraction", async () => {
  // The classifier taking a fraction is worthless if nothing passes one. Asserted
  // on source because these are .tsx call sites the runner cannot render.
  const { readFileSync } = await import("node:fs");
  const card = readFileSync(
    new URL(
      "../src/features/hub/catalog/gguf-download-card.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    card,
    /useVramBudgetFraction\(\)/,
    "the Hub card must read the saved budget, not rely on the default",
  );
  // Every fit-scoring call on the row, not just the badge: the sort ranks with
  // the same classifier and would otherwise order rows against a different line.
  const passes = card.match(/budgetFraction,/g) ?? [];
  assert.ok(
    passes.length >= 3,
    `expected the fraction at the badge, the sort and the menu; found ${passes.length}`,
  );
});

test("the offload band credits the budget, not the whole card", async () => {
  // Once layers spill, what the GPU can still hold is what it is ALLOWED to hold.
  // The reserve is precisely what the model and KV cache may not use, so adding
  // the raw card to the RAM allowance invented capacity the loader will not give.
  //
  // Driven at 0.80, the LEGAL minimum (VRAM_FRACTION_MIN in
  // vram_budget_settings.py). The originally reported example used 0.50, which
  // the settings route rejects, so the real defect is smaller than it looked --
  // 4.8 GiB of phantom GPU on a 24 GiB card -- and still large enough to mislabel
  // four whole quant sizes.
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
  // Deliberately NOT scored against the budget. `fits` already returns for
  // everything at or under the budget, so scoring this against the budget too
  // would make the branch dead code. The band means "over your budget, still
  // card-sized", which is a warning and not a promise of admission.
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
  // The Hub download card got the live fraction; the On Device card did not, and
  // it renders a memory bar (which uses the saved value) directly above a quant
  // menu sorted by classifyGgufFit (which did not). One card, two budgets.
  const { readFileSync } = await import("node:fs");
  const read = (rel: string) =>
    readFileSync(new URL(rel, import.meta.url), "utf8");
  for (const rel of [
    "../src/features/hub/catalog/gguf-download-card.tsx",
    "../src/features/hub/catalog/local-on-device-card.tsx",
  ]) {
    const source = read(rel);
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
  // loadVramBudgetSettings deliberately has no read-through cache and clears its
  // shared promise once each request settles, so it coalesces only callers whose
  // requests overlap in time. A Hub catalog mounts a card per repo progressively
  // through scrolling and filtering, so a per-card call is a GET per card, and a
  // 404 per card on a backend predating the route.
  const source = await import("node:fs").then(({ readFileSync }) =>
    readFileSync(
      new URL("../src/hooks/use-vram-budget-fraction.ts", import.meta.url),
      "utf8",
    ),
  );
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
  // And the cache must still be invalidated by a save, or a shared cache is just
  // a stale one.
  assert.match(
    source,
    /subscribeVramBudgetSettings\(\([\s\S]{0,200}?cachedFraction = settings\.fraction/,
    "the change event must refresh the cache, or a save never reaches the cards",
  );
});
